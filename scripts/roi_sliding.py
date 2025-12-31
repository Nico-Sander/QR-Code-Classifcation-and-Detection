import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass
class QRROIConfig:
    patch_size: int = 256
    max_slidingwindow_size = 1280
    overlap: float = 0.5
    min_area: int = 500
    top_k: int = 50
    debug_view: bool = True

def pad_and_crop(img, cx, cy, size):
    """Schneidet ein Quadrat aus und füllt Ränder bei Bedarf mit Schwarz auf."""
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + size, y0 + size
    h, w = img.shape[:2]
    
    pad_top = max(0, -y0)
    pad_bottom = max(0, y1 - h)
    pad_left = max(0, -x0)
    pad_right = max(0, x1 - w)
    
    # Falls wir außerhalb des Bildes sind, erweitern wir das Bild mit schwarzen Rändern
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])
        x0 += pad_left; x1 += pad_left; y0 += pad_top; y1 += pad_top
        
    patch = img[y0:y1, x0:x1]
    
    # Sicherheitscheck: Falls der Patch trotzdem leer ist (sollte nicht passieren)
    if patch.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return patch

def generate_patches(img, cand, cfg: QRROIConfig):
    """Erzeugt eine Liste von 256x256 Patches für einen Kandidaten."""
    patches = []
    S = cfg.patch_size
    
    # Fall A: Klein -> Zentrieren
    if cand["w"] <= S and cand["h"] <= S:
        patches.append(pad_and_crop(img, cand["cx"], cand["cy"], S))
    
    # Fall B: Groß -> Sliding Window
    else:   
        W = cfg.max_slidingwindow_size
        
        if not(cand["w"] <= W and cand["h"] <= W):
            max_size = max(cand["w"], cand["h"])
            S = int(np.ceil(max_size / 5))

        stride = max(1, int(S * (1 - cfg.overlap)))
        for y_s in range(cand["y"], cand["y"] + cand["h"] - S + stride, stride):
            for x_s in range(cand["x"], cand["x"] + cand["w"] - S + stride, stride):
                # Wir berechnen die obere linke Ecke des Fensters
                ax = min(x_s, cand["x"] + cand["w"] - S)
                ay = min(y_s, cand["y"] + cand["h"] - S)
                    
                # WICHTIG: Wir nutzen pad_and_crop mit dem Zentrum des Fensters,
                # um sicherzustellen, dass das Bild niemals leer ist!
                cx_patch = ax + S // 2
                cy_patch = ay + S // 2

                patch = pad_and_crop(img, cx_patch, cy_patch, S)
                if S != 256:
                    patch = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_AREA)

                patches.append(patch)
    return patches

def detect_candidates(img, cfg: QRROIConfig):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    thresh_val = 100 if avg_brightness < 70 else 180
    
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        if cv2.contourArea(cnt) < cfg.min_area: continue
        x, y, w, h = cv2.boundingRect(cnt)
        candidates.append({
            "cx": x + w // 2, 
            "cy": y + h // 2, 
            "x": x, "y": y, "w": w, "h": h, 
            "score": cv2.contourArea(cnt)
        })
    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:cfg.top_k], mask, closed

def main(input_dir, output_dir="extraktion_ergebnis"):
    cfg = QRROIConfig()
    in_path = Path(input_dir)
    out_root = Path(output_dir)
    out_patches = out_root / "patches"
    out_patches.mkdir(parents=True, exist_ok=True)

    img_files = [f for f in in_path.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    print(f"--- Starte Verarbeitung von {len(img_files)} Bildern ---")

    for img_file in img_files:
        img = cv2.imread(str(img_file))
        if img is None: continue
        
        candidates, mask, closed = detect_candidates(img, cfg)
        print(f"\n[BILD] {img_file.name}: {len(candidates)} ROIs gefunden.")
        
        display_img = img.copy()
        S = cfg.patch_size
        total_patches_for_image = 0

        for i, cand in enumerate(candidates):
            roi_patches = generate_patches(img, cand, cfg)
            total_patches_for_image += len(roi_patches)

            for p_idx, patch in enumerate(roi_patches):
                patch_name = f"{img_file.stem}_roi{i:02d}_p{p_idx:02d}_0.jpg"
                cv2.imwrite(str(out_patches / patch_name), patch)

            # Visualisierung zeichnen
            cv2.rectangle(display_img, (cand["x"], cand["y"]), 
                          (cand["x"] + cand["w"], cand["y"] + cand["h"]), (0, 255, 0), 2)
            
            if cand["w"] <= S and cand["h"] <= S:
                px, py = cand["cx"] - S // 2, cand["cy"] - S // 2
                cv2.rectangle(display_img, (px, py), (px + S, py + S), (0, 255, 255), 2)
            else:
                stride = int(S * (1 - cfg.overlap))
                for ys in range(cand["y"], cand["y"] + cand["h"] - S + stride, stride):
                    for xs in range(cand["x"], cand["x"] + cand["w"] - S + stride, stride):
                        ax, ay = min(xs, cand["x"] + cand["w"] - S), min(ys, cand["y"] + cand["h"] - S)
                        cv2.rectangle(display_img, (ax, ay), (ax + S, ay + S), (0, 0, 255), 1)

        print(f"-> Insgesamt {total_patches_for_image} Patches gespeichert.")

        if cfg.debug_view and len(candidates) > 0:
            res_small = cv2.resize(display_img, (800, 600))
            cv2.imshow("Vorschau", res_small)
            if cv2.waitKey(0) & 0xFF == ord('q'): return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("example_pictures")