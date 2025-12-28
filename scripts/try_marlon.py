import cv2
import numpy as np
import csv
from pathlib import Path
from dataclasses import dataclass

@dataclass
class QRROIConfig:
    patch_size: int = 256        # Größe der Ausschnitte
    overlap:  float = 0.5         # 50% Überlappungsd
    min_area: int = 500          # Mindestgröße eines weißen Objekts
    top_k: int = 50              # Maximale Anzahl an Patches pro Bild
    debug_view: bool = True      # Zeigt das Vorschaufenster

def pad_and_crop(img, cx, cy, size):
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + size, y0 + size
    h, w = img.shape[:2]
    
    pad_top = max(0, -y0)
    pad_bottom = max(0, y1 - h)
    pad_left = max(0, -x0)
    pad_right = max(0, x1 - w)
    
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])
        x0 += pad_left; x1 += pad_left; y0 += pad_top; y1 += pad_top
        
    return img[y0:y1, x0:x1]



def generate_patches(img, cand, cfg: QRROIConfig):
    patches = []
    S = cfg.patch_size

    # Fall A: Das Objekt ist kleiner oder gleich 256x256
    if cand["w"] <= S and cand["h"] <= S:
        patch = pad_and_crop(img, cand["cx"], cand["cy"], S)
        patches.append(patch)
    
    # Fall B: Das Objekt ist größer -> Durchführung des sliding windows
    else:
        # Schrittweite berechnen
        stride = int( S* (1 - cfg.overlap))

        for y_start in range(cand["y"], cand["y"]+ cand["h"] - S + stride, stride):
            for x_start in range(cand["x"], cand["x"] + cand["w"] - S + stride, stride):
                actual_x = min(x_start, cand["x"] + cand["w"] - S)
                actual_y = min(y_start, cand["y"] + cand["h"] - S)

                patch = img[actual_y: actual_y + S, actual_x: actual_x + S]
                patches.append(patch)

    return patches


def propose_qr_centers_with_debug(img, cfg: QRROIConfig):

    # 1. In Graustufen umwandeln
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Durchschnittliche Helligkeit berechnen
    avg_brightness = np.mean(gray)
    
    # 3. Dynamische Threshold-Logik
    # Standardmäßig 180, bei dunklen Bildern (Schnitt < 70) senken auf 100
    if avg_brightness < 70:
        thresh_val = 100
        status_text = f"DUNKEL (Thresh: {thresh_val})"
    else:
        thresh_val = 180
        status_text = f"NORMAL (Thresh: {thresh_val})"

    # 4. Maske erstellen
    _, bright_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # 5. Morphologie: QR-Code zu einem soliden weißen Block verschmelzen
    # Kernel-Größe (25, 25) für gute Clusterbildung
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

    # 6. Konturen auf der Maske finden
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    debug_canvas = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Nur Flächen nehmen, die groß genug für einen QR-Code sind
        if area < cfg.min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        
        candidates.append({"cx": cx, "cy": cy, 
                           "x": x, "y": y,
                           "w": w, "h": h,                           
                           "score": area})
        
        # Grünes Rechteck für die Vorschau zeichnen
        cv2.rectangle(debug_canvas, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # 7. Debug-Monitor zusammenbauen
    if cfg.debug_view:
        s = (640, 420)
        v_bright = cv2.resize(bright_mask, s)
        v_mask = cv2.resize(closed, s)
        v_res = cv2.resize(debug_canvas, s)
        
        # Info-Text auf das Ergebnisbild schreiben
        info = f"Avg Bright: {avg_brightness:.1f} | {status_text}"
        cv2.putText(v_res, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Horizontaler Stack: Maske | Cluster-Maske | Ergebnis mit Boxen
        display = np.hstack((cv2.cvtColor(v_bright, cv2.COLOR_GRAY2BGR), 
                             cv2.cvtColor(v_mask, cv2.COLOR_GRAY2BGR), 
                             v_res))
        
        cv2.imshow("Helligkeit | Cluster | Ergebnis", display)
        
        # Warten auf Tastendruck (beliebige Taste = weiter, 'q' = abbruch)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            return None 

    # Nach Größe sortieren (größte Flächen zuerst)
    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:cfg.top_k]


def main(input_dir, output_dir):
    cfg = QRROIConfig()
    in_path = Path(input_dir)
    out_root = Path(output_dir)
 
    img_files = [f for f in in_path.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    for img_file in img_files:
        img = cv2.imread(str(img_file))
        if img is None: continue
        
        # Die Detektion läuft weiterhin:
        centers = propose_qr_centers_with_debug(img, cfg)
        
        if centers is None: 
            print("Abbruch durch Nutzer.")
            break
        
        for i, cand in enumerate(centers):
            roi_patches = generate_patches(img, cand, cfg)

            # --- NEU: Visualisierung der tatsächlichen 256x256 Patches ---
            S = cfg.patch_size
            if cand["w"] <= S and cand["h"] <= S:
                # Zeichne ein gelbes Quadrat für zentrierte Patches
                x_draw = cand["cx"] - S // 2
                y_draw = cand["cy"] - S // 2
                cv2.rectangle(img, (x_draw, y_draw), (x_draw + S, y_draw + S), (0, 255, 255), 2)
            else:
                # Zeichne rote Quadrate für Sliding Window Patches
                # (Gleiche Logik wie in generate_patches)
                stride = int(S * (1 - cfg.overlap))
                for y_s in range(cand["y"], cand["y"] + cand["h"] - S + stride, stride):
                    for x_s in range(cand["x"], cand["x"] + cand["w"] - S + stride, stride):
                        ax = min(x_s, cand["x"] + cand["w"] - S)
                        ay = min(y_s, cand["y"] + cand["h"] - S)
                        cv2.rectangle(img, (ax, ay), (ax + S, ay + S), (0, 0, 255), 1)

            for patch_idx, patch in enumerate(roi_patches):
                # Hier kannst du mit print prüfen:
                # print(f"Patch Shape: {patch.shape}") # Sollte (256, 256, 3) sein
                pass

                # Zum Testen/Speichern:
                # cv2.imwrite(str(out_patches / patch_name), patch)
                # writer.writerow([img_file.name, patch_name, cand["cx"], cand["cy"], int(cand['score'])])
            
        print(f"[INFO] {img_file.name}: ROI {i} erzeugte {len(roi_patches)} Patches.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("example_pictures", "extraktion_ergebnis")