import cv2
import numpy as np
import csv
from pathlib import Path
from dataclasses import dataclass

@dataclass
class QRROIConfig:
    patch_size: int = 320        # Größe der Ausschnitte
    min_area: int = 500          # Mindestgröße eines weißen Objekts
    top_k: int = 25              # Maximale Anzahl an Patches pro Bild
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

def propose_qr_centers_with_debug(img, cfg: QRROIConfig):
    # 1. Maske für helle Bereiche (Papier des QR-Codes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # 2. Morphologie: Schließt Löcher (schwarze Pixel im QR), um Cluster zu bilden
    # Wir nehmen einen großen Kernel, damit der QR-Code als EIN weißer Klumpen erscheint
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Alle weißen Cluster finden
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    debug_canvas = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        
        candidates.append({"cx": cx, "cy": cy, "score": area})
        # Zeichne grünes Rechteck für die Vorschau
        cv2.rectangle(debug_canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 4. Anzeige & Warten auf Nutzer
    if cfg.debug_view:
        s = (640, 420)
        v_bright = cv2.resize(bright_mask, s)
        v_mask = cv2.resize(closed, s)
        v_res = cv2.resize(debug_canvas, s)
        
        display = np.hstack((cv2.cvtColor(v_bright, cv2.COLOR_GRAY2BGR), 
                             cv2.cvtColor(v_mask, cv2.COLOR_GRAY2BGR), 
                             v_res))
        
        cv2.imshow("Helligkeit | Verbundene Cluster | Vorschau ROIs", display)
        
        # WARTE AUF TASTENDRUCK
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27: # 'q' oder ESC zum Abbrechen
            return None 

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:cfg.top_k]


def propose_qr_centers_with_debug2(img, cfg: QRROIConfig):
    """
    Detektiert Kandidaten-Zentren basierend auf hellen Clustern.
    Passt den Schwellenwert automatisch an, wenn das Bild zu dunkel ist.
    """
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
        
        candidates.append({"cx": cx, "cy": cy, "score": area})
        
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
    out_patches = out_root / "patches"
    out_patches.mkdir(parents=True, exist_ok=True)
    
    meta_path = out_root / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_image", "patch_file", "cx", "cy", "area_size"])

        img_files = [f for f in in_path.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            if img is None: continue
            
            centers = propose_qr_centers_with_debug2(img, cfg)
            
            # Falls None zurückgegeben wurde (Nutzer hat 'q' gedrückt)
            if centers is None: 
                print("Abbruch durch Nutzer.")
                break
            
            for i, c in enumerate(centers):
                patch = pad_and_crop(img, c["cx"], c["cy"], cfg.patch_size)
                patch_name = f"{img_file.stem}_roi_{i:02d}.jpg"
                cv2.imwrite(str(out_patches / patch_name), patch)
                writer.writerow([img_file.name, patch_name, c["cx"], c["cy"], int(c['score'])])
            
            print(f"[OK] {img_file.name}: {len(centers)} ROIs extrahiert. Beliebige Taste fuer nächstes Bild...")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("example_pictures", "extraktion_ergebnis")