import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_qr_candidates(image_path, debug=True):
    # 1. Bild laden
    img = cv2.imread(image_path)
    if img is None:
        print("Fehler: Bild konnte nicht geladen werden.")
        return []
    
    original = img.copy()
    
    # 2. Preprocessing
    # Graustufen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur (Entfernt Rauschen/Wassertropfen)
    # (5,5) ist die Kernelgröße. Bei viel Unschärfe/Regen evtl. auf (7,7) oder (9,9) erhöhen.
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Kanten detektieren (Canny)
    # Die Werte 50 und 150 sind Schwellenwerte für Hysteresis. 
    # Experimentiert hiermit! Bei schlechtem Licht evtl. senken.
    edges = cv2.Canny(blur, 50, 150)
    
    # Morphologische Operationen: Lücken schließen
    # Das macht die weißen Kanten "dicker", damit sich Konturen schließen
    kernel = np.ones((3, 3), np.uint8)
    # Dilate = Erweitern, Close = Schließen von Löchern innerhalb von Objekten
    morph = cv2.dilate(edges, kernel, iterations=2) 
    
    # 3. Konturen finden
    # RETR_EXTERNAL sucht nur die äußeren Konturen (wir wollen keine kleinen Details im QR Code, sondern den Rahmen)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    rois_coords = [] # Zum Zeichnen der Boxen im Debug-Modus
    
    # Parameter für Filterung (Müssen an eure Bilder angepasst werden!)
    min_area = 20   # Zu kleine Flecken ignorieren (in Pixeln)
    max_area = (img.shape[0] * img.shape[1]) * 0.9 # Wenn es fast das ganze Bild ist, ist es auch falsch
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area > min_area and area < max_area:
            # Annäherung an ein Polygon (um zu prüfen, ob es ca. 4 Ecken hat)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # Bounding Box berechnen
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # Filter-Logik:
            # 1. Hat es 4 Ecken? (QR Codes haben 4, aber durch Verzerrung manchmal mehr Punkte im Approx)
            # 2. Ist das Seitenverhältnis annähernd quadratisch? (0.5 bis 1.5 Toleranz für Neigung)
            # Wir sind hier lieber etwas großzügiger, das CNN sortiert später aus.
            if len(approx) >= 4 and len(approx) <= 10: 
                if 0.5 < aspect_ratio < 2.0:
                    
                    # ROI ausschneiden (mit kleinem Padding/Rand, damit nichts abgeschnitten wird)
                    pad = 10
                    x_start = max(0, x - pad)
                    y_start = max(0, y - pad)
                    x_end = min(img.shape[1], x + w + pad)
                    y_end = min(img.shape[0], y + h + pad)
                    
                    roi = original[y_start:y_end, x_start:x_end]
                    
                    # Speichern
                    rois.append(roi)
                    rois_coords.append((x, y, w, h))

    # 4. Visualisierung (Nur wenn debug=True)
    if debug:
        debug_img = original.copy()
        for (x, y, w, h) in rois_coords:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
        # Plotten mit Matplotlib
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        
        plt.subplot(1, 4, 2)
        plt.title("Edges (Canny)")
        plt.imshow(edges, cmap='gray')
        
        plt.subplot(1, 4, 3)
        plt.title("Morphology")
        plt.imshow(morph, cmap='gray')
        
        plt.subplot(1, 4, 4)
        plt.title(f"Detections ({len(rois)})")
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        
        plt.show()

    return rois

def extract_qr_candidates_v2(image_path, debug=True):
    img = cv2.imread(image_path)
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Morphologischer Gradient
    # Das berechnet die Differenz zwischen Dilation und Erosion.
    # Es hebt Bereiche hervor, in denen sich Helligkeit stark ändert (Kanten),
    # ignoriert aber glatte Flächen (Motorhaube, Himmel).
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_grad)

    # 2. Binarisierung (Otsu's Method)
    # Automatische Schwellenwert-Findung. Alles was "unruhig" ist, wird weiß.
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 3. Lücken schließen (Closing)
    # Jetzt verbinden wir die einzelnen QR-Pixel zu einem großen weißen Block.
    # Wir nehmen ein rechteckiges Kernel, da QR Codes rechteckig sind.
    # (21, 21) ist aggressiv, um den ganzen Code zu füllen.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # 4. Rauschen entfernen (Erosion)
    # Entfernt kleine weiße Punkte, die keine großen Blöcke sind.
    closed = cv2.erode(closed, None, iterations=4)
    # Dilation, um die verbliebenen Blöcke wieder auf Originalgröße zu bringen
    closed = cv2.dilate(closed, None, iterations=4)

    # 5. Konturen finden
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    rois_coords = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filterung anpassen: QR Codes sind meist signifikant groß
        if area > 1000: 
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            # QR Codes sind meist quadratisch (0.5 - 1.5 Toleranz)
            if 0.5 < aspect_ratio < 2.0:
                # Padding hinzufügen
                pad = 15
                x_start = max(0, x - pad)
                y_start = max(0, y - pad)
                x_end = min(img.shape[1], x + w + pad)
                y_end = min(img.shape[0], y + h + pad)

                roi = original[y_start:y_end, x_start:x_end]
                rois.append(roi)
                rois_coords.append((x, y, w, h))

    if debug:
        # Visualisierung der neuen Schritte
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.title("Gradient (Kanten)")
        plt.imshow(grad, cmap='gray')
        
        plt.subplot(1, 4, 2)
        plt.title("Otsu Binary")
        plt.imshow(binary, cmap='gray')
        
        plt.subplot(1, 4, 3)
        plt.title("Closed & Eroded (Blob)")
        plt.imshow(closed, cmap='gray')
        
        # Zeichne Boxen ins Original
        debug_img = original.copy()
        for (x, y, w, h) in rois_coords:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        plt.subplot(1, 4, 4)
        plt.title(f"Resultat ({len(rois)})")
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.show()

    return rois

def detect_rectangles_on_gradient(image_path, debug=True):
    img = cv2.imread(image_path)
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Textur hervorheben (Gradient)
    # Ein kleiner Kernel (3,3) reicht, um Kanten zu finden
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_grad)

    # 2. Weichzeichnen VOR dem Thresholding
    # Das ist der Trick: Wir verschmieren die Kanten des QR-Codes zu einer "Wolke".
    # Dadurch entsteht beim Thresholding eher eine Fläche als 1000 Pünktchen.
    # (9, 9) ist stark genug zum Verschmieren, aber erhält die grobe Form.
    blurred = cv2.blur(grad, (9, 9))

    # 3. Binarisierung (Otsu)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 4. Kleines Closing (Lücken füllen)
    # Statt (21, 21) nehmen wir hier weniger, z.B. (5, 5) oder (7, 7).
    # Wir wollen nur die "Löcher" im QR-Code füllen.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # 5. Konturen finden & Geometrie prüfen
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    rois_coords = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Winzige Flecken ignorieren
        if area > 500:
            
            # A. Polygon-Approximation (Wieviele Ecken hat das Ding?)
            peri = cv2.arcLength(cnt, True)
            # epsilon ist die Toleranz. Je größer, desto "eckiger" wird die Form gemacht.
            # 0.02 * Umfang ist ein guter Standardwert.
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # B. Geometrie-Check: Hat es 4 Ecken?
            # Wir erlauben 4 bis 6 Ecken, da durch Perspektive/Unschärfe oft eine Ecke "abgeschnitten" wirkt.
            if 4 <= len(approx) <= 6:
                
                # C. Check: Ist es konvex? (Ein QR Code hat keine Einbuchtungen nach innen)
                if cv2.isContourConvex(approx):
                    
                    # D. Bounding Box für Aspect Ratio Check
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    
                    # Ein QR-Code ist quadratisch (Aspect Ratio ~1.0)
                    # Wir erlauben Toleranz (0.5 bis 2.0) wegen Neigung der Scheibe
                    if 0.5 < aspect_ratio < 2.0:
                        
                        # E. Solidity Check (Optional, aber gut)
                        # Verhältnis von Konturfläche zur Fläche der BoundingBox.
                        # Ein Rechteck füllt seine Box fast ganz aus (> 0.7).
                        rect_area = w * h
                        solidity = area / float(rect_area)
                        
                        if solidity > 0.4: # Wert niedrig angesetzt wegen Rotation/Verzerrung
                            # --- TREFFER! ---
                            pad = 10
                            x_start = max(0, x - pad)
                            y_start = max(0, y - pad)
                            x_end = min(img.shape[1], x + w + pad)
                            y_end = min(img.shape[0], y + h + pad)
                            
                            roi = original[y_start:y_end, x_start:x_end]
                            rois.append(roi)
                            rois_coords.append((x, y, w, h))

    if debug:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.title("Blurred Gradient")
        plt.imshow(blurred, cmap='gray')
        
        plt.subplot(1, 4, 2)
        plt.title("Otsu Binary")
        plt.imshow(binary, cmap='gray')

        plt.subplot(1, 4, 3)
        plt.title("Closed (Small Kernel)")
        plt.imshow(closed, cmap='gray')

        debug_img = original.copy()
        for (x, y, w, h) in rois_coords:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Zeichne auch die approximierte Form (Poloygon) in Blau
            # cv2.drawContours(debug_img, [approx], -1, (255, 0, 0), 2) 

        plt.subplot(1, 4, 4)
        plt.title(f"Resultat ({len(rois)})")
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.show()

    return rois

# --- Deine Funktion (Korrigiert mit np.int64 und Logik) ---
def detect_blobs_robust(image_path, debug=True):
    img = cv2.imread(image_path)
    if img is None: return []
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Gradient & Blur (wie gehabt, das war gut)
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_grad)
    blurred = cv2.blur(grad, (3, 3))
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 2. WICHTIG: Trennen statt Verbinden!
    # Wir benutzen EROSION, um dünne Verbindungen (Brücken) zum Scheibenwischer zu kappen.
    # Dadurch schrumpfen die QR-Codes etwas, aber sie werden isoliert.
    kernel_erode = np.ones((2, 2), np.uint8)
    # Iterations=2 oder 3 probieren, bis die Verbindung bricht
    separated = cv2.erode(binary, kernel_erode, iterations=2)
    
    # Optional: Danach wieder etwas aufblasen (Dilation), um die Originalgröße zu bekommen,
    # aber ohne die Brücken wiederherzustellen. (Das nennt man "Opening")
    separated = cv2.dilate(separated, kernel_erode, iterations=2)

    # 3. Konturen finden
    contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    rois_coords = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter: Zu klein/groß weg
        if area > 400 and area < (img.shape[0]*img.shape[1]*0.5):
            
            # NEUER ANSATZ: Rotated Rectangle (minAreaRect)
            # Das ist viel robuster als approxPolyDP. Es legt das kleinstmögliche 
            # gedrehte Rechteck um den Blob, egal wie ausgefranst er ist.
            rect = cv2.minAreaRect(cnt) # Liefert ((center_x, center_y), (width, height), angle)
            (center, (w, h), angle) = rect
            
            # Verhindern von Division durch Null
            if w == 0 or h == 0: continue

            # A. Aspect Ratio Check (Quadratisch?)
            # minAreaRect sortiert w/h manchmal nach Länge, daher ist w/h oder h/w möglich.
            # Wir nehmen einfach min/max.
            aspect_ratio = min(w, h) / max(w, h)
            
            # Ein QR Code ist quadratisch -> Ratio nahe 1. 
            # Wir erlauben bis 0.4 (sehr schräg)
            if aspect_ratio > 0.4:
                
                # B. Extent / Füllgrad
                # Wie viel von dem gedrehten Rechteck ist wirklich mit weißen Pixeln gefüllt?
                # Ein Rechteck-Blob füllt sein minAreaRect fast ganz aus.
                # Ein "L"-förmiger Blob oder ein Kreis füllt es schlechter aus.
                box_area = w * h
                extent = area / float(box_area)
                
                # Wenn > 0.6, dann ist der Blob "kastenförmig" genug
                if extent > 0.6:
                    
                    # --- TREFFER ---
                    # Box Punkte berechnen für Visualisierung
                    box = cv2.boxPoints(rect)
                    box = np.int64(box)
                    
                    # ROI ausschneiden (gerades Bounding Rect für einfacheres Cropping)
                    x, y, w_straight, h_straight = cv2.boundingRect(cnt)
                    
                    pad = 15
                    x_start = max(0, x - pad)
                    y_start = max(0, y - pad)
                    x_end = min(img.shape[1], x + w_straight + pad)
                    y_end = min(img.shape[0], y + h_straight + pad)
                    
                    roi = original[y_start:y_end, x_start:x_end]
                    rois.append(roi)
                    rois_coords.append(box) # Wir speichern die gedrehte Box zum Zeichnen

    if debug:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Otsu Binary (Verklebt?)")
        plt.imshow(binary, cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title("Separated (Erode/Open)")
        plt.imshow(separated, cmap='gray')

        debug_img = original.copy()
        for box in rois_coords:
            cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 3)

        plt.subplot(1, 3, 3)
        plt.title(f"Resultat ({len(rois)})")
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.show()

    return rois



def detect_blobs_robust_with_clahe(image_path, debug=True):
    img = cv2.imread(image_path)
    if img is None: return []
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # NEU: CLAHE Implementierung
    # ---------------------------------------------------------
    # Wir erhöhen den lokalen Kontrast VOR der Gradientenberechnung.
    # Das macht die Kanten im QR-Code "aggressiver", selbst bei Spiegelungen.
    # clipLimit=2.0 ist Standard, bei Bedarf auf 3.0 oder 4.0 erhöhen.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # ---------------------------------------------------------

    # 1. Gradient & Blur
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Jetzt wird der Gradient auf dem kontrastverstärkten Bild berechnet
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_grad)
    
    blurred = cv2.blur(grad, (9, 9))
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 2. Trennen (Erosion) und Reinigen (Opening)
    # Brücken zum Scheibenwischer kappen
    kernel_erode = np.ones((3, 3), np.uint8)
    separated = cv2.erode(binary, kernel_erode, iterations=2)
    
    # Wieder auf Originalgröße bringen (optional, hier als Opening zusammengefasst)
    separated = cv2.dilate(separated, kernel_erode, iterations=2)

    # 3. Konturen finden
    contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    rois_coords = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter: Zu klein/groß weg
        if area > 400 and area < (img.shape[0]*img.shape[1]*0.5):
            
            # Rotated Rectangle (minAreaRect)
            rect = cv2.minAreaRect(cnt) 
            (center, (w, h), angle) = rect
            
            if w == 0 or h == 0: continue

            # A. Aspect Ratio Check
            aspect_ratio = min(w, h) / max(w, h)
            
            # QR Code ist quadratisch -> Ratio nahe 1. 
            if aspect_ratio > 0.4:
                
                # B. Extent / Füllgrad
                box_area = w * h
                extent = area / float(box_area)
                
                # Wenn > 0.6, dann ist der Blob "kastenförmig" genug
                if extent > 0.6:
                    
                    # --- TREFFER ---
                    box = cv2.boxPoints(rect)
                    box = np.int64(box)
                    
                    # ROI ausschneiden (gerades Bounding Rect)
                    x, y, w_straight, h_straight = cv2.boundingRect(cnt)
                    
                    pad = 15
                    x_start = max(0, x - pad)
                    y_start = max(0, y - pad)
                    x_end = min(img.shape[1], x + w_straight + pad)
                    y_end = min(img.shape[0], y + h_straight + pad)
                    
                    roi = original[y_start:y_end, x_start:x_end]
                    rois.append(roi)
                    rois_coords.append(box)

    if debug:
        plt.figure(figsize=(15, 5))
        
        # Zeige hier zum Vergleich das CLAHE-Gradienten Bild
        plt.subplot(1, 3, 1)
        plt.title("Gradient (auf CLAHE Basis)")
        plt.imshow(gray, cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title("Separated (Binary)")
        plt.imshow(separated, cmap='gray')

        debug_img = original.copy()
        for box in rois_coords:
            cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 3)

        plt.subplot(1, 3, 3)
        plt.title(f"Resultat ({len(rois)})")
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.show()

    return rois

def analyze_steps(image_path):
    # 1. Bild laden
    img = cv2.imread(image_path)
    if img is None:
        print("Fehler: Bild konnte nicht geladen werden.")
        return
    
    # BGR zu RGB für Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ---------------------------------------------------------
    # SCHRITT 1: Graustufen & CLAHE (Kontrastverbesserung)
    # ---------------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE: Erhöht lokalen Kontrast. Das hilft extrem bei Reflexionen auf der Scheibe.
    # clipLimit begrenzt die Verstärkung von Rauschen.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)

    # ---------------------------------------------------------
    # SCHRITT 2: Adaptive Thresholding
    # ---------------------------------------------------------
    # Im Gegensatz zu Otsu wird der Schwellenwert für jeden kleinen Bereich (Block) berechnet.
    # blockSize=19: Größe der Nachbarschaft (muss ungerade sein).
    # C=5: Konstante, die vom Mittelwert abgezogen wird.
    # THRESH_BINARY_INV: Wir wollen weiße Features auf schwarzem Grund für die Konturensuche.
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_enhanced, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        29, # BlockSize: Größer = unempfindlicher gegen Rauschen, kleiner = mehr Details
        10   # C: Je höher, desto weniger Rauschen, aber feine Linien könnten verschwinden
    )

    # ---------------------------------------------------------
    # SCHRITT 3: Morphological Closing (Lücken schließen)
    # ---------------------------------------------------------
    # Ein QR-Code besteht aus vielen kleinen Punkten. Wir wollen aber EINEN großen Block.
    # Wir benutzen einen Kernel, um die Lücken zwischen den Punkten zu füllen.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) # Größe an Bildauflösung anpassen!
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    # ---------------------------------------------------------
    # SCHRITT 4: Konturen finden & Filtern
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = img_rgb.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        
        # Filter:
        # 1. Zu kleine Bereiche ignorieren (Rauschen)
        # 2. Zu große Bereiche ignorieren (ganzes Auto)
        # 3. Seitenverhältnis grob quadratisch (QR Codes sind meist quadratisch)
        if area > 500 and area < (img.shape[0]*img.shape[1] * 0.8): 
             # Wir zeichnen erstmal ALLES ein, was halbwegs groß genug ist (= hoher Recall)
             # Grün: Die gefundene Box
             cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # ---------------------------------------------------------
    # VISUALISIERUNG
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(gray_enhanced, cmap='gray')
    axes[0].set_title("1. CLAHE (Kontrast)")
    
    axes[1].imshow(adaptive_thresh, cmap='gray')
    axes[1].set_title("2. Adaptive Thresh (Inv)")
    
    axes[2].imshow(morph, cmap='gray')
    axes[2].set_title("3. Morph Close (Verbinden)")
    
    axes[3].imshow(result_img)
    axes[3].set_title("4. Ergebnis ROIs")
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def detect_qr_regions_high_recall(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None: return []
    
    original = img.copy()
    
    # --- STEP 1: PRE-PROCESSING ---
    # Convert to HSV to isolate "White-ish" things (The paper)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Sensitivity for "White": Low Saturation, High Value
    # S: 0-60 (very low color), V: 130-255 (bright)
    lower_white = np.array([0, 0, 130])
    upper_white = np.array([180, 80, 255])
    mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

    # --- STEP 2: ADAPTIVE THRESHOLDING ---
    # Good for glare/shadow variance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Block size 51, C=10 (tunable)
    thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 51, 10)

    # --- STEP 3: COMBINE & CLEAN ---
    # We want regions that are EITHER white-ish OR high local contrast
    # But usually, AND logic works better to remove noise, 
    # however, for High Recall, let's stick to the HSV mask mainly, 
    # refined by the threshold.
    
    # Let's combine: The region must be bright (HSV) AND high contrast (Adaptive)
    # This helps remove flat bright reflections.
    combined = cv2.bitwise_and(mask_hsv, thresh_adapt)

    # --- STEP 4: MORPHOLOGY (CRITICAL STEP) ---
    # 1. Close: Dilate then Erode. This fills the black gaps INSIDE the QR code 
    #    to make it a solid block.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. Open: Erode then Dilate. This removes thin connections (wipers).
    #    We use a smaller kernel here.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # --- STEP 5: CONTOURS ---
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    debug_boxes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Relaxed Area Filter
        img_area = img.shape[0] * img.shape[1]
        if area > 800 and area < (img_area * 0.4):
            
            rect = cv2.minAreaRect(cnt)
            (center, (w, h), angle) = rect
            
            # Sanity checks
            if w == 0 or h == 0: continue
            
            # Aspect Ratio (Relaxed: 0.3 to 3.0)
            # QR codes are 1:1, but perspective makes them skewed.
            aspect_ratio = min(w, h) / max(w, h)
            
            if aspect_ratio > 0.3: # Very loose to allow tilted codes
                
                # Extent check (how rectangular is the blob?)
                box_area = w * h
                extent = area / float(box_area)
                
                # Lowered extent threshold to 0.45 because glare 
                # cuts chunks out of the rectangle
                if extent > 0.45:
                    
                    # --- SUCCESS ---
                    box = cv2.boxPoints(rect)
                    box = np.int64(box)
                    debug_boxes.append(box)
                    
                    # Crop logic (same as before)
                    x, y, w_straight, h_straight = cv2.boundingRect(cnt)
                    pad = 20
                    x_start = max(0, x - pad)
                    y_start = max(0, y - pad)
                    x_end = min(img.shape[1], x + w_straight + pad)
                    y_end = min(img.shape[0], y + h_straight + pad)
                    
                    roi = original[y_start:y_end, x_start:x_end]
                    rois.append(roi)

    if debug:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.title("HSV Mask"); plt.imshow(mask_hsv, cmap='gray')
        plt.subplot(1, 3, 2); plt.title("Morph Cleaned"); plt.imshow(opened, cmap='gray')
        
        debug_img = original.copy()
        cv2.drawContours(debug_img, debug_boxes, -1, (0, 255, 0), 3)
        
        plt.subplot(1, 3, 3); plt.title(f"Result ({len(rois)})"); plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.show()

    return rois

def detect_via_variance(image_path):
    img = cv2.imread(image_path)
    if img is None: return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Calculate Standard Deviation in a sliding window
    # We define a window size (must be odd). 
    # 21x21 or 25x25 is usually good for QR codes at this resolution.
    ksize = 21
    
    # Calculate Mean and Mean-of-Squares to derive Variance/StdDev quickly
    img_float = gray.astype(np.float32)
    mu = cv2.blur(img_float, (ksize, ksize))
    mu2 = cv2.blur(img_float * img_float, (ksize, ksize))
    
    # Variance = E[X^2] - (E[X])^2
    variance = mu2 - mu * mu
    
    # Standard Deviation (scaling for visualization)
    std_dev = np.sqrt(np.abs(variance))
    
    # Normalize to 0-255 so we can threshold it
    std_dev_norm = cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Threshold the Variance
    # We only want areas with HIGH 'busyness'. 
    # Otsu usually works great on the variance map.
    _, binary_variance = cv2.threshold(std_dev_norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 3. Morphological Cleanup (Close gaps, remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    clean = cv2.morphologyEx(binary_variance, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

    # ... (From here, proceed to FindContours as usual) ...
    
    return clean # Returning the mask for you to visualizeHere is the complete Python code for the Variance/Texture Filter.



def detect_and_visualize_variance(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Compute Variance (Standard Deviation)
    # The QR code has high local "busyness" (texture), while glare/glass is smooth.
    # We calculate standard deviation in a 25x25 window.
    ksize = 25
    img_float = gray.astype(np.float32)
    
    # E[X]
    mu = cv2.blur(img_float, (ksize, ksize))
    # E[X^2]
    mu2 = cv2.blur(img_float * img_float, (ksize, ksize))
    
    # Variance = E[X^2] - (E[X])^2
    variance = mu2 - mu * mu
    
    # Standard Deviation = sqrt(Variance)
    # We clip negative values (floating point errors) to 0
    variance[variance < 0] = 0
    std_dev = np.sqrt(variance)

    # Normalize to 0-255 for thresholding
    std_dev_norm = cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Thresholding
    # Otsu works brilliantly here because the "texture" (QR) is distinct from "smooth" (Glass)
    _, binary_variance = cv2.threshold(std_dev_norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 4. Morphological Cleanup
    # "Close" (Dilate -> Erode) to fill gaps inside the QR code
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask_closed = cv2.morphologyEx(binary_variance, cv2.MORPH_CLOSE, kernel_close)
    
    # "Open" (Erode -> Dilate) to remove small noise (like edges of the car)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    final_mask = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)

    # --- VISUALIZATION ---
    plt.figure(figsize=(18, 6))

    # A. Original
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # B. The Variance Map (Heatmap of Texture)
    # Brighter pixels = More texture (QR code edges)
    plt.subplot(1, 3, 2)
    plt.title("Variance Map (Texture Heatmap)")
    plt.imshow(std_dev_norm, cmap='inferno') # 'inferno' makes high values glow nicely
    plt.axis('off')

    # C. Final Binary Mask
    plt.subplot(1, 3, 3)
    plt.title("Final Binary Mask")
    plt.imshow(final_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return final_mask

def detect_and_visualize_optimized(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    # --- STEP 1: COLOR FILTER (Find "White-ish" areas) ---
    # We use HSV to find things that are bright (High Value) and not colorful (Low Saturation).
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # S: 0-60 (very low color), V: 130-255 (bright)
    lower_white = np.array([0, 0, 130])
    upper_white = np.array([180, 60, 255])
    color_mask = cv2.inRange(hsv, lower_white, upper_white)

    # --- STEP 2: TEXTURE FILTER (Your Variance Method) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ksize = 25
    img_float = gray.astype(np.float32)
    mu = cv2.blur(img_float, (ksize, ksize))
    mu2 = cv2.blur(img_float * img_float, (ksize, ksize))
    variance = mu2 - mu * mu
    # Normalize and Threshold
    variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, variance_mask = cv2.threshold(variance_norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # --- STEP 3: COMBINE (The crucial optimization) ---
    # We keep pixels that are BOTH white AND highly textured.
    # This removes grass (textured but not white) and reflections (white but smooth).
    combined_mask = cv2.bitwise_and(color_mask, variance_mask)

    # --- STEP 4: MORPHOLOGY & GEOMETRIC FILTERING ---
    # 1. Close gaps to make the QR code a solid blob.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    morph_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

    # 2. Find Contours and filter by shape properties.
    contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(gray)
    final_boxes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter by Area
        if area < 500: continue 

        # Get rotated bounding box
        rect = cv2.minAreaRect(cnt)
        (center, (w, h), angle) = rect
        if w == 0 or h == 0: continue
        
        # Filter by Aspect Ratio (QR codes are roughly square, allowing for perspective)
        aspect_ratio = min(w, h) / max(w, h)
        if aspect_ratio < 0.3: continue # Too skinny

        # Filter by "Extent" (Is it a solid block or a sparse shape?)
        # The morphological closing should have made it a solid block.
        box_area = w * h
        extent = area / box_area
        if extent < 0.5: continue # Not solid enough

        # --- SUCCESS ---
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        final_boxes.append(box)
        # Draw the filled contour on the final mask
        cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    # --- VISUALIZATION ---
    plt.figure(figsize=(15, 10))

    # Row 1: The Intermediate Masks
    plt.subplot(2, 3, 1)
    plt.title("1. Color Mask (White Paper)")
    plt.imshow(color_mask, cmap='gray'), plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("2. Variance Mask (High Texture)")
    plt.imshow(variance_mask, cmap='gray'), plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("3. Combined (White & Textured)")
    plt.imshow(combined_mask, cmap='gray'), plt.axis('off')
    
    # Row 2: Final Processing
    plt.subplot(2, 3, 4)
    plt.title("4. After Morphology (Closing)")
    plt.imshow(morph_mask, cmap='gray'), plt.axis('off')
    
    # Final Result on Original Image
    result_img = img.copy()
    for box in final_boxes:
        cv2.drawContours(result_img, [box], 0, (0, 255, 0), 4)
        
    plt.subplot(2, 3, 5)
    plt.title(f"5. Final Detected Regions ({len(final_boxes)})")
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)), plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Final Refined Mask")
    plt.imshow(final_mask, cmap='gray'), plt.axis('off')

    plt.tight_layout()
    plt.show()

def contour_detection(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply binary thresholding
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                      
    # draw contours on the original image
    image_copy = img.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
    # see the results
    cv2.imshow('None approximation', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_quadrilaterals(gradient_img):
    # 1. Binarize the gradient image
    # The gradient image is already mostly black/white, but we need a strict binary mask.
    # Otsu's thresholding works well here to adapt to different lighting levels.
    _, binary = cv2.threshold(gradient_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Find Contours
    # RETR_EXTERNAL gets the outer boundaries (good if the QR is a solid block)
    # RETR_LIST gets everything (better if the QR looks like a hollow box)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rois = []

    for cnt in contours:
        # 3. Filter by Area (Remove small noise)
        area = cv2.contourArea(cnt)
        if area < 500:  # Adjust this based on your image resolution
            continue

        # 4. Approximate the Polygon
        # perimeter is the arc length of the contour
        perimeter = cv2.arcLength(cnt, True)
        
        # epsilon is the accuracy parameter. 
        # 0.02 (2%) is standard. Larger = clumsier shape; Smaller = strictly follows curves.
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 5. The "Rectangle" Check
        # If the approximated contour has exactly 4 points, it's a quadrilateral.
        # For high recall, we might accept 4, 5, or 6 points (in case a corner is rounded/noisy).
        if 4 <= len(approx) <= 6:
            
            # Get the bounding box for aspect ratio filtering
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Sanity Check: Is it vaguely square-ish? (0.5 to 2.0)
            if 0.5 < aspect_ratio < 2.0:
                rois.append(cnt)
                
                # Visual Debug: Draw the simplified polygon
                cv2.drawContours(gradient_img, [approx], -1, (255, 255, 255), 3)

    return gradient_img

def detect_qr_regions(image_path):
    # 1. Load and Preprocess
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image")
        return

    # Resize if image is massive (speed up processing)
    scale_percent = 100 
    # img = cv2.resize(img, None, fx=scale_percent/100, fy=scale_percent/100)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Texture Detection (High Recall Step)
    # We use a Morphological Gradient (Dilation - Erosion) to find edges.
    # This works better than Canny for variable lighting.
    kernel_size = 5 # Size of the "probe"
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # "morph_gradient" highlights areas with high local contrast (edges)
    morph_gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, rect_kernel)

    # 3. Binarize
    # Otsu's threshold automatically finds the split between "smooth" and "textured"
    _, binary = cv2.threshold(morph_gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 4. Connect the Dots (The "Blobbing" Step)
    # We dilate heavily to merge the QR code modules into one solid white square.
    # The kernel size here defines how far apart elements can be and still be merged.
    # For a dashboard, a larger kernel is safer for high recall.
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) 
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, connect_kernel)

    # 5. Find Contours (ROI Candidates)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    
    for cnt in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 6. Geometric Filters (High Recall Settings)
        # We are very loose with these filters to ensure we don't miss the QR code.
        
        aspect_ratio = w / float(h)
        area = w * h
        
        # Filter 1: Area (Ignore tiny noise like dead pixels or bugs)
        if area < 80: 
            continue

        # Filter 2: Aspect Ratio (QR codes are 1:1, but perspective makes them rectangles)
        # We accept anything from 0.5 (tall) to 2.0 (wide).
        if 0.5 < aspect_ratio < 2.0:
            rois.append((x, y, w, h))
            
            # Visualization: Draw rectangle on original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display Results
    quadrilaterals = detect_quadrilaterals(morph_gradient)
    cv2.imshow("1. Gradient", morph_gradient)
    cv2.imshow("2. Connected", connected)
    cv2.imshow("3. Detected ROIs", img)
    cv2.imshow("4. Quadrilaterals", quadrilaterals)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_high_recall_rois(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Edge Detection (Sobel)
    # We use Sobel instead of Canny because it preserves intensity information
    # (stronger edges = brighter pixels), which is useful for the next step.
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    
    # Calculate gradient magnitude
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.convertScaleAbs(magnitude)

    # 3. The "Smear" (Texture Energy)
    # We blur the edge map significantly. 
    # This connects the tiny edges of the QR code into one solid "island" of white.
    # Adjust (21, 21) -> Larger values merge more objects together (higher recall).
    blurred = cv2.blur(magnitude, (23, 23))

    # 4. Thresholding
    # Any area with sufficient "busyness" becomes white.
    # Threshold 50 is very low (permissive), ensuring we catch even faint QR codes.
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    # 5. Morphological Closing (Optional Clean-up)
    # Fills small holes inside the blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 6. Extract ROIs
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    output_img = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Minimal size filter (just to remove single speckles)
        if w * h < 500: 
            continue
            
        rois.append((x, y, w, h))
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Visual Debugging
    cv2.imshow("1. Gradient Magnitude", magnitude)
    cv2.imshow("2. Texture Energy (Blurred)", blurred)
    cv2.imshow("3. Final ROIs", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_blobs_robust("/home/nico/Pictures/Screenshots/038.png", debug = True)



'''

# Hier den Pfad zu deinem Bild einfügen
# analyze_steps('/DatenUbuntu/Studium/1. Semester/KI-Projekt/einzelbilder_mitqr/001.png')


# --- ANWENDUNG ---
# Ersetze 'test_bild.jpg' mit einem Pfad zu einem eurer Bilder
extracted_rois = detect_qr_regions_high_recall('/home/nico/Pictures/Screenshots/030.png', debug=True)

# Um zu sehen was passiert:
for i, roi in enumerate(extracted_rois):
    cv2.imshow(f"ROI {i}", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
