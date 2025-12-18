# KI-Project-WS2526

## Daten
- ausgeglichene Anzahl Bilder mit und ohne QR Codes unter der Windschutzscheibe
- verschiedene Abstände zur Windschutzscheibe
- verschiedene Winkel zur Windschutzscheibe
- verschiedene Verschmutzungsgrade der Windschutzscheibe
- verschiedene Größen von QR Codes
- verschiedene Farben der QR Codes

## Preprocessing
- Schwarz-Weiß
- Konstante Skalierung

## Data Augmentation
- Blur
- Crop
- vertical Mirror
- Contrast
- Rotation (max +- 30 Grad)

## Aufgabe 1

### Erster Ansatz

- Bilder direkt in CNN
- Hyperparamter optimieren und dokumentieren.

### Zweiter Ansatz (wenn 1. zu schlecht)

- konstante Auflösung festlegen

- YOLO benutzen um Regions of Interests (ROI) zu erkennen.
    - Windschutzscheibe ausschneiden, 
    - wenn keine Windschutzscheibe erkannt -> ganzes Bild in eigenes CNN.

- konstante Auflösung festlegen

- ROIs sind dann Input für eigenes CNN 
    - Trifft Aussage of QR-Code oder nicht (Einfluss von mehreren QR Codes)

### 3 Transfer Modelle nutzen (kein YOLO)

## Aufgabe 2

- Bounding Boxen mit Yolo o.Ä. erkennen
- Mit QR-Code Scanner versunche QR Codes in den Bounding zu scannen
    - wenn möglich -> grüne Bounding Box
    - wenn nicth möglich -> rote Bounding Box

- Winkel anhand der Bounding Box Koordinaten bestimmen.
