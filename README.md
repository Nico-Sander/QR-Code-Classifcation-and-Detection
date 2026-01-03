# KI-Project-WS2526

## Dataset Generation

This project uses an automated pipeline to combine real and synthetic data into a final training set. The pipeline ensures a balanced dataset by generating synthetic patches to fill the gap between available real images and the target dataset size.

### Data Setup (Prerequisite)
Before running the pipeline, ensure your real data is organized in the following structure:
```text
data/
├── real_patches/
│   ├── positive/   # Place real images containing QR codes here
│   └── negative/   # Place real background images here
└── backgrounds/    # Place full-size images without QR codes used for generating synthetic backgrounds
```

### How to Generate the Dataset

1. **Configure**: Open `config/dataset_config.yaml` and set your desired parameters:
    - `total_images`: The total number of patches (Real + Synthetic) you want.
    - `positive_ratio`: The split between positive (QR) and negative (Background) images (e.g., `0.5`).
    - `cleanup_intermediate`: Set to `false` to cache synthetic images for faster re-runs.

2. **Run the Pipeline**:
    ```bash
    python scripts/build_dataset.py
    ```

## How to train a Model

1. **Configure**: Open `config/model_config.yaml' and set your desired parameters

2. **Run the Pipeline**:
- run all the cells in the `notebooks/01_train_model.ipynb` notebook.

3. The best and final model and the logging data will be saved to `runs/<model_name>` where the modelname can be specified in the config.

4. **See interactive model parameters**:
- run this command from the the root of the project:
    ```bash
    tensorboard --logdir runs
    ```
- then open http://localhost:6006 in your browser.

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
