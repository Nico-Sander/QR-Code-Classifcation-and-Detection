# QR-Code Erkennung und Analyse für Waschanlagen

Ein KI-basiertes System zur robusten Detektion, Dekodierung und geometrischen Analyse von QR-Code-Belegen hinter Windschutzscheiben.

## Über das Projekt

In modernen Waschanlagen werden Waschprozesse oft über Belege mit QR-Codes gesteuert, die hinter der Windschutzscheibe platziert sind. Die automatisierte Erfassung dieser Codes stellt eine große Herausforderung dar:

* Variable Distanzen: Codes müssen aus bis zu 3 Metern Entfernung erkannt werden.

* Störeinflüsse: Nässe, Schaum, Spiegelungen auf der Scheibe und "Distractors" (andere Zettel, Vignetten).

* Perspektive: Starke Verzerrung durch den Kamerawinkel und die Neigung der Windschutzscheibe.

Dieses Projekt implementiert eine vollständige Pipeline – von der Datenerhebung über das Training neuronaler Netze bis hin zur geometrischen Auswertung für die Hardware-Justierung.

## Projektstruktur

```
.
├── config/             # Konfigurationsdateien
│   ├── dataset_config.yaml # Einstellungen für die Generierung des Datensatzes
│   └── model_config.yaml   # Modellarchitektur und Hyperparameter für das Training
├── dataset/            # Datenverzeichnis (Rohbilder, Patches, Labels)
├── notebooks/          # Jupyter Notebooks (Der Kern des Projekts)
│   └── dataset_pipeline.ipynb   # Schritt 1: Generierung des Datensatzes
│   ├── training_pipeline.ipynb  # Schritt 2: Training der CNNs
│   └── inferenz_pipeline.ipynb  # Schritt 3: Inferenz, Decodierung und geometrische Analyse
├── runs/               # Logs und gespeicherte Modelle
├── scripts/            # Hilfsskripte (Generator, Visualisierung, Utils)
├── images/            
│   └── samples/        # Testbilder für die Inferenz
└── pyproject.toml      # Projekt-Abhängigkeiten (uv / pip)
```

## Funktionsweise

Die Inferenz Pipeline implementiert die folgenden Schritte:

1. **Detektion / Klassifikation (Sliding Window & CNN)**
    
    Das QR-Codes im Gesamtbild sehr klein sein können, wird ein **Multi-Scale Sliding Window** Verfahren verwendet.

    * Das Bild wird in viele kleine Ausschnitte zerlegt (Patches)

    * Ein trainiertes **CNN (Convolutional Neural Network)** klassifiziert jeden Patch (QR-Code vorhanden: Ja/Nein)

    * Verwendete Modelle: Eigene Architektur, MobileNetV2, MobileNetV3Small und EfficientNetB0 

    * Ergebnis: Eine **Heatmap** über deren Werte auf das Vorhandensein von QR-Codes und deren Positionen geschlossen wird.

2. **Dekodierung & Rektifizierung**

    Klassische Scanner scheitern oft and der perspektivischen Verzerrung. Daher nutzen wir:

    * **QReader (YOLOv8-basiert)**: Erkennt die Eckpunkte des QR-Codes auch unter schwierigen Bedingungen.

    * **Rektifizierung**: Der QR-Code wird mathematisch "geradegezogen" (entzerrt), bevor er gelesen wird.

3. **Geometrische Analyse**

    Für die optimale Ausrichtung der Hardware in einer Waschanlage wird der Neigungswinkel der QR-Codes relativ zur Kamera mittels **SolvePnP** berechnet.

    * Ergebnis der Analyse: Ein Montagewinkel von 44° zur horizontalen maximiert die Chance der Erkennung der QR-Codes.

## Inbetriebnahme

Dieses Projekt nutzt moderne Python-Tools für das Dependency Management.

### Vorraussetzungen

* Python 3.11

* [uv](https://github.com/astral-sh/uv) (empfohlen für die Installation der benötigten Packages) oder pip

### Installation

1. **Repository klonen:**

    ```shell
    git clone https://github.com/Nico-Sander/QR-Code-Classifcation-and-Detection.git
    ```

    ```shell
    cd QR-Code-Classification-and-Detection
    ```
2. **Abhängigkeiten Installieren**: 

    Mit `uv` wird automatisch eine virtuelle Python Umgebung erstellt und alle Pakete aus `pyproject.toml` installiert.

    ```shell
    uv sync
    ```

    Alternativ mit pip:

    ```shell
    pip install .
    ```

### Nutzung

Das Projekt wird hauptsächlich über die Jupyter Notebooks im `notebooks/` Verzeichnis und die Konfigurationsdateien im `config/` Verzeichnis genutzt.

* **Vorbereitung des Datensatzes**

    Um eigene Modelle trainieren zu können, wird der Datensatz benötigt. Falls Interesse besteht, kontaktieren Sie uns gerne über GitHub.

    Falls eigene Daten genutzt werden möchten, müssen die folgenden Schritte befolgt werden:
    
    1. Annotieren der Bilder mit einem Tool wie [Roboflow](https://roboflow.com/). Markieren von QR-Codes und *Distractors* mit Bounding Boxes.

    2. Exportieren der annotierten Bilder im **PyTorch v5 Format**

    3. Herstellung dieser Ordnerstruktur:

    ```
    .
    └── dataset/   
        ├── backgrounds/    # Hintergründe für synthetische Daten
        │   ├── high_res/   # Hochauflösende Hintergrundbilder
        │   └── dtd/        # DTD Datensatz
        ├── full_sized/     # Annotierter Datensatz
        │   ├── images/     # Vollbilder
        └── └── labels/     # Koordinaten & Klassen der Bounding Boxen

    ```

    4. Prüfen der Einstellungen in `config/dataset_config.yaml`

    5. Ausführen der Zellen in `notebooks/dataset_pipeline.ipynb`. Dies erstellt das `dataset/patches/` Verzeichnis, wo sich dann die Bilder befinden, die zum Training genutzt werden können. Mit den unveränderten Einstellungen, werden so 100.000 Patches generiert.

* **Training von Modellen**

    * Steht ein verarbeiteter Datensatz zur Verfügung, können Modelle Trainiert werden. Um die Modellarchitektur einzustellen und die Hyperparameter festzulegen wird `config/model_config.yaml` genutzt.

    * Sind alle Einstellungen vorgenommen, müssen die Zellen aus `notebooks/training_pipeline.ipynb` ausgeführt werden.

    * Das erfolgreiche Ausführen resultiert in einem neuen Unterordner im `runs/` Verzeichnis. Dort können verschiedene Informationen zum Training sowie das finale Model gefunden werden.

* **Inferenz mit Testbildern**

    * Im Verzeichnis `images/samples` befinden sich beispielhafte Testbilder, die vom `notebooks/inferenz_pipeline.ipynb` Notebook analysiert werden können.

## Verwendete Skripte und Logik

Im Verzeichnis `scripts/` sind Python Skripte zu finden. In diesen ist die eigentliche Logik implementiert, die von den *Pipelines* genutzt wird. Im folgenden eine kurze Übersichst mit Beschreibungen:

* **`roi_testing/`**:

    Skripte die verwendet wurden, um einen *Region of Interest* Ansatz zu evaluieren. Dieser Ansatz führte nicht zum Erfolg und diese Skripte sind für das Projekt nichtmehr notwendig.

* **`callbacks.py`**:

    Hier werden verschiedene *Callbacks* definiert die beim Training genutzt werden können. Beispiele sind `EarlyStopping` und `ReduceLRonPlateau`.


* **`create_patches.py`**:

    Die *Multi-Scale Sliding Window* Logik, die die Bildausschnitte aus den Vollbildern für den finalen Datensatz erstellt.

* **`create_splits.py`**:

    Logik, die die Aufteilung des Datensatzes in Train, Validation und Test implementiert.

* **`data_generator.py`**:

    Funktionen für die automatische Generierung von synthetischen Daten.

* **`deduplication_fullsize_interactive.py`**:

    Ein Skript, das ausgeführt werden kann um Duplikate und ähnliche **Vollbilder** im Datensatz zu identifizieren. Ähliche Bilder werden ge-clustert und dem Nutzer in eine GUI präsentiert. Der Nutzer kann dann per Keyboard Inputs entscheiden, welche Bilder aus einem Cluster behalten werden sollen und somit für die Erstellung des Patches genutzt werden.

* **`model_builder.py`**:

    Der Parser, der `config/model_config.yaml` liest und aus der dort vorgegebenen Architektur und Hyperparamter das Modell zusammenbaut.

* **`project_paths.py`** und **`repo_paths.py`**:

    Hier werden die Pfade zu den relevanten Verzeichnissen des Projektes definiert, sodass andere Funktionen des Projekt wissen, wo z.B. der Datensatz zu finden ist.

* **`reverse_splits.py`**:

    Ein Skript das ausgeführt werden kann, um die Aufteilung in *Train-, Validation- und Test*daten rückgängig zu machen.

* **`synthetic_data_visualizer.py`**:

    Ein ausführbares Skript, welches synthetische Trainingsdaten produziert und dem Nutzer zur Inspektion und Validierung der Qualität zeigt.



