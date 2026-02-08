# Experimente: MobileNetV2 (Transfer Learning)

## Iteration 1: Full Fine-Tuning (Baseline)

### Parameter

In diesem Experiment wird das **MobileNetV2** evaluiert, welches als robuster Industriestandard für effiziente Bildverarbeitung gilt. Basierend auf den Erkenntnissen der vorangegangenen MobileNetV3-Experimente wird die Strategie angepasst: Auf eine reine Feature-Extraction-Phase wird verzichtet, stattdessen wird das Modell direkt vollständig auf die Zieldomäne angepasst.

* **Modellarchitektur:**
    * Als Basis dient das **MobileNetV2**, das im Vergleich zu V3Small eine höhere Parameterzahl und Komplexität aufweist.
    * Es wird direkt der Ansatz des **Full Fine-Tuning** verfolgt: Die Option `freeze_base` wird auf `false` gesetzt. Zwar erfolgt die Initialisierung mit ImageNet-Gewichten, jedoch sind sämtliche Schichten (inklusive der tiefen Convolutional-Layer) von Beginn an trainierbar. Dies ermöglicht eine sofortige Spezialisierung der Filter auf geometrische QR-Code-Strukturen.

* **Datenverarbeitung & Augmentierung:**
    * Die Eingabe erfolgt weiterhin als **Graustufenbild** ($256 \times 256 \times 1$), um die Konsistenz der Versuchsreihe zu wahren. Eine vorgeschaltete, trainierbare Konvolutionsschicht adaptiert diese Eingabe auf die vom Modell erwarteten drei Kanäle.
    * Zur Erhöhung der Robustheit wird die etablierte Augmentierungs-Pipeline (Rotation, Spiegelung, Gaußsches Rauschen $\sigma=0.025$, Kontrastvarianz) beibehalten.

* **Klassifikator-Kopf (Head):**
    * Die extrahierten Merkmalskarten werden mittels **Global Average Pooling** auf einen Vektor reduziert.
    * Es folgt eine dichte Schicht mit 64 Neuronen (ReLU-Aktivierung), die durch **Dropout** (Rate: 0,5) regularisiert wird.
    * Die finale Klassifikation erfolgt durch ein einzelnes Neuron mit Sigmoid-Aktivierung.

* **Trainingsparameter:**
    * Das Training ist auf maximal 50 Epochen angesetzt.
    * Aufgrund der sofortigen Freigabe aller Gewichte wird eine konservative **Lernrate von $5 \times 10^{-5}$** (Adam-Optimierer) gewählt. Dies verhindert das sogenannte "Catastrophic Forgetting" der nützlichen Initialgewichte zu Beginn des Trainings.
    * Ein **Early-Stopping**-Mechanismus und ein Learning-Rate-Scheduler sorgen für effiziente Konvergenz.
    * **Anpassung der Batch-Größe:**
    Aufgrund der höheren Modellkomplexität von MobileNetV2 (tieferes Netzwerk, mehr Parameter) und dem Speicherbedarf für die Gradientenberechnung aller Schichten (*Full Fine-Tuning*), wurde die **Batch-Größe auf 32 reduziert**. Dies war notwendig, um Speicherüberläufe (Out-of-Memory Errors) auf der verfügbaren GPU-Hardware zu vermeiden, die bei der ursprünglichen Größe von 64 auftraten.

### Ergebnisse

Das Experiment mit MobileNetV2 (Full Fine-Tuning) lieferte hervorragende Ergebnisse und übertraf die Leistung des MobileNetV3Small nochmals deutlich. Das Training wurde nach 43 Epochen durch Early-Stopping beendet, da keine weitere signifikante Verbesserung des Validierungsverlusts erzielt wurde. Die besten Modellgewichte wurden aus **Epoche 35** gesichert.

* **Quantitative Performance (Best Model - Epoche 35):**
    * **Validation Accuracy:** **99,86%**.
        Das Modell erreicht eine nahezu perfekte Klassifikationsgenauigkeit. Im Vergleich zu MobileNetV3Small (~99,1%) konnte die Fehlerrate nochmals massiv reduziert werden.
    * **Validation Recall:** **99,59%**.
        Dies ist das wichtigste Ergebnis für die Anwendungssicherheit. Das Modell übersieht praktisch keine QR-Codes mehr (weniger als 1 von 200). Zum Vergleich: Das V3-Modell lag hier bei ~96,7%.
    * **Validation Precision:** **99,84%**.
        Die Wahrscheinlichkeit für einen Fehlalarm (False Positive) ist verschwindend gering.
    * **Validation Loss:** **0,0054**.
        Der extrem niedrige Verlustwert bestätigt, dass das Modell sehr "sicher" in seinen Entscheidungen ist und klare Trenngrenzen gelernt hat.

* **Lernverhalten & Stabilität:**
    * **Hohe Stabilität zu Beginn:** Im Gegensatz zum kleineren V3-Modell, das beim "Unfreeze" zunächst einbrach, startete MobileNetV2 bereits in der ersten Epoche mit einer Validierungsgenauigkeit von über 94% und einem Recall von ~77%. Die höhere Parameterzahl scheint das Modell robuster gegen das initiale Aufbrechen der Gewichte zu machen.
    * **Konvergenz:** Das Modell lernte extrem schnell. Bereits nach 8 Epochen lag der Validierungsverlust unter 0,03. Der Learning-Rate-Scheduler griff in Epoche 34 und 43 ein, um die letzten Prozentpunkte an Genauigkeit herauszuholen.

* **Fazit:**
    Mit einer Genauigkeit und einem Recall von jeweils fast 100% hat sich das MobileNetV2 als die überlegene Architektur erwiesen. Die Entscheidung, direkt mit dem Fine-Tuning aller Schichten zu starten, war hocheffizient. Das Modell ist in diesem Zustand optimal für den produktiven Einsatz im Sliding-Window-Detektor geeignet, da es die notwendige Zuverlässigkeit bietet, die bei der reinen Bildklassifikation gefordert ist.