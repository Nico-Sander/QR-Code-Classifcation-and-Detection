# Experimente: MobileNetV3Small (Transfer Learning)

## Iteration 1: Baseline Model

### Parameter

Das Ziel dieses Experiments ist die Evaluierung einer parametereffizienten Architektur mittels Transfer-Learning, um die Eignung für ressourcenbeschränkte Umgebungen zu prüfen. Die Konfiguration gliedert sich wie folgt:

* **Modellarchitektur:**
    * Als Basis dient das **MobileNetV3Small**, welches speziell für mobile Anwendungen mit geringer Latenz optimiert ist.
    * Es wird der Ansatz der **Feature Extraction** verfolgt: Die auf ImageNet vortrainierten Gewichte des Basismodells bleiben während des Trainings fixiert (`freeze_base: true`), sodass ausschließlich die neu hinzugefügten Schichten trainiert werden.

* **Datenverarbeitung & Augmentierung:**
    * Die Eingabe erfolgt weiterhin als **Graustufenbild** ($256 \times 256 \times 1$), um die Vergleichbarkeit mit dem Basismodell zu wahren. Eine vorgeschaltete, trainierbare Konvolutionsschicht adaptiert diese Eingabe auf die vom Modell erwarteten drei Kanäle.
    * Zur Erhöhung der Robustheit gegen Aufnahmefehler werden geometrische Transformationen (Rotation, Spiegelung) sowie Simulationen von Bildqualitätsverlusten (Gaußsches Rauschen $\sigma=0.025$, Kontrastvarianz) angewendet.

* **Klassifikator-Kopf (Head):**
    * Die extrahierten Merkmalskarten werden mittels **Global Average Pooling** auf einen Vektor reduziert.
    * Es folgt eine dichte Schicht mit 64 Neuronen (ReLU-Aktivierung), die durch **Dropout** (Rate: 0,5) regularisiert wird, um Overfitting zu vermeiden.
    * Die finale Klassifikation erfolgt durch ein einzelnes Neuron mit Sigmoid-Aktivierung.

* **Trainingsparameter:**
    * Das Training ist auf maximal 50 Epochen angesetzt und nutzt den **Adam-Optimierer** mit einer initialen Lernrate von $0,0005$.
    * Ein **Early-Stopping**-Mechanismus (Geduld: 7 Epochen) und eine dynamische Reduktion der Lernrate bei Stagnation verhindern unnötige Rechenzeit und optimieren die Konvergenz.

### Ergebnisse

Das erste Experiment mit dem MobileNetV3Small zeigte ein sehr solides Lernverhalten, benötigte jedoch deutlich mehr Zeit zur Konvergenz als ursprünglich angenommen. Das Training wurde durch Early-Stopping nach **45 Epochen** beendet, wobei die besten Modellgewichte aus **Epoche 38** wiederhergestellt wurden.

* **Quantitative Performance (Best Model - Epoche 38):**
    * **Validation Accuracy:** Das Modell erreichte eine Genauigkeit von **95,51%**.
    * **Validation Loss:** Der Verlust auf dem Validierungsdatensatz sank auf ein Minimum von **0,1184**.
    * **Validation Recall:** Mit **88,86%** ist die Erkennungsrate für QR-Codes gut, zeigt aber noch Raum für Verbesserungen im Vergleich zu reinen Custom-CNNs.
    * **Validation Precision:** Die Präzision ist mit **92,88%**, das Modell erkennt noch viel zu oft QR-Codes, wo keine sind.

* **Lernverhalten & Dynamik:**
    * Der Start war langsam: In der ersten Epoche lag der Recall bei nur ~4,5%, stieg aber bis Epoche 5 schnell auf ~79% an.
    * **Plateaus & Lernrate:** Das Modell stieß ab ca. Epoche 30 auf ein Plateau. Der Learning-Rate-Scheduler griff korrigierend ein:
        * In Epoche 34 wurde die Lernrate auf $2,5 \times 10^{-4}$ halbiert.
        * In Epoche 42 erfolgte eine weitere Reduktion auf $1,25 \times 10^{-4}$.
    * Diese Eingriffe stabilisierten das Training, führten jedoch nicht mehr zu signifikanten Sprüngen in der Validierungsgenauigkeit, was zum Early-Stopping führte.

* **Fazit des Laufs:** Das Modell lernt die Aufgabe erfolgreich, konvergiert aber langsamer und erreicht "nur" ~95.5% Accuracy im Vergleich zu den ~98-99%, die mit spezialisierteren Architekturen oft möglich sind. Dies deutet darauf hin, dass die "eingefrorenen" ImageNet-Features für QR-Codes zwar brauchbar, aber nicht perfekt optimiert sind.

## Iteration 2: Convolutional Layer auch trainieren

## Änderungen
Um die Limitierungen der ersten Iteration (insbesondere den stagnierenden Recall bei ~88%) zu überwinden, wird die Strategie von reiner Feature-Extraction auf **Full Fine-Tuning** geändert.

* **Freigabe der Basis-Schichten (`freeze_base: false`):**
    Sämtliche Schichten des MobileNetV3Small werden für das Training freigegeben. Dies ermöglicht dem Netzwerk, die tiefen Convolutional-Filter, die ursprünglich auf natürliche Objekte (ImageNet) trainiert wurden, spezifisch an die harten, geometrischen Strukturen von QR-Codes anzupassen (Domain Adaptation).

* **Reduktion der Lernrate:**
    Da nun das gesamte Netzwerk trainiert wird, wurde die Lernrate drastisch auf **$5 \times 10^{-5}$** ($0,00005$) gesenkt. Dies ist notwendig, um "Catastrophic Forgetting" zu verhindern – also das Zerstören der bereits nützlichen, vortrainierten Merkmale durch zu aggressive Gewichts-Updates zu Beginn des Trainings.

## Ergebnisse

Der Wechsel auf **Full Fine-Tuning** erwies sich als der entscheidende Durchbruch. Die Hypothese, dass die auf ImageNet vortrainierten Filter nicht ideal für QR-Codes geeignet sind, wurde bestätigt: Durch die Freigabe aller Gewichte konnte das Modell die notwendige geometrische Präzision erlernen, was zu einer massiven Leistungssteigerung führte.

Das Training wurde durch Early-Stopping in Epoche 28 beendet, wobei die **besten Gewichte aus Epoche 20** wiederhergestellt wurden.

### Quantitative Analyse (Best Model - Epoche 20)
Im Vergleich zur ersten Iteration (Feature Extraction) zeigen alle Metriken signifikante Verbesserungen:

* **Validation Recall:** **96,73%** (vgl. ~88% in Iteration 1).
    Dies ist der wichtigste Erfolg dieser Iteration. Das Modell übersieht nun kaum noch QR-Codes, was für die Zuverlässigkeit in der Waschanlage kritisch ist.
* **Validation Precision:** **99,58%**.
    Die Falsch-Positiv-Rate ist auf ein Minimum gesunken. Das Modell ist extrem sicher in seiner Entscheidung.
* **Validation Accuracy:** **99,08%** (vgl. ~95,5% in Iteration 1).
    Das Modell erreicht nun ein Leistungsniveau, das für den produktiven Einsatz geeignet ist.
* **Validation Loss:** **0,0356**.
    Der Fehlerwert ist im Vergleich zum ersten Versuch (0,1184) um fast den Faktor 4 gesunken.

### Lernverhalten & Dynamik
* **Der "Unfreeze-Schock":** In der allerersten Epoche (Epoch 1) brachen die Validierungswerte kurzzeitig ein (Recall: 0,00%), da sich die Gewichte erst auf die neue Freiheit "einstellen" mussten. Doch bereits in Epoche 2 stabilisierte sich das Netz, und ab Epoche 5 lag der Recall bereits wieder über 88%.
* **Effektiver Scheduler:** Der Learning-Rate-Scheduler spielte eine wichtige Rolle bei der Feinjustierung. Die Reduktion der Lernrate in Epoche 13 (auf $2,5 \times 10^{-5}$) brachte den entscheidenden Sprung von ~95% auf über 96% Recall.
* **Stabilität:** Das Modell zeigte ab Epoche 20 leichte Anzeichen von Overfitting (steigender Val-Loss), weshalb der Early-Stopping-Mechanismus korrekt eingriff und das Training in Epoche 28 beendete, um die Generalisierungsfähigkeit zu bewahren.

### Fazit
Mit einer Genauigkeit von über 99% und einem Recall von fast 97% ist das MobileNetV3Small nun **bereit für die Integration in den Sliding-Window-Algorithmus**. Die Anpassung der tiefen Convolutional-Filter war der Schlüssel, um die feinen geometrischen Details der QR-Codes robust zu erfassen.


