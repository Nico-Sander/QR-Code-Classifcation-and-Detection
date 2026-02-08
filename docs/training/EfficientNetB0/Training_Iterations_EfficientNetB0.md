# Experimente: EfficientNetB0 (Transfer Learning)

## Iteration 1: Full Fine-Tuning (Baseline)

### Parameter

In diesem finalen Experiment wird das **EfficientNetB0** evaluiert. Diese Architektur gilt als besonders leistungsfähig im Verhältnis zur Parameteranzahl und nutzt fortschrittliche Skalierungsmethoden ("Compound Scaling"). Ziel ist es zu prüfen, ob die komplexeren Features des EfficientNets die bereits nahezu perfekten Ergebnisse des MobileNetV2 nochmals stabilisieren oder übertreffen können.

* **Modellarchitektur:**
    * Als Basis dient das **EfficientNetB0**. Es ist strukturell komplexer und rechenintensiver als die MobileNet-Familie, bietet jedoch potenziell robustere Feature-Extraktoren.
    * Es wird analog zum erfolgreichen MobileNetV2-Versuch direkt der Ansatz des **Full Fine-Tuning** verfolgt: Die Option `freeze_base` wird auf `false` gesetzt. Sämtliche Schichten sind von Beginn an trainierbar, um die vortrainierten ImageNet-Gewichte optimal an die geometrischen Strukturen der QR-Codes anzupassen.

* **Datenverarbeitung & Augmentierung:**
    * Die Eingabe erfolgt weiterhin als **Graustufenbild** ($256 \times 256 \times 1$), um die Vergleichbarkeit innerhalb der Versuchsreihe zu wahren. Eine vorgeschaltete, trainierbare Konvolutionsschicht adaptiert diese Eingabe auf die vom Modell erwarteten drei Kanäle.
    * Die bewährte Augmentierungs-Pipeline (Rotation, Spiegelung, Gaußsches Rauschen $\sigma=0.025$, Kontrastvarianz) wird identisch beibehalten.

* **Klassifikator-Kopf (Head):**
    * Die extrahierten Merkmalskarten werden mittels **Global Average Pooling** auf einen Vektor reduziert.
    * Es folgt eine dichte Schicht mit 64 Neuronen (ReLU-Aktivierung), die durch **Dropout** (Rate: 0,5) regularisiert wird.
    * Die finale Klassifikation erfolgt durch ein einzelnes Neuron mit Sigmoid-Aktivierung.

* **Trainingsparameter:**
    * Das Training ist auf maximal 50 Epochen angesetzt.
    * Die **Lernrate beträgt $5 \times 10^{-5}$** (Adam-Optimierer). Diese niedrige Rate ist essenziell, um das sogenannte "Catastrophic Forgetting" beim Trainieren des gesamten, tiefen Netzwerks zu verhindern.
    * Ein **Early-Stopping**-Mechanismus und ein Learning-Rate-Scheduler sorgen für effiziente Konvergenz.
    * **Batch-Größe:** Es wird weiterhin eine **Batch-Größe von 32** verwendet. Dies stellt aufgrund der höheren Speicherauslastung des EfficientNetB0 das Limit der verfügbaren Hardware dar, ermöglicht jedoch eine effizientere GPU-Auslastung im Vergleich zu einer weiteren Reduktion.

### Ergebnisse

Das finale Experiment mit dem EfficientNetB0 bestätigte das enorme Potenzial dieser Architektur, zeigte jedoch auch die erwartete höhere Sensibilität im Training. Das Modell erreichte nach einer volatilen Startphase absolute Spitzenwerte. Das Training wurde nach 29 Epochen beendet, wobei die **besten Gewichte aus Epoche 24** (Index 24) gesichert wurden.

* **Quantitative Performance (Best Model - Epoche 24):**
    * **Validation Accuracy:** **99,91%**.
        Das EfficientNetB0 setzt eine neue Bestmarke und übertrifft knapp die bereits exzellenten 99,86% des MobileNetV2. Es ist das präziseste Modell der Versuchsreihe.
    * **Validation Recall:** **99,76%**.
        Die Erkennungsrate ist extrem hoch. Es werden faktisch keine QR-Codes übersehen.
    * **Validation Precision:** **99,88%**.
        Fehlalarme sind bei diesem Modell fast ausgeschlossen.
    * **Validation Loss:** **0,0027**.
        Dieser extrem niedrige Verlustwert (nahezu Null) ist bemerkenswert und halbierte sich im Vergleich zum MobileNetV2 (0,0054) nochmals. Das Modell ist sich seiner Vorhersagen extrem sicher.

* **Lernverhalten & Stabilität:**
    * **Volatile Startphase:** Im starken Kontrast zum stabilen MobileNetV2 zeigte das EfficientNetB0 zu Beginn massive Instabilitäten. In den ersten 7 Epochen schwankte der Validierungsverlust stark (Spitzen bis 1,88 in Epoche 3 und 6), und die Accuracy fiel zeitweise auf unter 68% zurück. Dies ist typisch für komplexere Netze, die beim *Full Fine-Tuning* mit hoher Lernrate zunächst "aus dem Tritt" geraten können.
    * **Stabilisierung:** Ab Epoche 10 stabilisierte sich das Training signifikant.
    * **Feinjustierung:** Der Learning-Rate-Scheduler spielte eine entscheidende Rolle. Nach der Reduktion der Lernrate in Epoche 14 und 20 sanken Verlust und Fehlerrate rapide auf das finale Rekordniveau.

* **Fazit:**
    Das EfficientNetB0 liefert die **beste rohe Performance** aller getesteten Modelle. Der Validierungsverlust von 0,0027 ist unerreicht. Allerdings erkauft man sich diesen minimalen Gewinn gegenüber dem MobileNetV2 durch eine deutlich höhere Trainingsinstabilität und einen größeren Ressourcenbedarf (Speicher). Für den produktiven Einsatz ist das MobileNetV2 möglicherweise der robustere "Allrounder", während das EfficientNetB0 die Wahl für kompromisslose Präzision ist.