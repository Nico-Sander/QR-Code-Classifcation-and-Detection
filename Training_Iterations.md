## 2. Iteration

### Changes

- Lower learning rate, because spikes in validation loss
- Use ReduceLearningRateOnPlateau callback to dynamically adjust learning rate
- Use batch normilaziation in 3rd convolutional block to increase convergence and stability
- increase number of real patches in dataset to ~4000

### Results
- Loss curve is now smoother
- Performand decreased slightly, val_acc now 95.2% vs 96.2% of the baseline model

## 3. Iteration

### Changes

- **Architecture:** Switched to a Deep CNN with **5 Convolutional Blocks** (previously 3) to learn more complex features.
- **Pooling Strategy:** Replaced Global Average Pooling (GAP) with **Flatten**. Since the 5 blocks reduce the image to 8x8, Flattening is now efficient and preserves spatial information.
- **Parameter Optimization:** - Capped deep filters at 128 to prevent parameter explosion.
    - Reduced Dense layer size to 64 units.
    - Total parameters targeted at ~1 million (optimal balance).
- **Hyperparameters:** - Increased batch size to **64** (optimized for GPU).
    - Increased learning rate to **0.0005** to help the deeper network converge.

### Results
- **Performance:** Reached a new peak validation accuracy of **~96.0%** fast (at Epoch 5), surpassing Iteration 2.
- **Issue:** Significant **Overfitting** observed after Epoch 5.
  - Training loss continued to decrease (0.09 -> 0.07).
  - Validation loss spiked dramatically (0.12 -> 0.37) and became unstable.
- **Conclusion:** The deeper architecture is effective at feature extraction but requires stronger regularization to prevent memorization.

## 4. Iteration

### Changes

- **Strategy Shift:** Moved from "Offline Augmentation" (fixed images) to **"Online Augmentation"** (GPU-based).
  - *Reasoning:* Prevents the model from memorizing specific synthetic images. Now, the model sees a slightly different variation of every image in every single epoch (infinite dataset).

- **New Augmentation Layers:** Added `preprocessing` layers at the start of the model:
  - **Geometry:** `RandomFlip`, `RandomRotation` (20%), and `RandomZoom` (10%) to handle real-world camera angles.
  - **Blur Proxy:** Combined `RandomContrast` + `RandomZoom` to simulate out-of-focus or bad lighting conditions.
  - **Artifacts Proxy:** Added `GaussianNoise` to simulate sensor grain, dust, and printing dots.
  - **Obstruction Proxy:** Added `RandomTranslation` to shift images off-center, forcing the model to recognize QR codes even when cut off by the frame edge.

- **Implementation details:**
  - Added `Rescaling(1./255)` layer to normalize inputs, ensuring `GaussianNoise` works correctly.
  - Updated `model_builder.py` to support these new layer types.
  - Increased `early_stopping_patience` to 10, as augmented training is harder and takes longer to converge.

### Results
- **Performance:** Peak Validation Accuracy dropped significantly to **~88.8%** (Epoch 9).
- **Issue:** The model suffered from **Domain Shift**.
  - Training Accuracy reached **95%**, proving the model could learn the difficult augmented data.
  - Validation Accuracy plateaued at **~88%**, indicating that the heavy noise/distortion during training made the model less effective on the "cleaner" validation data.
  - Validation Loss started increasing after Epoch 3 (from 0.30 to 0.53), a sign that the aggressive augmentations were hindering convergence rather than helping it.
- **Conclusion:** The augmentation parameters (specifically noise and contrast) were too strong. We need to dial them back to find the sweet spot between robustness and accuracy.

## 5. Iteration

### Changes

- **Dataset Quality Control:**
  - **Cleaned Real Negatives:** Removed duplicates and low-value images (e.g., solid colors like sky/car hoods) that were biasing the model.
  - **Cleaned Real Positives:** Removed images where the QR code was barely visible or indistinguishable to the human eye, as these were acting as "label noise."

- **Augmentation Refinement (Fixing Domain Shift):**
  - **Reduced Intensity:** Dialed back augmentation parameters by ~50-75% to fix the accuracy drop seen in Iteration 4. The previous settings were creating "noisy" images that looked too different from the clean validation data.
    - `gaussian_noise`: Reduced from 0.1 to **0.025** (subtle grain instead of static).
    - `random_rotation`: Reduced from 0.2 to **0.1** (limiting to realistic scanning angles).
    - `random_contrast`: Reduced from 0.2 to **0.1**.
    - `random_zoom`: Removed to avoid cutting QR codes out of images.
    - `random_translation`: Removed to avoid cutting QR codes out of images.

### Results

- **Performance (Best Saved Model):** The model converged to its optimal state at **Epoch 17** (lowest Validation Loss).
  - **Validation Accuracy:** **97.5%**
  - **Validation Loss:** **0.0847**
- **Stability:**
  - The "Domain Shift" issue from Iteration 4 is completely resolved.
  - **Perfect Generalization:** The gap between Training Accuracy (**97.8%**) and Validation Accuracy (**97.5%**) is negligible (< 0.3%), proving the model is not overfitting.
- **Conclusion:** This configuration (Deep CNN + Gentle Online Augmentation + Clean Data) yielded the most robust and stable model so far.

## 6. Iteration

### Changes

- **Real Dataset Expansion:**
  - **Increased Volume:** Added significantly more real positive and negative patches.
  - **Hard Negatives:** Specifically added "Distractor" negatives containing objects often confused for QR codes (Euro4 Stickers, Highway Vignettes, text fragments, random paper scraps).

- **Data Quality Pipeline (Deduplication):**
  - **Perceptual Hashing:** Integrated a `DuplicateCleaner` into the build pipeline using `ImageHash`.
  - **Human-in-the-Loop:** Implemented an interactive GUI that flags images with a Hamming Distance lower than a configurable threshhold. This allows manual selection of the "cleanest" version to keep, preventing data leakage (train/val overlap) and class imbalance.

- **Synthetic Generator Overhaul (Bug Fixes & Logic):**
  - **Fixed "Invisible" QR Codes:** Changed the Lens Distortion logic (`cv2.remap`) to use `BORDER_REFLECT` instead of black borders. Previously, distortion was pulling black pixels over the QR code, effectively erasing it.
  - **Contrast Safety:** Added a constraint to ensure a minimum brightness difference (>50) between QR modules and the background, preventing unreadable low-contrast generations.
  - **Variable Background Zoom:** Implemented dynamic background cropping (random chunks from 128px to 512px resized to 256px). This simulates different camera distances and varies the texture scale (coarse vs. fine grain) of the background.

- **Model Evaluation:**
  - **New Metrics:** Added **Precision** and **Recall** to the monitored metrics (alongside Accuracy). This is crucial for evaluating the new "Hard Negatives"—we need to ensure the model isn't generating False Positives on stickers (Precision) or missing difficult codes (Recall).

### Results
- **Performance (Best Saved Model - Epoch 38):**
  - **Validation Accuracy:** **96.43%** (Slight drop from 97.5% in It.5, expected due to harder dataset)
  - **Validation Loss:** **0.1049**
  - **Validation Precision:** **95.08%**
  - **Validation Recall:** **97.86%**
- **Analysis:**
  - **Impact of Hard Negatives:** The drop in raw accuracy (from 97.5% to 96.4%) and the Precision score (95%) indicates the model is struggling slightly with the new "Distractor" images (stickers/text). It is occasionally flagging them as QR codes.
  - **Strong Recall:** The high Recall (97.9%) confirms that the synthetic generator fixes worked—the model is correctly identifying almost all real QR codes, even difficult ones.
  - **Conclusion:** We successfully increased the problem difficulty. The model is now robust to "easy" scenes but needs fine-tuning to better distinguish between a QR code and a shipping label (to improve Precision).