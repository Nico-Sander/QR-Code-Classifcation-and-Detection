import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data_generator import SyntheticGenerator
from repo_paths import get_repo_paths

# --- SETUP ---

# 1. Load Configuration
paths = get_repo_paths()
config_path = paths["config_dir"] / "dataset_config.yaml"

if not config_path.exists():
    print(f"Error: Config not found at {config_path}")
    exit()

with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)

# 2. Initialize Generator with the FULL config
# (Note: Ensure your SyntheticGenerator init accepts 'full_config' as discussed)
generator = SyntheticGenerator(full_config=full_config)

# --- CAPTURE LOGIC ---

# Storage for images intercepted from the generator
captured_images = []

def mock_imwrite(filename, img):
    """
    Instead of saving to disk, we store the image in a list.
    filename: str (the dummy path passed)
    img: numpy array (the BGR image)
    """
    # Convert BGR (OpenCV) to RGB (Matplotlib) for correct viewing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Label based on the filename we passed
    label = "Positive" if "pos" in str(filename) else "Negative"
    captured_images.append({"label": label, "img": img_rgb})
    return True

# --- EXECUTION ---

def generate_and_view(num_samples=4):
    print(f"Generating {num_samples} pairs of synthetic images...")
    
    # 1. Hijack cv2.imwrite
    original_imwrite = cv2.imwrite
    cv2.imwrite = mock_imwrite
    
    try:
        # 2. Run the generator logic
        for i in range(num_samples):
            # We pass dummy paths; they won't actually be used on disk
            generator.generate_single_positive(f"dummy_pos_{i}.png")
            generator.generate_single_negative(f"dummy_neg_{i}.png")
            
    finally:
        # 3. Restore cv2.imwrite (Good practice!)
        cv2.imwrite = original_imwrite

    # 4. Visualization with Matplotlib
    if not captured_images:
        print("No images were generated (check if generator returned False).")
        return

    rows = num_samples
    cols = 2 # Pos vs Neg
    
    plt.figure(figsize=(10, 4 * rows))
    plt.suptitle("Synthetic Data Preview (Not Saved to Disk)", fontsize=16)

    # We expect captured_images to be [Pos, Neg, Pos, Neg, ...]
    # because of the loop order above.
    for i in range(len(captured_images)):
        item = captured_images[i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(item["img"])
        plt.title(item["label"])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_and_view(num_samples=4)