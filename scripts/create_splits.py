import os
import random
import shutil
import sys
import yaml
import logging
from pathlib import Path
from tqdm import tqdm

# Add current directory to path to import repo_paths
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
from repo_paths import get_repo_paths

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_dataset(config):
    """
    Splits the dataset into train, val, and test subsets based on the config.
    Moves files from:
        dataset/patches/positive -> dataset/patches/{split}/positive
        dataset/patches/negative -> dataset/patches/{split}/negative
    """
    paths = get_repo_paths()
    base_patches_dir = paths["patches_dir"]
    
    # 1. Validation & Setup
    split_cfg = config.get("split", {"train": 0.8, "val": 0.19, "test": 0.01})
    
    total_ratio = sum(split_cfg.values())
    if abs(total_ratio - 1.0) > 1e-6:
        logger.error(f"Split ratios must sum to 1.0. Found: {total_ratio}")
        return

    # Define the source directories (The current location of your data)
    sources = {
        paths["class_name_pos"]: paths["patches_pos"],
        paths["class_name_neg"]: paths["patches_neg"]
    }

    # Define the splits we want to create
    splits = ["train", "val", "test"]
    
    # Check if sources exist and are not empty
    for label, path in sources.items():
        if not path.exists() or not any(path.iterdir()):
            logger.warning(f"Source directory for '{label}' is empty or missing: {path}")
            # Check if maybe the user already ran the split?
            if (base_patches_dir / "train" / label).exists():
                logger.error("It looks like the dataset is already split. Check 'dataset/patches/train'. Aborting.")
                return

    logger.info(f"Starting Dataset Split: {split_cfg}")

    # 2. Processing Loop
    for class_name, source_dir in sources.items():
        # Get all valid image files
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        files = []
        for ext in extensions:
            files.extend(list(source_dir.glob(ext)))
            
        # Sort first to ensure deterministic order before shuffling
        files.sort()
        
        # Shuffle deterministically
        random.seed(42)
        random.shuffle(files)
        
        n_total = len(files)
        if n_total == 0:
            continue

        # Calculate split indices
        n_train = int(n_total * split_cfg["train"])
        n_val = int(n_total * split_cfg["val"])
        # Test gets the remainder to ensure no off-by-one errors
        n_test = n_total - n_train - n_val

        # Slice the file list
        split_files = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }

        logger.info(f"Processing Class: {class_name.upper()} ({n_total} images)")
        logger.info(f"  -> Train: {n_train} | Val: {n_val} | Test: {n_test}")

        # Move files
        for split_name in splits:
            dest_dir = base_patches_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            file_list = split_files[split_name]
            
            # Use TQDM for progress bar
            for f in tqdm(file_list, desc=f"  Moving to {split_name}/{class_name}", leave=False):
                shutil.move(str(f), str(dest_dir / f.name))

    # 3. Cleanup
    # Optional: Remove the original class folders if they are now empty
    for label, path in sources.items():
        try:
            path.rmdir() # Only removes if empty
            logger.info(f"Cleaned up empty source folder: {path}")
        except OSError:
            logger.info(f"Source folder not empty, kept: {path}")

    logger.info("Dataset split complete!")
    logger.info(f"Structure created at: {base_patches_dir}")

if __name__ == "__main__":
    # Load Config
    paths = get_repo_paths()
    config_path = paths["config_dir"] / "dataset_config.yaml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        split_dataset(cfg)
    else:
        logger.error(f"Config not found at {config_path}")