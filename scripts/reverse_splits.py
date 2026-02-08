import shutil
import sys
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

def reverse_split():
    """
    Reverses the dataset split operation.
    Moves files from:
        dataset/patches/{split}/{class_name}
    Back to:
        dataset/patches/{class_name}
    And removes the empty {split} directories.
    """
    paths = get_repo_paths()
    base_patches_dir = paths["patches_dir"]
    
    # 1. Define source (splits) and destination (base class folders)
    splits = ["train", "val", "test"]
    class_names = [paths["class_name_pos"], paths["class_name_neg"]]
    
    # Mapping for destination folders
    # e.g., 'positive' -> dataset/patches/positive
    destinations = {
        paths["class_name_pos"]: paths["patches_pos"],
        paths["class_name_neg"]: paths["patches_neg"]
    }
    
    # Ensure destination folders exist
    for dest in destinations.values():
        dest.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Reversal of Splits...")

    # 2. Iterate through splits and move files back
    moved_count = 0
    
    for split in splits:
        split_dir = base_patches_dir / split
        
        if not split_dir.exists():
            logger.info(f"Split folder not found, skipping: {split}")
            continue
            
        for class_name in class_names:
            src_class_dir = split_dir / class_name
            dest_class_dir = destinations[class_name]
            
            if not src_class_dir.exists():
                continue
                
            # Get all files
            files = list(src_class_dir.glob("*.*")) # Catch all image extensions
            
            if not files:
                continue
                
            logger.info(f"Moving {len(files)} images from '{split}/{class_name}' back to base folder...")
            
            for f in tqdm(files, desc=f"Reversing {split}/{class_name}", leave=False):
                try:
                    shutil.move(str(f), str(dest_class_dir / f.name))
                    moved_count += 1
                except shutil.Error:
                    # Occurs if file already exists in destination (rare safety check)
                    logger.warning(f"File already exists in destination: {f.name}")

    # 3. Cleanup empty directories
    logger.info("Cleaning up empty folders...")
    for split in splits:
        split_dir = base_patches_dir / split
        if split_dir.exists():
            try:
                # remove tree if empty, or handle specific subfolders
                for class_name in class_names:
                    subdir = split_dir / class_name
                    if subdir.exists() and not any(subdir.iterdir()):
                        subdir.rmdir()
                
                # Try removing the split root if empty
                if not any(split_dir.iterdir()):
                    split_dir.rmdir()
                    logger.info(f"Removed empty split folder: {split}")
            except OSError as e:
                logger.warning(f"Could not remove folder {split}: {e}")

    logger.info(f"Reversal Complete. Moved {moved_count} images back to '{base_patches_dir}'.")

if __name__ == "__main__":
    reverse_split()