import sys
import shutil
from pathlib import Path
import imagehash
from PIL import Image
from tqdm import tqdm
import yaml

# Force QtAgg for interactive windows
import matplotlib
try:
    matplotlib.use('QtAgg')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Add current directory to path to find sibling modules
sys.path.append(str(Path(__file__).parent))
from project_paths import resolve_path, CONFIG_DIR

class DuplicateCleaner:
    def __init__(self, trash_dir, threshold=7):
        self.trash_dir = Path(trash_dir)
        self.threshold = threshold
        self.trash_dir.mkdir(parents=True, exist_ok=True)
        self.user_decision = None # Will store 'delete_existing', 'delete_new', or 'keep_both'

    def on_key(self, event):
        """Callback for keyboard events."""
        if event.key == '1':
            self.user_decision = 'delete_existing'
            plt.close()
        elif event.key == '2':
            self.user_decision = 'delete_new'
            plt.close()
        elif event.key == 'escape' or event.key == 'q':
            self.user_decision = 'keep_both'
            plt.close()

    def show_comparison_and_wait(self, existing_path, new_path, distance):
        """
        Shows images side-by-side and blocks until user decides.
        """
        self.user_decision = 'keep_both' # Default if window closed
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # LEFT IMAGE (Existing)
        try:
            img1 = mpimg.imread(str(existing_path))
            axes[0].imshow(img1, cmap='gray')
            axes[0].set_title(f"[1] DELETE LEFT (Existing)\n{existing_path.name}", color='red', fontweight='bold')
            axes[0].axis('off')
        except Exception:
            axes[0].text(0.5, 0.5, "Error Loading", ha='center')

        # RIGHT IMAGE (New Match)
        try:
            img2 = mpimg.imread(str(new_path))
            axes[1].imshow(img2, cmap='gray')
            axes[1].set_title(f"[2] DELETE RIGHT (New Candidate)\n{new_path.name}", color='red', fontweight='bold')
            axes[1].axis('off')
        except Exception:
            axes[1].text(0.5, 0.5, "Error Loading", ha='center')

        fig.suptitle(f"DUPLICATE DETECTED (Dist: {distance})\nPress '1' to delete LEFT, '2' to delete RIGHT, 'Esc' to keep both", fontsize=14)
        
        # Hook up the keyboard listener
        fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.tight_layout()
        plt.show() # Blocks here until closed

    def process_directory(self, directory):
        directory = Path(directory)
        if not directory.exists():
            return

        print(f"\n🔍 Scanning for duplicates in: {directory.name}")
        
        images = sorted(list(directory.glob("*.png")) + list(directory.glob("*.jpg")))
        known_hashes = [] # List of (hash, path)
        
        duplicates_handled = 0
        
        # Iterate through images
        # We assume known_hashes represents the 'clean' set we are building up.
        for img_path in tqdm(images, desc="Checking"):
            # Check if file still exists (might have been deleted in previous loop if specific logic was complex, mostly safety check)
            if not img_path.exists(): 
                continue

            try:
                with Image.open(img_path) as img:
                    current_hash = imagehash.phash(img)
            except Exception as e:
                print(f"Error hashing {img_path.name}: {e}")
                continue

            match_found = False
            
            # Compare against clean list
            # We iterate backwards so if we delete from known_hashes it doesn't break? 
            # Actually we just need to find ONE match.
            for i, (existing_hash, existing_path) in enumerate(known_hashes):
                distance = current_hash - existing_hash
                
                if distance <= self.threshold:
                    # DUPLICATE FOUND
                    # Ask user what to do
                    self.show_comparison_and_wait(existing_path, img_path, distance)
                    
                    if self.user_decision == 'delete_existing':
                        # Delete the one currently in our list
                        shutil.move(str(existing_path), str(self.trash_dir / existing_path.name))
                        print(f"   🗑️ Moved to trash: {existing_path.name}")
                        
                        # Replace the entry in known_hashes with the new image (since we kept the new one)
                        known_hashes[i] = (current_hash, img_path)
                        match_found = True
                        duplicates_handled += 1
                        
                    elif self.user_decision == 'delete_new':
                        # Delete the current image
                        shutil.move(str(img_path), str(self.trash_dir / img_path.name))
                        print(f"   🗑️ Moved to trash: {img_path.name}")
                        match_found = True # We handled it, don't add to known_hashes
                        duplicates_handled += 1
                        
                    else:
                        # Keep Both
                        print(f"   Unknown decision, keeping both.")
                        pass
                    
                    # If we found a match and handled it, stop checking other hashes for this image
                    if match_found:
                        break
            
            # If no match was found (or we kept the new one after deleting the old one), add to known hashes
            if not match_found:
                known_hashes.append((current_hash, img_path))

        print(f"✅ Deduplication complete. Moved {duplicates_handled} images to {self.trash_dir}")

# Helper for standalone usage
def main():
    with open(CONFIG_DIR / "dataset_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    cleaner = DuplicateCleaner(
        trash_dir=resolve_path(config['deduplication']['trash_dir']),
        threshold=config['deduplication']['threshold']
    )
    
    raw_real = resolve_path(config['paths']['raw_real'])
    cleaner.process_directory(raw_real / "positive")
    cleaner.process_directory(raw_real / "negative")

if __name__ == "__main__":
    main()