import yaml
import imagehash
from PIL import Image
import os
import shutil
import networkx as nx
import sys
from pathlib import Path

# --- GUI BACKEND SETUP ---
# Must be set before importing pyplot
import matplotlib
try:
    matplotlib.use('QtAgg') 
except Exception:
    print("Warning: Could not set QtAgg backend. Visualization might fail.")

import matplotlib.pyplot as plt

# Add src to path to import repo_paths
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
from repo_paths import get_repo_paths

def load_config():
    """Loads the dataset_config.yaml file."""
    paths = get_repo_paths()
    config_path = paths["repo_root"] / "dataset_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def calculate_hash(image_path, method="dhash", hash_size=8):
    """Calculates the hash of an image."""
    try:
        with Image.open(image_path) as img:
            if method == "dhash":
                return imagehash.dhash(img, hash_size=hash_size)
            elif method == "phash":
                return imagehash.phash(img, hash_size=hash_size)
            else:
                return imagehash.average_hash(img, hash_size=hash_size)
    except Exception as e:
        print(f"Warning: Could not hash {image_path.name}: {e}")
        return None

def show_cluster(cluster_files, img_dir, cluster_id):
    """
    Displays the cluster in a non-blocking window.
    Returns the figure object so we can close it later.
    """
    num_imgs = len(cluster_files)
    # Dynamic figsize: wider for more images
    fig, axes = plt.subplots(1, num_imgs, figsize=(4 * num_imgs, 5))
    
    if num_imgs == 1: axes = [axes] # Handle edge case
    
    fig.canvas.manager.set_window_title(f"Cluster {cluster_id}")
    fig.suptitle(f"Cluster {cluster_id}: Which to KEEP?", fontsize=16, color='red')
    
    for i, (ax, fname) in enumerate(zip(axes, cluster_files)):
        img_path = img_dir / fname
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            # Add big red number ID
            ax.text(10, 50, str(i), fontsize=40, color='red', weight='bold', 
                    bbox=dict(facecolor='white', alpha=0.5))
            ax.set_title(f"ID: {i}\n{fname}", fontsize=10)
            ax.axis('off')
        except Exception:
            ax.text(0.5, 0.5, "Error loading", ha='center')

    plt.tight_layout()
    plt.show(block=False) # Important: allows the terminal to accept input
    plt.pause(0.1)        # Give the window a moment to render
    return fig

def run_interactive_deduplication(config=None):
    """
    Main entry point for interactive deduplication.
    
    Args:
        config (dict, optional): Loaded configuration. If None, loads from yaml.
    
    Returns:
        int: Number of images removed.
    """
    # 1. Setup
    paths = get_repo_paths()
    if config is None:
        config = load_config()
        
    dedup_cfg = config["deduplication"]
    threshold = dedup_cfg.get("similarity_threshold", 8) 
    
    img_dir = paths["train_images"]
    label_dir = paths["train_labels"]
    trash_dir = paths["data_dir"] / "trash"
    trash_dir.mkdir(exist_ok=True)
    
    print(f"--- Starting Interactive Cleanup (Threshold: {threshold}) ---")
    print(f"Removed images will be moved to: {trash_dir}")
    
    # 2. Hashing
    print("Hashing images (this may take a moment)...")
    image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    hashes = {}
    
    for p in image_files:
        h = calculate_hash(p, method=dedup_cfg["method"], hash_size=dedup_cfg["hash_size"])
        if h:
            hashes[p.name] = h
            
    # 3. Clustering
    print("Grouping duplicates...")
    G = nx.Graph()
    filenames = list(hashes.keys())
    
    for f in filenames: G.add_node(f)
    
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            f1 = filenames[i]
            f2 = filenames[j]
            # Calculate Hamming distance
            if hashes[f1] - hashes[f2] <= threshold:
                G.add_edge(f1, f2)
    
    clusters = [list(c) for c in nx.connected_components(G) if len(c) > 1]
    print(f"Found {len(clusters)} clusters to review.")
    
    if not clusters:
        print("Dataset is clean! No duplicates found.")
        return 0

    # 4. Interactive Loop
    removed_count = 0
    
    try:
        for i, cluster in enumerate(clusters):
            print(f"\n--- Cluster {i+1}/{len(clusters)} ---")
            
            # Show images
            fig = show_cluster(cluster, img_dir, i+1)
            
            # Get User Input
            valid_input = False
            keep_indices = []
            
            while not valid_input:
                user_input = input(f"Enter IDs to KEEP (e.g. '0', '0 2', 'all', 'none', 'q' to quit): ").strip().lower()
                
                if user_input == 'q':
                    print("Quitting early.")
                    plt.close('all')
                    return removed_count
                
                if user_input == 'all':
                    keep_indices = range(len(cluster))
                    valid_input = True
                elif user_input == 'none':
                    keep_indices = []
                    valid_input = True
                else:
                    try:
                        parts = user_input.replace(',', ' ').split()
                        keep_indices = [int(p) for p in parts]
                        valid_input = True
                    except ValueError:
                        print("Invalid input. Please type numbers separated by spaces.")

            # Close the window
            plt.close(fig)

            # PROCESS REMOVAL
            for idx, fname in enumerate(cluster):
                if idx not in keep_indices:
                    # Move to Trash
                    src_img = img_dir / fname
                    dst_img = trash_dir / fname
                    
                    src_label = label_dir / f"{Path(fname).stem}.txt"
                    dst_label = trash_dir / f"{Path(fname).stem}.txt"
                    
                    if src_img.exists():
                        shutil.move(str(src_img), str(dst_img))
                        removed_count += 1
                        print(f"  -> Removed: {fname}")
                    
                    if src_label.exists():
                        shutil.move(str(src_label), str(dst_label))
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    
    print(f"\n--- Done. Removed {removed_count} images to '{trash_dir.name}' ---")
    return removed_count

if __name__ == "__main__":
    # Standard boilerplate to run standalone
    run_interactive_deduplication()