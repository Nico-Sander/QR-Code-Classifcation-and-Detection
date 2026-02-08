import cv2
import os
import random
import sys
import yaml
import numpy as np
from pathlib import Path
import hashlib
from tqdm import tqdm

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

def get_ground_truth_boxes(label_path, img_width, img_height):
    """Parses a YOLO format .txt file into a list of objects."""
    objects = []
    if not label_path.exists(): return objects
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = list(map(float, line.split()))
            class_id = int(parts[0])
            x, y, w, h = parts[1], parts[2], parts[3], parts[4]
            
            x1 = int((x - w/2) * img_width)
            y1 = int((y - h/2) * img_height)
            x2 = int((x + w/2) * img_width)
            y2 = int((y + h/2) * img_height)
            
            objects.append({'class_id': class_id, 'box': [x1, y1, x2, y2]})
    return objects

def get_intersection_area(patch_box, gt_box):
    """Calculates the area of overlap (intersection) between patch and GT."""
    xA = max(patch_box[0], gt_box[0])
    yA = max(patch_box[1], gt_box[1])
    xB = min(patch_box[2], gt_box[2])
    yB = min(patch_box[3], gt_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea

def get_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

# --- PERCEPTUAL HASHING HELPERS ---

def dhash(image, hash_size=8):
    """
    Calculates a 'Difference Hash' for an image.
    Robust to slight color changes and noise/shifts.
    """
    # 1. Resize to (hash_size + 1, hash_size)
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # 3. Compute differences between adjacent pixels
    diff = gray[:, 1:] > gray[:, :-1]
    # 4. Convert boolean array to integer
    return sum([2**i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_distance(hash1, hash2):
    """Count bits that are different between two hashes."""
    return bin(hash1 ^ hash2).count('1')

# --- MAIN GENERATION ---

def create_training_patches(config=None):
    """
    Main function to generate patches with integrated EXACT (MD5) 
    and PERCEPTUAL (dHash) deduplication.
    """
    # 1. Setup paths and config
    paths = get_repo_paths()
    if config is None:
        config = load_config()
    
    cfg_patch = config["patch_creation"]
    IMAGE_DIR = paths["train_images"]
    LABEL_DIR = paths["train_labels"]
    OUTPUT_DIR = paths["patches_dir"]

    # Ensure output directories exist
    for cat in [paths["class_name_pos"], paths["class_name_neg"]]:
        (OUTPUT_DIR / cat).mkdir(parents=True, exist_ok=True)

    # --- DEDUPLICATION INITIALIZATION ---
    seen_md5_hashes = set()
    
    # We store dHashes for BOTH flat and textured backgrounds
    # This ensures we don't get sliding-window duplicates of concrete/grass either
    background_dhashes = [] 
    
    print("Pre-scanning output directory for existing patches...")
    # Scan existing files so multiple runs don't create exact duplicates
    for ext in ["*.jpg", "*.png"]:
        for existing_file in OUTPUT_DIR.rglob(ext):
            with open(existing_file, "rb") as f:
                seen_md5_hashes.add(hashlib.md5(f.read()).hexdigest())
                
    print(f"  Found {len(seen_md5_hashes)} existing unique patches (MD5).")

    images = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))
    print(f"Starting Patch Generation...")
    print(f"  Input: {IMAGE_DIR}")
    print(f"  Images Found: {len(images)}")

    stats = {
        "pos": 0, "neg_hard": 0, "neg_flat": 0, "neg_textured": 0,
        "discarded_ambiguous": 0, "duplicates_skipped": 0,
        "skipped_perceptual_dup": 0 
    }

    # 2. Main Processing Loop
    for img_path in tqdm(images, desc="Processing Images"):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        h_orig, w_orig, _ = img.shape
        label_path = LABEL_DIR / f"{img_path.stem}.txt"
        gt_objects = get_ground_truth_boxes(label_path, w_orig, h_orig)
        base_name = img_path.stem

        base_size = min(h_orig, w_orig)
        
        # Scale Loop
        for divisor in cfg_patch["scale_divisors"]:
            win_size = max(int(base_size / divisor), 256)
            win_size = min(win_size, base_size)
            
            overlap = cfg_patch["overlap"]
            stride = max(1, int(win_size * (1 - overlap)))
            
            pad = win_size
            img_padded = cv2.copyMakeBorder(img, 0, pad, 0, pad, cv2.BORDER_CONSTANT, value=[0,0,0])

            # Sliding Window
            for y in range(0, h_orig - win_size + stride, stride):
                for x in range(0, w_orig - win_size + stride, stride):
                    
                    patch_box = [x, y, x + win_size, y + win_size]
                    patch_area = win_size * win_size
                    
                    qr_max_containment = 0.0
                    qr_max_coverage = 0.0
                    distractor_pixel_sum = 0
                    
                    # Check Overlaps
                    for obj in gt_objects:
                        inter_area = get_intersection_area(patch_box, obj['box'])
                        if inter_area == 0: continue
                        
                        obj_area = get_area(obj['box'])
                        
                        if obj['class_id'] == cfg_patch["class_ids"]["qr_code"]:
                            containment = inter_area / obj_area
                            if containment > qr_max_containment: qr_max_containment = containment
                            coverage = inter_area / patch_area
                            if coverage > qr_max_coverage: qr_max_coverage = coverage
                            
                        elif obj['class_id'] == cfg_patch["class_ids"]["distractor"]:
                            distractor_pixel_sum += inter_area

                    distractor_coverage = distractor_pixel_sum / patch_area

                    # --- DECISION TREE ---
                    is_positive = False
                    is_ambiguous = False
                    is_hard_negative = False
                    is_flat = False 

                    if (qr_max_containment > cfg_patch["thresholds"]["qr_positive"] or 
                        qr_max_coverage > cfg_patch["thresholds"]["qr_positive"]):
                        
                        if qr_max_coverage >= cfg_patch["thresholds"]["qr_min_coverage"]:
                            is_positive = True
                        else:
                            is_ambiguous = True
                        
                    elif (qr_max_containment > cfg_patch["thresholds"]["qr_ambiguous"] or 
                          qr_max_coverage > cfg_patch["thresholds"]["qr_ambiguous"]):
                        is_ambiguous = True

                    if distractor_coverage > cfg_patch["thresholds"]["distractor"]:
                        is_hard_negative = True

                    if is_ambiguous and not is_positive:
                        stats["discarded_ambiguous"] += 1
                        continue 

                    patch = img_padded[y : y + win_size, x : x + win_size]
                    final_class = paths["class_name_neg"]
                    should_keep = False
                    
                    # Determine Keep Logic & Texture Analysis
                    if is_positive:
                        final_class = paths["class_name_pos"]
                        should_keep = True
                    elif is_hard_negative:
                        should_keep = random.random() < cfg_patch["sampling"]["keep_rate_distractor"]
                    else:
                        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                        std_dev = np.std(gray)
                        
                        if std_dev < cfg_patch["sampling"]["texture_threshold"]:
                            is_flat = True
                            should_keep = random.random() < cfg_patch["sampling"]["keep_rate_flat"]
                        else:
                            # It is textured
                            should_keep = random.random() < cfg_patch["sampling"]["keep_rate_textured"]

                    # --- SAVE AND DEDUPLICATE ---
                    if should_keep:
                        target_s = cfg_patch["target_size"]
                        patch_resized = cv2.resize(patch, (target_s, target_s))
                        
                        # 1. MD5 Check (Exact Duplicates)
                        patch_hash = hashlib.md5(patch_resized.tobytes()).hexdigest()
                        if patch_hash in seen_md5_hashes:
                            stats["duplicates_skipped"] += 1
                            continue
                        
                        # 2. dHash Check (Perceptual Duplicates)
                        # Applied to ALL background negatives (Flat OR Textured)
                        # We exclude Positives (always keep) and Hard Negatives (keep based on distractor logic)
                        if not is_positive and not is_hard_negative:
                            curr_dhash = dhash(patch_resized)
                            is_perceptual_dup = False
                            
                            # Check against existing background hashes
                            for existing_h in background_dhashes:
                                if hamming_distance(curr_dhash, existing_h) <= 6:
                                    is_perceptual_dup = True
                                    break
                            
                            if is_perceptual_dup:
                                stats["skipped_perceptual_dup"] += 1
                                continue
                                
                            # If unique, add to list
                            background_dhashes.append(curr_dhash)
                        
                        # Save valid patch
                        seen_md5_hashes.add(patch_hash)
                        
                        if is_positive: stats["pos"] += 1
                        elif is_hard_negative: stats["neg_hard"] += 1
                        elif is_flat: stats["neg_flat"] += 1
                        else: stats["neg_textured"] += 1

                        fname = f"{base_name}_div{divisor}_x{x}_y{y}.jpg"
                        cv2.imwrite(str(OUTPUT_DIR / final_class / fname), patch_resized)

    return stats

if __name__ == "__main__":
    config = load_config()
    final_stats = create_training_patches(config)
    print("\n--- Processing Complete ---")
    print(f"Positives: {final_stats['pos']}")
    print(f"Hard Negatives: {final_stats['neg_hard']}")
    print(f"Backgrounds (Textured): {final_stats['neg_textured']}")
    print(f"Backgrounds (Flat): {final_stats['neg_flat']}")
    print(f"Discarded (Ambiguous): {final_stats['discarded_ambiguous']}")
    print(f"Skipped (Exact Duplicates): {final_stats['duplicates_skipped']}")
    print(f"Skipped (Perceptual Duplicates): {final_stats['skipped_perceptual_dup']}")