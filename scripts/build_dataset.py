import yaml
import shutil
import cv2
import sys
from pathlib import Path
from tqdm import tqdm

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

import training_data_generator
from remove_duplicates import DuplicateCleaner
from verify_dataset import DatasetVerifier 
from project_paths import ROOT_DIR, CONFIG_DIR, resolve_path

def load_config(path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def count_images(directory):
    if not directory.exists(): return 0
    return len(list(directory.glob("*.png")) + list(directory.glob("*.jpg")) + list(directory.glob("*.jpeg")))

def main():
    # 1. Load Configuration
    config_path = CONFIG_DIR / "dataset_config.yaml"
    config = load_config(config_path)
    cfg_data = config['dataset']
    cfg_paths = config['paths']

    print(f"🚀 Starting Pipeline: {config['project_name']}")

    raw_real = resolve_path(cfg_paths['raw_real'])
    raw_synth = resolve_path(cfg_paths['raw_synthetic'])
    processed_dir = resolve_path(cfg_paths['processed'])
    misclassified_dir = resolve_path(cfg_paths['misclassified']) # From updated config

    # --- PHASE 1: DEDUPLICATION ---
    if config.get('deduplication', {}).get('enabled', False):
        print("\n🕵️  1. Checking for Duplicates (Interactive)...")
        trash_dir = resolve_path(config['deduplication']['trash_dir'])
        cleaner = DuplicateCleaner(trash_dir, config['deduplication']['threshold'])
        cleaner.process_directory(raw_real / "positive")
        cleaner.process_directory(raw_real / "negative")
    else:
        print("\n⏭️  1. Deduplication skipped.")

    # --- PHASE 2: VERIFY REAL DATA (Cleaning Source) ---
    print("\n🧠 2. Verifying Real Data Integrity...")
    verifier = DatasetVerifier(config, trash_root=misclassified_dir)
    
    real_dirs = [
        (raw_real / "positive", "pos"),
        (raw_real / "negative", "neg")
    ]
    
    for r_dir, label in real_dirs:
        if not r_dir.exists(): continue
        files = sorted(list(r_dir.glob("*")))
        
        # Use a manual loop index so we can handle file moves safely
        print(f"   Checking {label} images in {r_dir.name}...")
        cleaned_count = 0
        for fpath in tqdm(files):
            # verify_and_move returns False if the user discards (moves) the file
            keep = verifier.verify_and_move(fpath, label)
            if not keep:
                cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"   ⚠️  Moved {cleaned_count} misclassified images to {misclassified_dir}")

    # --- PHASE 3: CALCULATE REQUIREMENTS ---
    print("\n📊 3. Calculating Dataset Balance...")
    # Now we count ONLY the files that survived verification
    real_pos_count = count_images(raw_real / "positive")
    real_neg_count = count_images(raw_real / "negative")
    
    total_target = cfg_data['total_images']
    pos_ratio = cfg_data['positive_ratio']
    
    target_pos = int(total_target * pos_ratio)
    target_neg = total_target - target_pos

    needed_syn_pos = max(0, target_pos - real_pos_count)
    needed_syn_neg = max(0, target_neg - real_neg_count)

    print(f"   Real Pos: {real_pos_count} | Need Synthetic: {needed_syn_pos}")
    print(f"   Real Neg: {real_neg_count} | Need Synthetic: {needed_syn_neg}")

    # --- PHASE 4: GENERATE & VERIFY SYNTHETIC ---
    print("\n🏭 4. Generating & Verifying Synthetic Data...")
    
    # Generate the batch
    training_data_generator.generate_synthetic_data(
        config=config, 
        output_dir=raw_synth,
        num_positives=needed_syn_pos,
        num_negatives=needed_syn_neg
    )

    # Immediately verify the NEW synthetic data
    # We verify the raw_synthetic folder now
    syn_dirs = [
        (raw_synth / "positive", "pos"),
        (raw_synth / "negative", "neg")
    ]

    for s_dir, label in syn_dirs:
        if not s_dir.exists(): continue
        files = sorted(list(s_dir.glob("*.png"))) 
        
        # Only check the newly generated ones if you want to save time, 
        # but checking all in that folder ensures consistency.
        print(f"   Verifying synthetic {label} quality...")
        bad_syn_count = 0
        for fpath in tqdm(files):
            # If user discards synthetic, we move it to misclassified too (or you could just delete)
            keep = verifier.verify_and_move(fpath, label)
            if not keep:
                bad_syn_count += 1
        
        if bad_syn_count > 0:
             print(f"   ✂️  Pruned {bad_syn_count} bad synthetic images.")

    # --- PHASE 5: ASSEMBLY ---
    print("\n📦 5. Assembling Final Dataset...")
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    
    (processed_dir / "positive").mkdir(parents=True)
    (processed_dir / "negative").mkdir(parents=True)

    # Re-list files because some might have been moved during verification
    sources = [
        {"path": raw_real / "positive", "dest": processed_dir / "positive"},
        {"path": raw_real / "negative", "dest": processed_dir / "negative"},
        {"path": raw_synth / "positive", "dest": processed_dir / "positive"},
        {"path": raw_synth / "negative", "dest": processed_dir / "negative"},
    ]

    img_size = (cfg_data['img_size'], cfg_data['img_size'])

    for src in sources:
        if not src["path"].exists(): continue
        
        # We take everything that currently exists in the folders 
        # (since we already cleaned them)
        files = sorted(list(src["path"].glob("*")))
        
        for fpath in tqdm(files, desc=f"Copying {src['path'].name}"):
            try:
                img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                if img.shape != img_size:
                    img = cv2.resize(img, img_size)
                
                # Prefix filename with origin type to avoid collisions
                origin_prefix = "real" if "real" in str(src["path"]) else "syn"
                label_prefix = "pos" if "positive" in str(src["path"]) else "neg"
                fname = f"{origin_prefix}_{label_prefix}_{fpath.name}"
                
                cv2.imwrite(str(src["dest"] / fname), img)
            except Exception as e:
                print(f"Error copying {fpath}: {e}")

    print(f"\n✅ Pipeline Complete! Clean dataset in {processed_dir}")

if __name__ == "__main__":
    main()