import yaml
import shutil
import cv2
import sys
from pathlib import Path
from tqdm import tqdm

# Add current directory to path to find sibling modules
sys.path.append(str(Path(__file__).parent))

import training_data_generator
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
    print(f"   Project Root: {ROOT_DIR}")

    # 2. Resolve Paths Smartly
    raw_real = resolve_path(cfg_paths['raw_real'])
    raw_synth = resolve_path(cfg_paths['raw_synthetic'])
    processed_dir = resolve_path(cfg_paths['processed'])

    # 3. Analyze Real Data
    print("\n📊 1. Analyzing Real Data...")
    real_pos_count = count_images(raw_real / "positive")
    real_neg_count = count_images(raw_real / "negative")
    print(f"   Found Real Pos: {real_pos_count}")
    print(f"   Found Real Neg: {real_neg_count}")

    # 4. Calculate Requirements
    total_target = cfg_data['total_images']
    pos_ratio = cfg_data['positive_ratio']
    
    target_pos = int(total_target * pos_ratio)
    target_neg = total_target - target_pos

    needed_syn_pos = max(0, target_pos - real_pos_count)
    needed_syn_neg = max(0, target_neg - real_neg_count)

    print("\n🧮 2. Dataset Plan:")
    print(f"   Target Total: {total_target}")
    print(f"   Positives: {target_pos} (Real: {real_pos_count} + Synthetic Needed: {needed_syn_pos})")
    print(f"   Negatives: {target_neg} (Real: {real_neg_count} + Synthetic Needed: {needed_syn_neg})")

    # 5. Generate Synthetic Data
    print("\n🏭 3. Synthetic Generation Phase...")
    training_data_generator.generate_synthetic_data(
        config=config, 
        output_dir=raw_synth,
        num_positives=needed_syn_pos,
        num_negatives=needed_syn_neg
    )

    # 6. Build Final Processed Dataset
    print("\n📦 4. Assembling Final Dataset...")
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    
    (processed_dir / "positive").mkdir(parents=True)
    (processed_dir / "negative").mkdir(parents=True)

    sources = [
        {"path": raw_real / "positive", "origin": "real", "label": "pos", "dest": processed_dir / "positive"},
        {"path": raw_real / "negative", "origin": "real", "label": "neg", "dest": processed_dir / "negative"},
        {"path": raw_synth / "positive", "origin": "syn",  "label": "pos", "dest": processed_dir / "positive"},
        {"path": raw_synth / "negative", "origin": "syn",  "label": "neg", "dest": processed_dir / "negative"},
    ]

    img_size = (cfg_data['img_size'], cfg_data['img_size'])

    for src in sources:
        if not src["path"].exists(): continue
        
        # Sort files to ensure the same subset is picked every time
        all_files = sorted(list(src["path"].glob("*")))
        
        if src["origin"] == "syn":
            limit = needed_syn_pos if src["label"] == "pos" else needed_syn_neg
            files_to_process = all_files[:limit]
            
            if len(all_files) > limit:
                print(f"   Note: Reusing {limit} of {len(all_files)} available synthetic {src['label']} images.")
        else:
            files_to_process = all_files

        desc = f"Processing {src['origin']}_{src['label']}"
        for fpath in tqdm(files_to_process, desc=desc):
            try:
                img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                if img.shape != img_size:
                    img = cv2.resize(img, img_size)
                
                fname = f"{src['origin']}_{src['label']}_{fpath.stem}.png"
                cv2.imwrite(str(src["dest"] / fname), img)
            except Exception as e:
                print(f"Error: {e}")

    # 7. Cleanup Phase
    if cfg_data.get('cleanup_intermediate', False):
        print("\n🧹 5. Cleaning up intermediate synthetic data...")
        if raw_synth.exists():
            shutil.rmtree(raw_synth)
            print(f"   Deleted {raw_synth} to save space.")
    else:
        print("\n✨ 5. Intermediate data preserved for faster re-runs.")

    print(f"\n✅ Pipeline Complete! Final dataset in {processed_dir}")

if __name__ == "__main__":
    main()