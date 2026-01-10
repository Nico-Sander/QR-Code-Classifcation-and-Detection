import numpy as np
import cv2
import shutil
import matplotlib
# Force QtAgg for interactive windows
try:
    matplotlib.use('QtAgg')
except:
    pass
import matplotlib.pyplot as plt
import tensorflow as tf 
from pathlib import Path
from project_paths import ROOT_DIR  # <--- Needed to find runs/

class DatasetVerifier:
    def __init__(self, config, trash_root):
        """
        trash_root: Path where misclassified images should be moved.
        """
        self.cfg = config
        self.enabled = config['verification']['enabled']
        self.trash_root = Path(trash_root)
        
        (self.trash_root / "positive").mkdir(parents=True, exist_ok=True)
        (self.trash_root / "negative").mkdir(parents=True, exist_ok=True)
        
        if self.enabled:
            # --- AUTO-DETECT LOGIC ---
            model_path = self._find_latest_model()
            
            if model_path is None:
                print(f"⚠️  Warning: No 'final_model.keras' found in {ROOT_DIR / 'runs'}. Verification disabled.")
                self.enabled = False
            else:
                print(f"🧠 Loading latest model: {model_path.parent.name}")
                self.model = tf.keras.models.load_model(model_path)
                self.img_size = config['dataset']['img_size']
                
        self.user_decision = 'keep' 

    def _find_latest_model(self):
        """
        Scans ROOT_DIR/runs for folders starting with 'run_'.
        Returns path to 'final_model.keras' in the directory that sorts last (latest date/run ID).
        """
        runs_dir = ROOT_DIR / "runs"
        if not runs_dir.exists():
            return None
            
        candidates = []
        
        # Iterating over directories in runs/
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("run_"):
                model_file = d / "final_model.keras"
                if model_file.exists():
                    candidates.append(d)
        
        if not candidates:
            return None
            
        # Sort by directory name. 
        # Since your format is 'run_XX_..._YYYYMMDD_HHMMSS', 
        # alphabetical sort correctly puts the latest timestamp/run-id last.
        latest_run_dir = sorted(candidates, key=lambda x: x.name)[-1]
        return latest_run_dir / "final_model.keras"

    def preprocess(self, img_array):
        # Resize to match model input
        img = cv2.resize(img_array, (self.img_size, self.img_size))
        
        # The model has a Rescaling(1./255) layer built-in.
        img = img.astype('float32') 
        
        # Add channel dimension (H, W, 1)
        img = np.expand_dims(img, axis=-1)
        
        # Add batch dimension (1, H, W, 1)
        img = np.expand_dims(img, axis=0)
        
        return img

    def on_key(self, event):
        if event.key == 'd':
            self.user_decision = 'discard'
            plt.close()
        elif event.key == 'k':
            self.user_decision = 'keep'
            plt.close()

    def show_ui(self, image, filename, label, prediction, reason):
        self.user_decision = 'keep' 
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap='gray')
        
        pred_pct = prediction * 100
        
        title = (f"SUSPICIOUS: {filename}\n"
                 f"Label: {label.upper()} | Model: {pred_pct:.1f}% QR\n"
                 f"{reason}\n"
                 f"Press 'k' to KEEP, 'd' to DISCARD")
        
        ax.set_title(title, color='red', fontweight='bold', fontsize=12)
        ax.axis('off')

        fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.tight_layout()
        plt.show() 
        
        return self.user_decision

    def verify_and_move(self, image_path, label):
        if not self.enabled:
            return True

        image_path = Path(image_path)
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None: return False
        except:
            return False

        input_tensor = self.preprocess(img)
        prediction = float(self.model.predict(input_tensor, verbose=0)[0][0])
        
        is_suspicious = False
        reason = ""

        threshold_pos = self.cfg['verification']['suspicious_pos_threshold']
        threshold_neg = self.cfg['verification']['suspicious_neg_threshold']

        if label == 'pos' and prediction < threshold_pos:
            is_suspicious = True
            reason = f"Low confidence positive (<{threshold_pos})"
            
        elif label == 'neg' and prediction > threshold_neg:
            is_suspicious = True
            reason = f"High confidence negative (>{threshold_neg})"

        if is_suspicious:
            decision = self.show_ui(img, image_path.name, label, prediction, reason)
            
            if decision == 'discard':
                dest = self.trash_root / ("positive" if label == 'pos' else "negative") / image_path.name
                shutil.move(str(image_path), str(dest))
                print(f"      Moved to {dest}")
                return False
            
        return True