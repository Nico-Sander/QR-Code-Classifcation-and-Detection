import cv2
import numpy as np
import qrcode
import random
import string
import sys
import barcode
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from faker import Faker

# Add current directory to path to find repo_paths
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
from repo_paths import get_repo_paths

class SyntheticGenerator:
    def __init__(self, full_config):
        """
        Initializes the generator with separate pools for High-Res and DTD.
        Accesses nested config for both generation and patch_creation logic.
        """
        self.cfg_gen = full_config.get('generation', {})
        self.cfg_patch = full_config.get('patch_creation', {})
        self.img_size = self.cfg_gen.get('img_size', 256)
        self.fake = Faker()
        
        paths = get_repo_paths()
        self.backgrounds_dir = paths["backgrounds"]
        
        # Two distinct pools for 50/50 selection logic
        self.high_res_files = []
        self.dtd_groups = {} 
        
        if self.backgrounds_dir.exists():
            extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            all_files = []
            for ext in extensions:
                all_files.extend(list(self.backgrounds_dir.rglob(ext)))
            
            for f in all_files:
                relative_parts = f.relative_to(self.backgrounds_dir).parts
                # Identify High-Res images specifically
                if "high_res" in relative_parts:
                    self.high_res_files.append(f)
                # Group DTD images by their specific texture folder
                elif "dtd" in relative_parts:
                    group_name = f.parent.name
                    if group_name not in self.dtd_groups:
                        self.dtd_groups[group_name] = []
                    self.dtd_groups[group_name].append(f)
        
        print(f"Generator initialized: {len(self.high_res_files)} High-Res images, {len(self.dtd_groups)} DTD categories.")

    # --- HELPER METHODS ---

    def get_random_string(self, length=12):
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for _ in range(length))

    def random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def get_contrasting_colors(self):
        while True:
            bg = self.random_color()
            fg = self.random_color()
            lum_bg = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
            lum_fg = 0.114 * fg[0] + 0.587 * fg[1] + 0.299 * fg[2]
            if abs(lum_bg - lum_fg) > 70:
                return fg, bg

    def random_place(self, bg_img, overlay_img):
        h_bg, w_bg = bg_img.shape[:2]
        h_ov, w_ov = overlay_img.shape[:2]
        if h_ov >= h_bg or w_ov >= w_bg: return bg_img
        x = random.randint(0, w_bg - w_ov)
        y = random.randint(0, h_bg - h_ov)
        if overlay_img.shape[2] == 4:
            alpha = overlay_img[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis]
            fg = overlay_img[:, :, :3]
            bg_slice = bg_img[y:y+h_ov, x:x+w_ov]
            blended = (fg * alpha + bg_slice * (1.0 - alpha)).astype(np.uint8)
            bg_img[y:y+h_ov, x:x+w_ov] = blended
        else:
            bg_img[y:y+h_ov, x:x+w_ov] = overlay_img
        return bg_img

    def add_iso_noise(self, image):
        """
        Adds subtle Gaussian noise to simulate camera sensor grain.
        This helps blend the "perfect" digital QR code with the "noisy" background.
        """
        row, col, ch = image.shape
        mean = 0
        # Low sigma to ensure this is "sensor noise" not "white noise augmentation"
        sigma = random.uniform(2, 8) 
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image.astype(np.float32) + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
        
    def add_glare(self, image, probability=0.10):
        """
        Rarely adds a specular highlight (bright spot) to simulate reflection.
        Kept rare (10%) to avoid destroying too many training features.
        """
        if random.random() > probability:
            return image
            
        h, w = image.shape[:2]
        # Random position for the light source
        x = random.randint(0, w)
        y = random.randint(0, h)
        
        # Random size of the glare
        radius = random.randint(30, 80)
        
        # Create a radial gradient mask
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)
        
        # Soft falloff
        mask = 1 - np.clip(dist_from_center / radius, 0, 1)
        
        # Apply intense brightness (255) weighted by the mask
        intensity = random.uniform(0.2, 0.5)
        glare = np.full((h, w, 3), 255, dtype=np.float32) * intensity
        
        mask = mask[:, :, np.newaxis]
        result = image.astype(np.float32) + (glare * mask)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def degrade_qr(self, qr_image, probability=0.15):
        """
        Rarely adds physical damage (scratches) to the QR code pattern.
        """
        if random.random() > probability:
            return qr_image
            
        # Convert to PIL for easy drawing
        img_pil = Image.fromarray(qr_image)
        draw = ImageDraw.Draw(img_pil)
        w, h = img_pil.size
        
        # Draw 1-3 random scratch lines
        num_scratches = random.randint(1, 3)
        for _ in range(num_scratches):
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            width = random.randint(1, 3)
            # Scratches can be black (ink smear) or white (paper tear)
            color = random.choice([(0, 0, 0), (255, 255, 255)])
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
            
        return np.array(img_pil)

    # --- DISTRACTORS ---

    def generate_random_text(self):
        w, h = random.randint(80, 200), random.randint(30, 60)
        img = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        templates = [
            f"SN:{self.fake.random_number(digits=6)}",
            f"LOT {self.fake.random_number(digits=4)}",
            f"EXP {random.randint(2023, 2030)}",
            f"{self.fake.bothify(text='??-####')}",
            "WARNING"
        ]
        text = random.choice(templates)
        draw.text((5, 5), text, fill=(random.randint(0, 50), random.randint(0, 50), random.randint(0, 50), 255))
        img = img.rotate(random.randint(-90, 90), expand=True)
        return np.array(img)

    def generate_geometric_shape(self):
        s = random.randint(30, 80)
        img = Image.new('RGBA', (s, s), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        color = (*self.random_color(), 255)
        shape_type = random.choice(['circle', 'triangle', 'rect', 'cross'])
        if shape_type == 'circle': draw.ellipse([0, 0, s-1, s-1], fill=color)
        elif shape_type == 'rect': draw.rectangle([0, 0, s-1, s-1], fill=color)
        elif shape_type == 'triangle': draw.polygon([(s//2, 0), (0, s), (s, s)], fill=color)
        elif shape_type == 'cross':
            t = s // 3
            draw.rectangle([t, 0, 2*t, s], fill=color)
            draw.rectangle([0, t, s, 2*t], fill=color)
        return np.array(img)

    def generate_fake_datamatrix(self):
        s = random.randint(30, 80)
        noise = np.random.randint(0, 2, (s, s), dtype=np.uint8) * 255
        img = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGRA)
        img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255,255,255,255))
        center = (img.shape[1]//2, img.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, random.randint(0, 360), 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        return img

    def generate_barcode(self):
        try:
            code_type = random.choice(['ean13', 'code128'])
            num = ''.join(random.choices(string.digits, k=12))
            writer = ImageWriter()
            bc = barcode.get(code_type, num, writer=writer)
            pil_img = bc.render(writer_options={'module_height': 8.0, 'module_width': 0.3, 'write_text': False, 'quiet_zone': 1.0})
            img = np.array(pil_img.convert("RGBA"))
            angle = random.randint(-90, 90)
            h, w = img.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - center[0]
            M[1, 2] += (nH / 2) - center[1]
            img = cv2.warpAffine(img, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255,0))
            scale = random.uniform(0.3, 0.6)
            new_w = int(nW * scale)
            new_h = int(nH * scale)
            return cv2.resize(img, (new_w, new_h))
        except Exception:
            return None

    def generate_shipping_label(self):
        w, h = random.randint(120, 220), random.randint(60, 120)
        img = Image.new('RGBA', (w, h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        text = f"{self.fake.bs().upper()}\n{self.fake.address()}"
        draw.text((5, 5), text, fill=(0, 0, 0, 255))
        draw.rectangle([(0,0), (w-1, h-1)], outline="black", width=2)
        img = img.rotate(random.randint(-5, 5), expand=True, fillcolor=(0,0,0,0))
        return np.array(img)

    def get_distractor(self):
        dist_funcs = [
            self.generate_barcode,
            self.generate_shipping_label,
            self.generate_random_text,
            self.generate_geometric_shape,
            self.generate_fake_datamatrix
        ]
        weights = [0.3, 0.3, 0.15, 0.15, 0.1]
        choice = random.choices(dist_funcs, weights=weights, k=1)[0]
        return choice()

    # --- BACKGROUND LOGIC ---

    def get_background(self):
        """
        Selects a background with a 50/50 chance between High-Res and DTD.
        Implements sliding-window crop logic for High-Res images.
        """
        # Fallback if no images found
        if not self.high_res_files and not self.dtd_groups:
            bg = np.random.randint(50, 200, (self.img_size, self.img_size, 3), dtype=np.uint8)
            return cv2.GaussianBlur(bg, (55, 55), 0)

        # --- 50/50 Selection Logic ---
        # If one pool is empty, default to the other
        use_high_res = random.random() < 0.5
        if not self.high_res_files: use_high_res = False
        if not self.dtd_groups: use_high_res = True

        if use_high_res:
            # 1. High-Res Logic: Sliding Window Simulation
            fname = random.choice(self.high_res_files)
            img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
            if img is None: return self.get_background() # Retry on read error

            h, w = img.shape[:2]
            base_size = min(h, w)
            
            # Select random divisor from config
            divisors = self.cfg_patch.get("scale_divisors", [1.5, 2.0, 3.0, 4.0, 6.0])
            divisor = random.choice(divisors)
            
            # Calculate window size based on divisor and pixel floor
            floor = self.cfg_patch.get("absolute_pixel_floor", 128)
            crop_size = max(int(base_size / divisor), floor)
            crop_size = min(crop_size, base_size)

            # Random position (sliding window simulation)
            y = random.randint(0, h - crop_size)
            x = random.randint(0, w - crop_size)
            crop = img[y : y + crop_size, x : x + crop_size]

        else:
            # 2. DTD Logic: Category-based sampling
            group_name = random.choice(list(self.dtd_groups.keys()))
            fname = random.choice(self.dtd_groups[group_name])
            img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
            if img is None: return self.get_background()

            h, w = img.shape[:2]
            limit = min(h, w)
            # Standard texture cropping
            crop_min, crop_max = (512, 1024) if limit > 1000 else (128, 512)
            crop_size = random.randint(min(crop_min, limit), min(crop_max, limit))
            
            y = random.randint(0, h - crop_size)
            x = random.randint(0, w - crop_size)
            crop = img[y : y + crop_size, x : x + crop_size]

        # Resize the crop to the target patch size
        return cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

    def apply_lighting_gradient(self, image):
        h, w = image.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        a = random.uniform(-0.001, 0.001)
        b = random.uniform(-0.001, 0.001)
        c = random.uniform(0.7, 1.3)
        lighting = a * X + b * Y + c
        lighting = lighting[:, :, np.newaxis]
        return np.clip(image.astype(np.float32) * lighting, 0, 255).astype(np.uint8)

    # --- QR GENERATION ---

    def create_qr_with_alpha(self):
        qr = qrcode.QRCode(version=random.randint(1, 4), box_size=10, border=2)
        qr.add_data(self.get_random_string())
        qr.make(fit=True)
        img_pil = qr.make_image(fill_color="black", back_color="white")
        img_arr = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

        color_prob = self.cfg_gen.get('qr_color_prob', 0.2)
        if random.random() < color_prob:
            color_fg, color_bg = self.get_contrasting_colors()
        else:
            val_dark = random.randint(0, 60)
            val_light = random.randint(180, 255)
            c_dark = (val_dark, val_dark, val_dark)
            c_light = (val_light, val_light, val_light)
            if random.random() < self.cfg_gen.get('qr_invert_prob', 0.1):
                color_fg, color_bg = c_light, c_dark
            else:
                color_fg, color_bg = c_dark, c_light

        colored_qr = np.where(img_arr < 128, color_fg, color_bg).astype(np.uint8)
        mask = np.full((colored_qr.shape[0], colored_qr.shape[1]), 255, dtype=np.uint8)
        return colored_qr, mask

    def transform_qr(self, image, mask):
        h, w = image.shape[:2]
    
        # 1. Perspective Transform
        p_min = self.cfg_gen.get('qr_perspective_range', [0.8, 1.0])[0]
        p_max = self.cfg_gen.get('qr_perspective_range', [0.8, 1.0])[1]
        perspective_limit = w * (1.0 - random.uniform(p_min, p_max))
        
        pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
        pts2 = np.float32([
            [random.uniform(0, perspective_limit), random.uniform(0, perspective_limit)],
            [w - random.uniform(0, perspective_limit), random.uniform(0, perspective_limit)],
            [random.uniform(0, perspective_limit), h - random.uniform(0, perspective_limit)],
            [w - random.uniform(0, perspective_limit), h - random.uniform(0, perspective_limit)]
        ])
        M_p = cv2.getPerspectiveTransform(pts1, pts2)
        warped_img = cv2.warpPerspective(image, M_p, (w, h), borderValue=(255,255,255))
        warped_mask = cv2.warpPerspective(mask, M_p, (w, h), borderValue=0)
        
        # 2. Rotation WITH Canvas Expansion
        rot_range = self.cfg_gen.get('qr_rotation_range', [-180, 180])
        angle = random.uniform(rot_range[0], rot_range[1])
        
        # Calculate new bounding dimensions
        (cX, cY) = (w // 2, h // 2)
        M_r = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M_r[0, 0])
        sin = np.abs(M_r[0, 1])
        
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        
        # Adjust the rotation matrix to take into account translation
        M_r[0, 2] += (nW / 2) - cX
        M_r[1, 2] += (nH / 2) - cY

        final_img = cv2.warpAffine(warped_img, M_r, (nW, nH), borderValue=(255,255,255))
        final_mask = cv2.warpAffine(warped_mask, M_r, (nW, nH), borderValue=0)
        
        return final_img, final_mask

    # --- PIPELINE ---

    def generate_single_positive(self, output_path):
        bg = self.get_background()

        # 1. Background Distractions
        if random.random() < 0.6:
            dist = self.get_distractor()
            if dist is not None:
                bg = self.random_place(bg, dist)

        # 2. QR Creation
        qr, mask = self.create_qr_with_alpha()
        
        #  Rare Physical Damage (Scratches)
        qr = self.degrade_qr(qr, probability=0.15)
        
        qr, mask = self.transform_qr(qr, mask)

        min_s = self.cfg_gen.get('min_scale', 0.3)
        max_s = self.cfg_gen.get('max_scale', 0.8)
        scale = random.triangular(min_s, max_s, min_s)
        
        new_w = int(self.img_size * scale)
        if new_w <= 0: return False
        new_h = int(qr.shape[0] * (new_w / qr.shape[1]))
        
        qr_resized = cv2.resize(qr, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Structural Blending: Match QR sharpness to background
        # Apply slight blur to QR to kill "digital perfection"
        if random.random() < 0.8:
            sigma = random.uniform(0.5, 1.0) # Subtle blur
            qr_resized = cv2.GaussianBlur(qr_resized, (0, 0), sigma)

        qr_h, qr_w = qr_resized.shape[:2]

        # Placement Logic
        margin_x, margin_y = int(qr_w * 0.1), int(qr_h * 0.1)
        raw_min_x, raw_max_x = -margin_x, self.img_size - qr_w + margin_x
        raw_min_y, raw_max_y = -margin_y, self.img_size - qr_h + margin_y
        
        min_x, max_x = sorted([raw_min_x, raw_max_x])
        min_y, max_y = sorted([raw_min_y, raw_max_y])
        
        placed = False
        for _ in range(self.cfg_gen.get('placement_retries', 25)): 
            x_off = random.randint(min_x, max_x)
            y_off = random.randint(min_y, max_y)
            x1, y1 = max(0, x_off), max(0, y_off)
            x2, y2 = min(self.img_size, x_off + qr_w), min(self.img_size, y_off + qr_h)
            qx1, qy1 = max(0, -x_off), max(0, -y_off)
            qx2, qy2 = qx1 + (x2 - x1), qy1 + (y2 - y1)
            
            if x2 > x1 and y2 > y1:
                mask_crop = mask_resized[qy1:qy2, qx1:qx2]
                threshold = self.cfg_gen.get('min_visible_percent', 0.7)
                if np.count_nonzero(mask_crop) > (np.count_nonzero(mask_resized) * threshold):
                    alpha = (mask_crop.astype(float) / 255.0)[:, :, np.newaxis]
                    fg = qr_resized[qy1:qy2, qx1:qx2].astype(float)
                    bg_slice = bg[y1:y2, x1:x2].astype(float)
                    bg[y1:y2, x1:x2] = (fg * alpha + bg_slice * (1.0 - alpha)).astype(np.uint8)
                    placed = True
                    break
        
        if placed:
            # Global Effects applied to the fused image
            bg = self.add_iso_noise(bg) # Unify noise layers
            bg = self.add_glare(bg, probability=0.10) # Rare lighting effect
            final = self.apply_lighting_gradient(bg)
            cv2.imwrite(str(output_path), final)
            return True
        return False

    def generate_single_negative(self, output_path):
        bg = self.get_background()
        
        # Hard Negative: Partial/Fragmented QR (15% chance)
        # This teaches the model that "Just a corner" is NOT a QR code.
        if random.random() < 0.15:
            qr, mask = self.create_qr_with_alpha()
            qr, mask = self.transform_qr(qr, mask)
            
            # Crop just a corner of the QR (e.g., top-left 20%)
            h, w = qr.shape[:2]
            crop_h, crop_w = int(h * 0.2), int(w * 0.2)
            if crop_h > 10 and crop_w > 10:
                fragment = qr[0:crop_h, 0:crop_w]
                # Place this fragment randomly on the background
                bg = self.random_place(bg, fragment)

        # Standard Distractors
        num = random.randint(1, 4)
        for _ in range(num):
            dist = self.get_distractor()
            if dist is not None:
                bg = self.random_place(bg, dist)
                
        # Apply lighting/noise to negatives too
        bg = self.add_iso_noise(bg)
        final = self.apply_lighting_gradient(bg)
        cv2.imwrite(str(output_path), final)

# --- ENTRY POINT ---

def count_images(directory):
    """Counts files with valid image extensions."""
    if not directory.exists(): return 0
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]
    count = 0
    for ext in extensions:
        count += len(list(directory.glob(ext)))
    return count

def get_next_start_index(directory, prefix):
    """
    Looks for the highest index among synthetic files (syn_*.png) 
    to append new ones without overwriting.
    """
    if not directory.exists(): return 0
    # We only care about previous synthetic files (png) for indexing
    files = list(directory.glob("*.png"))
    if not files: return 0
    max_idx = -1
    for f in files:
        try:
            parts = f.stem.split('_')
            idx = int(parts[-1])
            if idx > max_idx: max_idx = idx
        except (ValueError, IndexError): continue
    return max_idx + 1

def generate_synthetic_data(config, output_dir=None, num_positives=100, num_negatives=100):
    paths = get_repo_paths()
    
    if output_dir:
        base = Path(output_dir)
        pos_dir = base / paths["class_name_pos"]
        neg_dir = base / paths["class_name_neg"]
    else:
        pos_dir = paths["patches_pos"]
        neg_dir = paths["patches_neg"]
        
    generator = SyntheticGenerator(full_config=config)
    
    print(f"--- Generator ---")
    print(f"Output: {pos_dir.parent}")
    
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Positives (Gap Filling)
    # Count ALL images (jpg + png) to see if we met the target
    current_pos_total = count_images(pos_dir)
    needed_pos = max(0, num_positives - current_pos_total)
    
    if needed_pos > 0:
        start = get_next_start_index(pos_dir, "syn_pos")
        print(f"Positives: Found {current_pos_total} total. Generating {needed_pos} new synthetic images.")
        count = 0
        pbar = tqdm(total=needed_pos, desc="Positives")
        while count < needed_pos:
            success = generator.generate_single_positive(pos_dir / f"syn_pos_{start + count:05d}.png")
            if success:
                count += 1
                pbar.update(1)
        pbar.close()
    else:
        print(f"Positives: Found {current_pos_total} total >= {num_positives} requested. Skipping generation.")

    # 2. Negatives (Gap Filling)
    current_neg_total = count_images(neg_dir)
    needed_neg = max(0, num_negatives - current_neg_total)
    
    if needed_neg > 0:
        start = get_next_start_index(neg_dir, "syn_neg")
        print(f"Negatives: Found {current_neg_total} total. Generating {needed_neg} new synthetic images.")
        for i in tqdm(range(needed_neg), desc="Negatives"):
            generator.generate_single_negative(neg_dir / f"syn_neg_{start + i:05d}.png")
    else:
        print(f"Negatives: Found {current_neg_total} total >= {num_negatives} requested. Skipping generation.")

if __name__ == "__main__":
    import yaml
    paths = get_repo_paths()
    config_path = paths["config_dir"] / "dataset_config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f: cfg = yaml.safe_load(f)
        generate_synthetic_data(cfg, num_positives=5, num_negatives=5)
    else:
        print("Config not found.")