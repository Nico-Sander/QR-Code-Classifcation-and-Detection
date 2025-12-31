import cv2
import numpy as np
import qrcode
import random
import string
import sys
from tqdm import tqdm
from pathlib import Path

# Add current directory to path to find sibling modules
sys.path.append(str(Path(__file__).parent))
from project_paths import resolve_path

class SyntheticGenerator:
    def __init__(self, config, backgrounds_dir, img_size=256):
        """
        config: Dictionary containing the 'generation' section of dataset_config.yaml
        """
        self.cfg = config
        self.backgrounds_dir = Path(backgrounds_dir)
        self.img_size = img_size
        
        # Pre-load backgrounds list
        self.bg_files = []
        if self.backgrounds_dir.exists():
            self.bg_files = list(self.backgrounds_dir.glob("*.jpg")) + \
                            list(self.backgrounds_dir.glob("*.png"))

    def get_random_string(self, length=12):
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for _ in range(length))

    # --- AUGMENTATION METHODS ---

    def add_specular_highlight(self, image):
        if random.random() > self.cfg['glare_prob']:
            return image
        h, w = image.shape
        overlay = np.zeros((h, w), dtype=np.uint8)
        
        radius_range = self.cfg['glare_radius_range']
        radius = random.randint(radius_range[0], radius_range[1])
        
        center_x = random.randint(0, w)
        center_y = random.randint(0, h)
        cv2.circle(overlay, (center_x, center_y), radius, 255, -1)
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=radius/2, sigmaY=radius/2)
        
        intensity = random.uniform(self.cfg['glare_intensity_range'][0], self.cfg['glare_intensity_range'][1])
        combined = cv2.add(image.astype(np.float32), overlay.astype(np.float32) * intensity)
        return np.clip(combined, 0, 255).astype(np.uint8)

    def apply_lens_distortion(self, image):
        if random.random() > self.cfg['distortion_prob']:
            return image
        h, w = image.shape
        strength = random.uniform(self.cfg['distortion_strength_range'][0], self.cfg['distortion_strength_range'][1])
        
        grid_y, grid_x = np.indices((h, w), dtype=np.float32)
        center_x, center_y = w / 2.0, h / 2.0
        
        delta_x = grid_x - center_x
        delta_y = grid_y - center_y
        distance_sq = delta_x**2 + delta_y**2
        factor = 1.0 + strength * distance_sq
        
        map_x = center_x + delta_x / factor
        map_y = center_y + delta_y / factor
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    def add_noise_and_blur(self, image):
        img_float = image.astype(np.float32)
        
        sigma_range = self.cfg['noise_sigma_range']
        noise = np.random.normal(0, random.randint(sigma_range[0], sigma_range[1]), img_float.shape)
        img_float += noise
        
        if random.random() < self.cfg['blur_prob']:
            k_choices = [k for k in self.cfg['blur_kernels'] if k % 2 == 1]
            if k_choices:
                k_size = random.choice(k_choices)
                img_float = cv2.GaussianBlur(img_float, (k_size, k_size), 0)
        
        if random.random() < self.cfg['downscale_prob']:
            factor = random.uniform(self.cfg['downscale_factor_range'][0], self.cfg['downscale_factor_range'][1])
            small = cv2.resize(img_float, (0,0), fx=factor, fy=factor)
            img_float = cv2.resize(small, (self.img_size, self.img_size))

        return np.clip(img_float, 0, 255).astype(np.uint8)

    def apply_lighting_gradient(self, image):
        h, w = image.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        
        slope_range = self.cfg['lighting_slope_range']
        base_range = self.cfg['lighting_base_range']
        
        a = random.uniform(slope_range[0], slope_range[1])
        b = random.uniform(slope_range[0], slope_range[1])
        c = random.uniform(base_range[0], base_range[1])
        
        lighting = a * X + b * Y + c
        return np.clip(image.astype(np.float32) * lighting, 0, 255).astype(np.uint8)

    def process_final_image(self, image):
        image = self.apply_lighting_gradient(image)
        image = self.add_specular_highlight(image)
        image = self.add_noise_and_blur(image)
        image = self.apply_lens_distortion(image)
        return image

    # --- BACKGROUND LOGIC ---

    def get_background(self):
        # Try to load Real Background
        if self.bg_files:
            fname = random.choice(self.bg_files)
            img = cv2.imread(str(fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                if h > self.img_size and w > self.img_size:
                    y = random.randint(0, h - self.img_size)
                    x = random.randint(0, w - self.img_size)
                    bg = img[y:y+self.img_size, x:x+self.img_size]
                else:
                    bg = cv2.resize(img, (self.img_size, self.img_size))
                return bg
        
        # Fallback to Synthetic Noise
        bg = np.random.randint(50, 200, (self.img_size, self.img_size), dtype=np.uint8)
        return cv2.GaussianBlur(bg, (55, 55), 0)

    # --- QR LOGIC ---

    def create_qr_with_alpha(self):
        qr = qrcode.QRCode(version=random.randint(1, 3), box_size=10, border=2)
        qr.add_data(self.get_random_string())
        qr.make(fit=True)
        img_pil = qr.make_image(fill_color="black", back_color="white")
        img_arr = np.array(img_pil.convert('L'))
        
        dark_range = self.cfg['qr_dark_range']
        light_range = self.cfg['qr_light_range']

        if random.random() > self.cfg['qr_invert_prob']:
            v1 = random.randint(dark_range[0], dark_range[1])
            v2 = random.randint(light_range[0], light_range[1])
        else:
            v1 = random.randint(light_range[0], light_range[1])
            v2 = random.randint(dark_range[0], dark_range[1])
            
        colored_qr = np.where(img_arr == 0, v1, v2).astype(np.uint8)
        mask = np.full_like(colored_qr, 255)
        return colored_qr, mask

    def transform_qr(self, image, mask):
        h, w = image.shape
        src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
        
        p_range = self.cfg['qr_perspective_range']
        scale = random.uniform(p_range[0], p_range[1])
        
        dst_pts = np.float32([
            [random.uniform(0, w*scale), random.uniform(0, h*scale)],
            [random.uniform(w*(1-scale), w), random.uniform(0, h*scale)],
            [random.uniform(0, w*scale), random.uniform(h*(1-scale), h)],
            [random.uniform(w*(1-scale), w), random.uniform(h*(1-scale), h)]
        ])
        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_img = cv2.warpPerspective(image, M_persp, (w, h))
        warped_mask = cv2.warpPerspective(mask, M_persp, (w, h))
        
        r_range = self.cfg['qr_rotation_range']
        angle = random.randint(r_range[0], r_range[1])
        
        center = (w//2, h//2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = np.abs(M_rot[0, 0]), np.abs(M_rot[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M_rot[0, 2] += (nW / 2) - center[0]
        M_rot[1, 2] += (nH / 2) - center[1]
        
        final_img = cv2.warpAffine(warped_img, M_rot, (nW, nH), flags=cv2.INTER_LINEAR)
        final_mask = cv2.warpAffine(warped_mask, M_rot, (nW, nH), flags=cv2.INTER_NEAREST)
        return final_img, final_mask

    # --- GENERATION METHODS ---

    def generate_single_positive(self, output_path):
        while True:
            bg = self.get_background()
            qr, mask = self.create_qr_with_alpha()
            qr, mask = self.transform_qr(qr, mask)
            
            scale = random.uniform(self.cfg['min_scale'], self.cfg['max_scale'])
            new_w = int(self.img_size * scale)
            if new_w <= 0: continue
            new_h = int(qr.shape[0] * (new_w / qr.shape[1]))
            
            qr_resized = cv2.resize(qr, (new_w, new_h))
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            placed = False
            qr_h, qr_w = qr_resized.shape
            
            for _ in range(self.cfg['placement_retries']):
                x_off = random.randint(-int(qr_w*0.5), self.img_size - int(qr_w*0.2))
                y_off = random.randint(-int(qr_h*0.5), self.img_size - int(qr_h*0.2))
                
                x1, y1 = max(0, x_off), max(0, y_off)
                x2, y2 = min(self.img_size, x_off + qr_w), min(self.img_size, y_off + qr_h)
                qx1, qy1 = max(0, -x_off), max(0, -y_off)
                qx2, qy2 = qx1 + (x2 - x1), qy1 + (y2 - y1)
                
                if x2 > x1 and y2 > y1:
                    mask_crop = mask_resized[qy1:qy2, qx1:qx2]
                    vis = np.count_nonzero(mask_crop)
                    tot = np.count_nonzero(mask_resized)
                    
                    if tot > 0 and (vis / tot) > self.cfg['min_visible_percent']:
                        alpha = mask_crop.astype(float) / 255.0
                        fg = qr_resized[qy1:qy2, qx1:qx2].astype(float)
                        bg_slice = bg[y1:y2, x1:x2].astype(float)
                        bg[y1:y2, x1:x2] = (fg * alpha + bg_slice * (1.0 - alpha)).astype(np.uint8)
                        placed = True
                        break
            
            if placed:
                final = self.process_final_image(bg)
                if final.std() > self.cfg['contrast_threshold']:
                    cv2.imwrite(str(output_path), final)
                    return

    def generate_single_negative(self, output_path):
        bg = self.get_background()
        final = self.process_final_image(bg)
        cv2.imwrite(str(output_path), final)


# --- MODULE INTERFACE ---

def get_next_start_index(directory, prefix):
    """
    Scans directory for files like '{prefix}_00123.png' and returns the highest index + 1.
    Prevents overwriting existing files during incremental runs.
    """
    if not directory.exists():
        return 0
    
    files = list(directory.glob("*.png"))
    if not files:
        return 0
        
    max_idx = -1
    for f in files:
        try:
            # Assumes format: prefix_XXXXX.png
            idx_str = f.stem.split('_')[-1]
            idx = int(idx_str)
            if idx > max_idx:
                max_idx = idx
        except (ValueError, IndexError):
            continue
            
    return max_idx + 1

def generate_synthetic_data(config, output_dir, num_positives, num_negatives):
    output_dir = Path(output_dir)
    gen_config = config['generation']
    
    # Resolve background path using project_paths logic
    bg_dir = resolve_path(config['paths']['backgrounds'])
    
    generator = SyntheticGenerator(
        config=gen_config, 
        backgrounds_dir=bg_dir, 
        img_size=config['dataset']['img_size']
    )
    
    (output_dir / "positive").mkdir(parents=True, exist_ok=True)
    (output_dir / "negative").mkdir(parents=True, exist_ok=True)

    # 1. Positives
    pos_dir = output_dir / "positive"
    existing_pos_count = len(list(pos_dir.glob("*.png")))
    needed_pos = max(0, num_positives - existing_pos_count)
    
    if needed_pos > 0:
        start_idx = get_next_start_index(pos_dir, "syn_pos")
        print(f"   [Gen] Positives: Found {existing_pos_count}, generating {needed_pos} new starting from ID {start_idx}...")
        for i in tqdm(range(needed_pos), desc="Generating Synthetic Pos"):
            idx = start_idx + i
            generator.generate_single_positive(pos_dir / f"syn_pos_{idx:05d}.png")
    else:
        print(f"   [Gen] Positives: Found {existing_pos_count}, enough for target {num_positives}. Reuse enabled.")

    # 2. Negatives
    neg_dir = output_dir / "negative"
    existing_neg_count = len(list(neg_dir.glob("*.png")))
    needed_neg = max(0, num_negatives - existing_neg_count)
    
    if needed_neg > 0:
        start_idx = get_next_start_index(neg_dir, "syn_neg")
        print(f"   [Gen] Negatives: Found {existing_neg_count}, generating {needed_neg} new starting from ID {start_idx}...")
        for i in tqdm(range(needed_neg), desc="Generating Synthetic Neg"):
            idx = start_idx + i
            generator.generate_single_negative(neg_dir / f"syn_neg_{idx:05d}.png")
    else:
        print(f"   [Gen] Negatives: Found {existing_neg_count}, enough for target {num_negatives}. Reuse enabled.")