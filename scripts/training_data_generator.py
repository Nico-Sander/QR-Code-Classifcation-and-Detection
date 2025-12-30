from pathlib import Path
import cv2
import numpy as np
import qrcode
import random
import os
import string
import glob

# --- 1. Automatic Repo Root Detection ---
def get_repo_root():
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / '.git').exists():
            return parent
    print("Warning: No .git directory found. Using script directory as root.")
    return current_path.parent

REPO_ROOT = get_repo_root()

# --- Configuration ---
BACKGROUNDS_DIR = REPO_ROOT / "data" / "backgrounds"
OUTPUT_DIR = REPO_ROOT / "data" / "synthetic_patches"

IMG_SIZE = 256
NUM_IMAGES = 40
POSITIVE_RATIO = 0.5 

# Constraints
MIN_VISIBLE_PERCENT = 0.65
MIN_SCALE = 0.4
MAX_SCALE = 1.4

def ensure_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "positive"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "negative"), exist_ok=True)

def get_random_string(length=12):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

# --- NEW: Advanced Artifacts (Glare & Distortion) ---

def add_specular_highlight(image):
    """
    Simulates a bright spot of light (sun glare) on the dashboard.
    """
    # 30% chance to add glare
    if random.random() > 0.3:
        return image

    h, w = image.shape
    
    # Create a separate layer for the light
    overlay = np.zeros((h, w), dtype=np.uint8)
    
    # Random position for the glare center
    center_x = random.randint(0, w)
    center_y = random.randint(0, h)
    
    # Random radius (large soft spot)
    radius = random.randint(30, 80)
    
    # Draw a solid white circle
    cv2.circle(overlay, (center_x, center_y), radius, 255, -1)
    
    # Heavily blur the circle to make it look like light falloff
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=radius/2, sigmaY=radius/2)
    
    # Blend: image + overlay (with clamping)
    intensity = random.uniform(0.1, 0.5)
    
    img_float = image.astype(np.float32)
    overlay_float = overlay.astype(np.float32) * intensity
    
    combined = cv2.add(img_float, overlay_float)
    return np.clip(combined, 0, 255).astype(np.uint8)

def apply_lens_distortion(image):
    """
    Simulates wide-angle / fisheye distortion common in dashcams.
    """
    # 40% chance to apply distortion
    if random.random() > 0.4:
        return image
        
    h, w = image.shape
    center_x, center_y = w / 2.0, h / 2.0
    
    # Reduced strength for subtle curvature
    strength = random.uniform(0.000001, 0.000005)

    grid_y, grid_x = np.indices((h, w), dtype=np.float32)
    
    delta_x = grid_x - center_x
    delta_y = grid_y - center_y
    distance_sq = delta_x**2 + delta_y**2
    
    # Radial distortion formula
    factor = 1.0 + strength * distance_sq
    
    # Map old coordinates to new
    map_x = center_x + delta_x / factor
    map_y = center_y + delta_y / factor
    
    # Use standard linear interpolation
    distorted_img = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    return distorted_img

# --- Distractor Generators (Text & Barcodes) ---
def get_distractor_vocabulary():
    """Returns a large list of words found in cars or on packages."""
    car_terms = [
        "AIRBAG", "VOL", "TEMP", "A/C", "MODE", "SET", "RESET", "12V", 
        "PASSENGER", "WARNING", "FUSE", "OBD", "CHECK ENGINE", "MPH", "km/h",
        "RPM", "OIL", "BRAKE", "ABS", "TCS", "AUTO", "MENU", "INFO"
    ]
    shipping_terms = [
        "FRAGILE", "PRIORITY", "EXPEDITE", "HANDLE WITH CARE", "DO NOT DROP",
        "THIS SIDE UP", "KEEP DRY", "URGENT", "DELIVERY", "LOGISTICS",
        "ZONE 1", "ZONE 5", "GROUND", "AIR", "PART NO.", "S/N:", "REF:",
        "BATCH:", "LOT:", "QTY:", "WEIGHT:", "INSPECTED BY"
    ]
    # Generate some random alphanumeric strings
    random_codes = [get_random_string(random.randint(4, 12)) for _ in range(20)]
    
    return car_terms + shipping_terms + random_codes

def make_paper_texture(h, w):
    """
    Creates a background that looks like a sticker or paper label.
    Instead of pure white, it has noise and slight gray variations.
    """
    # Base off-white color (230-255)
    base = np.random.randint(230, 256, (h, w), dtype=np.uint8)
    
    # Add subtle grain/noise
    noise = np.random.normal(0, 5, (h, w))
    texture = np.clip(base + noise, 0, 255).astype(np.uint8)
    
    # Optional: Add a subtle border 50% of the time
    if random.random() < 0.5:
        border_thickness = random.randint(1, 5)
        cv2.rectangle(texture, (0,0), (w-1, h-1), random.randint(0, 100), border_thickness)
        
    return texture

def generate_realistic_barcode():
    """
    Generates a barcode that looks like a printed sticker
    with numbers at the bottom.
    """
    # Random dimensions for a label shape
    w = random.randint(150, 300)
    h = random.randint(80, 150)
    
    # Create paper background
    img = make_paper_texture(h, w)
    
    # Define barcode area (leave space at bottom for numbers)
    bar_h = int(h * 0.7)
    
    # Draw vertical bars
    x = random.randint(10, 20)
    end_x = w - random.randint(10, 20)
    
    while x < end_x:
        # Variable bar width
        thickness = random.choice([1, 2, 3, 4])
        if x + thickness >= end_x:
            break
            
        # Draw black bar with some noise (not perfect black)
        color = random.randint(0, 50) 
        cv2.rectangle(img, (x, 10), (x + thickness, 10 + bar_h), color, -1)
        
        # Variable gap
        gap = random.choice([1, 2, 3])
        x += thickness + gap

    # Add "Human Readable" numbers below
    font = cv2.FONT_HERSHEY_PLAIN
    text = get_random_string(random.randint(8, 12))
    scale = random.uniform(0.8, 1.2)
    thk = 1
    (fw, fh), _ = cv2.getTextSize(text, font, scale, thk)
    
    # Center text below bars
    text_x = (w - fw) // 2
    text_y = h - 5
    if text_x > 0:
        cv2.putText(img, text, (text_x, text_y), font, scale, 0, thk)

    # Mask: The whole label is the object
    mask = np.full((h, w), 255, dtype=np.uint8)
    
    return img, mask

def generate_realistic_text_label():
    """
    Generates either a single button label OR a multi-line shipping label.
    """
    vocab = get_distractor_vocabulary()
    
    # Mode 1: Shipping Label (Multi-line) - 40% chance
    if random.random() < 0.4:
        w = random.randint(120, 200)
        h = random.randint(80, 150)
        img = make_paper_texture(h, w)
        
        # Pick 2-4 words
        num_lines = random.randint(2, 4)
        y = 25
        for _ in range(num_lines):
            text = random.choice(vocab)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = random.uniform(0.3, 0.5)
            cv2.putText(img, text, (10, y), font, scale, 0, 1)
            y += 25
            
    # Mode 2: Dashboard Button/Warning (Single large word) - 60% chance
    else:
        text = random.choice(vocab)
        font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX])
        scale = random.uniform(1.0, 2.5)
        thickness = random.randint(2, 4)
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        
        # Add padding for the sticker background
        w = tw + 40
        h = th + 40
        img = make_paper_texture(h, w)
        
        # Draw text centered
        cv2.putText(img, text, (20, h - 20), font, scale, 0, thickness)

    mask = np.full((h, w), 255, dtype=np.uint8)
    return img, mask

# --- ADDED: The missing function ---
def superimpose_element(bg, element, mask):
    """
    Warps and blends an element (barcode/text) onto the background.
    """
    h_bg, w_bg = bg.shape
    h_elem, w_elem = element.shape
    
    # 1. Random Perspective Warp
    src_pts = np.float32([[0, 0], [w_elem, 0], [w_elem, h_elem], [0, h_elem]])
    
    # Destination: Random scale and location
    scale = random.uniform(0.1, 0.4) 
    new_w = int(w_bg * scale)
    new_h = int(new_w * (h_elem / w_elem))
    
    x_offset = random.randint(0, w_bg - new_w)
    y_offset = random.randint(0, h_bg - new_h)
    
    tilt = random.randint(0, int(new_w * 0.2))
    dst_pts = np.float32([
        [x_offset + random.randint(0, tilt), y_offset + random.randint(0, tilt)],
        [x_offset + new_w - random.randint(0, tilt), y_offset + random.randint(0, tilt)],
        [x_offset + new_w - random.randint(0, tilt), y_offset + new_h - random.randint(0, tilt)],
        [x_offset + random.randint(0, tilt), y_offset + new_h - random.randint(0, tilt)]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_elem = cv2.warpPerspective(element, M, (w_bg, h_bg), borderValue=0)
    warped_mask = cv2.warpPerspective(mask, M, (w_bg, h_bg), borderValue=0)
    
    # 2. Blend
    mask_bool = warped_mask > 127
    out = bg.copy()
    bg_slice = out[mask_bool]
    elem_slice = warped_elem[mask_bool]
    
    # Blend with slight transparency
    alpha = 0.95
    blended = (elem_slice.astype(float) * alpha + bg_slice.astype(float) * (1 - alpha)).astype(np.uint8)
    
    out[mask_bool] = blended
    return out

def add_distractors(bg):
    """Adds random realistic labels or barcodes to the background."""
    num_distractors = random.randint(0, 3)
    for _ in range(num_distractors):
        if random.random() < 0.5:
            elem, mask = generate_realistic_barcode()
        else:
            elem, mask = generate_realistic_text_label()
        
        bg = superimpose_element(bg, elem, mask)
    return bg

# --- Visual Artifacts & Degradations ---

def add_noise_and_blur(image):
    img_float = image.astype(np.float32)
    noise_sigma = random.randint(5, 20)
    noise = np.random.normal(0, noise_sigma, img_float.shape)
    img_float += noise
    
    if random.random() < 0.6:
        k_size = random.choice([3, 7])
        img_float = cv2.GaussianBlur(img_float, (k_size, k_size), 0)
        
    if random.random() < 0.4:
        factor = random.uniform(0.5, 0.8)
        small_h = int(IMG_SIZE * factor)
        small_w = int(IMG_SIZE * factor)
        down = cv2.resize(img_float, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        img_float = cv2.resize(down, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    return np.clip(img_float, 0, 255).astype(np.uint8)

def apply_lighting_gradient(image):
    h, w = image.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    a = random.uniform(-0.002, 0.002)
    b = random.uniform(-0.002, 0.002)
    c = random.uniform(0.7, 1.3)
    lighting = a * X + b * Y + c
    image_lit = image.astype(np.float32) * lighting
    return np.clip(image_lit, 0, 255).astype(np.uint8)

# --- Background Generators ---

def load_real_background():
    if BACKGROUNDS_DIR and os.path.exists(BACKGROUNDS_DIR):
        files = glob.glob(str(BACKGROUNDS_DIR / "*.jpg")) + glob.glob(str(BACKGROUNDS_DIR / "*.png"))
        if files:
            fname = random.choice(files)
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                if h > IMG_SIZE and w > IMG_SIZE:
                    y = random.randint(0, h - IMG_SIZE)
                    x = random.randint(0, w - IMG_SIZE)
                    return img[y:y+IMG_SIZE, x:x+IMG_SIZE]
                else:
                    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return None

def generate_synthetic_background():
    bg = np.random.randint(50, 200, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (55, 55), 0)
    return bg

def get_background():
    bg = load_real_background()
    if bg is None:
        bg = generate_synthetic_background()
    bg = add_distractors(bg)
    return bg

# --- QR Generation ---

def create_qr_with_alpha():
    qr = qrcode.QRCode(version=random.randint(1, 3), box_size=10, border=2)
    qr.add_data(get_random_string())
    qr.make(fit=True)
    img_pil = qr.make_image(fill_color="black", back_color="white")
    img_arr = np.array(img_pil.convert('L'))
    
    if random.random() < 0.90:
        v1 = random.randint(0, 60)
        v2 = random.randint(200, 255)
    else:
        v1 = random.randint(200, 255)
        v2 = random.randint(0, 60)
        
    colored_qr = np.where(img_arr == 0, v1, v2).astype(np.uint8)
    
    mask = np.full_like(colored_qr, 255)
    return colored_qr, mask

def transform_qr(image, mask):
    h, w = image.shape
    src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    scale = random.uniform(0.0, 0.2)
    dst_pts = np.float32([
        [random.uniform(0, w*scale), random.uniform(0, h*scale)],
        [random.uniform(w*(1-scale), w), random.uniform(0, h*scale)],
        [random.uniform(0, w*scale), random.uniform(h*(1-scale), h)],
        [random.uniform(w*(1-scale), w), random.uniform(h*(1-scale), h)]
    ])
    M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(image, M_persp, (w, h))
    warped_mask = cv2.warpPerspective(mask, M_persp, (w, h))
    
    angle = random.randint(0, 360)
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

# --- Main Generators ---

def process_final_image(image):
    image = apply_lighting_gradient(image)
    image = add_specular_highlight(image)
    image = add_noise_and_blur(image)
    image = apply_lens_distortion(image)
    return image

def generate_positive(index):
    while True:
        bg = get_background()
        
        qr, mask = create_qr_with_alpha()
        qr, mask = transform_qr(qr, mask)
        
        scale = random.uniform(MIN_SCALE, MAX_SCALE)
        new_w = int(IMG_SIZE * scale)
        new_h = int(qr.shape[0] * (new_w / qr.shape[1]))
        qr_resized = cv2.resize(qr, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        placed = False
        qr_h, qr_w = qr_resized.shape
        
        for _ in range(20):
            x_off = random.randint(-int(qr_w*0.5), IMG_SIZE - int(qr_w*0.2))
            y_off = random.randint(-int(qr_h*0.5), IMG_SIZE - int(qr_h*0.2))
            
            x1, y1 = max(0, x_off), max(0, y_off)
            x2, y2 = min(IMG_SIZE, x_off + qr_w), min(IMG_SIZE, y_off + qr_h)
            qx1, qy1 = max(0, -x_off), max(0, -y_off)
            qx2, qy2 = qx1 + (x2 - x1), qy1 + (y2 - y1)
            
            if x2 > x1 and y2 > y1:
                mask_crop = mask_resized[qy1:qy2, qx1:qx2]
                vis_pixels = np.count_nonzero(mask_crop)
                total_pixels = np.count_nonzero(mask_resized)
                
                if total_pixels > 0 and (vis_pixels / total_pixels) > MIN_VISIBLE_PERCENT:
                    alpha = mask_crop.astype(float) / 255.0
                    fg = qr_resized[qy1:qy2, qx1:qx2].astype(float)
                    bg_slice = bg[y1:y2, x1:x2].astype(float)
                    
                    blended = fg * alpha + bg_slice * (1.0 - alpha)
                    bg[y1:y2, x1:x2] = blended.astype(np.uint8)
                    placed = True
                    
                    final_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                    final_mask[y1:y2, x1:x2] = mask_crop
                    break
        
        if placed:
            final_img = process_final_image(bg)
            
            qr_pixels = final_img[final_mask > 0]
            
            if qr_pixels.size > 0:
                contrast = np.std(qr_pixels)
                
                if contrast > 30:
                    path = os.path.join(OUTPUT_DIR, "positive", f"pos_{index:04d}.png")
                    cv2.imwrite(path, final_img)
                    print(f"[POS] Generated {path} (Contrast: {contrast:.1f})")
                    return
                else:
                    print(f"   [Retry] Discarding washed out image (Contrast: {contrast:.1f})")

def generate_negative(index):
    bg = get_background()
    final = process_final_image(bg)
    path = os.path.join(OUTPUT_DIR, "negative", f"neg_{index:04d}.png")
    cv2.imwrite(path, final)
    print(f"[NEG] Generated {path}")

if __name__ == "__main__":
    ensure_dirs()
    print("Starting generation...")
    for i in range(NUM_IMAGES):
        if random.random() < POSITIVE_RATIO:
            generate_positive(i)
        else:
            generate_negative(i)
    print(f"Done. Check folder: {OUTPUT_DIR}")