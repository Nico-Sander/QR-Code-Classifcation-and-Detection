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
OUTPUT_DIR = REPO_ROOT / "data" / "synthetic_output"

IMG_SIZE = 256
NUM_IMAGES = 20
POSITIVE_RATIO = 0.5 

# Constraints
MIN_VISIBLE_PERCENT = 0.25
MIN_SCALE = 0.3
MAX_SCALE = 1.2

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
    # Sigma is high to create a soft gradient
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=radius/2, sigmaY=radius/2)
    
    # Blend: image + overlay (with clamping)
    # The 'intensity' controls how "washed out" the glare is.
    intensity = random.uniform(0.3, 0.7)
    
    # Convert to float to avoid overflow before clipping
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
    
    # Camera matrix (simulated)
    cam_matrix = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]])
    
    # Distortion coefficients (k1, k2, p1, p2, k3)
    # Positive k1 = Barrel distortion (fisheye), Negative k1 = Pincushion
    # Dashcams usually have Barrel distortion.
    k1 = random.uniform(0.1, 0.3) 
    k2 = random.uniform(0.01, 0.05)
    dist_coeffs = np.array([k1, k2, 0, 0, 0])
    
    # We use undistort with the SAME matrix to simulate the distortion effect inverse
    # or typically remapping. A quick trick for synthesis is simply running un-distortion
    # with inverted parameters, but OpenCV's undistort is for REMOVING it.
    # To ADD distortion, we can map coordinates manually or cheat by using 'projectPoints'.
    # However, the easiest stable way in OpenCV is strictly remapping.
    
    # Let's use a faster approximation: shrinking the image slightly and pinning corners? 
    # No, let's do it properly with initUndistortRectifyMap but invert the logic roughly
    # by treating the input as the "undistorted" result we want to distort.
    # Actually, simpler approach for data aug:
    
    mapx, mapy = cv2.initUndistortRectifyMap(cam_matrix, dist_coeffs, None, cam_matrix, (w,h), 5)
    # This usually removes distortion. To ADD it, we'd need the inverse mapping.
    # But often, random small warping is enough.
    
    # Alternative: Simple localized warp (Bulge effect)
    # This is more robust for synthesis than fighting with calibration matrices.
    
    # create mapping grid
    flex_x = np.zeros((h, w), np.float32)
    flex_y = np.zeros((h, w), np.float32)
    
    center_x, center_y = w/2, h/2
    strength = random.uniform(0.00001, 0.00005) # Magnitude of bulge

    grid_y, grid_x = np.indices((h, w))
    
    # Radial distortion formula: r_new = r * (1 + k*r^2)
    # We calculate distance from center
    delta_x = grid_x - center_x
    delta_y = grid_y - center_y
    distance_sq = delta_x**2 + delta_y**2
    
    # Apply factor
    factor = 1 + strength * distance_sq
    
    # Map old coordinates to new
    map_x = center_x + delta_x / factor # Divide by factor to pull pixels IN (fisheye look)
    map_y = center_y + delta_y / factor
    
    distorted_img = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)
    
    return distorted_img

# --- Distractor Generators (Text & Barcodes) ---

def generate_synthetic_barcode():
    h, w = 100, 300
    img = np.full((h, w), 255, dtype=np.uint8)
    x = 10
    while x < w - 10:
        thickness = random.randint(1, 4)
        if x + thickness >= w - 10: break
        img[:, x:x+thickness] = 0
        x += thickness + random.randint(1, 6)
    mask = np.full((h, w), 255, dtype=np.uint8)
    return img, mask

def generate_random_text_img():
    text = random.choice(["AIRBAG", "VOL", "TEMP", "A/C", "MODE", "SET", "RESET", "12V", "PASSENGER", "WARNING", get_random_string(5)])
    font_scale = random.uniform(1.0, 3.0)
    thickness = random.randint(2, 5)
    font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX])
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    canvas_w, canvas_h = w + 20, h + 20
    img = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    cv2.putText(img, text, (10, h + 5), font, font_scale, 0, thickness)
    mask = cv2.bitwise_not(img)
    return img, mask

def superimpose_element(bg, element, mask):
    h_bg, w_bg = bg.shape
    h_elem, w_elem = element.shape
    src_pts = np.float32([[0, 0], [w_elem, 0], [w_elem, h_elem], [0, h_elem]])
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
    warped_elem = cv2.warpPerspective(element, M, (w_bg, h_bg), borderValue=255)
    warped_mask = cv2.warpPerspective(mask, M, (w_bg, h_bg), borderValue=0)
    
    mask_bool = warped_mask > 127
    out = bg.copy()
    bg_slice = out[mask_bool]
    elem_slice = warped_elem[mask_bool]
    blended = (bg_slice.astype(float) * (elem_slice.astype(float) / 255.0)).astype(np.uint8)
    out[mask_bool] = blended
    return out

def add_distractors(bg):
    num_distractors = random.randint(0, 3)
    for _ in range(num_distractors):
        if random.random() < 0.5:
            elem, mask = generate_synthetic_barcode()
        else:
            elem, mask = generate_random_text_img()
        bg = superimpose_element(bg, elem, mask)
    return bg

# --- Visual Artifacts & Degradations ---

def add_noise_and_blur(image):
    img_float = image.astype(np.float32)
    noise_sigma = random.randint(5, 25)
    noise = np.random.normal(0, noise_sigma, img_float.shape)
    img_float += noise
    
    if random.random() < 0.6:
        k_size = random.choice([3, 5])
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
    
    v1, v2 = random.randint(0, 255), random.randint(0, 255)
    while abs(v1 - v2) < 50: v2 = random.randint(0, 255)
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
    """Applies all final "lens and light" effects in order."""
    image = apply_lighting_gradient(image)  # Base shadow/gradient
    image = add_specular_highlight(image)   # Bright glare
    image = add_noise_and_blur(image)       # Sensor noise/blur
    image = apply_lens_distortion(image)    # Optical distortion
    return image

def generate_positive(index):
    bg = get_background()
    
    qr, mask = create_qr_with_alpha()
    qr, mask = transform_qr(qr, mask)
    
    scale = random.uniform(MIN_SCALE, MAX_SCALE)
    new_w = int(IMG_SIZE * scale)
    new_h = int(qr.shape[0] * (new_w / qr.shape[1]))
    qr_resized = cv2.resize(qr, (new_w, new_h))
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    placed = False
    attempts = 0
    qr_h, qr_w = qr_resized.shape
    
    while not placed and attempts < 20:
        attempts += 1
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

    if placed:
        # Apply the full chain of artifacts
        final = process_final_image(bg)
        
        path = os.path.join(OUTPUT_DIR, "positive", f"pos_{index:04d}.png")
        cv2.imwrite(path, final)
        print(f"[POS] Generated {path}")

def generate_negative(index):
    bg = get_background()
    
    # Apply the full chain of artifacts
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