import cv2
import numpy as np
import qrcode
import random
import os
import string
import math

# --- Configuration ---
OUTPUT_DIR = "/home/nico/workspace/github.com/Nico-Sander/KI-Project-WS2526/data/synthetic_qr_dataset"
NUM_IMAGES = 20
IMG_SIZE = 256

# Constraints
MIN_QR_AREA_RATIO = 0.2   # QR must be at least 15% of the total image area (before crop)
MIN_VISIBLE_PERCENT = 0.3 # At least 25% of the actual QR pixels must be visible

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_random_string(length=12):
    """Generates random data for the QR code."""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def generate_dynamic_background(shape):
    """Generates a 256x256 grayscale background with varying structure/texture."""
    h, w = shape
    mode = random.choice(['noise', 'gradient', 'shapes', 'checkerboard', 'flat'])
    
    if mode == 'noise':
        mean = random.randint(50, 200)
        sigma = random.randint(10, 50)
        bg = np.random.normal(mean, sigma, (h, w)).astype(np.uint8)
    
    elif mode == 'gradient':
        bg = np.zeros((h, w), dtype=np.uint8)
        v1 = random.randint(0, 255)
        v2 = random.randint(0, 255)
        if random.choice([True, False]): # Vertical
            for i in range(h):
                bg[i, :] = v1 + (v2 - v1) * i / h
        else: # Horizontal
            for i in range(w):
                bg[:, i] = v1 + (v2 - v1) * i / w
                
    elif mode == 'shapes':
        bg_color = random.randint(0, 255)
        bg = np.full((h, w), bg_color, dtype=np.uint8)
        for _ in range(random.randint(5, 15)):
            color = random.randint(0, 255)
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (random.randint(0, w), random.randint(0, h))
            if random.choice([True, False]):
                cv2.rectangle(bg, pt1, pt2, color, -1)
            else:
                radius = random.randint(10, 50)
                cv2.circle(bg, pt1, radius, color, -1)
                
    elif mode == 'checkerboard':
        bg = np.zeros((h, w), dtype=np.uint8)
        tile_size = random.randint(20, 60)
        c1 = random.randint(0, 150)
        c2 = random.randint(100, 255)
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                if (x // tile_size + y // tile_size) % 2 == 0:
                    bg[y:y+tile_size, x:x+tile_size] = c1
                else:
                    bg[y:y+tile_size, x:x+tile_size] = c2
    else: # Flat
        bg = np.full((h, w), random.randint(0, 255), dtype=np.uint8)

    return bg

def create_random_qr():
    """Generates a QR code image with randomized foreground/background intensities."""
    qr = qrcode.QRCode(
        version=random.randint(1, 3), # Keep version lower for clearer features
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=2,
    )
    qr.add_data(get_random_string())
    qr.make(fit=True)

    img_pil = qr.make_image(fill_color="black", back_color="white")
    img_arr = np.array(img_pil.convert('L'))
    
    # Randomize contrast
    val1 = random.randint(0, 255)
    val2 = random.randint(0, 255)
    
    # Ensure distinct contrast (at least 60 difference)
    while abs(val1 - val2) < 60:
        val2 = random.randint(0, 255)
    
    colored_qr = np.where(img_arr == 0, val1, val2).astype(np.uint8)
    return colored_qr

def apply_perspective_transform(image):
    """
    Applies a random perspective transform (homography) to simulate camera angles.
    """
    h, w = image.shape
    
    # Define source points (corners of the image)
    src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    
    # Define destination points (perturbed corners)
    # Distortion scale determines how "extreme" the angle is (0.0 to 0.3 recommended)
    distortion_scale = random.uniform(0.0, 0.3) 
    
    dx = w * distortion_scale
    dy = h * distortion_scale
    
    # Randomly move corners inward
    dst_pts = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],             
        [random.uniform(w-dx, w), random.uniform(0, dy)],           
        [random.uniform(0, dx), random.uniform(h-dy, h)],           
        [random.uniform(w-dx, w), random.uniform(h-dy, h)]          
    ])
    
    # Get Perspective Matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Warp Image
    warped_img = cv2.warpPerspective(image, M, (w, h), borderValue=0)
    
    # Warp Mask (to distinguish QR from padding)
    mask = np.ones((h, w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, M, (w, h), borderValue=0)
    
    return warped_img, warped_mask

def rotate_image(image, mask, angle):
    """Rotates image and mask."""
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated_img = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    
    return rotated_img, rotated_mask

def generate_synthetic_image(index):
    bg = generate_dynamic_background((IMG_SIZE, IMG_SIZE))
    qr_img = create_random_qr()
    
    # 1. Perspective Transform (Camera Angle)
    qr_warped, qr_mask = apply_perspective_transform(qr_img)
    
    # 2. Rotation
    angle = random.randint(0, 360)
    qr_rotated, qr_mask_rotated = rotate_image(qr_warped, qr_mask, angle)
    
    # 3. Resize (Scaling) with Minimum Size Constraints
    # Calculate scale needed to meet minimum area constraint
    qr_area = qr_rotated.shape[0] * qr_rotated.shape[1]
    bg_area = IMG_SIZE * IMG_SIZE
    
    # Enforce minimum size logic
    min_scale = math.sqrt((bg_area * MIN_QR_AREA_RATIO) / qr_area)
    max_scale = 1.3 
    
    if min_scale > max_scale: min_scale = max_scale 
    
    scale_factor = random.uniform(min_scale, max_scale)
    
    new_w = int(qr_rotated.shape[1] * scale_factor)
    new_h = int(qr_rotated.shape[0] * scale_factor)
    
    qr_final = cv2.resize(qr_rotated, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_final = cv2.resize(qr_mask_rotated, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 4. Placement Loop (Retry until 25% visibility constraint is met)
    valid_placement = False
    qr_h, qr_w = qr_final.shape
    
    max_attempts = 50
    attempts = 0
    
    while not valid_placement and attempts < max_attempts:
        attempts += 1
        
        # Random Coordinates (allowing bleed off canvas)
        x_offset = random.randint(-int(qr_w * 0.7), IMG_SIZE - int(qr_w * 0.1))
        y_offset = random.randint(-int(qr_h * 0.7), IMG_SIZE - int(qr_h * 0.1))
        
        # Calculate Intersection
        x1 = max(x_offset, 0)
        y1 = max(y_offset, 0)
        x2 = min(x_offset + qr_w, IMG_SIZE)
        y2 = min(y_offset + qr_h, IMG_SIZE)
        
        # QR internal coordinates for intersection
        qr_x1 = max(0, -x_offset)
        qr_y1 = max(0, -y_offset)
        qr_x2 = qr_x1 + (x2 - x1)
        qr_y2 = qr_y1 + (y2 - y1)
        
        if x2 > x1 and y2 > y1:
            # Check Pixel Visibility
            # We count how many 'mask' pixels (255) actually ended up inside the canvas
            mask_crop = mask_final[qr_y1:qr_y2, qr_x1:qr_x2]
            
            total_mask_pixels = np.count_nonzero(mask_final)
            visible_mask_pixels = np.count_nonzero(mask_crop)
            
            if total_mask_pixels > 0:
                ratio = visible_mask_pixels / total_mask_pixels
                
                if ratio >= MIN_VISIBLE_PERCENT:
                    valid_placement = True
                    
                    # Perform Blending
                    bg_crop = bg[y1:y2, x1:x2]
                    qr_crop = qr_final[qr_y1:qr_y2, qr_x1:qr_x2]
                    
                    # Create alpha map
                    alpha = mask_crop.astype(float) / 255.0
                    
                    fg = qr_crop.astype(float)
                    bkg = bg_crop.astype(float)
                    
                    blended = (fg * alpha + bkg * (1.0 - alpha)).astype(np.uint8)
                    bg[y1:y2, x1:x2] = blended

    if valid_placement:
        filename = os.path.join(OUTPUT_DIR, f"sample_{index:04d}.png")
        cv2.imwrite(filename, bg)
        print(f"Generated: {filename} (Visible: {ratio:.2%})")
    else:
        print(f"Skipped {index}: Could not place QR with >{MIN_VISIBLE_PERCENT:.0%} visibility.")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Generating {NUM_IMAGES} synthetic images with perspective transforms...")
    for i in range(NUM_IMAGES):
        generate_synthetic_image(i)
    print("Done.")