import cv2
import numpy as np
import argparse
from pathlib import Path

def normalize01(x: np.ndarray) -> np.ndarray:
    """Normalize array to range [0, 1]."""
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def compute_roi_score(
    gray: np.ndarray,
    st_sigma: float = 2.0,
    w_grad: float = 0.5,
    w_aniso: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a score map for Regions of Interest based on:
    1. Contrast (Gradient Magnitude)
    2. Parallel Lines (Structure Tensor Anisotropy)

    Args:
        gray: Grayscale input image (uint8).
        st_sigma: Smoothing sigma for structure tensor (controls scale of "parallelism").
        w_grad: Weight for gradient magnitude contribution (0.0 to 1.0).
        w_aniso: Weight for anisotropy contribution (0.0 to 1.0).

    Returns:
        tuple: (score_map, gradient_norm, anisotropy_norm)
            - score_map: Combined score (0.0 to 1.0)
            - gradient_norm: Normalized gradient magnitude (debug)
            - anisotropy_norm: Normalized anisotropy (debug)
    """
    # Normalize image to 0..1 float
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Gradient Magnitude (Contrast)
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(dx, dy)
    mag01 = normalize01(mag)

    # 2. Structure Tensor Anisotropy (Parallel Lines / Orientation stability)
    # J = Gaussian * (grad * grad.T)
    jxx = dx * dx
    jyy = dy * dy
    jxy = dx * dy
    
    # Smooth the tensor elements
    # ksize must be odd and roughly 3*sigma to capture the window
    ksize = int(max(3, round(st_sigma * 6))) | 1
    jxx = cv2.GaussianBlur(jxx, (ksize, ksize), st_sigma)
    jyy = cv2.GaussianBlur(jyy, (ksize, ksize), st_sigma)
    jxy = cv2.GaussianBlur(jxy, (ksize, ksize), st_sigma)

    # Eigenvalues of structure tensor
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    root = np.sqrt(disc)
    
    l1 = 0.5 * (tr + root)
    l2 = 0.5 * (tr - root)
    
    # Anisotropy = (l1 - l2) / (l1 + l2). 
    # High when dominant direction exists (lines), Low when isotropic (flat or corners)
    aniso = (l1 - l2) / (l1 + l2 + 1e-6)
    aniso01 = normalize01(aniso)

    # Combine
    score = (w_grad * mag01 + w_aniso * aniso01)
    score = normalize01(score)

    return score, mag01, aniso01

    return score, mag01, aniso01

def extract_patches(
    img: np.ndarray,
    score_map: np.ndarray,
    patch_size: int = 128,
    max_patches: int = 50,
    nms_threshold: float = 0.3
) -> list[tuple[tuple[int, int, int, int], float]]:
    """
    Extract patches from image based on score map maxima with NMS.
    
    Returns:
        List of tuples ((x1, y1, x2, y2), score)
    """
    h, w = img.shape[:2]
    # Working copy of score map to modify during NMS
    s_map = score_map.copy()
    
    patches = []
    half_size = patch_size // 2
    
    for _ in range(max_patches):
        # Find maximum value location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(s_map)
        
        if max_val < 0.1:  # Stop if score is too low
            break
            
        cx, cy = max_loc
        
        # Define patch coordinates
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, x1 + patch_size)
        y2 = min(h, y1 + patch_size)
        
        # Adjust if close to border to keep fixed size if possible, 
        # but simplistic crop is safest:
        if x2 - x1 < patch_size:
            if x1 == 0: x2 = min(w, patch_size)
            if x2 == w: x1 = max(0, w - patch_size)
        if y2 - y1 < patch_size:
            if y1 == 0: y2 = min(h, patch_size)
            if y2 == h: y1 = max(0, h - patch_size)
        
        # We don't extract the image anymore, just coordinates
        if (x2 > x1) and (y2 > y1):
            patches.append(((x1, y1, x2, y2), max_val))
        
        # NMS: Zero out the region in score map
        # We zero out a slightly larger region to avoid overlaps
        nms_pad = int(patch_size * 0.8) 
        n_x1 = max(0, cx - nms_pad // 2)
        n_y1 = max(0, cy - nms_pad // 2)
        n_x2 = min(w, cx + nms_pad // 2)
        n_y2 = min(h, cy + nms_pad // 2)
        
        s_map[n_y1:n_y2, n_x1:n_x2] = 0
        
    return patches

    return patches

def process_image(
    image_path: Path, 
    output_path: str = None, 
    display: bool = False,
    show_patches: bool = False
) -> np.ndarray:
    """
    Process a single image and return the visualization image.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute interest map
    score, mag, aniso = compute_roi_score(gray, st_sigma=2.0, w_grad=0.4, w_aniso=0.6)

    final_vis = None

    # 1. Visualization Type A: Bounding Boxes (Priority if show_patches is True)
    if show_patches:
        print(f"[{image_path.name}] Extracting patches...")
        patch_coords = extract_patches(img, score, patch_size=128, max_patches=50)
        print(f"[{image_path.name}] Found {len(patch_coords)} regions.")
        
        boxes_img = img.copy()
        for i, ((x1, y1, x2, y2), p_score) in enumerate(patch_coords):
             cv2.rectangle(boxes_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
             cv2.putText(boxes_img, f"{p_score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        final_vis = boxes_img

    # 2. Visualization Type B: Heatmap Composite (Default/Debug)
    elif display:
        score_uint8 = (score * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(score_uint8, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        # Stack for nice view: Original | Gradient | Anisotropy | Result
        h, w = img.shape[:2]
        target_h = 400
        scale = target_h / float(h)
        w_new = int(w * scale)
        def rz(x): return cv2.resize(x, (w_new, target_h))
        
        show_img = rz(img)
        show_mag = cv2.applyColorMap((rz(mag) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        show_aniso = cv2.applyColorMap((rz(aniso) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        show_res = rz(vis)
        
        final_vis = np.hstack([show_img, show_mag, show_aniso, show_res])

    # Save output if requested (Logic slightly changed: we save the specific vis type)
    if output_path and final_vis is not None:
        # If batch processing, we might need output directory logic, 
        # but for now keep it simple: only save if single file or specific logic.
        # If output_path is provided for a directory input, it might overwrite.
        # Let's assume output_path is only for single file mode or handled by caller.
        cv2.imwrite(output_path, final_vis)
        print(f"Saved result to {output_path}")

    return final_vis

def main():
    parser = argparse.ArgumentParser(description="Detect ROI based on contrast and parallel lines.")
    parser.add_argument("input", help="Path to input image or directory")
    parser.add_argument("--output", "-o", help="Path to save output visualization (single file mode only)")
    parser.add_argument("--no-display", action="store_true", help="Do not open a window")
    parser.add_argument("--show-patches", action="store_true", help="Extract and show ROI patches bounding boxes")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist.")
        return

    # Gather images
    files = []
    if input_path.is_dir():
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            files.extend(sorted(input_path.glob(ext)))
    else:
        files = [input_path]

    if not files:
        print("No images found.")
        return

    print(f"Processing {len(files)} images...")

    window_name = "ROI Detection"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for i, fpath in enumerate(files):
        print(f"--- Processing {i+1}/{len(files)}: {fpath.name} ---")
        
        # For batch output saving, we could derive filename, but user didn't explicitly ask for batch save.
        # We only pass output_path if it's a single file, to avoid overwriting.
        out_p = args.output if len(files) == 1 else None

        vis_img = process_image(fpath, output_path=out_p, display=not args.no_display, show_patches=args.show_patches)

        if not args.no_display and vis_img is not None:
             # Resize for display if huge
            h, w = vis_img.shape[:2]
            if h > 900:
                scale = 900 / h
                vis_img = cv2.resize(vis_img, (int(w*scale), 900))
            
            cv2.imshow(window_name, vis_img)
            
            print("Press SPACE/ENTER for next, ESC to quit.")
            k = cv2.waitKey(0) & 0xFF
            if k == 27: # ESC
                print("Quitting.")
                break
    
    if not args.no_display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
