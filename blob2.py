import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter

def single_scale_retinex(img, sigma):
    """Single Scale Retinex (SSR)"""
    # Convert to float and add small epsilon to avoid log(0)
    img_float = img.astype(np.float64) + 1.0
    
    # Apply Gaussian filter (surround function)
    img_blur = gaussian_filter(img_float, sigma=sigma)
    
    # SSR: log(I) - log(I * G)
    retinex = np.log10(img_float) - np.log10(img_blur)
    
    return retinex

def multi_scale_retinex(img, sigmas=[15, 80, 250]):
    """Multi Scale Retinex (MSR)"""
    retinex_scales = []
    
    for sigma in sigmas:
        retinex_scales.append(single_scale_retinex(img, sigma))
    
    # Average all scales
    msr = np.mean(retinex_scales, axis=0)
    
    return msr

def retinex_with_color_restore(img, sigmas=[15, 80, 250], alpha=125, beta=46):
    """Multi Scale Retinex with Color Restoration (MSRCR)"""
    # Apply MSR
    msr = multi_scale_retinex(img, sigmas)
    
    # Color restoration
    img_float = img.astype(np.float64) + 1.0
    img_sum = np.sum(img_float, axis=2, keepdims=True)
    
    # Avoid division by zero
    img_sum = np.where(img_sum == 0, 1, img_sum)
    
    color_restoration = beta * (np.log10(alpha * img_float) - np.log10(img_sum))
    
    # Combine MSR with color restoration
    msrcr = color_restoration * msr
    
    return msrcr

def normalize_retinex(retinex_img):
    """Normalize Retinex output to 0-255 range"""
    # Remove extreme values (optional)
    retinex_img = np.clip(retinex_img, -4, 4)
    
    # Normalize to 0-255
    min_val = np.min(retinex_img)
    max_val = np.max(retinex_img)
    
    if max_val > min_val:
        normalized = (retinex_img - min_val) / (max_val - min_val) * 255
    else:
        normalized = np.zeros_like(retinex_img)
    
    return normalized.astype(np.uint8)

def preprocess_image(img):
    """ Enhanced preprocessing with both CLAHE and Retinex """
    # Extract green channel (most informative for fundus images)
    green = img[:, :, 1] if len(img.shape) == 3 else img
    
    # 1. CLAHE Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(green)
    
    # 2. Retinex Enhancement (for illumination correction)
    # Apply Multi-Scale Retinex on green channel
    retinex_result = multi_scale_retinex(green, sigmas=[15, 80, 200])
    retinex_normalized = normalize_retinex(retinex_result)
    
    # 3. Combine CLAHE and Retinex
    # Weighted combination for better results
    combined_enhancement = cv2.addWeighted(clahe_enhanced, 0.6, retinex_normalized, 0.4, 0)
    
    # 4. Median filter (reduce background noise, preserve small spots)
    median = cv2.medianBlur(combined_enhancement, 5)
    
    # 5. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    # Top-hat (highlight bright spots)
    tophat = cv2.morphologyEx(median, cv2.MORPH_TOPHAT, kernel)
    
    # Black-hat (highlight dark spots, like microaneurysms)
    blackhat = cv2.morphologyEx(median, cv2.MORPH_BLACKHAT, kernel)
    
    # 6. Combine both morphological results
    combined = cv2.addWeighted(tophat, 0.7, blackhat, 0.7, 0)
    
    # 7. Gamma correction (enhance contrast for small lesions)
    gamma = 1.2
    combined = np.array(255 * ((combined / 255) ** gamma), dtype=np.uint8)
    
    return combined, clahe_enhanced, retinex_normalized

def detect_microaneurysms(img, min_radius=2, max_radius=7):
    """ Enhanced microaneurysm detection with adjustable parameters """
    # LoG blob detection with fine-tuned parameters
    blobs = blob_log(img, 
                     min_sigma=1, max_sigma=6, 
                     num_sigma=15, threshold=0.02)
    
    if len(blobs) > 0:
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)  # Convert sigma to radius
        
        # Filter by size (typical microaneurysm size)
        blobs = [b for b in blobs if min_radius <= b[2] <= max_radius]
    
    return blobs

def detect_microaneurysms_adaptive_threshold(img):
    """ Alternative detection method using adaptive thresholding """
    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and circularity
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 10 < area < 150:  # Typical MA area range
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.5:  # Reasonably circular
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        radius = np.sqrt(area / np.pi)
                        candidates.append([cy, cx, radius])
    
    return candidates

# --------------------------
# Load and Process Image
# --------------------------
try:
    img = cv2.imread('img5.png')
    if img is None:
        raise FileNotFoundError("Could not load image 'img.jpg'")
    
    # Convert BGR to RGB for proper display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess with enhanced method
    preprocessed, clahe_only, retinex_only = preprocess_image(img_rgb)
    
    # Detect microaneurysms using LoG method only
    blobs_log = detect_microaneurysms(preprocessed)
    
    print(f"LoG method detected: {len(blobs_log)} microaneurysms")
    
    # Simplified Visualization - Only 3 outputs
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(rgb2gray(img_rgb), cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # CLAHE enhanced
    axes[1].imshow(clahe_only, cmap='gray')
    axes[1].set_title("CLAHE Enhanced")
    axes[1].axis('off')
    
    # LoG detection results on original
    axes[2].imshow(rgb2gray(img_rgb), cmap='gray')
    for y, x, r in blobs_log:
        circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=1.5)
        axes[2].add_patch(circle)
    axes[2].set_title(f"LoG Detection ({len(blobs_log)} detections)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Optional: Save results
    # cv2.imwrite('enhanced_fundus.jpg', cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR))
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure 'img.jpg' exists in the current directory")
except Exception as e:
    print(f"An error occurred: {e}")
