import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging
import os

def single_scale_retinex(img: np.ndarray, variance: float) -> np.ndarray:
    """Apply Single-Scale Retinex processing to an image."""
    try:
        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
        return retinex
    except Exception as e:
        logging.error(f"Error in Single-Scale Retinex: {e}")
        return img

def multi_scale_retinex(img: np.ndarray, variance_list: List[float]) -> np.ndarray:
    """Apply Multi-Scale Retinex processing to an image."""
    try:
        retinex = np.zeros_like(img)
        for variance in variance_list:
            retinex += single_scale_retinex(img, variance)
        retinex = retinex / len(variance_list)
        return retinex
    except Exception as e:
        logging.error(f"Error in Multi-Scale Retinex: {e}")
        return img

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    
    Args:
        img: Input image (BGR or grayscale)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of the neighborhood for adaptive histogram equalization
    
    Returns:
        CLAHE enhanced image
    """
    try:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        if len(img.shape) == 3:  # Color image
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:  # Grayscale image
            return clahe.apply(img)
    except Exception as e:
        logging.error(f"Error in CLAHE processing: {e}")
        return img

def normalize_retinex_output(img_retinex: np.ndarray) -> np.ndarray:
    """Normalize Retinex output to 0-255 range."""
    try:
        normalized = np.zeros_like(img_retinex)
        for i in range(img_retinex.shape[2]):
            unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
            zero_count = next((c for u, c in zip(unique, count) if u == 0), 0)
            low_val = unique[0] / 100.0
            high_val = unique[-1] / 100.0
            
            for u, c in zip(unique, count):
                if u < 0 and c < zero_count * 0.1:
                    low_val = u / 100.0
                if u > 0 and c < zero_count * 0.1:
                    high_val = u / 100.0
                    break
            
            channel = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
            channel_min = np.min(channel)
            channel_max = np.max(channel)
            if channel_max > channel_min:
                normalized[:, :, i] = (channel - channel_min) / (channel_max - channel_min) * 255
            else:
                normalized[:, :, i] = channel
                
        return np.uint8(normalized)
    except Exception as e:
        logging.error(f"Error in normalization: {e}")
        return img_retinex.astype(np.uint8)

def process_msr(img: np.ndarray, variance_list: List[float]) -> np.ndarray:
    """Process image using Multi-Scale Retinex and normalize output."""
    try:
        img_float = np.float64(img) + 1.0
        img_retinex = multi_scale_retinex(img_float, variance_list)
        return normalize_retinex_output(img_retinex)
    except Exception as e:
        logging.error(f"Error processing MSR: {e}")
        return img.astype(np.uint8)

def process_ssr(img: np.ndarray, variance: float) -> np.ndarray:
    """Process image using Single-Scale Retinex and normalize output."""
    try:
        img_float = np.float64(img) + 1.0
        img_retinex = single_scale_retinex(img_float, variance)
        
        # Expand dimensions if needed for normalization function
        if len(img_retinex.shape) == 2:
            img_retinex = np.expand_dims(img_retinex, axis=2)
        
        normalized = normalize_retinex_output(img_retinex)
        
        # Return to original dimensions
        if len(img.shape) == 2 and len(normalized.shape) == 3:
            normalized = np.squeeze(normalized, axis=2)
            
        return normalized
    except Exception as e:
        logging.error(f"Error processing SSR: {e}")
        return img.astype(np.uint8)

def enhance_with_retinex_clahe(img: np.ndarray, 
                               method: str = 'msr',
                               variance_list: Optional[List[float]] = None,
                               variance: float = 80,
                               clahe_clip_limit: float = 2.0,
                               clahe_tile_size: Tuple[int, int] = (8, 8),
                               apply_clahe_first: bool = False) -> np.ndarray:
    """
    Enhanced image processing combining Retinex and CLAHE algorithms.
    
    Args:
        img: Input image
        method: 'msr' for Multi-Scale Retinex or 'ssr' for Single-Scale Retinex
        variance_list: List of variances for MSR (default: [15, 80, 250])
        variance: Single variance for SSR
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_size: CLAHE tile grid size
        apply_clahe_first: Whether to apply CLAHE before or after Retinex
    
    Returns:
        Enhanced image
    """
    try:
        if variance_list is None:
            variance_list = [15, 80, 250]
            
        if apply_clahe_first:
            img_clahe = apply_clahe(img, clahe_clip_limit, clahe_tile_size)
            if method.lower() == 'msr':
                result = process_msr(img_clahe, variance_list)
            else:
                result = process_ssr(img_clahe, variance)
        else:
            if method.lower() == 'msr':
                img_retinex = process_msr(img, variance_list)
            else:
                img_retinex = process_ssr(img, variance)
            result = apply_clahe(img_retinex, clahe_clip_limit, clahe_tile_size)
            
        return result
    except Exception as e:
        logging.error(f"Error in combined enhancement: {e}")
        return img

def adaptive_enhance(img: np.ndarray) -> np.ndarray:
    """
    Adaptive enhancement that automatically selects best parameters based on image characteristics.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        mean_brightness = np.mean(gray)
        contrast = np.std(gray)
        
        if mean_brightness < 80:  # Dark image
            clahe_clip_limit = 3.0
            variance_list = [15, 80, 200]
            apply_clahe_first = True
        elif mean_brightness > 180:  # Bright image
            clahe_clip_limit = 1.5
            variance_list = [30, 120, 300]
            apply_clahe_first = False
        else:  # Normal brightness
            clahe_clip_limit = 2.0
            variance_list = [15, 80, 250]
            apply_clahe_first = False
            
        if contrast < 30:
            clahe_clip_limit += 1.0
            
        return enhance_with_retinex_clahe(
            img, 
            method='msr',
            variance_list=variance_list,
            clahe_clip_limit=clahe_clip_limit,
            apply_clahe_first=apply_clahe_first
        )
    except Exception as e:
        logging.error(f"Error in adaptive enhancement: {e}")
        return img

if __name__ == "__main__":
    import os
    import time
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define input and output directories
    input_dir = r'C:\Users\HP\OneDrive\Desktop\retinex\img.jpg'
    output_dir = r'C:\Users\HP\OneDrive\Desktop\retinex\output11'
    
    # Get the current working directory and constructed paths for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image files from input directory
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    input_images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not input_images:
        print(f"No images found in {input_dir}. Please ensure images are present and the path is correct.")
    else:
        for img_name in input_images:
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load image at {img_path}")
                continue
            
            print(f"Processing {img_name} with shape: {img.shape}")
            
            # Apply optimal enhancement (MSR + CLAHE)
            print(f"Processing {img_name} with MSR + CLAHE enhancement...")
            start_time = time.time()
            
            enhanced_img = enhance_with_retinex_clahe(
                img,
                method='msr',
                variance_list=[15, 80, 250],
                clahe_clip_limit=2.0,
                apply_clahe_first=False
            )
            
            end_time = time.time()
            print(f"Processing {img_name} completed in {end_time - start_time:.2f} seconds")
            
            # Save the enhanced image to the output directory
            output_path = os.path.join(output_dir, f"enhanced_{img_name}")
            cv2.imwrite(output_path, enhanced_img)
            print(f"Enhanced image saved to: {output_path}")