import cv2
import numpy as np
from sklearn.cluster import KMeans

def detect_optic_disc_medical(image):
    """
    Medical-grade optic disc detection for fundus images.
    """
    height, width = image.shape[:2]
    
    # Use red channel where OD appears brightest
    red_channel = image[:, :, 2]
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_red = clahe.apply(red_channel)
    
    # Heavy Gaussian blur to find the general bright region
    blurred = cv2.GaussianBlur(enhanced_red, (71, 71), 0)
    
    # Find the brightest point
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
    
    # Create circular mask for OD
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # OD radius estimation based on typical fundus image proportions
    od_radius = int(min(width, height) * 0.08)  # About 8% of image dimension
    
    # Ensure the radius is reasonable
    od_radius = max(30, min(od_radius, 150))
    
    cv2.circle(mask, max_loc, od_radius, 255, -1)
    
    return mask, max_loc, od_radius

def create_vessel_suppression_mask(image):
    """
    Create a mask to suppress vessel-related false positives.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Use green channel (vessels appear dark here)
    green_channel = image[:, :, 1]
    
    # Invert green channel to make vessels bright
    inverted_green = cv2.bitwise_not(green_channel)
    
    # Apply morphological opening to detect vessel-like structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(inverted_green, cv2.MORPH_OPEN, kernel)
    
    # Threshold to create vessel mask
    _, vessel_mask = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Dilate vessel mask to ensure coverage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_mask = cv2.dilate(vessel_mask, kernel, iterations=1)
    
    return vessel_mask

def detect_exudates_medical(image_path, output_path='medical_exudate_detection.jpg', 
                           min_size=20, max_size=1000, brightness_threshold=160):
    """
    Medical-grade exudate detection similar to clinical annotation tools.
    
    Parameters:
    - min_size: Minimum area of exudates (pixels)
    - max_size: Maximum area of exudates (pixels)  
    - brightness_threshold: Minimum brightness for detection (0-255)
    """
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return 0
    
    original = img.copy()
    height, width = img.shape[:2]
    
    # Step 1: Detect optic disc
    od_mask, od_center, od_radius = detect_optic_disc_medical(img)
    
    # Step 2: Create vessel suppression mask
    vessel_mask = create_vessel_suppression_mask(img)
    
    # Step 3: Enhance image contrast
    # Convert to LAB for better brightness analysis
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    
    # Also work with green channel (exudates appear bright here)
    green_channel = img[:, :, 1]
    enhanced_green = clahe.apply(green_channel)
    
    # Step 4: Create candidate masks
    
    # Primary mask: Brightness-based detection
    _, bright_mask_l = cv2.threshold(enhanced_l, brightness_threshold, 255, cv2.THRESH_BINARY)
    _, bright_mask_green = cv2.threshold(enhanced_green, brightness_threshold - 10, 255, cv2.THRESH_BINARY)
    
    # Combine brightness masks
    brightness_mask = cv2.bitwise_or(bright_mask_l, bright_mask_green)
    
    # Secondary mask: Very bright regions (regardless of other criteria)
    _, very_bright_mask = cv2.threshold(enhanced_l, min(brightness_threshold + 40, 220), 255, cv2.THRESH_BINARY)
    
    # Color-based mask for yellowish regions
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for yellowish-white exudates
    lower_exudate = np.array([10, 20, brightness_threshold])
    upper_exudate = np.array([40, 120, 255])
    color_mask = cv2.inRange(hsv, lower_exudate, upper_exudate)
    
    # Step 5: Combine detection criteria
    # Primary detection: bright AND yellowish
    primary_candidates = cv2.bitwise_and(brightness_mask, color_mask)
    
    # Secondary detection: very bright regions
    secondary_candidates = very_bright_mask
    
    # Combine both
    all_candidates = cv2.bitwise_or(primary_candidates, secondary_candidates)
    
    # Step 6: Remove optic disc and vessels
    all_candidates = cv2.bitwise_and(all_candidates, cv2.bitwise_not(od_mask))
    all_candidates = cv2.bitwise_and(all_candidates, cv2.bitwise_not(vessel_mask))
    
    # Step 7: Morphological operations to clean up
    # Remove small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(all_candidates, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Fill small holes
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    
    # Step 8: Find and filter contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Scale size constraints based on image size
    image_scale = (width * height) / (1024 * 1024)  # Normalize to 1MP
    scaled_min_size = max(min_size * image_scale, 15)
    scaled_max_size = max_size * image_scale
    
    valid_exudates = []
    detection_info = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if scaled_min_size <= area <= scaled_max_size:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Shape analysis
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Allow various shapes but exclude very elongated regions (likely vessels)
            if 0.2 <= aspect_ratio <= 5.0:
                # Verify brightness within the contour
                mask_roi = np.zeros_like(enhanced_l)
                cv2.fillPoly(mask_roi, [contour], 255)
                mean_brightness = cv2.mean(enhanced_l, mask_roi)[0]
                
                if mean_brightness >= brightness_threshold - 30:
                    # Calculate distance from optic disc (for analysis)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        distance_from_od = np.sqrt((cx - od_center[0])**2 + (cy - od_center[1])**2)
                        
                        valid_exudates.append(contour)
                        detection_info.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'brightness': mean_brightness,
                            'center': (cx, cy),
                            'distance_from_od': distance_from_od
                        })
    
    # Step 9: Create medical-style visualization
    result = original.copy()
    
    # Draw clean bounding boxes like in medical software
    for i, info in enumerate(detection_info):
        x, y, w, h = info['bbox']
        
        # Draw bounding rectangle in yellow (like your reference image)
        cv2.rectangle(result, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 255), 2)  # Yellow box
        
        # Optional: Add small number label
        cv2.putText(result, str(i+1), (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Draw optic disc boundary for reference (like your image shows)
    cv2.circle(result, od_center, od_radius, (0, 0, 255), 2)  # Red circle for OD
    
    # Save result
    cv2.imwrite(output_path, result)
    
    return len(valid_exudates), detection_info

def batch_detect_with_different_settings(image_path, base_output='medical_detection'):
    """
    Run detection with different parameter settings to find optimal configuration.
    """
    # Different parameter combinations
    settings = [
        {'brightness_threshold': 140, 'min_size': 15, 'max_size': 800, 'name': 'sensitive'},
        {'brightness_threshold': 150, 'min_size': 25, 'max_size': 600, 'name': 'strict_size'},
    ]
    
    results = {}
    
    for setting in settings:
        name = setting.pop('name')
        output_path = f"{base_output}_{name}.jpg"
        
        count, info = detect_exudates_medical(image_path, output_path, **setting)
        results[name] = {'count': count, 'info': info}
        
    return results

# Main execution
if __name__ == "__main__":
    image_path = r'C:\Users\HP\OneDrive\Desktop\retinex\originalImages\1052.jpg'
    
    # Test different parameter combinations
    all_results = batch_detect_with_different_settings(image_path)