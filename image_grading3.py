import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.util import img_as_float
from skimage.filters import sobel
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Load the Fundus Image
# -------------------------
image_path = r"C:\Users\HP\OneDrive\Desktop\retinex\input\enhanced_1029_right.jpeg"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Image not found. Check your path!")

# Convert to grayscale and float
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_f = img_as_float(gray)

# -------------------------
# Advanced Quality Metrics
# -------------------------

# 1. Sharpness Metrics (Multiple methods for robustness)
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
sobel_mean = np.mean(sobel(gray_f))
gradient = np.mean(np.gradient(gray_f))

# 2. Contrast Metrics
std_intensity = np.std(gray_f)
rms_contrast = np.sqrt(np.mean((gray_f - np.mean(gray_f)) ** 2))
michelson_contrast = (np.max(gray_f) - np.min(gray_f)) / (np.max(gray_f) + np.min(gray_f) + 1e-10)

# 3. Exposure & Brightness
mean_intensity = np.mean(gray_f)
median_intensity = np.median(gray_f)

# 4. Dynamic Range
dynamic_range = np.percentile(gray_f, 99) - np.percentile(gray_f, 1)

# 5. Information Content
entropy_val = shannon_entropy(gray_f)

# 6. Saturation Analysis (over/underexposure)
overexposed = np.sum(gray >= 245) / gray.size * 100
underexposed = np.sum(gray <= 10) / gray.size * 100
total_saturation = overexposed + underexposed

# 7. Histogram Analysis
hist_std = np.std(cv2.calcHist([gray], [0], None, [256], [0, 256]))

# 8. Local Contrast (using patches)
h, w = gray.shape
patches = []
for i in range(0, h-50, 50):
    for j in range(0, w-50, 50):
        patch = gray_f[i:i+50, j:j+50]
        patches.append(np.std(patch))
local_contrast_mean = np.mean(patches)

# -------------------------
# Adaptive Normalization
# -------------------------
def adaptive_normalize(value, good_min, good_max, excellent_max=None):
    """
    Normalize with consideration for excellent range
    """
    if excellent_max and value >= excellent_max:
        return 100
    if value >= good_max:
        return 95
    if value <= good_min:
        return 20
    # Linear interpolation between good_min and good_max
    return 20 + (value - good_min) / (good_max - good_min) * 75

# -------------------------
# Scoring Individual Components
# -------------------------

# Sharpness Score (most important for fundus)
sharp_score_lap = adaptive_normalize(laplacian_var, 50, 150, 300)
sharp_score_sobel = adaptive_normalize(sobel_mean * 1000, 5, 15, 25)
sharpness_score = (sharp_score_lap * 0.6 + sharp_score_sobel * 0.4)

# Contrast Score
contrast_score_std = adaptive_normalize(std_intensity, 0.08, 0.20, 0.30)
contrast_score_rms = adaptive_normalize(rms_contrast, 0.08, 0.20, 0.30)
contrast_score = (contrast_score_std * 0.5 + contrast_score_rms * 0.5)

# Exposure Score (penalize deviation from optimal 0.35-0.55 range)
exposure_deviation = min(abs(mean_intensity - 0.35), abs(mean_intensity - 0.55))
if 0.35 <= mean_intensity <= 0.55:
    exposure_score = 100
elif 0.25 <= mean_intensity <= 0.65:
    exposure_score = 85 - (exposure_deviation * 100)
else:
    exposure_score = max(20, 70 - (exposure_deviation * 150))

# Dynamic Range Score
dr_score = adaptive_normalize(dynamic_range, 0.3, 0.6, 0.8)

# Entropy Score (information content)
entropy_score = adaptive_normalize(entropy_val, 6.0, 7.0, 7.5)

# Saturation Penalty
if total_saturation < 1:
    sat_score = 100
elif total_saturation < 3:
    sat_score = 90
elif total_saturation < 5:
    sat_score = 70
else:
    sat_score = max(20, 70 - (total_saturation - 5) * 8)

# Local Contrast Score
local_contrast_score = adaptive_normalize(local_contrast_mean, 0.02, 0.08, 0.15)

# -------------------------
# Weighted Final Score
# -------------------------
quality_score = (
    0.30 * sharpness_score +      # Sharpness is critical
    0.20 * contrast_score +        # Overall contrast
    0.15 * exposure_score +        # Proper exposure
    0.12 * dr_score +              # Dynamic range
    0.10 * entropy_score +         # Information content
    0.08 * sat_score +             # No saturation
    0.05 * local_contrast_score    # Local detail
)

quality_score = np.clip(quality_score, 0, 100)

# -------------------------
# Quality Classification
# -------------------------
if quality_score >= 80:
    quality_label = "EXCELLENT"
    icon = "✅"
    color_code = "\033[92m"  # Green
    description = "Outstanding image quality. Clear, well-focused with excellent detail and proper exposure."
elif quality_score >= 65:
    quality_label = "GOOD"
    icon = "✓"
    color_code = "\033[94m"  # Blue
    description = "Good image quality. Suitable for diagnostic purposes with minor imperfections."
elif quality_score >= 50:
    quality_label = "ACCEPTABLE"
    icon = "⚠"
    color_code = "\033[93m"  # Yellow
    description = "Acceptable quality. Usable but may benefit from improved lighting or focus."
else:
    quality_label = "POOR"
    icon = "✗"
    color_code = "\033[91m"  # Red
    description = "Poor image quality. Retake recommended for reliable diagnosis."

reset_code = "\033[0m"

# -------------------------
# Output Report
# -------------------------
print("\n" + "="*60)
print(f"{color_code}🩺 FUNDUS IMAGE QUALITY ASSESSMENT{reset_code}")
print("="*60)
print(f"\n{color_code}{icon} Overall Quality: {quality_label} ({quality_score:.1f}/100){reset_code}")
print(f"\n{description}")
print("\n" + "-"*60)
print("📊 Quality Breakdown:")
print(f"  • Sharpness:        {sharpness_score:.1f}/100")
print(f"  • Contrast:         {contrast_score:.1f}/100")
print(f"  • Exposure:         {exposure_score:.1f}/100")
print(f"  • Dynamic Range:    {dr_score:.1f}/100")
print(f"  • Detail Level:     {entropy_score:.1f}/100")
print("="*60 + "\n")