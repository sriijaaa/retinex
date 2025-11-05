import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.util import img_as_float
import matplotlib.pyplot as plt

# -------------------------
# Load the Fundus Image
# -------------------------
image_path = r"C:\Users\HP\OneDrive\Desktop\retinex\pictures\img6.png"  # Change path to your image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Image not found. Check your path!")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_f = img_as_float(gray)

# -------------------------
# Quality Feature Extraction
# -------------------------

# 1. Mean Intensity (Exposure)
mean_intensity = np.mean(gray_f)

# 2. Standard Deviation (Contrast)
std_intensity = np.std(gray_f)

# 3. Skewness (Lighting balance)
skewness_val = skew(gray_f.flatten())

# 4. Kurtosis (Sharpness measure)
kurtosis_val = kurtosis(gray_f.flatten())

# 5. Entropy (Information content)
entropy_val = shannon_entropy(gray_f)

# 6. Focus Measure (Variance of Laplacian)
lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

# 7. Illumination: Mean brightness & saturation check
mean_brightness = np.mean(gray)
sat_pixels = np.sum((gray >= 250) | (gray <= 5)) / gray.size * 100

# 8. RMS Contrast
rms_contrast = np.sqrt(np.mean((gray_f - np.mean(gray_f)) ** 2))

# -------------------------
# Normalized Scoring (0–100)
# -------------------------
def normalize(value, min_val, max_val):
    return np.clip(100 * (value - min_val) / (max_val - min_val), 0, 100)

sharp_score = normalize(lap_var, 20, 200)
contrast_score = normalize(std_intensity, 0.05, 0.25)
entropy_score = normalize(entropy_val, 5.0, 7.5)
exposure_score = 100 - normalize(abs(mean_intensity - 0.5), 0, 0.25)
sat_penalty = max(0, (sat_pixels - 5) * 5)

# Weighted total quality score
quality_score = (
    0.3 * sharp_score +
    0.25 * contrast_score +
    0.25 * entropy_score +
    0.2 * exposure_score
) - sat_penalty

quality_score = np.clip(quality_score, 0, 100)

# -------------------------
# Quality Category
# -------------------------
if quality_score >= 75:
    quality_label = "✅ GOOD"
    message = "Image is well-focused, properly exposed, and detailed."
elif 50 <= quality_score < 75:
    quality_label = "⚠️ MODERATE"
    message = "Image is usable but may have slight blur or low contrast."
else:
    quality_label = "❌ POOR"
    message = "Image is blurry, dark, or lacks sufficient details. Retake recommended."

# -------------------------
# Display Results
# -------------------------
print("\n🩺 FUNDUS IMAGE QUALITY REPORT 🩺")
print(f"Mean Intensity (Exposure): {mean_intensity:.4f}")
print(f"Std Dev (Contrast): {std_intensity:.4f}")
print(f"Skewness: {skewness_val:.4f}")
print(f"Kurtosis: {kurtosis_val:.4f}")
print(f"Entropy: {entropy_val:.4f}")
print(f"Laplacian Variance (Sharpness): {lap_var:.2f}")
print(f"Saturated Pixels: {sat_pixels:.2f}%")
print(f"RMS Contrast: {rms_contrast:.4f}")
print(f"\nComposite Quality Score: {quality_score:.2f}/100")
print(f"Overall Quality: {quality_label}")
print(f"Summary: {message}")

