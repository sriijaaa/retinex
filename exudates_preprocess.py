import cv2
import numpy as np

img_path = r"C:\Users\HP\OneDrive\Desktop\retinex\originalImages\1012.jpg"
img = cv2.imread(img_path)

if img is None:
    print("Image not found!")
    exit()

img = cv2.resize(img, (512, 512))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, fundus_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# Fill holes and smooth the mask
fundus_mask = cv2.morphologyEx(
    fundus_mask, 
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    iterations=2
)

# Erode to remove boundary
fundus_mask = cv2.erode(
    fundus_mask,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
    iterations=1
)


green = img[:, :, 1].astype(np.float32)
green_masked = cv2.bitwise_and(green.astype(np.uint8), green.astype(np.uint8), mask=fundus_mask)

# Multi-scale illumination correction
background1 = cv2.GaussianBlur(green_masked, (71, 71), 0)
background2 = cv2.GaussianBlur(green_masked, (31, 31), 0)

# Combine multiple scales for better illumination estimation
background = cv2.addWeighted(background1, 0.6, background2, 0.4, 0)

# Normalize
normalized = cv2.subtract(green_masked, background)
normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# Apply CLAHE with optimized parameters
clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
enhanced = clahe.apply(normalized)

# Apply mask
enhanced = cv2.bitwise_and(enhanced, enhanced, mask=fundus_mask)

# Create blurred version
blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

# Unsharp mask: original + (original - blurred) * amount
sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

# Clip values to valid range
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

# Apply mask
sharpened = cv2.bitwise_and(sharpened, sharpened, mask=fundus_mask)


kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
tophat = cv2.morphologyEx(sharpened, cv2.MORPH_TOPHAT, kernel_tophat)

# Blend top-hat with enhanced image for better exudate visibility
preprocessed = cv2.add(sharpened, tophat)

# Apply mask one final time
preprocessed = cv2.bitwise_and(preprocessed, preprocessed, mask=fundus_mask)

preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)

cv2.imshow("Preprocessed Image", preprocessed_bgr)
cv2.imwrite("preprocessed_image.png", preprocessed_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("✓ Done! Saved: preprocessed_image.png")