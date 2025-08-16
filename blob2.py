import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from skimage.color import rgb2gray
from skimage.measure import regionprops, label

def preprocess_image(img):
    """ Preprocess fundus image for microaneurysm detection """
    if len(img.shape) == 3:  # Color image
        green = img[:, :, 1]  # Green channel emphasizes lesions
    else:
        green = img

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # Gaussian blur (reduce vessel noise)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Morphological Top-hat (highlight small dark spots)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)

    return tophat

def detect_microaneurysms(img):
    """ Detect microaneurysms using Laplacian of Gaussian blob detection """
    blobs = blob_log(img, 
                     min_sigma=2, max_sigma=4,  # tighter range
                     num_sigma=8, threshold=0.07)

    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)  # radius

    # Filter by size (approximate MA size)
    blobs = [b for b in blobs if 3 <= b[2] <= 7]

    return blobs

# Load image
img = cv2.imread('img.jpg')

preprocessed = preprocess_image(img)
blobs = detect_microaneurysms(preprocessed)

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(preprocessed, cmap='gray')
ax[0].set_title("Preprocessed Fundus")

ax[1].imshow(rgb2gray(img), cmap='gray')
for y, x, r in blobs:
    c = plt.Circle((x, y), r, color='red', fill=False, linewidth=1.2)
    ax[1].add_patch(c)

ax[1].set_title("Detected Microaneurysms")
plt.show()
