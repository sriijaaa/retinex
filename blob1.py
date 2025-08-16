import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_local
from skimage.feature import blob_log, blob_dog, blob_doh

# ----------------------
# 1. Load image
# ----------------------
image = imread("img.jpg")

# ----------------------
# 2. Remove red channel (keep only green & blue)
# ----------------------
image_no_red = image.copy()
image_no_red[:, :, 0] = 0  # zero out red channel

# ----------------------
# 3. Extract green channel for analysis
# ----------------------
green_channel = image_no_red[:, :, 1]

# ----------------------
# 4. Adaptive thresholding for binarization
# ----------------------
block_size = 51  # must be odd
local_thresh = threshold_local(green_channel, block_size, offset=10)
binary_mask = green_channel > local_thresh

# ----------------------
# 5. Blob detection
# ----------------------
blobs_log = blob_log(binary_mask, min_sigma=3, max_sigma=10, num_sigma=10, threshold=0.02)
blobs_dog = blob_dog(binary_mask, min_sigma=3, max_sigma=10, threshold=0.02)
blobs_doh = blob_doh(binary_mask, min_sigma=3, max_sigma=30, threshold=0.02)

# Radius for each method (LoG returns sqrt(2)*sigma approx)
blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

# ----------------------
# 6. Visualization
# ----------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
titles = ["Binary Mask", "Blob Detection - LoG", "Blob Detection - DoG", "Blob Detection - DoH"]
blob_sets = [None, blobs_log, blobs_dog, blobs_doh]

for i, ax in enumerate(axes):
    if i == 0:
        ax.imshow(binary_mask, cmap='gray')
    else:
        ax.imshow(image)
        for blob in blob_sets[i]:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', fill=False, linewidth=1.5)
            ax.add_patch(c)
    ax.set_title(titles[i])
    ax.axis("off")

plt.tight_layout()
plt.show()