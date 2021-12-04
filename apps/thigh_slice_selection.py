"""
Created on Mon Jun 29 19:07:02 2020
@author: ynomura
"""

import numpy as np
from scipy import ndimage
from skimage import filters


def main(volume):

    for k in range(volume.shape[0]-2, -1, -1):

        img = volume[k, :, :]
        threshold = filters.threshold_otsu(img)

        binary_img = (img >= threshold).astype(np.uint8)
        binary_img = ndimage.morphology.binary_fill_holes(binary_img)

        labeled, num_labels = ndimage.label(binary_img)

        # Extract the largest and 2nd largest area
        hist = ndimage.histogram(labeled, 1, num_labels, num_labels)
        largest_idx = np.argmax(hist)

        if num_labels > 1:
            second_idx = np.where(hist == np.sort(hist)[-2])[0][0]

            if hist[largest_idx] // 10 > hist[second_idx]:
                return k

        else:
            return k


