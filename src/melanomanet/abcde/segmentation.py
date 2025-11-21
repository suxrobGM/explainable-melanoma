"""
Lesion segmentation utilities for ABCDE analysis.
"""

import cv2
import numpy as np


def extract_lesion_mask(image: np.ndarray) -> np.ndarray:
    """
    Extract lesion mask using image segmentation.

    Uses Otsu's thresholding on the grayscale image to separate
    the lesion from the background.

    Args:
        image: RGB image (H, W, 3) with values [0, 255]

    Returns:
        Binary mask (H, W) where 255 = lesion, 0 = background
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's thresholding
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if lesion is darker than background (common in dermoscopy)
    if np.mean(gray[mask > 0]) > np.mean(gray[mask == 0]):
        mask = cv2.bitwise_not(mask)

    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask
