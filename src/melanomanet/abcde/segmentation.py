import cv2
import numpy as np


def extract_lesion_mask(image: np.ndarray) -> np.ndarray:
    """
    Extract lesion mask using image segmentation.

    Uses multiple color space thresholding and morphological operations
    to robustly segment the lesion from the background.

    Args:
        image: RGB image (H, W, 3) with values [0, 255]

    Returns:
        Binary mask (H, W) where 255 = lesion, 0 = background
    """
    h, w = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's thresholding
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine if we need to invert based on multiple heuristics
    # Heuristic 1: Check if center region is darker than edges
    center_region = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    edge_region = np.concatenate(
        [
            gray[: h // 8, :].flatten(),
            gray[-h // 8 :, :].flatten(),
            gray[:, : w // 8].flatten(),
            gray[:, -w // 8 :].flatten(),
        ]
    )

    center_mean = float(np.mean(center_region))
    edge_mean = float(np.mean(edge_region))

    # Heuristic 2: Check which region is smaller (lesion is typically smaller than background)
    white_pixels = int(np.sum(mask > 0))
    black_pixels = int(np.sum(mask == 0))

    # Heuristic 3: Check mean intensity of each region
    mask_white_mean = float(np.mean(gray[mask > 0])) if white_pixels > 0 else 0.0
    mask_black_mean = float(np.mean(gray[mask == 0])) if black_pixels > 0 else 0.0

    # Decision: invert if lesion appears to be in the dark region
    should_invert = False

    # If center is darker and white region is larger, invert
    if center_mean < edge_mean and white_pixels > black_pixels:
        should_invert = True
    # If white region has higher intensity and is larger, invert
    elif mask_white_mean > mask_black_mean and white_pixels > (h * w * 0.5):
        should_invert = True

    if should_invert:
        mask = cv2.bitwise_not(mask)

    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find largest connected component (remove small artifacts)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels > 1:
        # Get largest component (excluding background at index 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Fill holes in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        # Fill the contour
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, [largest_contour], -1, 255, -1)
        mask = mask_filled

    return mask
