"""
ABCDE feature extraction functions.

Individual analyzers for each ABCDE criterion:
- Asymmetry
- Border irregularity
- Color variation
- Diameter
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def analyze_asymmetry(image: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Analyze asymmetry by comparing lesion halves.

    Compares the lesion along two perpendicular axes (horizontal and vertical)
    to measure asymmetry.

    Args:
        image: RGB image (H, W, 3)
        mask: Binary lesion mask (H, W)

    Returns:
        Tuple of (asymmetry_score, visualization)
        Score ranges from 0 (symmetric) to 1 (highly asymmetric)
    """
    # Find lesion centroid
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return 0.0, image.copy()

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    h, w = mask.shape

    # Split horizontally
    top_half = mask[:cy, :]
    bottom_half = cv2.flip(mask[cy:, :], 0)

    # Split vertically
    left_half = mask[:, :cx]
    right_half = cv2.flip(mask[:, cx:], 1)

    # Resize halves to match for comparison
    min_h = min(top_half.shape[0], bottom_half.shape[0])
    min_w = min(left_half.shape[1], right_half.shape[1])

    top_half_resized = cv2.resize(top_half, (w, min_h))
    bottom_half_resized = cv2.resize(bottom_half, (w, min_h))

    left_half_resized = cv2.resize(left_half, (min_w, h))
    right_half_resized = cv2.resize(right_half, (min_w, h))

    # Calculate asymmetry scores
    h_asymmetry = np.sum(
        np.abs(top_half_resized.astype(float) - bottom_half_resized.astype(float))
    ) / (w * min_h * 255)

    v_asymmetry = np.sum(
        np.abs(left_half_resized.astype(float) - right_half_resized.astype(float))
    ) / (min_w * h * 255)

    # Average asymmetry
    asymmetry_score = (h_asymmetry + v_asymmetry) / 2

    # Create visualization
    viz = image.copy()
    cv2.line(viz, (0, cy), (w, cy), (255, 0, 0), 2)  # Horizontal axis
    cv2.line(viz, (cx, 0), (cx, h), (255, 0, 0), 2)  # Vertical axis
    cv2.circle(viz, (cx, cy), 5, (0, 255, 0), -1)  # Centroid

    return asymmetry_score, viz


def analyze_border(mask: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Analyze border irregularity.

    Measures border irregularity using contour analysis and compactness.

    Args:
        mask: Binary lesion mask (H, W)

    Returns:
        Tuple of (border_score, visualization)
        Score ranges from 0 (smooth) to 1 (highly irregular)
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return 0.0, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, True)
    area = cv2.contourArea(largest_contour)

    if area == 0:
        return 0.0, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Compactness measure (circularity)
    # Perfect circle = 4π, higher values = more irregular
    compactness = (perimeter**2) / (4 * np.pi * area)

    # Normalize to 0-1 range (1 = circle, higher = more irregular)
    # Typical melanomas have compactness 1.5-3.0
    border_score = min((compactness - 1.0) / 2.0, 1.0)

    # Polygon approximation for irregularity
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    n_vertices = len(approx)

    # More vertices = more irregular border
    # Normalize: circle ≈ 8-12 vertices, irregular ≈ 30+
    vertex_score = min((n_vertices - 8) / 30.0, 1.0)

    # Combine scores
    border_score = (border_score + vertex_score) / 2

    # Create visualization
    viz = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(viz, [largest_contour], -1, (0, 255, 0), 2)
    cv2.drawContours(viz, [approx], -1, (255, 0, 0), 2)

    return border_score, viz


def analyze_color(image: np.ndarray, mask: np.ndarray) -> tuple[float, int, np.ndarray]:
    """
    Analyze color variation within the lesion.

    Uses K-means clustering to identify distinct colors.

    Args:
        image: RGB image (H, W, 3)
        mask: Binary lesion mask (H, W)

    Returns:
        Tuple of (color_variation_score, num_colors, visualization)
    """
    # Extract lesion pixels
    lesion_pixels = image[mask > 0]

    if len(lesion_pixels) < 10:
        return 0.0, 0, image.copy()

    # Perform K-means clustering to find dominant colors
    max_clusters = min(8, len(lesion_pixels) // 100)
    if max_clusters < 2:
        return 0.0, 1, image.copy()

    # Use K-means with optimal number of clusters (typically 3-6 for melanoma)
    n_colors = min(6, max_clusters)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(lesion_pixels)

    # Count significant colors (those covering >5% of lesion)
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    significant_colors = np.sum(counts > (len(labels) * 0.05))

    # Color variation score based on std dev of colors
    color_centers = kmeans.cluster_centers_
    color_std = np.mean(np.std(color_centers, axis=0))
    variation_score = min(color_std / 128.0, 1.0)  # Normalize to 0-1

    # Create color palette visualization
    viz = image.copy()
    palette_height = 50
    palette = np.zeros((palette_height, image.shape[1], 3), dtype=np.uint8)

    # Sort colors by frequency
    sorted_indices = np.argsort(counts)[::-1]
    x_start = 0
    for idx in sorted_indices:
        color = color_centers[idx].astype(int)
        width = int((counts[idx] / len(labels)) * image.shape[1])
        palette[:, x_start : x_start + width] = color
        x_start += width

    # Combine image with palette
    viz = np.vstack([viz, palette])

    return variation_score, int(significant_colors), viz


def analyze_diameter(mask: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Analyze lesion diameter.

    Calculates the maximum diameter of the lesion.

    Args:
        mask: Binary lesion mask (H, W)

    Returns:
        Tuple of (diameter_in_pixels, visualization)
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Find minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    diameter = 2 * radius

    # Also calculate bounding box diagonal as alternative measure
    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(largest_contour)
    bbox_diagonal = np.sqrt(w_rect**2 + h_rect**2)

    # Use average of both measures
    diameter = (diameter + bbox_diagonal) / 2

    # Create visualization
    viz = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    center = (int(x), int(y))
    cv2.circle(viz, center, int(radius), (0, 255, 0), 2)
    cv2.rectangle(
        viz, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (255, 0, 0), 2
    )

    # Draw diameter line
    angle = 45 * np.pi / 180  # 45-degree angle
    pt1 = (
        int(x - radius * np.cos(angle)),
        int(y - radius * np.sin(angle)),
    )
    pt2 = (
        int(x + radius * np.cos(angle)),
        int(y + radius * np.sin(angle)),
    )
    cv2.line(viz, pt1, pt2, (0, 0, 255), 2)

    return diameter, viz
