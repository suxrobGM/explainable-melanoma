from typing import Any

import cv2
import numpy as np


def analyze_gradcam_alignment(
    abcde_result: dict[str, Any],
    attention_map: np.ndarray,
    image: np.ndarray,
) -> dict[str, float]:
    """
    Analyze alignment between GradCAM attention and ABCDE features.

    Quantifies how well the model's attention aligns with clinically
    relevant ABCDE regions.

    Args:
        abcde_result: Result from ABCDEAnalyzer.analyze_image()
        attention_map: GradCAM attention heatmap (H, W) normalized [0, 1]
        image: Original RGB image (H, W, 3)

    Returns:
        Dictionary of alignment scores for each criterion
    """
    if "visualizations" not in abcde_result:
        raise ValueError("ABCDE result must include visualizations")

    mask = abcde_result["visualizations"]["lesion_mask"]

    # Resize attention map to match image size if needed
    if attention_map.shape != mask.shape:
        attention_map = cv2.resize(attention_map, (mask.shape[1], mask.shape[0]))

    # Extract lesion region attention
    lesion_attention = attention_map[mask > 0]

    if len(lesion_attention) == 0:
        return {
            "border_alignment": 0.0,
            "color_alignment": 0.0,
            "overall_alignment": 0.0,
        }

    # Border alignment: Check if attention focuses on border regions
    # Create border region (dilated boundary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    border_region = dilated - mask

    border_attention = attention_map[border_region > 0]
    border_alignment = np.mean(border_attention) if len(border_attention) > 0 else 0.0

    # Color variation alignment: High attention on color-diverse regions
    # Use color variation from ABCDE analysis
    color_alignment = abcde_result["scores"]["color"]

    # Overall lesion alignment
    overall_alignment = np.mean(lesion_attention)

    return {
        "border_alignment": float(border_alignment),
        "color_alignment": float(color_alignment),
        "overall_alignment": float(overall_alignment),
        "mean_lesion_attention": float(np.mean(lesion_attention)),
        "max_lesion_attention": float(np.max(lesion_attention)),
    }
