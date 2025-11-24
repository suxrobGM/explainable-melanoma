from typing import Any

import numpy as np

from .alignment import analyze_gradcam_alignment
from .features import analyze_asymmetry, analyze_border, analyze_color, analyze_diameter
from .segmentation import extract_lesion_mask


class ABCDEAnalyzer:
    """
    Analyzer for ABCDE melanoma detection criteria.

    Extracts clinical features from dermoscopic images and provides
    quantitative scores for each ABCDE criterion.

    Args:
        asymmetry_threshold: Threshold for asymmetry score (0-1). Default is 0.3.
        border_threshold: Threshold for border irregularity (0-1). Default is 0.4.
        color_threshold: Number of distinct colors indicating concern. Default is 3.
        diameter_threshold_px: Diameter threshold in pixels. Default is 50px.
    """

    def __init__(
        self,
        asymmetry_threshold: float = 0.3,
        border_threshold: float = 0.4,
        color_threshold: int = 3,
        diameter_threshold_px: int = 50,
    ):
        self.asymmetry_threshold = asymmetry_threshold
        self.border_threshold = border_threshold
        self.color_threshold = color_threshold
        self.diameter_threshold_px = diameter_threshold_px

    def analyze_image(
        self, image: np.ndarray, return_visualizations: bool = False
    ) -> dict[str, Any]:
        """
        Perform complete ABCDE analysis on a dermoscopic image.

        Args:
            image: RGB image array (H, W, 3) with values [0, 255]
            return_visualizations: Whether to return visualization images

        Returns:
            Dictionary containing:
                - scores: Dict of ABCDE scores
                - flags: Dict of boolean flags for each criterion
                - details: Detailed measurements
                - visualizations: Optional visualization images
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Extract lesion mask
        lesion_mask = extract_lesion_mask(image)

        # Analyze each criterion
        asymmetry_score, asym_viz = analyze_asymmetry(image, lesion_mask)
        border_score, border_viz = analyze_border(lesion_mask, image)
        color_score, num_colors, color_viz = analyze_color(image, lesion_mask)
        diameter_px, diameter_viz = analyze_diameter(lesion_mask, image)

        # Create result dictionary
        result = {
            "scores": {
                "asymmetry": float(asymmetry_score),
                "border": float(border_score),
                "color": float(color_score),
                "diameter": float(diameter_px),
            },
            "flags": {
                "asymmetry_flag": asymmetry_score > self.asymmetry_threshold,
                "border_flag": border_score > self.border_threshold,
                "color_flag": num_colors >= self.color_threshold,
                "diameter_flag": diameter_px > self.diameter_threshold_px,
            },
            "details": {
                "num_colors": int(num_colors),
                "diameter_pixels": float(diameter_px),
                "lesion_area_pixels": int(np.sum(lesion_mask > 0)),
            },
            "overall_risk": self._calculate_overall_risk(
                asymmetry_score,
                border_score,
                num_colors,
                diameter_px,
            ),
        }

        # Add visualizations if requested
        if return_visualizations:
            result["visualizations"] = {
                "asymmetry": asym_viz,
                "border": border_viz,
                "color": color_viz,
                "diameter": diameter_viz,
                "lesion_mask": lesion_mask,
            }

        return result

    def _calculate_overall_risk(
        self,
        asymmetry_score: float,
        border_score: float,
        num_colors: int,
        diameter: float,
    ) -> str:
        """
        Calculate overall risk assessment based on ABCDE scores.

        Args:
            asymmetry_score: Asymmetry score (0-1)
            border_score: Border irregularity score (0-1)
            num_colors: Number of distinct colors
            diameter: Diameter in pixels

        Returns:
            Risk level: "Low", "Medium", or "High"
        """
        risk_points = 0

        # Asymmetry
        if asymmetry_score > self.asymmetry_threshold:
            risk_points += 1

        # Border
        if border_score > self.border_threshold:
            risk_points += 1

        # Color
        if num_colors >= self.color_threshold:
            risk_points += 1

        # Diameter
        if diameter > self.diameter_threshold_px:
            risk_points += 1

        # Risk stratification
        if risk_points >= 3:
            return "High"
        elif risk_points >= 2:
            return "Medium"
        else:
            return "Low"

    def align_with_gradcam(
        self,
        abcde_result: dict[str, Any],
        attention_map: np.ndarray,
    ) -> dict[str, float]:
        """
        Analyze alignment between GradCAM attention and ABCDE features.

        Args:
            abcde_result: Result from analyze_image()
            attention_map: GradCAM attention heatmap (H, W) normalized [0, 1]

        Returns:
            Dictionary of alignment scores for each criterion
        """
        return analyze_gradcam_alignment(abcde_result, attention_map)
