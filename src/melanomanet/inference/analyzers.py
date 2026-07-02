# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

"""Analysis functions for inference (GradCAM, uncertainty, FastCAV, ABCDE)."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..abcde import ABCDEAnalyzer, create_abcde_report
from ..config import ABCDEConfig, FastCAVConfig, UncertaintyConfig
from ..explainability import FastCAV, create_fastcav_report
from ..explainability.models import FastCAVResult
from ..models.melanomanet import MelanomaNet
from ..uncertainty import MCDropoutEstimator, get_uncertainty_interpretation
from ..uncertainty.models import UncertaintyResult
from ..utils.console import console
from ..utils.gradcam import MelanomaGradCAM


def run_gradcam(
    gradcam: MelanomaGradCAM,
    image_tensor: torch.Tensor,
    original_np: np.ndarray,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Run GradCAM analysis.

    Args:
        gradcam: MelanomaGradCAM instance
        image_tensor: Preprocessed image tensor
        original_np: Original image as numpy array

    Returns:
        Tuple of (predicted class, confidence, attention map, visualization)
    """
    console.print("[cyan]Generating attention map...[/cyan]")
    pred_class, confidence = gradcam.get_prediction(image_tensor)
    attention_map = gradcam.generate_attention_map(image_tensor, pred_class)
    visualization = gradcam.visualize_attention(
        image_tensor, original_np, target_class=pred_class
    )
    return pred_class, confidence, attention_map, visualization


def run_uncertainty_analysis(
    model: MelanomaNet,
    image_tensor: torch.Tensor,
    uncertainty: UncertaintyConfig,
    device: torch.device,
) -> UncertaintyResult | None:
    """Run MC Dropout uncertainty estimation.

    Args:
        model: MelanomaNet model
        image_tensor: Preprocessed image tensor
        uncertainty: Uncertainty configuration
        device: Computation device

    Returns:
        UncertaintyResult or None if disabled
    """
    if not uncertainty.enable:
        return None

    console.print("[cyan]Estimating prediction uncertainty (MC Dropout)...[/cyan]")
    estimator = MCDropoutEstimator(
        model=model,
        n_samples=uncertainty.n_samples,
        uncertainty_threshold=uncertainty.uncertainty_threshold,
        device=device,
    )
    result = estimator.estimate(image_tensor)
    console.print(get_uncertainty_interpretation(result))
    return result


def run_fastcav_analysis(
    model: MelanomaNet,
    image_tensor: torch.Tensor,
    pred_class: int,
    class_name: str,
    fastcav_config: FastCAVConfig,
    device: torch.device,
) -> FastCAVResult | None:
    """Run FastCAV concept analysis.

    Args:
        model: MelanomaNet model
        image_tensor: Preprocessed image tensor
        pred_class: Predicted class index
        class_name: Name of predicted class
        fastcav_config: FastCAV configuration
        device: Computation device

    Returns:
        FastCAVResult or None if disabled or CAVs not found
    """
    if not fastcav_config.enable:
        return None

    console.print("[cyan]Analyzing concept importance (FastCAV)...[/cyan]")
    concepts_dir = Path(fastcav_config.concepts_dir)
    cavs_path = Path(fastcav_config.cavs_path)

    if not cavs_path.exists():
        console.print(f"[yellow]Warning: CAVs not found at {cavs_path}[/yellow]")
        return None

    fastcav = FastCAV(model=model, concepts_dir=concepts_dir, device=device)
    fastcav.load_cavs(cavs_path)

    result = fastcav.analyze_image(
        image_tensor, target_class=pred_class, class_name=class_name
    )
    console.print(create_fastcav_report(result))
    return result


def run_abcde_analysis(
    image_np: np.ndarray,
    attention_map: np.ndarray,
    abcde: ABCDEConfig,
) -> tuple[dict[str, Any] | None, dict[str, float] | None]:
    """Run ABCDE criterion analysis.

    Args:
        image_np: Image as numpy array normalized to [0,1]
        attention_map: GradCAM attention map
        abcde: ABCDE configuration

    Returns:
        Tuple of (ABCDE result dict, alignment scores dict) or (None, None) if disabled
    """
    if not abcde.enable:
        return None, None

    console.print("[cyan]Performing ABCDE criterion analysis...[/cyan]")
    analyzer = ABCDEAnalyzer(
        asymmetry_threshold=abcde.asymmetry_threshold,
        border_threshold=abcde.border_threshold,
        color_threshold=abcde.color_threshold,
        diameter_threshold_px=abcde.diameter_threshold_px,
    )

    image_uint8 = (image_np * 255).astype(np.uint8)
    abcde_result = analyzer.analyze_image(image_uint8, return_visualizations=True)

    alignment_scores = None
    if abcde.enable_alignment_analysis:
        alignment_scores = analyzer.align_with_gradcam(abcde_result, attention_map)

    console.print("\n" + create_abcde_report(abcde_result, alignment_scores))
    return abcde_result, alignment_scores
