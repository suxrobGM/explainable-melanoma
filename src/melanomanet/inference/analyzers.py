"""Analysis functions for inference (GradCAM, uncertainty, FastCAV, ABCDE)."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console

from ..abcde import ABCDEAnalyzer, create_abcde_report
from ..explainability import FastCAV, create_fastcav_report
from ..explainability.models import FastCAVResult
from ..models.melanomanet import MelanomaNet
from ..uncertainty import MCDropoutEstimator, get_uncertainty_interpretation
from ..uncertainty.models import UncertaintyResult
from ..utils.gradcam import MelanomaGradCAM

console = Console()


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
    config: dict,
    device: torch.device,
) -> UncertaintyResult | None:
    """Run MC Dropout uncertainty estimation.

    Args:
        model: MelanomaNet model
        image_tensor: Preprocessed image tensor
        config: Configuration dictionary
        device: Computation device

    Returns:
        UncertaintyResult or None if disabled
    """
    if not config.get("uncertainty", {}).get("enable", True):
        return None

    console.print("[cyan]Estimating prediction uncertainty (MC Dropout)...[/cyan]")
    n_samples = config.get("uncertainty", {}).get("n_samples", 10)
    threshold = config.get("uncertainty", {}).get("uncertainty_threshold", 0.5)

    estimator = MCDropoutEstimator(
        model=model,
        n_samples=n_samples,
        uncertainty_threshold=threshold,
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
    config: dict,
    device: torch.device,
) -> FastCAVResult | None:
    """Run FastCAV concept analysis.

    Args:
        model: MelanomaNet model
        image_tensor: Preprocessed image tensor
        pred_class: Predicted class index
        class_name: Name of predicted class
        config: Configuration dictionary
        device: Computation device

    Returns:
        FastCAVResult or None if disabled or CAVs not found
    """
    if not config.get("fastcav", {}).get("enable", True):
        return None

    console.print("[cyan]Analyzing concept importance (FastCAV)...[/cyan]")
    concepts_dir = Path(
        config.get("fastcav", {}).get("concepts_dir", "./data/concepts")
    )
    cavs_path = Path(
        config.get("fastcav", {}).get("cavs_path", "./checkpoints/cavs.pth")
    )

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
    config: dict,
) -> tuple[dict[str, Any] | None, dict[str, float] | None]:
    """Run ABCDE criterion analysis.

    Args:
        image_np: Image as numpy array normalized to [0,1]
        attention_map: GradCAM attention map
        config: Configuration dictionary

    Returns:
        Tuple of (ABCDE result dict, alignment scores dict) or (None, None) if disabled
    """
    if not config.get("abcde", {}).get("enable", True):
        return None, None

    console.print("[cyan]Performing ABCDE criterion analysis...[/cyan]")
    abcde_config = config.get("abcde", {})

    analyzer = ABCDEAnalyzer(
        asymmetry_threshold=abcde_config.get("asymmetry_threshold", 0.3),
        border_threshold=abcde_config.get("border_threshold", 0.4),
        color_threshold=abcde_config.get("color_threshold", 3),
        diameter_threshold_px=abcde_config.get("diameter_threshold_px", 50),
    )

    image_uint8 = (image_np * 255).astype(np.uint8)
    abcde_result = analyzer.analyze_image(image_uint8, return_visualizations=True)

    alignment_scores = None
    if abcde_config.get("enable_alignment_analysis", True):
        alignment_scores = analyzer.align_with_gradcam(abcde_result, attention_map)

    console.print("\n" + create_abcde_report(abcde_result, alignment_scores))
    return abcde_result, alignment_scores
