"""Visualization functions for inference results."""

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from ..explainability.models import FastCAVResult
from ..uncertainty.models import UncertaintyResult
from .models import InferenceResult

console = Console()


def create_uncertainty_text(result: UncertaintyResult) -> tuple[str, str]:
    """Create uncertainty panel text and color.

    Args:
        result: UncertaintyResult object

    Returns:
        Tuple of (text content, background color)
    """
    reliability = "RELIABLE" if result.is_reliable else "UNCERTAIN"
    color = "lightgreen" if result.is_reliable else "lightyellow"

    text = "UNCERTAINTY ANALYSIS\n" + "=" * 40 + "\n"
    text += (
        f"Predictive: {result.predictive_uncertainty:.3f} (total model uncertainty)\n"
    )
    text += f"Epistemic:  {result.epistemic_uncertainty:.3f} (model knowledge gaps)\n"
    text += f"Aleatoric:  {result.aleatoric_uncertainty:.3f} (inherent data noise)\n"
    text += "-" * 40 + "\n"
    text += f"Reliability: {reliability}\n"
    text += (
        "(Low uncertainty = confident prediction)"
        if result.is_reliable
        else "(High uncertainty = review recommended)"
    )
    return text, color


def create_fastcav_text(result: FastCAVResult) -> tuple[str, str]:
    """Create FastCAV panel text and color.

    Args:
        result: FastCAVResult object

    Returns:
        Tuple of (text content, background color)
    """
    if not result.concept_scores:
        text = "CONCEPT IMPORTANCE (FastCAV)\n" + "=" * 40 + "\n"
        text += "No concept scores available.\n"
        text += "Run 'pdm run train-fastcav' to train CAVs.\n"
        return text, "lightyellow"

    text = "CONCEPT IMPORTANCE (FastCAV)\n" + "=" * 40 + "\n"
    sorted_concepts = sorted(
        result.concept_scores.values(),
        key=lambda x: abs(x.tcav_score),
        reverse=True,
    )
    for cs in sorted_concepts[:4]:
        direction = "+" if cs.tcav_score > 0 else "-"
        text += f"{cs.concept_name}: {direction}{abs(cs.tcav_score):.2f}\n"
    text += "-" * 40 + "\n"
    text += "+ supports prediction\n"
    text += "- opposes prediction\n"
    text += "(higher magnitude = stronger influence)"
    return text, "lightblue"


def create_abcde_summary_text(abcde_result: dict) -> str:
    """Create ABCDE summary panel text.

    Args:
        abcde_result: ABCDE analysis result dictionary

    Returns:
        Formatted text summary
    """
    text = "ABCDE CRITERION SUMMARY\n" + "=" * 40 + "\n"
    for criterion, flag_key in [
        ("Asymmetry", "asymmetry_flag"),
        ("Border", "border_flag"),
        ("Color", "color_flag"),
        ("Diameter", "diameter_flag"),
    ]:
        status = "[!] CONCERN" if abcde_result["flags"][flag_key] else "[OK] OK"
        text += f"{criterion}: {status}\n"
    return text


def plot_main_row(
    fig,
    gs,
    original_np: np.ndarray,
    attention_map: np.ndarray,
    visualization: np.ndarray,
    pred_class: int,
    confidence: float,
    class_names: list[str],
    abcde_result: dict | None,
) -> None:
    """Plot main prediction row (original, gradcam, overlay, risk).

    Args:
        fig: Matplotlib figure
        gs: GridSpec for layout
        original_np: Original image array
        attention_map: GradCAM attention map
        visualization: GradCAM overlay visualization
        pred_class: Predicted class index
        confidence: Prediction confidence
        class_names: List of class names
        abcde_result: ABCDE result dict or None
    """
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_np)
    ax1.set_title("Original Image", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(attention_map, cmap="jet")
    ax2.set_title("GradCAM++ Heatmap", fontsize=12, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(visualization)
    ax3.set_title(
        f"Prediction: {class_names[pred_class]}\nConfidence: {confidence:.2%}",
        fontsize=12,
        fontweight="bold",
    )
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis("off")
    if abcde_result:
        risk_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(
            abcde_result["overall_risk"], "gray"
        )
        ax4.text(
            0.5,
            0.5,
            f"ABCDE Risk:\n{abcde_result['overall_risk']}",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
            color=risk_color,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=risk_color,
                linewidth=3,
            ),
        )


def plot_abcde_row(fig, gs, abcde_result: dict) -> None:
    """Plot ABCDE visualizations row.

    Args:
        fig: Matplotlib figure
        gs: GridSpec for layout
        abcde_result: ABCDE analysis result dictionary
    """
    viz = abcde_result["visualizations"]
    flags = abcde_result["flags"]
    scores = abcde_result["scores"]
    details = abcde_result["details"]

    criteria = [
        (
            "asymmetry",
            "A - Asymmetry",
            "asymmetry_flag",
            "[!] PRESENT",
            "[OK] Absent",
            f"Score: {scores['asymmetry']:.3f}",
        ),
        (
            "border",
            "B - Border",
            "border_flag",
            "[!] IRREGULAR",
            "[OK] Regular",
            f"Score: {scores['border']:.3f}",
        ),
        (
            "color",
            "C - Color",
            "color_flag",
            "[!] VARIED",
            "[OK] Uniform",
            f"Colors: {details['num_colors']}",
        ),
        (
            "diameter",
            "D - Diameter",
            "diameter_flag",
            "[!] LARGE",
            "[OK] Small",
            f"{scores['diameter']:.1f}px",
        ),
    ]

    for i, (key, title, flag_key, present, absent, detail) in enumerate(criteria):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(viz[key])
        status = present if flags[flag_key] else absent
        ax.set_title(f"{title}: {status}\n{detail}", fontsize=10)
        ax.axis("off")


def plot_analysis_row(
    fig,
    gs,
    uncertainty: UncertaintyResult | None,
    fastcav: FastCAVResult | None,
    abcde_result: dict | None,
    alignment_scores: dict | None,
) -> None:
    """Plot uncertainty and concept analysis row.

    Args:
        fig: Matplotlib figure
        gs: GridSpec for layout
        uncertainty: UncertaintyResult or None
        fastcav: FastCAVResult or None
        abcde_result: ABCDE result dict or None
        alignment_scores: Alignment scores dict or None
    """
    # Left panel: Uncertainty or ABCDE summary
    ax_left = fig.add_subplot(gs[2, :2])
    ax_left.axis("off")

    if uncertainty:
        text, color = create_uncertainty_text(uncertainty)
        ax_left.text(
            0.1,
            0.5,
            text,
            fontsize=10,
            fontfamily="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor=color, alpha=0.7),
        )
    elif abcde_result:
        text = create_abcde_summary_text(abcde_result)
        ax_left.text(
            0.1,
            0.5,
            text,
            fontsize=11,
            fontfamily="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="lightgray", alpha=0.5),
        )

    # Right panel: FastCAV or alignment scores
    ax_right = fig.add_subplot(gs[2, 2:])
    ax_right.axis("off")

    if fastcav:
        text, color = create_fastcav_text(fastcav)
        ax_right.text(
            0.1,
            0.5,
            text,
            fontsize=10,
            fontfamily="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor=color, alpha=0.5),
        )
    elif alignment_scores:
        text = "GradCAM ALIGNMENT\n" + "=" * 40 + "\n"
        text += f"Border Alignment: {alignment_scores['border_alignment']:.3f}\n"
        text += f"Overall Attention: {alignment_scores['overall_alignment']:.3f}\n"
        text += (
            f"Mean Lesion Attention: {alignment_scores['mean_lesion_attention']:.3f}\n"
        )
        ax_right.text(
            0.1,
            0.5,
            text,
            fontsize=11,
            fontfamily="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.5),
        )


def create_visualization(
    result: InferenceResult,
    original_np: np.ndarray,
    class_names: list[str],
    output_path: str,
) -> None:
    """Create and save comprehensive visualization.

    Args:
        result: InferenceResult containing all analysis results
        original_np: Original image as numpy array
        class_names: List of class names
        output_path: Path to save visualization image
    """
    if result.abcde:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        plot_main_row(
            fig,
            gs,
            original_np,
            result.attention_map,
            result.visualization,
            result.pred_class,
            result.confidence,
            class_names,
            result.abcde,
        )
        plot_abcde_row(fig, gs, result.abcde)
        plot_analysis_row(
            fig,
            gs,
            result.uncertainty,
            result.fastcav,
            result.abcde,
            result.alignment_scores,
        )

        plt.suptitle(
            "MelanomaNet: Explainable Melanoma Detection with Uncertainty & Concept Analysis",
            fontsize=14,
            fontweight="bold",
        )
    else:
        # Simple visualization without ABCDE
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_np)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(result.attention_map, cmap="jet")
        axes[1].set_title("GradCAM++ Heatmap")
        axes[1].axis("off")

        axes[2].imshow(result.visualization)
        axes[2].set_title(
            f"Prediction: {class_names[result.pred_class]}\n"
            f"Confidence: {result.confidence:.2%}"
        )
        axes[2].axis("off")
        plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Visualization saved to {output_path}[/green]")
