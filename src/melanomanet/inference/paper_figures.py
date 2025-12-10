# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

"""Paper-ready figure generation for reports."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Paper style settings
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
}

CONCEPT_ABBREV = {
    "large_diameter": "Diameter",
    "multicolor": "Multicolor",
    "asymmetry": "Asymmetry",
    "irregular_border": "Irreg. Border",
}

RISK_COLORS = {"Low": "#55A868", "Medium": "#FFA500", "High": "#C44E52"}


@dataclass
class PaperFigureData:
    """Data container for paper figure generation."""

    prediction: str
    confidence: float
    risk_level: str

    # Images (as numpy arrays)
    original: np.ndarray
    gradcam: np.ndarray
    overlay: np.ndarray
    asymmetry_img: np.ndarray
    border_img: np.ndarray
    color_img: np.ndarray
    diameter_img: np.ndarray

    # ABCDE scores
    asymmetry_score: float
    border_score: float
    n_colors: int
    diameter: float

    # Uncertainty
    predictive_unc: float
    epistemic_unc: float
    aleatoric_unc: float
    is_reliable: bool

    # FastCAV
    concepts: list[str]
    scores: list[float]


def _get_status_color(is_flagged: bool) -> tuple[str, str]:
    """Get status indicator and color."""
    return ("red", "!") if is_flagged else ("green", "OK")


def _plot_image_row(fig, gs, data: PaperFigureData) -> None:
    """Plot top row: original, gradcam, overlay, risk badge."""
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(data.original)
    ax1.set_title("Input", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(data.gradcam)
    ax2.set_title("GradCAM++", fontsize=10)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(data.overlay)
    ax3.set_title(
        f"{data.prediction} ({data.confidence:.1%})", fontsize=10, fontweight="bold"
    )
    ax3.axis("off")

    ax_risk = fig.add_subplot(gs[0, 3])
    ax_risk.axis("off")
    bbox_props = dict(
        boxstyle="round,pad=0.4",
        facecolor=RISK_COLORS.get(data.risk_level, "gray"),
        edgecolor="black",
        linewidth=2,
    )
    ax_risk.text(
        0.5,
        0.5,
        f"ABCDE Risk:\n{data.risk_level}",
        transform=ax_risk.transAxes,
        fontsize=12,
        fontweight="bold",
        va="center",
        ha="center",
        bbox=bbox_props,
        color="white",
    )


def _plot_abcde_row(fig, gs, data: PaperFigureData) -> None:
    """Plot ABCDE criterion visualizations."""
    criteria = [
        (
            data.asymmetry_img,
            "A",
            data.asymmetry_score,
            data.asymmetry_score > 0.3,
            f"{data.asymmetry_score:.2f}",
        ),
        (
            data.border_img,
            "B",
            data.border_score,
            data.border_score > 0.4,
            f"{data.border_score:.2f}",
        ),
        (
            data.color_img,
            "C",
            data.n_colors,
            data.n_colors > 3,
            f"{data.n_colors} colors",
        ),
        (
            data.diameter_img,
            "D",
            data.diameter,
            data.diameter > 114,
            f"{data.diameter:.0f}px",
        ),
    ]

    for i, (img, label, _, is_flagged, detail) in enumerate(criteria):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(img)
        color, status = _get_status_color(is_flagged)
        ax.set_title(
            f"{label} [{status}] {detail}", fontsize=10, color=color, fontweight="bold"
        )
        ax.axis("off")


def _plot_uncertainty_bar(ax, data: PaperFigureData) -> None:
    """Plot uncertainty horizontal bar chart."""
    categories = ["Pred.", "Epist.", "Aleat."]
    values = [data.predictive_unc, data.epistemic_unc, data.aleatoric_unc]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    y_pos = np.arange(len(categories))

    bars = ax.barh(y_pos, values, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Uncertainty", fontsize=9)

    for bar, val in zip(bars, values):
        if val > 0.15:
            ax.text(
                val - 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                ha="right",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        else:
            ax.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                ha="left",
                fontsize=8,
            )

    status = "RELIABLE" if data.is_reliable else "UNCERTAIN"
    color = "green" if data.is_reliable else "red"
    ax.set_title(f"Uncertainty ({status})", fontsize=10, color=color, fontweight="bold")


def _plot_fastcav_bar(ax, data: PaperFigureData) -> None:
    """Plot FastCAV concept importance bar chart."""
    if not data.concepts or not data.scores:
        ax.text(0.5, 0.5, "No FastCAV data", ha="center", va="center")
        ax.set_title(
            f"Concept Influence → {data.prediction}", fontsize=10, fontweight="bold"
        )
        return

    sorted_pairs = sorted(
        zip(data.concepts, data.scores), key=lambda x: abs(x[1]), reverse=True
    )
    concepts, scores = zip(*sorted_pairs)
    y_pos = np.arange(len(concepts))
    colors = ["#55A868" if s > 0 else "#C44E52" for s in scores]

    bars = ax.barh(y_pos, scores, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [CONCEPT_ABBREV.get(c, c.replace("_", " ").title()) for c in concepts],
        fontsize=8,
    )
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("TCAV Score", fontsize=9)

    max_score = max(abs(s) for s in scores)
    ax.set_xlim(-max_score - 0.8, max_score + 0.8)

    for bar, val in zip(bars, scores):
        if abs(val) > 0.5:
            x_pos = val - 0.1 if val > 0 else val + 0.1
            ha = "right" if val > 0 else "left"
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}",
                va="center",
                ha=ha,
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        else:
            x_pos = val + 0.1 if val > 0 else val - 0.1
            ha = "left" if val > 0 else "right"
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}",
                va="center",
                ha=ha,
                fontsize=8,
            )

    ax.set_title(
        f"Concept Influence → {data.prediction}", fontsize=10, fontweight="bold"
    )


def create_paper_figure(
    data: PaperFigureData, output_path: Path, dpi: int = 300
) -> None:
    """Create a paper-ready combined figure.

    Args:
        data: PaperFigureData containing all visualization data
        output_path: Path to save the figure
        dpi: Output resolution
    """
    plt.rcParams.update(PAPER_STYLE)

    fig = plt.figure(figsize=(8.5, 5.5))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.8], hspace=0.4, wspace=0.3)

    # Plot rows
    _plot_image_row(fig, gs, data)
    _plot_abcde_row(fig, gs, data)

    # Bottom row with manual positioning for proper spacing
    ax_unc = fig.add_axes([0.08, 0.08, 0.38, 0.18])
    _plot_uncertainty_bar(ax_unc, data)

    ax_cav = fig.add_axes([0.55, 0.08, 0.40, 0.18])
    _plot_fastcav_bar(ax_cav, data)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
