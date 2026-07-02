"""
Faithfulness evaluation for GradCAM++ attention maps.

Quantifies whether the attention maps actually explain the model, rather than
merely looking plausible, using three measures with a random-attention control:

* Deletion AOPC: progressively remove the most-attended pixels and measure how
  fast the target-class probability falls. Faithful maps drop it quickly, so
  higher is better.
* Insertion AOPC: progressively reveal the most-attended pixels on a blank
  baseline and measure how fast probability rises. Higher is better.
* Lesion IoU: overlap between thresholded attention and the segmented lesion,
  testing whether attention lands on the lesion rather than the background.

A random heatmap is scored identically to establish the baseline each real
measure must beat.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rich.table import Table

from ..abcde.segmentation import extract_lesion_mask
from ..config import Config
from ..data.transforms import get_val_transforms
from ..inference.loaders import load_model
from ..utils.console import console
from ..utils.env import resolve_device
from ..utils.gradcam import MelanomaGradCAM
from ..utils.images import iter_image_files


def _probability_curve(
    model: torch.nn.Module,
    x: torch.Tensor,
    order: np.ndarray,
    target_class: int,
    device: torch.device,
    mode: str,
    steps: int,
) -> np.ndarray:
    """
    Target-class probability as pixels are removed (deletion) or added
    (insertion) in the given importance order.

    Args:
        x: Normalized input, shape (1, 3, H, W).
        order: Flat pixel indices sorted most- to least-important.
        mode: "deletion" starts from the full image and blanks pixels;
            "insertion" starts from a blank baseline and reveals pixels.
        steps: Number of points along the curve.
    """
    _, c, h, w = x.shape
    n_pixels = h * w
    baseline = torch.zeros_like(x)  # zero == ImageNet mean in normalized space

    batch = []
    fractions = np.linspace(0.0, 1.0, steps + 1)
    for frac in fractions:
        k = int(frac * n_pixels)
        if mode == "deletion":
            img = x.clone()
            src = baseline
        else:
            img = baseline.clone()
            src = x
        if k > 0:
            idx = torch.from_numpy(order[:k]).long()
            rows, cols = idx // w, idx % w
            img[0, :, rows, cols] = src[0, :, rows, cols]
        batch.append(img)

    inputs = torch.cat(batch, dim=0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(inputs), dim=1)[:, target_class]
    return probs.cpu().numpy()


def _deletion_aopc(curve: np.ndarray) -> float:
    """Mean drop from the starting probability as pixels are removed."""
    return float(np.mean(curve[0] - curve))


def _insertion_aopc(curve: np.ndarray) -> float:
    """Mean rise from the blank-baseline probability as pixels are added."""
    return float(np.mean(curve - curve[0]))


def _iou(attention: np.ndarray, lesion_mask: np.ndarray, quantile: float) -> float:
    """IoU between high-attention pixels and the lesion mask."""
    thresh = np.quantile(attention, quantile)
    attn_bin = attention >= thresh
    lesion_bin = lesion_mask > 0
    inter = np.logical_and(attn_bin, lesion_bin).sum()
    union = np.logical_or(attn_bin, lesion_bin).sum()
    return float(inter / union) if union > 0 else 0.0


def evaluate_faithfulness(
    config: Config,
    checkpoint_path: str,
    input_dir: str,
    steps: int,
    iou_quantile: float,
    seed: int,
) -> None:
    """Score GradCAM++ faithfulness against a random-attention control."""
    device = resolve_device(config.device)
    img_size = config.data.image_size
    transform = get_val_transforms(img_size)

    model = load_model(config, checkpoint_path, device)
    gradcam = MelanomaGradCAM(model, device=device)
    rng = np.random.default_rng(seed)

    image_paths = list(iter_image_files(Path(input_dir)))
    if not image_paths:
        console.print(f"[red]No images found in {input_dir}[/red]")
        return

    rows: list[dict] = []
    for path in image_paths:
        original = Image.open(path).convert("RGB")
        x = transform(original).unsqueeze(0).to(device)

        with torch.no_grad():
            target_class = int(model(x).argmax(dim=1).item())

        attention = gradcam.generate_attention_map(x, target_class)  # (H, W) [0,1]
        if attention.shape != (img_size, img_size):
            attention = np.asarray(
                Image.fromarray(attention).resize((img_size, img_size))
            )

        random_attention = rng.random(attention.shape)
        gc_order = np.argsort(attention.ravel())[::-1]
        rnd_order = np.argsort(random_attention.ravel())[::-1]

        resized_rgb = np.array(original.resize((img_size, img_size)))
        lesion_mask = extract_lesion_mask(resized_rgb)

        row = {"image": path.name, "target_class": target_class}
        for label, order, attn in (
            ("gradcam", gc_order, attention),
            ("random", rnd_order, random_attention),
        ):
            del_curve = _probability_curve(
                model, x, order, target_class, device, "deletion", steps
            )
            ins_curve = _probability_curve(
                model, x, order, target_class, device, "insertion", steps
            )
            row[f"{label}_deletion_aopc"] = _deletion_aopc(del_curve)
            row[f"{label}_insertion_aopc"] = _insertion_aopc(ins_curve)
            row[f"{label}_lesion_iou"] = _iou(attn, lesion_mask, iou_quantile)
        rows.append(row)
        console.print(
            f"  {path.name}: del={row['gradcam_deletion_aopc']:.3f} "
            f"ins={row['gradcam_insertion_aopc']:.3f} "
            f"iou={row['gradcam_lesion_iou']:.3f}"
        )

    _report(rows, Path(config.paths.output_dir))


def _report(rows: list[dict], output_dir: Path) -> None:
    """Print mean GradCAM vs random and persist per-image results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = ["deletion_aopc", "insertion_aopc", "lesion_iou"]

    table = Table(title=f"Faithfulness over {len(rows)} images (mean)")
    table.add_column("Metric", style="cyan")
    table.add_column("GradCAM++", style="green")
    table.add_column("Random", style="yellow")

    summary = {}
    for key in metric_keys:
        gc_mean = float(np.mean([r[f"gradcam_{key}"] for r in rows]))
        rnd_mean = float(np.mean([r[f"random_{key}"] for r in rows]))
        summary[key] = {"gradcam": gc_mean, "random": rnd_mean}
        table.add_row(key, f"{gc_mean:.4f}", f"{rnd_mean:.4f}")

    console.print(table)

    out_path = output_dir / "faithfulness.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_image": rows}, f, indent=2)
    console.print(f"[green]Faithfulness results saved to {out_path}[/green]")
