"""Structured inference artifacts.

For an inference output ``S_result.png``, two machine-readable siblings are
saved: ``S_result.json`` (scalar results) and ``S_result_panels.npz`` (panel
images). Paper figures are built from these instead of parsing the rendered
outputs. This module owns the artifact naming scheme.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import InferenceResult, PaperFigureData

SCHEMA_VERSION = 1

RESULT_SUFFIX = "_result"

_ABCDE_PANEL_KEYS = ("asymmetry", "border", "color", "diameter")


def result_png_path(output_dir: Path, image_stem: str) -> Path:
    """Visualization PNG path for an input image stem."""
    return output_dir / f"{image_stem}{RESULT_SUFFIX}.png"


def image_id_from(artifact_path: Path) -> str:
    """Recover the input image stem from any result artifact path."""
    return artifact_path.stem.removesuffix(RESULT_SUFFIX)


def iter_artifact_jsons(output_dir: Path) -> list[Path]:
    """All result JSON artifacts in a directory, sorted by name."""
    return sorted(output_dir.glob(f"*{RESULT_SUFFIX}.json"))


def panels_path(path: Path) -> Path:
    """Panels npz path for a result artifact (png or json) path."""
    return path.with_name(f"{path.stem}_panels.npz")


def save_artifacts(
    result: InferenceResult,
    original_np: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Save the structured JSON result and panel images next to the PNG."""
    output_path = Path(output_path)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "prediction": class_names[result.pred_class],
        "pred_class": int(result.pred_class),
        "confidence": float(result.confidence),
        "uncertainty": None,
        "abcde": None,
        "alignment_scores": None,
        "fastcav": None,
    }

    if result.uncertainty is not None:
        unc = result.uncertainty
        payload["uncertainty"] = {
            "predictive": float(unc.predictive_uncertainty),
            "epistemic": float(unc.epistemic_uncertainty),
            "aleatoric": float(unc.aleatoric_uncertainty),
            "is_reliable": bool(unc.is_reliable),
        }

    if result.abcde is not None:
        abcde = result.abcde
        payload["abcde"] = {
            "overall_risk": abcde["overall_risk"],
            "scores": {k: float(v) for k, v in abcde["scores"].items()},
            "flags": {k: bool(v) for k, v in abcde["flags"].items()},
            "num_colors": int(abcde["details"]["num_colors"]),
        }

    if result.alignment_scores is not None:
        payload["alignment_scores"] = {k: float(v) for k, v in result.alignment_scores.items()}

    if result.fastcav is not None:
        payload["fastcav"] = {
            "target_class": result.fastcav.target_class,
            "concepts": {
                name: {
                    "tcav_score": float(score.tcav_score),
                    "accuracy": float(score.accuracy),
                    "p_value": float(score.p_value),
                    "is_significant": bool(score.is_significant),
                }
                for name, score in result.fastcav.concept_scores.items()
            },
        }

    output_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    panels: dict[str, Any] = {
        "original": (np.asarray(original_np) * 255).astype(np.uint8),
        "gradcam": np.asarray(result.attention_map, dtype=np.float32),
        "overlay": np.asarray(result.visualization).astype(np.uint8),
    }
    if result.abcde is not None and "visualizations" in result.abcde:
        viz = result.abcde["visualizations"]
        for key in _ABCDE_PANEL_KEYS:
            panels[key] = np.asarray(viz[key])

    np.savez_compressed(panels_path(output_path), **panels)


def load_paper_figure_data(json_path: Path) -> PaperFigureData | None:
    """Build PaperFigureData from saved artifacts; None if panels are missing."""
    npz_path = panels_path(json_path)
    if not npz_path.exists():
        return None

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    panels = np.load(npz_path)

    unc = payload.get("uncertainty") or {}
    abcde = payload.get("abcde") or {}
    fastcav = payload.get("fastcav") or {}
    scores_by_concept = {
        name: score["tcav_score"] for name, score in fastcav.get("concepts", {}).items()
    }

    blank = np.zeros((8, 8, 3), dtype=np.uint8)

    def panel(key: str) -> np.ndarray:
        return panels[key] if key in panels else blank

    abcde_scores = abcde.get("scores", {})
    abcde_flags = abcde.get("flags", {})
    return PaperFigureData(
        prediction=payload["prediction"],
        confidence=float(payload["confidence"]),
        risk_level=abcde.get("overall_risk", "Unknown"),
        original=panels["original"],
        gradcam=panels["gradcam"],
        overlay=panels["overlay"],
        asymmetry_img=panel("asymmetry"),
        border_img=panel("border"),
        color_img=panel("color"),
        diameter_img=panel("diameter"),
        asymmetry_score=float(abcde_scores.get("asymmetry", 0.0)),
        border_score=float(abcde_scores.get("border", 0.0)),
        n_colors=int(abcde.get("num_colors", 0)),
        diameter=float(abcde_scores.get("diameter", 0.0)),
        asymmetry_flag=bool(abcde_flags.get("asymmetry_flag", False)),
        border_flag=bool(abcde_flags.get("border_flag", False)),
        color_flag=bool(abcde_flags.get("color_flag", False)),
        diameter_flag=bool(abcde_flags.get("diameter_flag", False)),
        predictive_unc=float(unc.get("predictive", 0.0)),
        epistemic_unc=float(unc.get("epistemic", 0.0)),
        aleatoric_unc=float(unc.get("aleatoric", 0.0)),
        is_reliable=bool(unc.get("is_reliable", False)),
        concepts=list(scores_by_concept),
        scores=list(scores_by_concept.values()),
    )
