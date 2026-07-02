"""Round-trip test: inference artifacts -> PaperFigureData -> rendered figure."""

from pathlib import Path

import numpy as np

from melanomanet.explainability.models import ConceptScore, FastCAVResult
from melanomanet.inference.artifacts import load_paper_figure_data, save_artifacts
from melanomanet.inference.models import InferenceResult
from melanomanet.inference.paper_figures import create_paper_figure
from melanomanet.uncertainty.models import UncertaintyResult

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]


def _fake_result(size: int = 16) -> tuple[InferenceResult, np.ndarray]:
    rgb = np.random.default_rng(0).random((size, size, 3))
    panel = (rgb * 255).astype(np.uint8)
    result = InferenceResult(
        pred_class=0,
        # numpy scalars below deliberately violate the annotations: save_artifacts
        # must coerce them for JSON serialization.
        confidence=np.float32(0.93),  # type: ignore[arg-type]
        attention_map=rgb[:, :, 0].astype(np.float32),
        visualization=panel,
        uncertainty=UncertaintyResult(
            predicted_class=0,
            confidence=0.93,
            mean_probs=np.full(9, 1 / 9),
            predictive_uncertainty=np.float64(0.4),
            epistemic_uncertainty=0.1,
            aleatoric_uncertainty=0.3,
            is_reliable=np.bool_(True),  # type: ignore[arg-type]
        ),
        fastcav=FastCAVResult(
            target_class="MEL",
            concept_scores={
                "asymmetry": ConceptScore("asymmetry", 1.2, 0.8, 0.01, True),
                "multicolor": ConceptScore("multicolor", -0.5, 0.7, 0.2, False),
            },
            feature_dim=1280,
            n_samples_used=100,
        ),
        abcde={
            "overall_risk": "High",
            "scores": {
                "asymmetry": np.float64(0.5),
                "border": 0.6,
                "color": 0.7,
                "diameter": 120.0,
            },
            "flags": {
                "asymmetry_flag": np.bool_(True),
                "border_flag": True,
                "color_flag": True,
                "diameter_flag": True,
            },
            "details": {"num_colors": np.int64(4)},
            "visualizations": {
                "asymmetry": panel,
                "border": panel,
                "color": panel,
                "diameter": panel,
                "lesion_mask": (rgb[:, :, 0] > 0.5).astype(np.uint8),
            },
        },
        alignment_scores={"overall_alignment": np.float64(0.55)},
    )
    return result, rgb


def test_artifact_round_trip_and_render(tmp_path: Path) -> None:
    result, original_np = _fake_result()
    output_path = tmp_path / "ISIC_0000000_result.png"

    save_artifacts(result, original_np, CLASS_NAMES, output_path)

    json_path = tmp_path / "ISIC_0000000_result.json"
    assert json_path.exists()
    assert (tmp_path / "ISIC_0000000_result_panels.npz").exists()

    data = load_paper_figure_data(json_path)
    assert data is not None
    assert data.prediction == "MEL"
    assert data.risk_level == "High"
    assert data.n_colors == 4
    assert data.asymmetry_flag is True  # analyzer flags survive the round trip
    assert data.is_reliable is True
    assert data.gradcam.ndim == 2
    assert set(data.concepts) == {"asymmetry", "multicolor"}

    figure_path = tmp_path / "ISIC_0000000_paper.png"
    create_paper_figure(data, figure_path)
    assert figure_path.exists() and figure_path.stat().st_size > 0


def test_load_returns_none_when_panels_missing(tmp_path: Path) -> None:
    json_path = tmp_path / "orphan_result.json"
    json_path.write_text("{}", encoding="utf-8")
    assert load_paper_figure_data(json_path) is None
