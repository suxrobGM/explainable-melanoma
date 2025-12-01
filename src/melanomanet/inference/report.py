"""Report generation for inference results."""

from pathlib import Path

from rich.console import Console

from ..abcde import create_abcde_report
from ..explainability import create_fastcav_report
from .models import InferenceResult

console = Console()


def save_report(
    result: InferenceResult,
    image_path: str,
    class_names: list[str],
    output_path: str,
) -> None:
    """Save comprehensive text report.

    Args:
        result: InferenceResult containing all analysis results
        image_path: Path to the analyzed image
        class_names: List of class names
        output_path: Path to save the report (will use .txt extension)
    """
    report_path = Path(output_path).with_suffix(".txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MELANOMANET COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Prediction: {class_names[result.pred_class]}\n")
        f.write(f"Confidence: {result.confidence:.4f}\n\n")

        if result.uncertainty:
            f.write("UNCERTAINTY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"Predictive Uncertainty: {result.uncertainty.predictive_uncertainty:.4f} "
                "(total model uncertainty)\n"
            )
            f.write(
                f"Epistemic Uncertainty:  {result.uncertainty.epistemic_uncertainty:.4f} "
                "(model knowledge gaps - can improve with more data)\n"
            )
            f.write(
                f"Aleatoric Uncertainty:  {result.uncertainty.aleatoric_uncertainty:.4f} "
                "(inherent data noise - cannot be reduced)\n"
            )
            reliability = "RELIABLE" if result.uncertainty.is_reliable else "UNCERTAIN"
            f.write(f"Reliability: {reliability}\n\n")

        if result.abcde:
            f.write(create_abcde_report(result.abcde, result.alignment_scores))
            f.write("\n")

        if result.fastcav:
            f.write(create_fastcav_report(result.fastcav))

    console.print(f"[green]Report saved to {report_path}[/green]")
