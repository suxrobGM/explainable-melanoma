# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

from ..config import Config
from ..data.transforms import get_val_transforms
from ..utils.console import console
from ..utils.env import resolve_device
from ..utils.gradcam import MelanomaGradCAM
from .analyzers import (
    run_abcde_analysis,
    run_fastcav_analysis,
    run_gradcam,
    run_uncertainty_analysis,
)
from .artifacts import save_artifacts
from .loaders import load_image, load_model
from .models import InferenceResult
from .report import save_report
from .visualization import create_visualization


def run_inference(
    config: Config,
    checkpoint_path: str,
    image_path: str,
    output_path: str,
) -> InferenceResult:
    """
    The main inference function that runs all analyses and creates outputs.
    Run inference on a single image with comprehensive explainability.

    Args:
        config: Configuration
        checkpoint_path: Path to model checkpoint
        image_path: Path to input image
        output_path: Path for output visualization (report will be saved with .txt)

    Returns:
        The packaged inference result
    """
    device = resolve_device(config.device)
    class_names = config.data.class_names

    # Load model and image
    model = load_model(config, checkpoint_path, device)
    original_image, original_np = load_image(image_path, config.data.image_size)

    # Transform image for model
    transform = get_val_transforms(config.data.image_size)
    image_tensor = transform(original_image).unsqueeze(0)

    # Run GradCAM
    gradcam = MelanomaGradCAM(model, device=device)
    pred_class, confidence, attention_map, visualization = run_gradcam(
        gradcam, image_tensor, original_np
    )

    console.print(f"\n[bold green]Prediction: {class_names[pred_class]}[/bold green]")
    console.print(f"[bold green]Confidence: {confidence:.4f}[/bold green]")

    # Run analyses
    uncertainty = run_uncertainty_analysis(
        model, image_tensor, config.uncertainty, device
    )
    fastcav = run_fastcav_analysis(
        model, image_tensor, pred_class, class_names[pred_class], config.fastcav, device
    )
    abcde, alignment = run_abcde_analysis(original_np, attention_map, config.abcde)

    # Package results
    result = InferenceResult(
        pred_class=pred_class,
        confidence=confidence,
        attention_map=attention_map,
        visualization=visualization,
        uncertainty=uncertainty,
        fastcav=fastcav,
        abcde=abcde,
        alignment_scores=alignment,
    )

    # Create outputs
    create_visualization(result, original_np, class_names, output_path)
    save_report(result, image_path, class_names, output_path)
    save_artifacts(result, original_np, class_names, output_path)

    return result
