"""Single Typer CLI exposing all MelanomaNet workflows.

Installed as the ``melanoma`` console script; also runnable as
``python -m melanomanet.cli`` (used by ``melanoma run-seeds`` subprocesses).
"""

from pathlib import Path
from typing import Annotated

import typer

from .config import load_config

app = typer.Typer(
    help="MelanomaNet: explainable melanoma detection.", no_args_is_help=True
)

ConfigOpt = Annotated[str, typer.Option(help="Path to config file")]
CheckpointOpt = Annotated[str, typer.Option(help="Path to checkpoint")]


@app.command()
def train(
    config: ConfigOpt = "config.yaml",
    resume: Annotated[
        str | None,
        typer.Option(help="Path to checkpoint file to resume training from"),
    ] = None,
) -> None:
    """Train MelanomaNet."""
    from .engine.train import train as run_train

    run_train(load_config(config), resume_checkpoint=resume)


@app.command("eval")
def evaluate(
    config: ConfigOpt = "config.yaml",
    checkpoint: CheckpointOpt = "checkpoints/best_model.pth",
) -> None:
    """Evaluate MelanomaNet on the test set."""
    from .evaluation.evaluate import evaluate as run_evaluate

    run_evaluate(load_config(config), checkpoint)


@app.command("run-seeds")
def run_seeds(
    config: ConfigOpt = "config.yaml",
    seeds: Annotated[
        str, typer.Option(help="Comma-separated seeds, e.g. 25,26,27")
    ] = "25,26,27",
    output_dir: Annotated[
        str, typer.Option(help="Root directory for per-seed artifacts")
    ] = "results/seeds",
    skip_training: Annotated[
        bool, typer.Option(help="Only evaluate existing per-seed checkpoints")
    ] = False,
) -> None:
    """Run MelanomaNet across multiple seeds and aggregate metrics."""
    from .evaluation.seeds import run_seeds as run_seed_sweep

    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    run_seed_sweep(config, seed_list, Path(output_dir), skip_training)


@app.command()
def faithfulness(
    config: ConfigOpt = "config.yaml",
    checkpoint: CheckpointOpt = "checkpoints/best_model.pth",
    input_dir: Annotated[
        str, typer.Option(help="Directory of images to evaluate")
    ] = "data/sample_images",
    steps: Annotated[
        int, typer.Option(help="Points along deletion/insertion curves")
    ] = 20,
    iou_quantile: Annotated[
        float, typer.Option(help="Attention quantile treated as 'high' for IoU")
    ] = 0.8,
    seed: Annotated[
        int, typer.Option(help="Seed for the random-attention control")
    ] = 25,
) -> None:
    """Evaluate GradCAM++ faithfulness (deletion/insertion AOPC + lesion IoU)."""
    from .evaluation.faithfulness import evaluate_faithfulness

    evaluate_faithfulness(
        load_config(config), checkpoint, input_dir, steps, iou_quantile, seed
    )


@app.command()
def infer(
    config: ConfigOpt = "config.yaml",
    checkpoint: CheckpointOpt = "checkpoints/best_model.pth",
    input: Annotated[
        list[str] | None,
        typer.Option(help="Path to input image(s) - can be specified multiple times"),
    ] = None,
    input_dir: Annotated[
        str | None, typer.Option(help="Directory containing images")
    ] = None,
) -> None:
    """Run inference with comprehensive explainability features."""
    from .inference.batch import collect_image_paths, run_batch_inference

    if not input and not input_dir:
        raise typer.BadParameter("Either --input or --input-dir must be specified")

    image_paths = collect_image_paths(input, input_dir)
    if not image_paths:
        typer.echo("Error: No valid images found!", err=True)
        raise typer.Exit(code=1)

    run_batch_inference(load_config(config), checkpoint, image_paths)


@app.command("generate-concepts")
def generate_concepts(
    config: ConfigOpt = "config.yaml",
    output: Annotated[
        str | None,
        typer.Option(help="Output directory (default: fastcav.concepts_dir)"),
    ] = None,
    max_samples: Annotated[
        int, typer.Option(help="Maximum samples per concept class")
    ] = 200,
    min_samples: Annotated[
        int, typer.Option(help="Minimum samples required per class")
    ] = 50,
) -> None:
    """Generate concept datasets for FastCAV analysis."""
    from .explainability.concepts import generate_concept_dataset

    cfg = load_config(config)
    generate_concept_dataset(
        cfg,
        output_dir=Path(output or cfg.fastcav.concepts_dir),
        max_samples_per_concept=max_samples,
        min_samples_per_class=min_samples,
    )


@app.command("train-fastcav")
def train_fastcav(
    config: ConfigOpt = "config.yaml",
    checkpoint: CheckpointOpt = "checkpoints/best_model.pth",
) -> None:
    """Train FastCAV concept vectors."""
    from .explainability.train_cavs import train_fastcav as run_train_fastcav

    run_train_fastcav(load_config(config), checkpoint)


@app.command("paper-figures")
def paper_figures(
    config: ConfigOpt = "config.yaml",
    output_dir: Annotated[
        str | None,
        typer.Option(help="Directory with inference results (default: output_dir)"),
    ] = None,
    paper_output: Annotated[
        str, typer.Option(help="Output directory for paper figures")
    ] = "docs/report/figures",
) -> None:
    """Generate paper-ready figures from saved inference artifacts."""
    from .inference.paper_figures import generate_paper_figures

    cfg = load_config(config)
    generate_paper_figures(
        Path(output_dir or cfg.paths.output_dir), Path(paper_output)
    )


if __name__ == "__main__":
    app()
