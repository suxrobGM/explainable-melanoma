"""Generate paper-ready figures from existing inference results."""

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

from melanomanet.inference.paper_figures import PaperFigureData, create_paper_figure


def parse_result_file(txt_path: Path) -> dict | None:
    """Parse inference result text file for values."""
    content = txt_path.read_text()

    # Extract all values with regex
    patterns = {
        "prediction": r"Prediction: (\w+)",
        "confidence": r"Confidence: ([\d.]+)",
        "predictive_unc": r"Predictive Uncertainty: ([\d.]+)",
        "epistemic_unc": r"Epistemic Uncertainty:\s+([\d.]+)",
        "aleatoric_unc": r"Aleatoric Uncertainty:\s+([\d.]+)",
        "reliability": r"Reliability: (\w+)",
        "risk_level": r"OVERALL RISK ASSESSMENT: (\w+)",
        "asymmetry": r"A - ASYMMETRY:.*?Score: ([\d.]+)",
        "border": r"B - BORDER:.*?Score: ([\d.]+)",
        "n_colors": r"Distinct colors: (\d+)",
        "diameter": r"Diameter: ([\d.]+)",
    }

    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.DOTALL)
        result[key] = match.group(1) if match else None

    # Parse FastCAV scores
    fastcav_matches = re.findall(r"(\w+)\s+: ([+-] ?[\d.]+)", content)
    result["fastcav"] = [(m[0], float(m[1].replace(" ", ""))) for m in fastcav_matches]

    return result if result.get("prediction") else None


def extract_image_panels(png_path: Path) -> dict[str, np.ndarray]:
    """Extract image panels from existing result PNG."""
    full_img = np.array(Image.open(png_path))
    h, w = full_img.shape[:2]

    # Layout positions (approximate)
    row1_top, row1_bottom = int(h * 0.08), int(h * 0.38)
    row2_top, row2_bottom = int(h * 0.40), int(h * 0.70)
    col_width = w // 4

    return {
        "original": full_img[row1_top:row1_bottom, 0:col_width],
        "gradcam": full_img[row1_top:row1_bottom, col_width : 2 * col_width],
        "overlay": full_img[row1_top:row1_bottom, 2 * col_width : 3 * col_width],
        "asymmetry": full_img[row2_top:row2_bottom, 0:col_width],
        "border": full_img[row2_top:row2_bottom, col_width : 2 * col_width],
        "color": full_img[row2_top:row2_bottom, 2 * col_width : 3 * col_width],
        "diameter": full_img[row2_top:row2_bottom, 3 * col_width : w],
    }


def generate_paper_figures(output_dir: Path, paper_output_dir: Path) -> None:
    """Generate paper figures from existing inference results."""
    paper_output_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in output_dir.glob("ISIC_*_result.txt"):
        image_id = txt_file.stem.replace("_result", "")
        png_file = output_dir / f"{image_id}_result.png"

        if not png_file.exists():
            continue

        # Parse results
        parsed = parse_result_file(txt_file)
        if not parsed:
            print(f"Skipping {image_id}: couldn't parse results")
            continue

        # Extract image panels
        panels = extract_image_panels(png_file)

        # Build data object
        concepts, scores = zip(*parsed["fastcav"]) if parsed["fastcav"] else ([], [])

        data = PaperFigureData(
            prediction=parsed["prediction"],
            confidence=float(parsed["confidence"]),
            risk_level=parsed.get("risk_level") or "Unknown",
            original=panels["original"],
            gradcam=panels["gradcam"],
            overlay=panels["overlay"],
            asymmetry_img=panels["asymmetry"],
            border_img=panels["border"],
            color_img=panels["color"],
            diameter_img=panels["diameter"],
            asymmetry_score=float(parsed["asymmetry"] or 0),
            border_score=float(parsed["border"] or 0),
            n_colors=int(parsed["n_colors"] or 0),
            diameter=float(parsed["diameter"] or 0),
            predictive_unc=float(parsed["predictive_unc"] or 0),
            epistemic_unc=float(parsed["epistemic_unc"] or 0),
            aleatoric_unc=float(parsed["aleatoric_unc"] or 0),
            is_reliable=(parsed.get("reliability") == "RELIABLE"),
            concepts=list(concepts),
            scores=list(scores),
        )

        # Generate figure
        output_path = paper_output_dir / f"{image_id}_paper.png"
        create_paper_figure(data, output_path)
        print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready figures")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory with existing inference results",
    )
    parser.add_argument(
        "--paper-output",
        type=str,
        default="docs/report/figures",
        help="Output directory for paper figures",
    )
    parser.add_argument(
        "--from-existing",
        action="store_true",
        help="Generate from existing result files (default behavior)",
    )
    args = parser.parse_args()

    generate_paper_figures(Path(args.output_dir), Path(args.paper_output))


if __name__ == "__main__":
    main()
