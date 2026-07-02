# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

from typing import Any


def create_abcde_report(
    abcde_result: dict[str, Any], alignment_scores: dict[str, float] | None = None
) -> str:
    """
    Generate a human-readable ABCDE analysis report.

    Args:
        abcde_result: Result from ABCDEAnalyzer.analyze_image()
        alignment_scores: Optional GradCAM alignment scores

    Returns:
        Formatted text report
    """
    report = "=" * 60 + "\n"
    report += "ABCDE CRITERION ANALYSIS REPORT\n"
    report += "=" * 60 + "\n\n"

    # Overall risk
    report += f"OVERALL RISK ASSESSMENT: {abcde_result['overall_risk']}\n"
    report += "-" * 60 + "\n\n"

    # Asymmetry
    asym_score = abcde_result["scores"]["asymmetry"]
    asym_flag = abcde_result["flags"]["asymmetry_flag"]
    report += f"A - ASYMMETRY: {'[!] PRESENT' if asym_flag else '[OK] Absent'}\n"
    report += f"    Score: {asym_score:.3f}\n"
    asym_interp = "Lesion shows asymmetry" if asym_flag else "Lesion appears symmetric"
    report += f"    Interpretation: {asym_interp}\n\n"

    # Border
    border_score = abcde_result["scores"]["border"]
    border_flag = abcde_result["flags"]["border_flag"]
    report += f"B - BORDER: {'[!] IRREGULAR' if border_flag else '[OK] Regular'}\n"
    report += f"    Score: {border_score:.3f}\n"
    border_interp = (
        "Irregular, poorly defined borders" if border_flag else "Smooth, well-defined borders"
    )
    report += f"    Interpretation: {border_interp}\n\n"

    # Color
    color_score = abcde_result["scores"]["color"]
    num_colors = abcde_result["details"]["num_colors"]
    color_flag = abcde_result["flags"]["color_flag"]
    report += f"C - COLOR: {'[!] VARIED' if color_flag else '[OK] Uniform'}\n"
    report += f"    Distinct colors: {num_colors}\n"
    report += f"    Variation score: {color_score:.3f}\n"
    color_interp = "Multiple colors present" if color_flag else "Uniform coloration"
    report += f"    Interpretation: {color_interp}\n\n"

    # Diameter
    diameter = abcde_result["scores"]["diameter"]
    diameter_flag = abcde_result["flags"]["diameter_flag"]
    report += f"D - DIAMETER: {'[!] LARGE' if diameter_flag else '[OK] Small'}\n"
    report += f"    Diameter: {diameter:.1f} pixels\n"
    diameter_interp = (
        "Lesion exceeds size threshold" if diameter_flag else "Lesion below size threshold"
    )
    report += f"    Interpretation: {diameter_interp}\n\n"

    # Evolution (future work)
    report += "E - EVOLUTION: [Requires temporal data - future work]\n\n"

    # GradCAM alignment if provided
    if alignment_scores:
        report += "-" * 60 + "\n"
        report += "GRADCAM ATTENTION ALIGNMENT\n"
        report += "-" * 60 + "\n"
        report += f"Border alignment: {alignment_scores['border_alignment']:.3f}\n"
        report += f"Overall lesion attention: {alignment_scores['overall_alignment']:.3f}\n"
        report += f"Mean attention on lesion: {alignment_scores['mean_lesion_attention']:.3f}\n"
        report += f"Max attention on lesion: {alignment_scores['max_lesion_attention']:.3f}\n\n"

    report += "=" * 60 + "\n"

    return report
