"""
ABCDE analysis reporting utilities.
"""

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
    report += f"A - ASYMMETRY: {'⚠ PRESENT' if asym_flag else '✓ Absent'}\n"
    report += f"    Score: {asym_score:.3f}\n"
    report += f"    Interpretation: {'Lesion shows asymmetry' if asym_flag else 'Lesion appears symmetric'}\n\n"

    # Border
    border_score = abcde_result["scores"]["border"]
    border_flag = abcde_result["flags"]["border_flag"]
    report += f"B - BORDER: {'⚠ IRREGULAR' if border_flag else '✓ Regular'}\n"
    report += f"    Score: {border_score:.3f}\n"
    report += f"    Interpretation: {'Irregular, poorly defined borders' if border_flag else 'Smooth, well-defined borders'}\n\n"

    # Color
    color_score = abcde_result["scores"]["color"]
    num_colors = abcde_result["details"]["num_colors"]
    color_flag = abcde_result["flags"]["color_flag"]
    report += f"C - COLOR: {'⚠ VARIED' if color_flag else '✓ Uniform'}\n"
    report += f"    Distinct colors: {num_colors}\n"
    report += f"    Variation score: {color_score:.3f}\n"
    report += f"    Interpretation: {'Multiple colors present' if color_flag else 'Uniform coloration'}\n\n"

    # Diameter
    diameter = abcde_result["scores"]["diameter"]
    diameter_flag = abcde_result["flags"]["diameter_flag"]
    report += f"D - DIAMETER: {'⚠ LARGE' if diameter_flag else '✓ Small'}\n"
    report += f"    Diameter: {diameter:.1f} pixels\n"
    report += f"    Interpretation: {'Lesion exceeds size threshold' if diameter_flag else 'Lesion below size threshold'}\n\n"

    # Evolution (future work)
    report += "E - EVOLUTION: [Requires temporal data - future work]\n\n"

    # GradCAM alignment if provided
    if alignment_scores:
        report += "-" * 60 + "\n"
        report += "GRADCAM ATTENTION ALIGNMENT\n"
        report += "-" * 60 + "\n"
        report += f"Border alignment: {alignment_scores['border_alignment']:.3f}\n"
        report += (
            f"Overall lesion attention: {alignment_scores['overall_alignment']:.3f}\n"
        )
        report += f"Mean attention on lesion: {alignment_scores['mean_lesion_attention']:.3f}\n"
        report += f"Max attention on lesion: {alignment_scores['max_lesion_attention']:.3f}\n\n"

    report += "=" * 60 + "\n"

    return report
