# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

from .models import FastCAVResult


def create_fastcav_report(result: FastCAVResult) -> str:
    """
    Create human-readable FastCAV report.

    Args:
        result: FastCAVResult from analysis

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "CONCEPT ACTIVATION ANALYSIS (FastCAV)",
        "=" * 60,
        f"Target Class: {result.target_class}",
        f"Feature Dimension: {result.feature_dim}",
        "",
        "Concept Importance Scores:",
        "(+ supports prediction, - opposes prediction)",
        "(higher magnitude = stronger influence on prediction)",
        "-" * 40,
    ]

    # Sort concepts by absolute TCAV score
    sorted_concepts = sorted(
        result.concept_scores.values(),
        key=lambda x: abs(x.tcav_score),
        reverse=True,
    )

    for cs in sorted_concepts:
        direction = "+" if cs.tcav_score > 0 else "-"
        significance = "*" if cs.is_significant else ""
        lines.append(
            f"  {cs.concept_name:20s}: {direction}{abs(cs.tcav_score):6.3f} "
            f"(CAV acc: {cs.accuracy:.2f}){significance}"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "-" * 40,
        ]
    )

    # Top positive concepts
    positive = [cs for cs in sorted_concepts if cs.tcav_score > 0 and cs.is_significant]
    if positive:
        lines.append("Concepts SUPPORTING this prediction:")
        for cs in positive[:3]:
            lines.append(f"  - {cs.concept_name} (score: +{cs.tcav_score:.3f})")

    # Top negative concepts
    negative = [cs for cs in sorted_concepts if cs.tcav_score < 0 and cs.is_significant]
    if negative:
        lines.append("Concepts OPPOSING this prediction:")
        for cs in negative[:3]:
            lines.append(f"  - {cs.concept_name} (score: {cs.tcav_score:.3f})")

    if not positive and not negative:
        lines.append("No strongly significant concepts detected.")

    lines.append("")
    lines.append("Note: * indicates statistically significant concept (acc > 0.6)")
    lines.append("=" * 60)

    return "\n".join(lines)
