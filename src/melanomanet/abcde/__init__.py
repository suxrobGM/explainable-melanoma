"""
ABCDE Criterion Analysis Module.

Provides automated analysis of the ABCDE melanoma detection criteria:
- A: Asymmetry
- B: Border irregularity
- C: Color variation
- D: Diameter
- E: Evolution (future work)

This module also includes GradCAM alignment analysis to validate that
model attention focuses on clinically relevant features.
"""

from .alignment import analyze_gradcam_alignment
from .analyzer import ABCDEAnalyzer
from .features import analyze_asymmetry, analyze_border, analyze_color, analyze_diameter
from .reporting import create_abcde_report
from .segmentation import extract_lesion_mask
