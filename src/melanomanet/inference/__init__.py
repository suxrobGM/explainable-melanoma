# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

"""Inference module for MelanomaNet."""

from .core import run_inference
from .models import InferenceResult

__all__ = ["run_inference", "InferenceResult"]
