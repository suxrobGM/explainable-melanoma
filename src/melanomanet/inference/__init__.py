"""Inference module for MelanomaNet."""

from .core import run_inference
from .models import InferenceResult

__all__ = ["run_inference", "InferenceResult"]
