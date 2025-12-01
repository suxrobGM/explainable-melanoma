"""
Uncertainty quantification module for MelanomaNet.

Provides MC Dropout and temperature scaling for reliable confidence estimation.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class UncertaintyResult:
    """Container for uncertainty estimation results."""

    predicted_class: int
    confidence: float
    mean_probs: np.ndarray
    predictive_uncertainty: float  # Total uncertainty (entropy)
    epistemic_uncertainty: float  # Model uncertainty (reducible with more data)
    aleatoric_uncertainty: float  # Data uncertainty (irreducible)
    is_reliable: bool  # True if uncertainty is below threshold


class MCDropoutEstimator:
    """
    Monte Carlo Dropout uncertainty estimator.

    Uses multiple stochastic forward passes with dropout enabled
    to estimate epistemic (model) uncertainty.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        uncertainty_threshold: float = 0.5,
        device: torch.device | None = None,
    ):
        """
        Initialize MC Dropout estimator.

        Args:
            model: MelanomaNet model
            n_samples: Number of stochastic forward passes
            uncertainty_threshold: Threshold for reliable predictions
            device: Device to run inference on
        """
        self.model = model
        self.n_samples = n_samples
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def estimate(self, x: torch.Tensor) -> UncertaintyResult:
        """
        Estimate uncertainty for a single input.

        Args:
            x: Input tensor (1, 3, H, W) or (3, H, W)

        Returns:
            UncertaintyResult with predictions and uncertainty estimates
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        self.model.to(self.device)

        # Store original training state
        was_training = self.model.training

        # Enable dropout for MC sampling
        self.model.train()

        # Collect samples
        logits_samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x)
                logits_samples.append(logits)

        # Stack samples: (n_samples, batch, classes)
        logits_stack = torch.stack(logits_samples, dim=0)
        probs_stack = F.softmax(logits_stack, dim=-1)

        # Mean prediction
        mean_probs = probs_stack.mean(dim=0).squeeze(0)  # (classes,)

        # Predicted class and confidence
        predicted_class = mean_probs.argmax().item()
        confidence = mean_probs[predicted_class].item()

        # Predictive uncertainty: entropy of mean prediction
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-10)
        ).item()

        # Epistemic uncertainty: variance across samples
        epistemic = probs_stack.var(dim=0).mean().item()

        # Aleatoric uncertainty: mean entropy of individual predictions
        individual_entropies = -torch.sum(
            probs_stack * torch.log(probs_stack + 1e-10), dim=-1
        )
        aleatoric = individual_entropies.mean().item()

        # Restore model state
        self.model.train(was_training)

        # Determine reliability
        is_reliable = predictive_entropy < self.uncertainty_threshold

        return UncertaintyResult(
            predicted_class=predicted_class,
            confidence=confidence,
            mean_probs=mean_probs.cpu().numpy(),
            predictive_uncertainty=predictive_entropy,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            is_reliable=is_reliable,
        )

    def estimate_batch(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty for a batch of inputs.

        Args:
            x: Input tensor (batch, 3, H, W)

        Returns:
            Tuple of (mean_probs, predictive_uncertainty, epistemic_uncertainty)
        """
        x = x.to(self.device)
        self.model.to(self.device)

        was_training = self.model.training
        self.model.train()

        logits_samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x)
                logits_samples.append(logits)

        logits_stack = torch.stack(logits_samples, dim=0)
        probs_stack = F.softmax(logits_stack, dim=-1)

        mean_probs = probs_stack.mean(dim=0)
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-10), dim=-1
        )
        epistemic = probs_stack.var(dim=0).mean(dim=-1)

        self.model.train(was_training)

        return mean_probs, predictive_entropy, epistemic


class TemperatureScaling:
    """
    Temperature scaling for probability calibration.

    Post-hoc calibration method that learns a single temperature
    parameter to scale logits for better calibrated probabilities.
    """

    def __init__(self, model: nn.Module, device: torch.device | None = None):
        """
        Initialize temperature scaling.

        Args:
            model: Trained MelanomaNet model
            device: Device for computation
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.temperature = nn.Parameter(torch.ones(1, device=self.device))

    def calibrate(
        self,
        val_loader: DataLoader,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Learn optimal temperature on validation set.

        Args:
            val_loader: Validation data loader
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations

        Returns:
            Learned temperature value
        """
        self.model.eval()
        self.model.to(self.device)

        # Collect all logits and labels
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc="Collecting logits"):
                images = images.to(self.device)
                logits = self.model(images)
                all_logits.append(logits)
                all_labels.append(labels)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0).to(self.device)

        # Optimize temperature
        self.temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.5)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        return self.temperature.item()

    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw model logits

        Returns:
            Calibrated probabilities
        """
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)


def compute_calibration_metrics(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict:
    """
    Compute calibration metrics (ECE, MCE).

    Args:
        probs: Predicted probabilities (N, C)
        labels: True labels (N,)
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with ECE, MCE, and per-bin statistics
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    # Bin boundaries
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_stats = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            bin_error = np.abs(avg_accuracy - avg_confidence)

            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)

            bin_stats.append(
                {
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                    "accuracy": avg_accuracy,
                    "confidence": avg_confidence,
                    "count": in_bin.sum(),
                }
            )

    return {
        "ece": ece,
        "mce": mce,
        "bin_stats": bin_stats,
    }


def get_uncertainty_interpretation(result: UncertaintyResult) -> str:
    """
    Generate human-readable interpretation of uncertainty.

    Args:
        result: UncertaintyResult from MC Dropout estimation

    Returns:
        Interpretation string
    """
    lines = []

    # Confidence level
    if result.confidence > 0.9:
        conf_level = "Very High"
    elif result.confidence > 0.7:
        conf_level = "High"
    elif result.confidence > 0.5:
        conf_level = "Moderate"
    else:
        conf_level = "Low"

    lines.append(f"Confidence Level: {conf_level} ({result.confidence:.1%})")

    # Uncertainty interpretation
    if result.predictive_uncertainty < 0.3:
        unc_level = "Low uncertainty - prediction is reliable"
    elif result.predictive_uncertainty < 0.7:
        unc_level = "Moderate uncertainty - consider additional review"
    else:
        unc_level = "High uncertainty - manual review recommended"

    lines.append(f"Predictive Uncertainty: {unc_level}")

    # Source of uncertainty
    if result.epistemic_uncertainty > result.aleatoric_uncertainty:
        source = "Model uncertainty dominates - more training data may help"
    else:
        source = "Data uncertainty dominates - image quality may be limiting"

    lines.append(f"Uncertainty Source: {source}")

    # Reliability flag
    if result.is_reliable:
        lines.append("Status: RELIABLE - suitable for clinical decision support")
    else:
        lines.append("Status: UNCERTAIN - requires expert verification")

    return "\n".join(lines)
