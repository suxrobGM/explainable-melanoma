from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import UncertaintyResult

if TYPE_CHECKING:
    from ..models.melanomanet import MelanomaNet


class MCDropoutEstimator:
    """
    Monte Carlo Dropout uncertainty estimator.

    Uses multiple stochastic forward passes with dropout enabled
    to estimate epistemic (model) uncertainty.

    Applies additional dropout on features during MC sampling to generate
    meaningful epistemic uncertainty without requiring model retraining.
    """

    def __init__(
        self,
        model: "MelanomaNet",
        n_samples: int = 10,
        uncertainty_threshold: float = 0.5,
        device: torch.device | None = None,
        feature_dropout_rate: float = 0.2,
    ):
        """
        Initialize MC Dropout estimator.

        Args:
            model: MelanomaNet model
            n_samples: Number of stochastic forward passes
            uncertainty_threshold: Threshold for reliable predictions
            device: Device to run inference on
            feature_dropout_rate: Dropout rate applied to features during MC sampling
        """
        self.model = model
        self.n_samples = n_samples
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Additional dropout applied to features for better epistemic uncertainty
        self.feature_dropout = nn.Dropout(p=feature_dropout_rate)

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
        self.feature_dropout.to(self.device)

        # Store original training state
        was_training = self.model.training

        # Enable dropout for MC sampling
        self.model.train()
        self.feature_dropout.train()

        # Extract features once (deterministic)
        with torch.no_grad():
            features = self.model.backbone(x)
            pooled = self.model.global_pool(features).flatten(1)

        # Collect samples with dropout on features
        logits_samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                # Apply dropout to features (stochastic)
                dropped_features = self.feature_dropout(pooled)
                logits = self.model.classifier(dropped_features)
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
