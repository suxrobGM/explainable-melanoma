# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.melanomanet import MelanomaNet
from .models import UncertaintyResult

_EPS = 1e-10


class MCDropoutEstimator:
    """
    Monte Carlo Dropout uncertainty estimator.

    Runs several stochastic forward passes with the network's dropout layers
    active (following Gal and Ghahramani, 2016) and summarises the resulting
    predictive distribution. Batch-normalisation layers are kept in evaluation
    mode so that single-image inference does not depend on batch statistics;
    only dropout is made stochastic.

    Predictive uncertainty is the entropy of the mean prediction. It is
    decomposed as predictive = aleatoric + epistemic, where the aleatoric term
    is the mean entropy of the individual passes and the epistemic term is the
    mutual information between the prediction and the model parameters
    (predictive entropy minus the aleatoric term).
    """

    def __init__(
        self,
        model: MelanomaNet,
        n_samples: int = 10,
        uncertainty_threshold: float = 0.5,
        device: torch.device | None = None,
    ):
        """
        Initialize MC Dropout estimator.

        Args:
            model: MelanomaNet model
            n_samples: Number of stochastic forward passes
            uncertainty_threshold: Threshold on predictive entropy below which a
                prediction is considered reliable
            device: Device to run inference on
        """
        self.model = model
        self.n_samples = n_samples
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @staticmethod
    def _enable_dropout(model: nn.Module) -> None:
        """Set only dropout layers to training mode, leaving all others (in
        particular batch normalisation) in evaluation mode."""
        for module in model.modules():
            if isinstance(module, nn.modules.dropout._DropoutNd):
                module.train()

    def _sample_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Collect ``n_samples`` softmax outputs with dropout active.

        Returns a tensor of shape ``(n_samples, batch, classes)``.
        """
        was_training = self.model.training
        self.model.eval()
        self._enable_dropout(self.model)

        probs = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x)
                probs.append(F.softmax(logits, dim=-1))

        self.model.train(was_training)
        return torch.stack(probs, dim=0)

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

        probs_stack = self._sample_probs(x)  # (n_samples, 1, classes)
        mean_probs = probs_stack.mean(dim=0).squeeze(0)  # (classes,)

        predicted_class = int(mean_probs.argmax().item())
        confidence = mean_probs[predicted_class].item()

        # Predictive uncertainty: entropy of the mean prediction.
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + _EPS)
        ).item()

        # Aleatoric uncertainty: mean entropy of the individual passes.
        per_sample_entropy = -torch.sum(
            probs_stack * torch.log(probs_stack + _EPS), dim=-1
        )
        aleatoric = per_sample_entropy.mean().item()

        # Epistemic uncertainty: mutual information = predictive - aleatoric.
        epistemic = max(0.0, predictive_entropy - aleatoric)

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
            Tuple of (mean_probs, predictive_uncertainty, epistemic_uncertainty),
            where epistemic uncertainty is the per-sample mutual information.
        """
        x = x.to(self.device)
        self.model.to(self.device)

        probs_stack = self._sample_probs(x)  # (n_samples, batch, classes)
        mean_probs = probs_stack.mean(dim=0)  # (batch, classes)

        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + _EPS), dim=-1
        )
        per_sample_entropy = -torch.sum(
            probs_stack * torch.log(probs_stack + _EPS), dim=-1
        ).mean(dim=0)
        epistemic = torch.clamp(predictive_entropy - per_sample_entropy, min=0.0)

        return mean_probs, predictive_entropy, epistemic
