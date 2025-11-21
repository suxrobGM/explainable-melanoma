"""
Loss functions for imbalanced classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Focuses training on hard examples by down-weighting easy examples.

    Args:
        alpha: Weighting factor for class imbalance [0, 1]
        gamma: Focusing parameter (gamma > 0 reduces loss for well-classified)
        reduction: 'mean' or 'sum'

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def create_criterion(config: dict, class_weights: torch.Tensor) -> nn.Module:
    """
    Create loss criterion based on configuration.

    Args:
        config: Configuration dictionary
        class_weights: Class weights for imbalanced dataset

    Returns:
        Loss criterion module
    """
    if config["training"].get("focal_loss", False):
        # Focal loss
        criterion = FocalLoss(
            alpha=config["training"]["focal_alpha"],
            gamma=config["training"]["focal_gamma"],
        )
    else:
        # Weighted cross entropy
        if config["training"]["use_class_weights"]:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

    return criterion
