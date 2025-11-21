"""
MelanomaNet: Explainable melanoma detection model.

Architecture: EfficientNet backbone with custom classification head.
Supports GradCAM++ for attention visualization.
"""

import timm
import torch
import torch.nn as nn


class MelanomaNet(nn.Module):
    """
    MelanomaNet model for binary melanoma classification.

    Uses pretrained EfficientNet backbone with custom head for
    melanoma detection. Designed for attention-based explainability.

    Args:
        backbone: EfficientNet variant ('efficientnet_b0' to 'efficientnet_b4')
        num_classes: Number of output classes (default: 2 for binary)
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout rate in classification head
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # Load pretrained EfficientNet from timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
            global_pool="",  # Remove global pooling, we'll add custom
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Custom classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.feature_dim, num_classes)
        )

        # Store architecture info for GradCAM
        self.backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, 3, H, W)

        Returns:
            Logits tensor (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)  # (B, C, H, W)

        # Global pooling
        pooled = self.global_pool(features)  # (B, C, 1, 1)
        pooled = pooled.flatten(1)  # (B, C)

        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)

        return logits

    def get_last_conv_layer(self) -> nn.Module:
        """
        Get the last convolutional layer for GradCAM.

        Returns:
            Last conv layer module
        """
        # For EfficientNet, last conv is in blocks[-1]
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1]
        else:
            raise AttributeError(f"Cannot find conv layer in {self.backbone_name}")


def create_model(config: dict) -> MelanomaNet:
    """
    Factory function to create MelanomaNet model from config.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized MelanomaNet model
    """
    model = MelanomaNet(
        backbone=config["model"]["backbone"],
        num_classes=config["data"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout_rate"],
    )

    return model
