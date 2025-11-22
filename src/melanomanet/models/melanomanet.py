"""
MelanomaNet: Explainable melanoma detection model.

Architecture: EfficientNet V2 backbone with custom classification head.
Supports GradCAM++ for attention visualization.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class MelanomaNet(nn.Module):
    """
    MelanomaNet model for multi-class skin lesion classification.

    Uses pretrained EfficientNet V2 backbone with custom head for
    skin lesion detection. Designed for attention-based explainability.

    Args:
        backbone: EfficientNet V2 variant ('efficientnet_v2_s', 'efficientnet_v2_m',
                  'efficientnet_v2_l')
        num_classes: Number of output classes (default: 9 for ISIC 2019)
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout rate in classification head
    """

    def __init__(
        self,
        backbone: str = "efficientnet_v2_l",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.backbone_name = backbone

        # Load from torchvision
        if backbone == "efficientnet_v2_s":
            weights = (
                tv_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            model = tv_models.efficientnet_v2_s(weights=weights)
            self.feature_dim = 1280  # V2-S output channels
        elif backbone == "efficientnet_v2_m":
            weights = (
                tv_models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            model = tv_models.efficientnet_v2_m(weights=weights)
            self.feature_dim = 1280  # V2-M output channels
        elif backbone == "efficientnet_v2_l":
            weights = (
                tv_models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            model = tv_models.efficientnet_v2_l(weights=weights)
            self.feature_dim = 1280  # V2-L output channels
        else:
            raise ValueError(
                f"Unsupported EfficientNet V2 variant: {backbone}. "
                f"Choose from: efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l"
            )

        # Extract feature extractor (remove classifier)
        self.backbone = model.features

        # Custom classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.feature_dim, num_classes)
        )

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
        # For EfficientNet V2 (torchvision), backbone is nn.Sequential
        # Access the last layer
        layers = list(self.backbone.children())
        return layers[-1]


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
