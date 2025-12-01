import torch
import torch.nn as nn
import torchvision.models as tv_models


class MelanomaNet(nn.Module):
    """
    MelanomaNet model for multi-class skin lesion classification.

    Uses EfficientNet V2 backbone architectures from torchvision.

    Args:
        backbone: Backbone architecture name (efficientnet_v2_s/m/l)
        num_classes: Number of output classes (default: 9 for ISIC 2019)
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate in classification head
    """

    BACKBONE_DIMS = {
        "efficientnet_v2_s": 1280,
        "efficientnet_v2_m": 1280,
        "efficientnet_v2_l": 1280,
    }

    def __init__(
        self,
        backbone: str = "efficientnet_v2_m",
        num_classes: int = 9,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.dropout_rate = dropout_rate

        # Initialize EfficientNet backbone
        self._init_efficientnet(backbone, pretrained)

        # Custom classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
        )

    def _init_efficientnet(self, backbone: str, pretrained: bool) -> None:
        """Initialize EfficientNet V2 backbone."""
        if backbone == "efficientnet_v2_s":
            weights = (
                tv_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            model = tv_models.efficientnet_v2_s(weights=weights)
        elif backbone == "efficientnet_v2_m":
            weights = (
                tv_models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            model = tv_models.efficientnet_v2_m(weights=weights)
        elif backbone == "efficientnet_v2_l":
            weights = (
                tv_models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            model = tv_models.efficientnet_v2_l(weights=weights)
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported: efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l"
            )

        self.backbone = model.features
        self.feature_dim = self.BACKBONE_DIMS.get(backbone, 1280)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, 3, H, W)

        Returns:
            Logits tensor (batch_size, num_classes)
        """
        features = self.backbone(x)  # (B, C, H, W)
        pooled = self.global_pool(features)
        pooled = pooled.flatten(1)
        logits = self.classifier(pooled)
        return logits

    def forward_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with MC Dropout uncertainty estimation.

        Args:
            x: Input tensor (batch_size, 3, H, W)
            n_samples: Number of stochastic forward passes

        Returns:
            Tuple of (mean_logits, predictive_uncertainty, epistemic_uncertainty)
        """
        was_training = self.training
        self.train()  # Enable dropout

        logits_samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                logits = self.forward(x)
                logits_samples.append(logits)

        logits_stack = torch.stack(logits_samples, dim=0)  # (n_samples, B, C)

        # Mean prediction
        mean_logits = logits_stack.mean(dim=0)

        # Probabilities for each sample
        probs_stack = torch.softmax(logits_stack, dim=-1)
        mean_probs = probs_stack.mean(dim=0)

        # Predictive uncertainty (entropy of mean prediction)
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-10), dim=-1
        )

        # Epistemic uncertainty (mutual information / variance)
        epistemic_uncertainty = probs_stack.var(dim=0).mean(dim=-1)

        self.train(was_training)

        return mean_logits, predictive_entropy, epistemic_uncertainty

    def get_last_conv_layer(self) -> nn.Module:
        """
        Get the last convolutional layer for GradCAM.

        Returns:
            Last conv layer module
        """
        layers = list(self.backbone.children())
        return layers[-1]

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations (for CAV/TCAV analysis).

        Args:
            x: Input tensor (batch_size, 3, H, W)

        Returns:
            Feature tensor (batch_size, feature_dim)
        """
        with torch.no_grad():
            features = self.backbone(x)
            pooled = self.global_pool(features).flatten(1)
        return pooled


def create_model(config: dict) -> MelanomaNet:
    """
    Factory function to create MelanomaNet model from config.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized MelanomaNet model
    """
    model_config = config.get("model", {})

    model = MelanomaNet(
        backbone=model_config.get("backbone", "efficientnet_v2_m"),
        num_classes=config["data"]["num_classes"],
        pretrained=model_config.get("pretrained", True),
        dropout_rate=model_config.get("dropout_rate", 0.3),
    )

    return model
