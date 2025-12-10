# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-12-09

"""
GradCAM++ implementation for attention visualization.

Reference:
    Chattopadhay et al. "Grad-CAM++: Generalized Gradient-Based Visual
    Explanations for Deep Convolutional Networks" (2018)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class MelanomaGradCAM:
    """
    GradCAM++ wrapper for MelanomaNet.

    Generates attention maps showing which regions of the dermoscopic
    image most influence the model's melanoma prediction.

    Args:
        model: Trained MelanomaNet model
        target_layer: Target convolutional layer for GradCAM
        device: Device (cuda/cpu)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Get target layer (last conv layer by default)
        if target_layer is None:
            target_layer = model.get_last_conv_layer()

        self.target_layers = [target_layer]

        # Initialize GradCAM++
        self.cam = GradCAMPlusPlus(model=model, target_layers=self.target_layers)

    def generate_attention_map(
        self, image: torch.Tensor, target_class: int | None = None
    ) -> np.ndarray:
        """
        Generate attention heatmap for an image.

        Args:
            image: Input tensor (1, 3, H, W) or (3, H, W)
            target_class: Target class index (None = predicted class)

        Returns:
            Attention heatmap as numpy array (H, W) with values [0, 1]
        """
        # Ensure batch dimension
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Generate CAM
        # target_class=None means use the predicted class
        targets = (
            None if target_class is None else [ClassifierOutputTarget(target_class)]
        )

        grayscale_cam = self.cam(input_tensor=image, targets=targets)

        # Return first image's CAM
        return grayscale_cam[0]

    def visualize_attention(
        self,
        image: torch.Tensor,
        original_image: np.ndarray,
        target_class: int | None = None,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Create attention visualization overlay.

        Args:
            image: Preprocessed input tensor (1, 3, H, W)
            original_image: Original RGB image as numpy array (H, W, 3) [0, 1]
            target_class: Target class for attention
            colormap: OpenCV colormap for visualization

        Returns:
            Visualization with attention overlay (H, W, 3) [0, 255]
        """
        # Generate attention map
        attention_map = self.generate_attention_map(image, target_class)

        # Resize attention map to match original image
        if attention_map.shape != original_image.shape[:2]:
            attention_map = cv2.resize(
                attention_map, (original_image.shape[1], original_image.shape[0])
            )

        # Create visualization using pytorch_grad_cam utility
        visualization = show_cam_on_image(
            original_image, attention_map, use_rgb=True, colormap=colormap
        )

        return visualization

    def get_prediction(self, image: torch.Tensor) -> tuple[int, float]:
        """
        Get model prediction and confidence.

        Args:
            image: Input tensor (1, 3, H, W)

        Returns:
            Tuple of (predicted_class, confidence)
        """
        self.model.eval()

        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.model(image)
            probs = F.softmax(outputs, dim=1)

            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        return pred_class, confidence


def denormalize_image(
    image: torch.Tensor,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """
    Denormalize image tensor for visualization.

    Args:
        image: Normalized image tensor (3, H, W) or (1, 3, H, W)
        mean: Mean used for normalization. Default is ImageNet mean (0.485, 0.456, 0.406)
        std: Std used for normalization. Default is ImageNet std (0.229, 0.224, 0.225)

    Returns:
        Denormalized numpy array (H, W, 3) with values [0, 1]
    """
    if image.ndim == 4:
        image = image.squeeze(0)

    # Convert to numpy and transpose
    img = image.cpu().numpy().transpose(1, 2, 0)

    # Denormalize
    mean_np = np.array(mean)
    std_np = np.array(std)
    img = img * std_np + mean_np

    # Clip to [0, 1]
    img = np.clip(img, 0, 1)

    return img
