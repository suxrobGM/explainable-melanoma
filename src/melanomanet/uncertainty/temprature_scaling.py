import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader


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
