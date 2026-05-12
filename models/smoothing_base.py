import torch
import torch.nn as nn
from configs.settings import DEVICE
from data.cifar10_loader import CIFAR10_MEAN, CIFAR10_STD


class NormalizeWrapper(nn.Module):
    """
    Wraps a base classifier that was trained on normalized inputs
    so it accepts [0,1] inputs from the Smooth wrapper.

    Smooth._sample_noise feeds [0,1] images (after adding Gaussian
    noise in pixel space). This wrapper normalizes them before the
    forward pass, matching what the base saw during training.
    """
    def __init__(self, base_classifier):
        super().__init__()
        self.base = base_classifier
        self.register_buffer(
            "mean",
            torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # x in [0, 1] → normalize → base
        return self.base((x - self.mean) / self.std)