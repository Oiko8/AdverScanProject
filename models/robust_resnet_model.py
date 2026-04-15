import torch
import torch.nn as nn
from robustbench.utils import load_model
from configs.settings import DEVICE

# CIFAR-10 normalization parameters
_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


class NormalizedModel(nn.Module):
    """
    Wraps a RobustBench model that expects [0,1] input so that
    it correctly handles normalized input from our data loader.

    Our loader produces images normalized with CIFAR-10 mean/std.
    This wrapper undoes that normalization before passing to the
    underlying model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mean  = _MEAN.to(DEVICE)
        self.std   = _STD.to(DEVICE)

    def forward(self, x):
        # Denormalize: normalized -> [0, 1]
        return self.model(torch.clamp(x * self.std + self.mean, 0, 1))


def get_robust_model():
    """
    Loads Engstrom2019Robustness from RobustBench and wraps it
    to accept normalized input from our data loader.

    Model  : Engstrom2019Robustness
    Dataset: CIFAR-10
    Threat : Linf, epsilon = 8/255
    Arch   : ResNet-50
    Clean accuracy (RobustBench leaderboard): ~87.0%
    """
    base = load_model(
        model_name="Engstrom2019Robustness",
        dataset="cifar10",
        threat_model="Linf"
    ).to(DEVICE)

    base.eval()
    return NormalizedModel(base)