import torch.nn as nn
from torchvision.models import resnet18, resnet50
from configs.settings import NUM_CLASSES


def get_model():
    """
    ResNet-18 adapted for CIFAR-10 (32x32 input).
    ResNet-18 is designed normally for ImageNet(224x224, 1000 classes).

    Two changes from the standard ImageNet ResNet-18:
      1. First conv: 7x7 stride-2 -> 3x3 stride-1 (preserves spatial resolution)
      2. Remove the max pool after the first conv (same reason)
    """
    model = resnet18(weights=None, num_classes=NUM_CLASSES)

    # Adjust for 32x32 input
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()

    return model



def get_model_resnet50():
    """
    Standard ResNet-50 adapted for CIFAR-10 (32x32 input).
    Used as same-architecture surrogate for transfer attack Stage 4.

    Same CIFAR-10 adaptations as ResNet-18:
      1. First conv: 7x7 stride-2 -> 3x3 stride-1
      2. Remove maxpool
    """
    model = resnet50(weights=None, num_classes=NUM_CLASSES)

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()

    return model