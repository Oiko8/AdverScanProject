import torch
from models.resnet_model import get_model_resnet50
from models.smoothing_base import NormalizeWrapper
from models.smoothing import Smooth
from configs.settings import DEVICE
from data.cifar10_loader import CIFAR10_MEAN, CIFAR10_STD
from data.cifar10_loader import get_loaders

def run():
    # Load the σ=0.25 base
    base = get_model_resnet50().to(DEVICE)
    base.load_state_dict(torch.load(
        "models/model3_smoothing_base_sigma025_resnet50.pth",
        weights_only=True
    ))
    base.eval()

    # Wrap it so it accepts [0,1] input
    wrapped_base = NormalizeWrapper(base).to(DEVICE)
    wrapped_base.eval()

    # Now Smooth can feed [0,1] noisy samples to it correctly
    smoother = Smooth(wrapped_base, num_classes=10, sigma=0.25)


    correct = 0
    total = 200  # small subset
    _, test_loader = get_loaders()

    mean3 = torch.tensor(CIFAR10_MEAN).view(3,1,1).to(DEVICE)
    std3  = torch.tensor(CIFAR10_STD).view(3,1,1).to(DEVICE)

    count = 0
    for images, labels in test_loader:
        for i in range(images.size(0)):
            if count >= total: break
            img_normalized = images[i].to(DEVICE)
            img_01 = torch.clamp(img_normalized * std3 + mean3, 0, 1).unsqueeze(0)
            pred = smoother.predict(img_01, n=1000, alpha=0.001, batch_size=64)
            if pred == labels[i].item():
                correct += 1
            count += 1
        if count >= total: break

    print(f"Smoothed clean accuracy: {100*correct/total:.1f}%")


if __name__ == "__main__":
    run()