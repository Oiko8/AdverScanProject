import sys
from models.resnet_model import get_model_resnet50
from models.train import train
from data.cifar10_loader import get_loaders_with_noise
from configs.settings import DEVICE

def run(sigma=0.25):
    train_loader, test_loader = get_loaders_with_noise(sigma=sigma)
    model = get_model_resnet50().to(DEVICE)

    sigma_str = f"{int(sigma*100):03d}"   # 025, 050, 100
    save_name = f"model3_smoothing_base_sigma{sigma_str}_resnet50.pth"

    train(model, train_loader, test_loader, save_name=save_name)

if __name__ == "__main__":
    sigma = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    run(sigma=sigma)