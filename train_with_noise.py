from models.resnet_model import get_model_resnet50
from models.train import train
from data.cifar10_loader import get_loaders_with_noise
from configs.settings import DEVICE

def run():
    train_loader, test_loader = get_loaders_with_noise(sigma=0.25)
    model = get_model_resnet50().to(DEVICE)
    train(model, train_loader, test_loader,
        save_name='model3_smoothing_base_sigma025_resnet50.pth')

if __name__ == "__main__":
    run()