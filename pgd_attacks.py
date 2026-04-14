import torch
from models.resnet_model import get_model
from attacks.pgd import pgd_attack
from data.cifar10_loader import get_loaders
from configs.settings import DEVICE, MODEL_DIR
import os


def produce_perturbed_images():
    # Load trained model
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'model1_standard.pth'),
                                    weights_only=True))
    model.eval()

    # Grab one batch
    _, test_loader = get_loaders()
    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    # Run a quick PGD attack (fewer steps just to verify it runs)
    epsilon = 8/255
    adv = pgd_attack(model, images, labels, epsilon, steps=10, restarts=1)

    # Check the perturbation is within budget
    perturbation = (adv - images).abs().max().item()
    print(f'Max perturbation : {perturbation:.6f}')
    print(f'Epsilon budget   : {epsilon:.6f}')
    print(f'Within budget    : {perturbation <= epsilon + 1e-6}')
    print(f'Adv shape        : {adv.shape}')
    print(f'Pixel range      : [{adv.min().item():.3f}, {adv.max().item():.3f}]')

def main():
    produce_perturbed_images()

if __name__ == "__main__":
    main()