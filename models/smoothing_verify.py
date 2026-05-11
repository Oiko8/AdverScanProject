import torch
from robustbench.utils import load_model
from models.smoothing import Smooth
from data.cifar10_loader import get_loaders
from configs.settings import DEVICE

def run():
    # Load base model
    base = load_model(
        model_name='Gowal2020Uncovering',
        dataset='cifar10',
        threat_model='L2'
    ).to(DEVICE)
    base.eval()

    _, test_loader = get_loaders()

    # Denormalize helper — smoothing operates in [0,1] space
    mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1).to(DEVICE)
    std  = torch.tensor([0.2023,0.1994,0.2010]).view(1,3,1,1).to(DEVICE)

    # Test base model clean accuracy (raw input, no noise)
    correct_base = 0
    total        = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images_01      = torch.clamp(images * std + mean, 0, 1)
            out            = base(images_01)
            correct_base  += out.argmax(1).eq(labels).sum().item()
            total         += labels.size(0)

    acc_base = 100.0 * correct_base / total
    print(f'Base model clean accuracy     : {acc_base:.1f}%')
    print(f'RobustBench leaderboard       : 90.90%')

    # Test smoothed model at sigma=0.25
    # Use a small subset for speed — certification is slow
    smoother  = Smooth(base, num_classes=10, sigma=0.25)

    correct_smooth = 0
    total_smooth   = 0
    num_samples    = 200  # small subset just to verify wrapper works

    for images, labels in test_loader:
        if total_smooth >= num_samples:
            break
        for i in range(min(len(labels), num_samples - total_smooth)):
            img_normalized = images[i].to(DEVICE)
            # breakpoint()
            mean3 = torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1).to(DEVICE)
            std3  = torch.tensor([0.2023,0.1994,0.2010]).view(3,1,1).to(DEVICE)
            img_01         = torch.clamp(
                img_normalized * std3 + mean3, 0, 1
            ).unsqueeze(0)

            # Majority vote prediction (n=1000 samples, alpha=0.001)
            pred = smoother.predict(img_01, n=1000, alpha=0.001, batch_size=64)

            if pred != Smooth.ABSTAIN and pred == labels[i].item():
                correct_smooth += 1
            total_smooth += 1

    acc_smooth = 100.0 * correct_smooth / total_smooth
    print(f'\nSmoothed model (sigma=0.25)   : {acc_smooth:.1f}%')
    print(f'Expected range               : 70-78%')
    print(f'Note: noise reduces accuracy — this is expected')

    # Quick certification test on 5 images
    print(f'\n── Certification test (5 images) ────────────────────────')
    print(f'  {"Image":>6} {"Predicted":>10} {"True":>6} {"Radius":>8} {"Certified?":>12}')
    print(f'  {"-"*46}')

    count = 0
    for images, labels in test_loader:
        if count >= 5:
            break
        img_normalized = images[0].to(DEVICE)
        mean3 = torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1).to(DEVICE)
        std3  = torch.tensor([0.2023,0.1994,0.2010]).view(3,1,1).to(DEVICE)
        img_01 = torch.clamp(
            img_normalized * std3 + mean3, 0, 1
        ).unsqueeze(0)
        label = labels[0].item()

        pred, radius = smoother.certify(
            img_01, n0=100, n=1000, alpha=0.001, batch_size=64
        )

        certified = pred != Smooth.ABSTAIN and pred == label
        pred_str  = str(pred) if pred != Smooth.ABSTAIN else "ABSTAIN"
        print(f'  {count:>6} {pred_str:>10} {label:>6} {radius:>8.3f} {str(certified):>12}')
        count += 1
        images, labels = next(iter(test_loader))


if __name__ == "__main__":
    run()