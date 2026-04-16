import os
import torch
import matplotlib.pyplot as plt
from models.resnet_model import get_model
from models.train import evaluate
from attacks.pgd import pgd_attack
from data.cifar10_loader import get_loaders
from configs.settings import (
    DEVICE, MODEL_DIR, OUTPUT_DIR, EPSILONS,
    PGD_STEPS, PGD_RESTARTS, EVAL_NUM_SAMPLES, scale_epsilon
)


def evaluate_robust(model, loader, epsilon, num_samples=EVAL_NUM_SAMPLES):
    """
    Runs PGD at a given epsilon and returns robust accuracy.
    Stops after num_samples images to keep runtime manageable.
    """
    model.eval()
    correct = 0
    total   = 0

    for images, labels in loader:
        if total >= num_samples:
            break

        # Only take as many samples as needed
        remaining = num_samples - total
        images    = images[:remaining]
        labels    = labels[:remaining]

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # epsilon == 0 means clean accuracy — no attack needed
        if epsilon == 0:
            adv = images
        else:
            adv = pgd_attack(model, images, labels, epsilon,
                             steps=PGD_STEPS, restarts=PGD_RESTARTS)

        with torch.no_grad():
            outputs   = model(adv)
            _, predicted = outputs.max(1)
            correct  += predicted.eq(labels).sum().item()
            total    += labels.size(0)

    return 100.0 * correct / total


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load trained model
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "model1_standard.pth"),
        weights_only=True
    ))
    model.eval()

    _, test_loader = get_loaders()

    print("=" * 55)
    print(f"  Model 1 — Standard ResNet-18")
    print(f"  PGD steps: {PGD_STEPS}  |  Restarts: {PGD_RESTARTS}")
    print(f"  Samples  : {EVAL_NUM_SAMPLES}")
    print("=" * 55)

    accuracies = []

    for eps in EPSILONS:
        eps_scaled = scale_epsilon(eps) if eps > 0 else 0
        acc = evaluate_robust(model, test_loader, eps_scaled)
        accuracies.append(acc)
        eps_label = f"{round(eps * 255):2d}/255" if eps > 0 else " 0/255"
        print(f"  epsilon = {eps_label}  ->  robust accuracy = {acc:.1f}%")

    print("=" * 55)

    # ── Plot ────────────────────────────────────────────────────────
    eps_labels = [f"{round(e*255)}/255" for e in EPSILONS]

    plt.figure(figsize=(8, 5))
    plt.plot(eps_labels, accuracies, marker="o", linewidth=2,
             color="#2E75B6", markersize=8, label="PGD robust accuracy")

    for x, y in zip(eps_labels, accuracies):
        plt.annotate(f"{y:.1f}%", (x, y),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=10)

    plt.title("Model 1 — Standard ResNet-18\nAccuracy vs. PGD Epsilon (CIFAR-10)",
              fontsize=13)
    plt.xlabel("Perturbation budget (epsilon)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "model1_accuracy_vs_epsilon.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved to: {save_path}")


if __name__ == "__main__":
    run()