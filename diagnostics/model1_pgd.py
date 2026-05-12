import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.resnet_model import get_model, get_model_resnet50
from data.cifar10_loader import get_loaders
from configs.settings import (
    DEVICE, MODEL_DIR, OUTPUT_DIR, EPSILONS, EVAL_NUM_SAMPLES, scale_epsilon
)
from configs.save_report import save_report


# ── FGSM (single step) ───────────────────────────────────────────────────────

def fgsm_attack(model, images, labels, epsilon):
    """Single-step FGSM attack."""
    if epsilon == 0:
        return images.clone()
    
    scaled_epsilon = scale_epsilon(epsilon)

    images = images.clone().requires_grad_(True)
    criterion = nn.CrossEntropyLoss()

    outputs = model(images)
    loss    = criterion(outputs, labels)
    loss.backward()

    with torch.no_grad():
        adv = images + scaled_epsilon * images.grad.sign()
        adv = torch.clamp(adv, -2.4291, 2.7537)

    return adv.detach()


def evaluate_fgsm(model, loader, epsilon, num_samples=EVAL_NUM_SAMPLES):
    """Returns FGSM accuracy at a given epsilon."""
    model.eval()
    correct = 0
    total   = 0

    for images, labels in loader:
        if total >= num_samples:
            break
        remaining      = num_samples - total
        images, labels = images[:remaining].to(DEVICE), labels[:remaining].to(DEVICE)

        adv = fgsm_attack(model, images, labels, epsilon)

        with torch.no_grad():
            outputs      = model(adv)
            _, predicted = outputs.max(1)
            correct     += predicted.eq(labels).sum().item()
            total       += labels.size(0)

    return 100.0 * correct / total


# ── PGD with loss trajectory logging ─────────────────────────────────────────

def pgd_loss_trajectory(model, images, labels, epsilon, steps=100):
    """
    Runs PGD on a single batch and records the loss value at each step.
    Used to check D4: loss should increase monotonically.
    """
    from attacks.pgd import CIFAR10_MIN, CIFAR10_MAX

    if epsilon == 0:
        return []

    epsilon   = scale_epsilon(epsilon)

    step_size = epsilon / 10
    criterion = nn.CrossEntropyLoss()
    images    = images.to(DEVICE)
    labels    = labels.to(DEVICE)

    delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, CIFAR10_MIN, CIFAR10_MAX) - images
    delta.requires_grad_(True)

    trajectory = []

    for _ in range(steps):
        outputs = model(images + delta)
        loss    = criterion(outputs, labels)
        trajectory.append(loss.item())
        loss.backward()

        with torch.no_grad():
            delta.data = delta.data + step_size * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(
                images + delta.data, CIFAR10_MIN, CIFAR10_MAX
            ) - images

        delta.grad.zero_()

    return trajectory


# ── D1: FGSM vs PGD comparison ────────────────────────────────────────────────

def check_d1(model, test_loader, pgd_accuracies):
    """
    D1 fires if PGD robust accuracy > FGSM robust accuracy at any epsilon (FGSM breaking the model more than PGD signals gradient masking).
    pgd_accuracies: list of PGD accuracies matching EPSILONS order.
    """
    print("\n── D1: Gradient Masking Check (FGSM vs PGD) ──────────────────")
    print(f"  {'Epsilon':<12} {'FGSM':>10} {'PGD':>10} {'D1 signal':>12}")
    print(f"  {'-'*48}")

    d1_fired = False

    for i, eps in enumerate(EPSILONS):
        fgsm_acc = evaluate_fgsm(model, test_loader, eps)
        pgd_acc  = pgd_accuracies[i]
        d1_fired = d1_fired or (fgsm_acc < pgd_acc)
        flag     = "⚠ D1 FIRED" if d1_fired else "ok"


        eps_label = f"{round(eps*255)}/255"
        print(f"  {eps_label:<12} {fgsm_acc:>9.1f}% {pgd_acc:>9.1f}% {flag:>12}")

    print()
    if d1_fired:
        print("  D1 STATUS: FIRED — possible gradient masking detected.")
        print("  FGSM breaks the model more than PGD does, which should not happen on a clean model.")
        print("  Interpretation: the model may be obfuscating gradients.")
    else:
        print("  D1 STATUS: clear — PGD is stronger than FGSM at all epsilons.")
        print("  Interpretation: gradients are well-behaved, no masking detected.")

    return d1_fired


# ── D4: Loss trajectory check ─────────────────────────────────────────────────

def check_d4(model, test_loader, epsilon=8/255, steps=100):
    """
    D4 fires if the PGD loss trajectory is non-monotonic.
    We measure the number of steps where loss decreases from the previous step.
    """
    print("\n── D4: Loss Consistency Check (PGD trajectory) ───────────────")

    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    trajectory = pgd_loss_trajectory(model, images, labels, epsilon, steps)

    # Count non-monotonic steps (loss drops from previous step)
    drops = sum(
        1 for i in range(1, len(trajectory))
        if trajectory[i] < trajectory[i - 1]
    )
    drop_pct = 100.0 * drops / (len(trajectory) - 1)

    print(f"  Epsilon        : {round(epsilon*255)}/255")
    print(f"  Steps          : {steps}")
    print(f"  Loss drops     : {drops} / {steps - 1} steps ({drop_pct:.1f}%)")
    print(f"  Loss start     : {trajectory[0]:.4f}")
    print(f"  Loss end       : {trajectory[-1]:.4f}")
    print(f"  Loss max       : {max(trajectory):.4f}")
    print()

    # D4 fires if more than 20% of steps show a loss decrease
    d4_fired = drop_pct > 20.0

    if d4_fired:
        print("  D4 STATUS: FIRED — loss trajectory is non-monotonic.")
        print("  More than 20% of steps show a loss decrease.")
        print("  Interpretation: non-smooth loss surface, attack results")
        print("  may understate true vulnerability.")
    else:
        print("  D4 STATUS: clear — loss increases consistently.")
        print("  Interpretation: loss surface is well-behaved.")

    return d4_fired, trajectory


# ── Plot loss trajectory ───────────────────────────────────────────────────────

def plot_trajectory(trajectory, d4_fired, epsilon=8/255):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    steps = list(range(1, len(trajectory) + 1))

    plt.figure(figsize=(8, 4))
    plt.plot(steps, trajectory, linewidth=1.5, color="#2E75B6")
    plt.title(
        f"PGD Loss Trajectory — epsilon={round(epsilon*255)}/255\n"
        f"D4 status: {'FIRED' if d4_fired else 'clear'}",
        fontsize=12
    )
    plt.xlabel("PGD step", fontsize=11)
    plt.ylabel("Cross-entropy loss", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "model1_d4_loss_trajectory.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Trajectory plot saved to: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # PGD accuracies from week1_baseline — paste your results here
    pgd_accuracies = [92.6, 1.0, 0.0, 0.0, 0.0]

    model = get_model_resnet50().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "model1_standard_resnet50.pth"),
        weights_only=True
    ))
    model.eval()

    _, test_loader = get_loaders()


    # ── Report ────────────────────────────────────────────────────────────────────── 
    summary_lines = []
    summary_lines.append("=" * 55)
    summary_lines.append("  Model 1 — D1 and D4 Diagnostic Checks")
    summary_lines.append("=" * 55)

    d1_fired              = check_d1(model, test_loader, pgd_accuracies)
    d4_fired, trajectory  = check_d4(model, test_loader)
    plot_trajectory(trajectory, d4_fired)


    summary_lines.append("\n── Summary ───────────────────────────────────────────────────")
    summary_lines.append(f"  D1 (gradient masking) : {'FIRED' if d1_fired else 'clear'}")
    summary_lines.append(f"  D4 (loss consistency) : {'FIRED' if d4_fired else 'clear'}")
    summary_lines.append(f"  D5 (narrow robustness): FIRED")
    summary_lines.append("  (D5 confirmed from week1_baseline results)")

    summary = "\n".join(summary_lines)
    print(summary)

    save_report("part1_pgd.txt", summary)


if __name__ == "__main__":
    run()