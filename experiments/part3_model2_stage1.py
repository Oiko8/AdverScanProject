import os
import torch
import matplotlib.pyplot as plt
from models.robust_resnet_model import get_robust_model
from models.train import evaluate
from attacks.pgd import pgd_attack
from data.cifar10_loader import get_loaders
from configs.save_report import save_report
from configs.settings import (
    DEVICE, OUTPUT_DIR, EPSILONS,
    PGD_STEPS, PGD_RESTARTS, EVAL_NUM_SAMPLES, scale_epsilon
)

# Model 1 PGD results for comparison
MODEL1_PGD = {
    0/255: 92.6,
    2/255: 1.0,
    4/255: 0.0,
    8/255: 0.0,
   16/255: 0.0,
}


def evaluate_robust(model, loader, epsilon, num_samples=EVAL_NUM_SAMPLES):
    """Runs PGD at a given epsilon and returns robust accuracy."""
    model.eval()
    correct = 0
    total   = 0

    for images, labels in loader:
        if total >= num_samples:
            break

        remaining      = num_samples - total
        images, labels = images[:remaining], labels[:remaining]
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if epsilon == 0:
            adv = images
        else:
            adv = pgd_attack(model, images, labels, epsilon,
                             steps=PGD_STEPS, restarts=PGD_RESTARTS)

        with torch.no_grad():
            outputs      = model(adv)
            _, predicted = outputs.max(1)
            correct     += predicted.eq(labels).sum().item()
            total       += labels.size(0)

    return 100.0 * correct / total


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = get_robust_model()
    _, test_loader = get_loaders()

    print("=" * 60)
    print("  Model 2 — Stage 1: PGD Evaluation")
    print(f"  PGD steps: {PGD_STEPS}  |  Restarts: {PGD_RESTARTS}")
    print(f"  Samples  : {EVAL_NUM_SAMPLES}")
    print("=" * 60)

    accuracies = {}

    for eps in EPSILONS:
        # Scale epsilon from raw pixel space to normalized space
        eps_scaled = scale_epsilon(eps) if eps > 0 else 0

        acc = evaluate_robust(model, test_loader, eps_scaled)
        accuracies[eps] = acc
        eps_label = f"{round(eps*255):2d}/255" if eps > 0 else " 0/255"
        print(f"  epsilon = {eps_label}  ->  robust accuracy = {acc:.1f}%")

    # ── D5 check ──────────────────────────────────────────────────
    print("\n── D5: Narrow Robustness Check ──────────────────────────────")
    eps_list = list(EPSILONS)
    acc_list = [accuracies[e] for e in eps_list]
    drops    = [acc_list[i] - acc_list[i+1]
                for i in range(len(acc_list) - 1)]
    max_drop     = max(drops)
    max_drop_idx = drops.index(max_drop)
    eps_a = f"{round(eps_list[max_drop_idx]*255)}/255"
    eps_b = f"{round(eps_list[max_drop_idx+1]*255)}/255"

    print(f"  Largest drop: {max_drop:.1f}% "
          f"between epsilon={eps_a} and epsilon={eps_b}")

    d5_fired = max_drop > 30.0
    if d5_fired:
        print(f"  D5 STATUS: FIRED — narrow robustness detected.")
    else:
        print(f"  D5 STATUS: clear — robustness degrades gradually.")

    # ── Comparison table ──────────────────────────────────────────
    print("\n── Model 1 vs Model 2 — PGD Comparison ─────────────────────")
    print(f"  {'Epsilon':<12} {'Model 1':>10} {'Model 2':>10} {'Delta':>8}")
    print(f"  {'-'*44}")
    for eps in EPSILONS:
        eps_label = f"{round(eps*255)}/255"
        m1        = MODEL1_PGD[eps]
        m2        = accuracies[eps]
        delta     = m2 - m1
        sign      = "+" if delta >= 0 else ""
        print(f"  {eps_label:<12} {m1:>9.1f}% {m2:>9.1f}% {sign}{delta:>6.1f}%")

    # ── Save report ───────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("  Model 2 — Stage 1: PGD Results")
    report_lines.append("=" * 60)
    report_lines.append(f"  {'Epsilon':<12} {'Model 1':>10} {'Model 2':>10} {'Delta':>8}")
    report_lines.append(f"  {'-'*44}")
    for eps in EPSILONS:
        eps_label = f"{round(eps*255)}/255"
        m1        = MODEL1_PGD[eps]
        m2        = accuracies[eps]
        delta     = m2 - m1
        sign      = "+" if delta >= 0 else ""
        report_lines.append(
            f"  {eps_label:<12} {m1:>9.1f}% {m2:>9.1f}% {sign}{delta:>6.1f}%"
        )
    report_lines.append("")
    report_lines.append(
        f"  D5: {'FIRED' if d5_fired else 'clear'} "
        f"(max drop {max_drop:.1f}% between {eps_a} and {eps_b})"
    )
    report_lines.append("=" * 60)
    save_report("part3_model2_stage1.txt", "\n".join(report_lines))

    # ── Plot ──────────────────────────────────────────────────────
    eps_labels = [f"{round(e*255)}/255" for e in EPSILONS]
    m1_vals    = [MODEL1_PGD[e]  for e in EPSILONS]
    m2_vals    = [accuracies[e]  for e in EPSILONS]

    plt.figure(figsize=(8, 5))
    plt.plot(eps_labels, m1_vals, marker="o", linewidth=2,
             color="#C0392B", markersize=7, label="Model 1 — Standard ResNet-50")
    plt.plot(eps_labels, m2_vals, marker="s", linewidth=2,
             color="#2E75B6", markersize=7, label="Model 2 — Adv. Trained ResNet-50")

    for x, y in zip(eps_labels, m2_vals):
        plt.annotate(f"{y:.1f}%", (x, y),
                     textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    plt.title("Model 1 vs Model 2 — PGD Robust Accuracy\n"
              "Standard vs Adversarially Trained (CIFAR-10)", fontsize=12)
    plt.xlabel("Perturbation budget (epsilon)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "model2_vs_model1_pgd.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    run()