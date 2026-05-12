import os
import torch
import matplotlib.pyplot as plt
from models.resnet_model import get_model, get_model_resnet50
from models.robust_resnet_model import get_robust_model
from attacks.transfer_attack import generate_transfer_examples, evaluate_transfer
from data.cifar10_loader import get_loaders
from configs.settings import (
    DEVICE, MODEL_DIR, OUTPUT_DIR,
    EPSILONS, EVAL_NUM_SAMPLES, scale_epsilon
)
from configs.save_report import save_report


# PGD accuracies for Model 2 from Stage 1 — needed for D2 check
MODEL2_PGD = {
    0/255:  88.7,
    2/255:  82.2,
    4/255:  73.8,
    8/255:  53.7,
    16/255: 14.4,
}


def run_transfer_evaluation(surrogate, target, loader,
                             epsilon, num_samples=EVAL_NUM_SAMPLES):
    """
    Generates adversarial examples on surrogate and evaluates
    transfer to target. Returns transfer success rate and accuracy.
    """
    surrogate.eval()
    target.eval()

    all_adv    = []
    all_labels = []
    total      = 0

    for images, labels in loader:
        if total >= num_samples:
            break

        remaining      = num_samples - total
        images, labels = images[:remaining], labels[:remaining]
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        eps_scaled = scale_epsilon(epsilon)

        # Generate high-confidence adversarial examples on surrogate
        adv = generate_transfer_examples(
            surrogate, images, labels,
            epsilon=eps_scaled,
            kappa=20.0,
            steps=100
        )

        all_adv.append(adv)
        all_labels.append(labels)
        total += labels.size(0)

    all_adv    = torch.cat(all_adv,    dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Evaluate on surrogate first — sanity check
    surrogate_success, surrogate_acc = evaluate_transfer(
        surrogate, all_adv, all_labels
    )

    # Evaluate on target — the actual transfer measurement
    target_success, target_acc = evaluate_transfer(
        target, all_adv, all_labels
    )

    return surrogate_success, surrogate_acc, target_success, target_acc


def check_d2(pgd_accuracy, transfer_success_rate, epsilon):
    """
    D2 fires when:
      PGD robust accuracy > 60%
      AND transfer success rate > 40%

    This combination means the model looks robust to direct
    gradient attacks but is still exploitable via a surrogate.
    """
    d2_fired = pgd_accuracy > 60.0 and transfer_success_rate > 40.0
    return d2_fired

def run_kappa_sweep(surrogate, target, loader,
                    epsilon=8/255,
                    kappas=[5, 10, 20, 30, 40],
                    num_samples=500):
    """
    Sweeps kappa values at a fixed epsilon to show how
    attack confidence affects transferability.
    Reproduces the Carlini & Wagner Section VIII-D finding.
    """
    from attacks.transfer_attack import (
        generate_transfer_examples, evaluate_transfer
    )

    surrogate.eval()
    target.eval()

    # Collect images once
    all_images = []
    all_labels = []
    total      = 0

    for images, labels in loader:
        if total >= num_samples:
            break
        remaining      = num_samples - total
        all_images.append(images[:remaining])
        all_labels.append(labels[:remaining])
        total += images[:remaining].size(0)

    images = torch.cat(all_images, dim=0).to(DEVICE)
    labels = torch.cat(all_labels, dim=0).to(DEVICE)

    eps_scaled = scale_epsilon(epsilon)
    eps_label  = f"{round(epsilon*255)}/255"

    print(f"\n── Kappa Sweep at epsilon = {eps_label} ─────────────────────")
    print(f"  {'Kappa':>8} {'Surr. fool%':>13} {'Transfer%':>11} {'Interpretation':>20}")
    print(f"  {'-'*58}")

    results = {}

    for kappa in kappas:
        adv = generate_transfer_examples(
            surrogate, images, labels,
            epsilon=eps_scaled,
            kappa=kappa,
            steps=100
        )

        s_succ, s_acc = evaluate_transfer(surrogate, adv, labels)
        t_succ, t_acc = evaluate_transfer(target,    adv, labels)

        results[kappa] = {
            "surrogate_success" : s_succ,
            "transfer_success"  : t_succ,
        }

        if kappa == 0:
            interp = "barely adversarial"
        elif kappa <= 10:
            interp = "moderate confidence"
        elif kappa <= 20:
            interp = "high confidence"
        else:
            interp = "very high confidence"

        print(f"  {kappa:>8} {s_succ:>12.1f}% {t_succ:>10.1f}%"
              f" {interp:>20}")

    # ── Plot ──────────────────────────────────────────────────────
    kappa_vals  = list(results.keys())
    surr_vals   = [results[k]["surrogate_success"] for k in kappa_vals]
    trans_vals  = [results[k]["transfer_success"]  for k in kappa_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(kappa_vals, surr_vals, marker="o", linewidth=2,
             color="#C0392B", markersize=8,
             label="Surrogate fool rate (Model 1)")
    plt.plot(kappa_vals, trans_vals, marker="s", linewidth=2,
             color="#2E75B6", markersize=8,
             label="Transfer success (→ Model 2)")

    for k, t in zip(kappa_vals, trans_vals):
        plt.annotate(f"{t:.1f}%", (k, t),
                     textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    plt.axvline(x=20, color="gray", linestyle="--",
                alpha=0.5, label="kappa=20 (our default)")
    plt.title(
        f"Transfer Attack — Kappa Sweep at epsilon={eps_label}\n"
        f"Carlini & Wagner confidence vs. transferability",
        fontsize=12
    )
    plt.xlabel("Kappa (confidence margin)", fontsize=11)
    plt.ylabel("Attack success rate (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "part4_kappa_sweep.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved to: {save_path}")

    # ── Save to report ─────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"  Kappa Sweep — epsilon={eps_label}")
    report_lines.append(f"  Reproduces Carlini & Wagner Section VIII-D")
    report_lines.append("=" * 60)
    report_lines.append(
        f"  {'Kappa':>8} {'Surr. fool%':>13} {'Transfer%':>11}"
    )
    report_lines.append(f"  {'-'*36}")
    for k in kappa_vals:
        r = results[k]
        report_lines.append(
            f"  {k:>8} {r['surrogate_success']:>12.1f}%"
            f" {r['transfer_success']:>10.1f}%"
        )
    report_lines.append("=" * 60)
    save_report("part4_kappa_sweep.txt", "\n".join(report_lines))

    return results

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load surrogate (Model 1 — standard ResNet-50)
    surrogate = get_model_resnet50().to(DEVICE)
    surrogate.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "model1_standard_resnet50.pth"),
        weights_only=True
    ))
    surrogate.eval()

    # Load target (Model 2 — adversarially trained ResNet-50)
    target_wrapped = get_robust_model().to(DEVICE)
    target_wrapped.eval()

    _, test_loader = get_loaders()

    print("=" * 65)
    print("  Stage 4 — Transfer Attack Evaluation")
    print("  Surrogate : Model 1 (standard ResNet-50)")
    print("  Target    : Model 2 (adv. trained ResNet-50)")
    print(f"  Samples   : {EVAL_NUM_SAMPLES}")
    print(f"  Kappa     : 20.0 (high-confidence CW loss)")
    print("=" * 65)

    results  = {}
    d2_flags = {}

    for eps in EPSILONS:
        if eps == 0:
            continue

        eps_label = f"{round(eps*255)}/255"
        print(f"\n── Transfer at epsilon = {eps_label} ────────────────────────")

        s_succ, s_acc, t_succ, t_acc = run_transfer_evaluation(
            surrogate, target_wrapped, test_loader, eps
        )

        results[eps] = {
            "surrogate_success" : s_succ,
            "surrogate_acc"     : s_acc,
            "transfer_success"  : t_succ,
            "transfer_acc"      : t_acc,
        }

        pgd_acc  = MODEL2_PGD[eps]
        d2_fired = check_d2(pgd_acc, t_succ, eps)
        d2_flags[eps] = d2_fired

        print(f"  Surrogate (Model 1) fooled : {s_succ:.1f}%  "
              f"(sanity check — should be high)")
        print(f"  Transfer to Model 2        : {t_succ:.1f}% success  "
              f"/ {t_acc:.1f}% accuracy")
        print(f"  Model 2 PGD accuracy       : {pgd_acc:.1f}%")
        print(f"  D2 status                  : "
              f"{'⚠ FIRED' if d2_fired else 'clear'}")

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Stage 4 — Transfer Attack Summary")
    print("=" * 65)
    print(f"  {'Epsilon':<10} {'Surr. fool%':>12} {'Transfer%':>11} "
          f"{'PGD acc':>9} {'D2':>8}")
    print(f"  {'-'*54}")

    for eps in EPSILONS:
        if eps == 0:
            continue
        eps_label = f"{round(eps*255)}/255"
        r         = results[eps]
        d2        = "⚠ FIRED" if d2_flags[eps] else "clear"
        print(f"  {eps_label:<10} {r['surrogate_success']:>11.1f}% "
              f"{r['transfer_success']:>10.1f}% "
              f"{MODEL2_PGD[eps]:>8.1f}% {d2:>8}")

    # ── D2 overall ────────────────────────────────────────────────
    print("\n── D2 Diagnostic — Transfer Vulnerability ───────────────────")
    any_d2 = any(d2_flags.values())

    if any_d2:
        print("  D2 STATUS: FIRED")
        print("  Model 2 PGD accuracy is high BUT transfer success")
        print("  rate is also high. The model's direct gradient is")
        print("  not fully representative of its decision boundary.")
        print("  Recommendation: ensemble adversarial training with")
        print("  diverse surrogate models.")
    else:
        print("  D2 STATUS: clear")
        print("  Transfer success rate is low. Model 2's robustness")
        print("  holds even against surrogate-based attacks.")
        print("  Decision boundary hardening is genuine.")

    # ── Key insight ───────────────────────────────────────────────
    eps_8   = 8/255
    t_succ_8 = results[eps_8]["transfer_success"]
    print(f"\n── Key finding at epsilon=8/255 ────────────────────────────")
    print(f"  Transfer success rate : {t_succ_8:.1f}%")
    print(f"  PGD robust accuracy   : {MODEL2_PGD[eps_8]:.1f}%")
    print(f"  AutoAttack accuracy   : 50.9%")
    if t_succ_8 < 40.0:
        print(f"  Interpretation: adversarial training generalizes well.")
        print(f"  The decision boundary is genuinely hardened, not just")
        print(f"  resistant to gradient-based attacks specifically.")
    else:
        print(f"  Interpretation: some boundary vulnerability remains.")
        print(f"  Transfer examples exploit regions that PGD misses.")

    # ── Save report ───────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append("  Stage 4 — Transfer Attack Report")
    report_lines.append("  Surrogate: Model 1 (standard ResNet-50)")
    report_lines.append("  Target   : Model 2 (adv. trained ResNet-50)")
    report_lines.append("=" * 65)
    report_lines.append(
        f"  {'Epsilon':<10} {'Surr. fool%':>12} {'Transfer%':>11} "
        f"{'PGD acc':>9} {'D2':>8}"
    )
    report_lines.append(f"  {'-'*54}")
    for eps in EPSILONS:
        if eps == 0:
            continue
        eps_label = f"{round(eps*255)}/255"
        r         = results[eps]
        d2        = "FIRED" if d2_flags[eps] else "clear"
        report_lines.append(
            f"  {eps_label:<10} {r['surrogate_success']:>11.1f}% "
            f"{r['transfer_success']:>10.1f}% "
            f"{MODEL2_PGD[eps]:>8.1f}% {d2:>8}"
        )
    report_lines.append("")
    report_lines.append(
        f"  D2 overall: {'FIRED' if any_d2 else 'clear'}"
    )
    report_lines.append("=" * 65)
    save_report("part4_transfer.txt", "\n".join(report_lines))


    # ── Kappa sweep ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Kappa Sweep — Confidence vs. Transferability")
    print("  Fixed epsilon=8/255, varying kappa")
    print("=" * 65)

    run_kappa_sweep(
        surrogate, target_wrapped, test_loader,
        epsilon  = 8/255,
        kappas   = [5, 10, 20, 30, 40],
        num_samples = 500
    )

    # ── Plot ──────────────────────────────────────────────────────
    eps_list    = [e for e in EPSILONS if e > 0]
    eps_labels  = [f"{round(e*255)}/255" for e in eps_list]
    surr_vals   = [results[e]["surrogate_success"] for e in eps_list]
    trans_vals  = [results[e]["transfer_success"]  for e in eps_list]
    pgd_vals    = [100 - MODEL2_PGD[e]             for e in eps_list]

    plt.figure(figsize=(9, 5))
    plt.plot(eps_labels, surr_vals, marker="o", linewidth=2,
             color="#C0392B", markersize=7,
             label="Surrogate fool rate (Model 1)")
    plt.plot(eps_labels, trans_vals, marker="s", linewidth=2,
             color="#E67E22", markersize=7,
             label="Transfer success (→ Model 2)")
    plt.plot(eps_labels, pgd_vals, marker="^", linewidth=2,
             color="#2E75B6", markersize=7, linestyle="--",
             label="PGD attack rate on Model 2")

    plt.title("Stage 4 — Transfer Attack\n"
              "Model 1 (surrogate) → Model 2 (target)", fontsize=12)
    plt.xlabel("Perturbation budget (epsilon)", fontsize=11)
    plt.ylabel("Attack success rate (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "part4_transfer.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved to: {save_path}")
    print("=" * 65)


if __name__ == "__main__":
    run()