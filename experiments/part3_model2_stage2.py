import os
import torch
import matplotlib.pyplot as plt
from models.resnet_model import get_model
from models.robust_resnet_model import get_robust_model
from attacks.autoattack_eval import run_autoattack
from data.cifar10_loader import get_loaders
from configs.settings import (
    DEVICE, MODEL_DIR, OUTPUT_DIR, EVAL_NUM_SAMPLES
)
from configs.save_report import save_report

EVAL_EPSILONS = [2/255, 4/255, 8/255, 16/255]

# PGD results from Week 3 on model 2 for direct comparison
PGD_ACCURACIES = {
    2/255:  82.3,
    4/255:  73.9,
    8/255:  53.6,
    16/255: 14.4,
}


def check_d1_from_components(components, eps_label):
    """
    Secondary D1 check: if Square Attack finds more adversarial
    examples than APGD-CE, that is an independent masking signal.

    In AutoAttack, each component only attacks the images that
    survived the previous component. So a lower accuracy after
    Square means Square found additional failures beyond APGD.
    """
    apgd_acc  = components.get("APGD-CE",  None)
    square_acc = components.get("Square",  None)

    if apgd_acc is None or square_acc is None:
        return False, "insufficient component data"

    # Square ran and reduced accuracy further beyond APGD-CE
    if square_acc < apgd_acc - 2.0:
        return True, (
            f"Square Attack reduced accuracy from "
            f"{apgd_acc:.1f}% to {square_acc:.1f}% "
            f"beyond what APGD-CE found."
        )
    return False, (
        f"APGD-CE dominant. Square added no extra failures "
        f"(APGD-CE: {apgd_acc:.1f}%, Square: {square_acc:.1f}%)."
    )


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = get_robust_model().to(DEVICE)
   
    _, test_loader = get_loaders()

    print("=" * 65)
    print("  Model 2 — Week 3: AutoAttack Evaluation + Diagnostics")
    print(f"  Samples : {EVAL_NUM_SAMPLES}")
    print("=" * 65)

    aa_accuracies  = {}
    all_components = {}
    d1_secondary   = {}

    for eps in EVAL_EPSILONS:
        eps_label = f"{round(eps*255)}/255"
        print(f"\n── AutoAttack at epsilon = {eps_label} ─────────────────────────")

        aa_acc, components = run_autoattack(model, test_loader, eps)
        aa_accuracies[eps]  = aa_acc
        all_components[eps] = components

        # Secondary D1 check from component breakdown
        fired, reason = check_d1_from_components(components, eps_label)
        d1_secondary[eps]   = (fired, reason)

    # ── Stage 1 vs Stage 2 summary ────────────────────────────────
    print("\n" + "=" * 65)
    print("  Stage 1 (PGD) vs Stage 2 (AutoAttack) — Summary")
    print("=" * 65)
    print(f"  {'Epsilon':<12} {'PGD':>10} {'AutoAttack':>12} {'Gap':>8}")
    print(f"  {'-'*46}")

    for eps in EVAL_EPSILONS:
        eps_label = f"{round(eps*255)}/255"
        pgd_acc   = PGD_ACCURACIES[eps]
        aa_acc    = aa_accuracies[eps]
        gap       = pgd_acc - aa_acc
        print(f"  {eps_label:<12} {pgd_acc:>9.1f}% {aa_acc:>11.1f}% {gap:>7.1f}%")

    # ── Component breakdown ───────────────────────────────────────
    print("\n── Component Breakdown ──────────────────────────────────────")
    print(f"  {'Epsilon':<12} {'APGD-CE':>10} {'APGD-DLR':>10}"
          f" {'FAB':>8} {'Square':>10}")
    print(f"  {'-'*54}")

    for eps in EVAL_EPSILONS:
        eps_label = f"{round(eps*255)}/255"
        c         = all_components[eps]
        apgd_ce   = f"{c.get('APGD-CE',  'N/A'):.1f}%" if 'APGD-CE'  in c else "N/A"
        apgd_dlr  = f"{c.get('APGD-DLR', 'N/A'):.1f}%" if 'APGD-DLR' in c else "N/A"
        fab       = f"{c.get('FAB',       'N/A'):.1f}%" if 'FAB'       in c else "N/A"
        square    = f"{c.get('Square',    'N/A'):.1f}%" if 'Square'    in c else "N/A"
        print(f"  {eps_label:<12} {apgd_ce:>10} {apgd_dlr:>10}"
              f" {fab:>8} {square:>10}")

    # ── D1 secondary check ────────────────────────────────────────
    print("\n── D1 Secondary Check (Square vs APGD-CE dominance) ────────")
    any_d1 = False
    for eps in EVAL_EPSILONS:
        eps_label    = f"{round(eps*255)}/255"
        fired, reason = d1_secondary[eps]
        status       = "⚠ D1 signal" if fired else "ok"
        if fired:
            any_d1 = True
        print(f"  epsilon={eps_label}: {status}")
        print(f"    {reason}")

    if any_d1:
        print("\n  D1 SECONDARY: signal detected at one or more epsilons.")
        print("  Square Attack found failures that APGD missed.")
        print("  Possible gradient issues even on this standard model.")
    else:
        print("\n  D1 SECONDARY: clear — APGD-CE dominated throughout.")
        print("  White-box attacks were sufficient. No masking signal.")

    # ── D4 update ─────────────────────────────────────────────────
    print("\n── D4 Update (PGD vs AutoAttack gap) ───────────────────────")
    max_gap     = max(PGD_ACCURACIES[e] - aa_accuracies[e]
                      for e in EVAL_EPSILONS)
    max_gap_eps = max(EVAL_EPSILONS,
                      key=lambda e: PGD_ACCURACIES[e] - aa_accuracies[e])

    print(f"  Largest gap: {max_gap:.1f}% at "
          f"epsilon={round(max_gap_eps*255)}/255")

    if max_gap > 5.0:
        print(f"  ⚠ PGD fixed step size was ineffective.")
        print(f"  Root cause: fixed step_size = epsilon/10 does not")
        print(f"  adapt when the attack stalls. APGD's adaptive")
        print(f"  schedule finds the same adversarial examples much")
        print(f"  more reliably. PGD accuracy numbers should not be")
        print(f"  reported as robustness estimates for this model.")
    else:
        print(f"  PGD and AutoAttack agree. PGD was reliable here.")

    # ── Final diagnostic summary ──────────────────────────────────
# ── Final diagnostic summary ──────────────────────────────────
    summary_lines = []
    summary_lines.append("=" * 65)
    summary_lines.append("  Final Diagnostic Summary — Model 2")
    summary_lines.append("=" * 65)
    summary_lines.append(f"  D1 primary   (FGSM vs PGD)       : clear")
    summary_lines.append(f"  D1 secondary (Square vs APGD)     : "
                         f"{'FIRED' if any_d1 else 'clear'}")
    summary_lines.append(f"  D4 (PGD step size ineffective)    : "
                         f"{'FIRED' if max_gap > 5.0 else 'clear'}")
    summary_lines.append(f"  D5 (narrow robustness)            : FIRED")
    summary_lines.append("")
    summary_lines.append("  AutoAttack ground truth:")
    for eps in EVAL_EPSILONS:
        eps_label = f"{round(eps*255)}/255"
        summary_lines.append(
            f"    epsilon={eps_label} -> {aa_accuracies[eps]:.1f}% robust accuracy"
        )
    summary_lines.append("")
    summary_lines.append("  Component breakdown:")
    summary_lines.append(
        f"  {'Epsilon':<12} {'APGD-CE':>10} {'APGD-DLR':>10}"
        f" {'FAB':>8} {'Square':>10}"
    )
    summary_lines.append(f"  {'-'*54}")
    for eps in EVAL_EPSILONS:
        eps_label = f"{round(eps*255)}/255"
        c         = all_components[eps]
        apgd_ce  = f"{c['APGD-CE']:.1f}%"  if "APGD-CE"  in c else "N/A"
        apgd_dlr = f"{c['APGD-DLR']:.1f}%" if "APGD-DLR" in c else "N/A"
        fab      = f"{c['FAB']:.1f}%"       if "FAB"       in c else "N/A"
        square   = f"{c['Square']:.1f}%"    if "Square"    in c else "N/A"
        summary_lines.append(
            f"  {eps_label:<12} {apgd_ce:>10} {apgd_dlr:>10}"
            f" {fab:>8} {square:>10}"
        )
    summary_lines.append("")
 
    summary_lines.append("=" * 65)

    summary = "\n".join(summary_lines)
    print(summary)

    # Save report
    save_report("part3_autoattack.txt", summary)

    # ── Plot ──────────────────────────────────────────────────────
    eps_labels = [f"{round(e*255)}/255" for e in EVAL_EPSILONS]
    pgd_vals   = [PGD_ACCURACIES[e]  for e in EVAL_EPSILONS]
    aa_vals    = [aa_accuracies[e]   for e in EVAL_EPSILONS]

    x     = list(range(len(eps_labels)))
    width = 0.3

    plt.figure(figsize=(8, 5))
    bars_pgd = plt.bar(
        [i - width/2 for i in x], pgd_vals, width,
        label="PGD (Stage 1)", color="#2E75B6", alpha=0.85
    )
    bars_aa = plt.bar(
        [i + width/2 for i in x], aa_vals, width,
        label="AutoAttack (Stage 2)", color="#C0392B", alpha=0.85
    )

    for i, (p, a) in enumerate(zip(pgd_vals, aa_vals)):
        plt.text(i - width/2, p + 0.8, f"{p:.1f}%",
                 ha="center", fontsize=9)
        plt.text(i + width/2, a + 0.8, f"{a:.1f}%",
                 ha="center", fontsize=9)

    plt.xticks(x, eps_labels)
    plt.title(
        "Model 2 — Stage 1 (PGD) vs Stage 2 (AutoAttack)\n"
        "Robust Accuracy Comparison",
        fontsize=12
    )
    plt.xlabel("Perturbation budget (epsilon)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "model2_stage1_vs_stage2.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved to: {save_path}")
    print("=" * 65)


if __name__ == "__main__":
    run()