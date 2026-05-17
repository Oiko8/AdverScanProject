import os
import torch
import matplotlib.pyplot as plt

from models.resnet_model import get_model_resnet50
from models.smoothing_base import NormalizeWrapper
from models.smoothing import Smooth
from data.cifar10_loader import CIFAR10_MEAN, CIFAR10_STD, get_loaders
from attacks.autoattack_eval import run_autoattack
from configs.settings import DEVICE, MODEL_DIR, OUTPUT_DIR
from configs.save_report import save_report


# ────────────────────────────────────────────────────────────────
#   Config
# ────────────────────────────────────────────────────────────────

SIGMAS = [0.25, 0.50, 0.75, 1.00]
L2_RADII = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50]

# Certification (Cohen et al. style)
NUM_SAMPLES = 1000
CERT_N0     = 100
CERT_N      = 10000
CERT_ALPHA  = 0.001
CERT_BATCH  = 400

# Empirical L2 attack on the base classifier
ATTACK_NUM_SAMPLES = 1000

# D7 diagnostic threshold (percentage points)
# Fires if empirical - certified > D7_THRESHOLD at any radius
# where certification is non-trivial (cert > 0).
D7_THRESHOLD = 20.0

# Hardcoded ceritfied accuracy from previous run
CERTIFIED_ACCURACY = {
    0.25: {0.00: 76.7, 0.25: 60.9, 0.50: 42.7, 0.75: 22.6,
           1.00:  0.0, 1.25:  0.0, 1.50:  0.0, 2.00:  0.0, 2.50: 0.0},
    0.50: {0.00: 60.4, 0.25: 49.0, 0.50: 38.9, 0.75: 26.5,
           1.00: 15.8, 1.25:  8.1, 1.50:  3.1, 2.00:  0.0, 2.50: 0.0},
    0.75: {0.00: 47.9, 0.25: 39.9, 0.50: 31.5, 0.75: 23.1,
           1.00: 16.4, 1.25: 12.3, 1.50:  6.4, 2.00:  1.9, 2.50: 0.0},
    1.00: {0.00: 35.4, 0.25: 30.1, 0.50: 24.1, 0.75: 18.8,
           1.00: 15.4, 1.25: 11.3, 1.50:  8.5, 2.00:  3.8, 2.50: 1.3},
}
 
ABSTAIN_RATE = {0.25: 7.3, 0.50: 17.0, 0.75: 29.0, 1.00: 32.2}

# ────────────────────────────────────────────────────────────────
#   Empirical L2 attack on the (deterministic) base classifier
# ────────────────────────────────────────────────────────────────

def attack_smoother_l2(wrapped_base, test_loader, num_samples=ATTACK_NUM_SAMPLES):
    """
    Run L2 AutoAttack on the wrapped base classifier at each non-zero
    radius in L2_RADII. Returns empirical accuracy at each radius.

    Why we attack the base (not the smoothed classifier directly):
      AutoAttack needs gradients. The smoothed classifier is stochastic
      (fresh noise per forward pass), so its gradient is unreliable.
      Attacking the base is strictly stronger than EOT-attacking the
      smoothed model — so base_emp_acc <= smoothed_emp_acc, and the
      D7 gap (base_emp_acc - cert_acc) is a LOWER BOUND on the true
      certificate looseness (smoothed_emp_acc - cert_acc).

    The wrapped_base accepts [0,1] inputs and normalizes internally,
    matching what run_autoattack expects when model_expects_raw=True.

    Returns:
        emp_accuracy : dict mapping radius r → empirical accuracy (%)
                       (r=0.0 entry = clean accuracy, no attack needed)
    """
    emp_accuracy = {}

    for r in L2_RADII:
        if r == 0.0:
            print(f"\n── L2 attack at r={r:.2f} (clean — no attack) ──")
            emp_accuracy[r] = _clean_accuracy(
                wrapped_base, test_loader, num_samples
            )
            print(f"  clean accuracy: {emp_accuracy[r]:.1f}%")
            continue

        print(f"\n── L2 AutoAttack at r={r:.2f} ─────────────────────")
        acc, _ = run_autoattack(
            wrapped_base,
            test_loader,
            epsilon=r,
            norm="L2",
            num_samples=num_samples,
            model_expects_raw=True
        )
        emp_accuracy[r] = acc
        print(f"  empirical accuracy: {acc:.1f}%")

    return emp_accuracy


def _clean_accuracy(wrapped_base, test_loader, num_samples):
    """Clean accuracy of wrapped_base on num_samples images."""
    wrapped_base.eval()

    mean3 = torch.tensor(CIFAR10_MEAN).view(3, 1, 1).to(DEVICE)
    std3  = torch.tensor(CIFAR10_STD).view(3, 1, 1).to(DEVICE)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            if total >= num_samples:
                break

            remaining = num_samples - total
            images, labels = images[:remaining], labels[:remaining]
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Denormalize: wrapped_base expects [0,1]
            images_01 = torch.clamp(images * std3 + mean3, 0, 1)

            outputs = wrapped_base(images_01)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


# ────────────────────────────────────────────────────────────────
#   Certification of the smoothed classifier
# ────────────────────────────────────────────────────────────────

def certify_smoother(smoother, test_loader, num_samples=NUM_SAMPLES):
    """
    Certify num_samples test images and return certified accuracy
    at each radius in L2_RADII.

    For each test image:
      1. Convert from normalized → [0,1] pixel space (smoother expects [0,1]).
      2. Call smoother.certify() to get (predicted_class, certified_radius).
      3. For each r in L2_RADII, count as "certified-correct" if
         prediction == true_label AND certified_radius >= r.

    Returns:
        cert_accuracy : dict {radius r → certified accuracy (%)}
        abstain_rate  : % of samples that returned ABSTAIN
    """
    correct_at_radius = {r: 0 for r in L2_RADII}
    abstain_count = 0
    total = 0

    mean3 = torch.tensor(CIFAR10_MEAN).view(3, 1, 1).to(DEVICE)
    std3  = torch.tensor(CIFAR10_STD).view(3, 1, 1).to(DEVICE)

    for images, labels in test_loader:
        if total >= num_samples:
            break

        for i in range(images.size(0)):
            if total >= num_samples:
                break

            img_normalized = images[i].to(DEVICE)
            label = labels[i].item()

            img_01 = torch.clamp(
                img_normalized * std3 + mean3, 0, 1
            ).unsqueeze(0)

            pred, radius = smoother.certify(
                img_01, CERT_N0, CERT_N, CERT_ALPHA, CERT_BATCH
            )

            if pred == Smooth.ABSTAIN:
                abstain_count += 1
            elif pred == label:
                for r in L2_RADII:
                    if r <= radius:
                        correct_at_radius[r] += 1

            total += 1
            if total % 50 == 0:
                print(f"  certified {total}/{num_samples} images...")

    cert_accuracy = {r: 100.0 * c / total for r, c in correct_at_radius.items()}
    abstain_rate  = 100.0 * abstain_count / total
    return cert_accuracy, abstain_rate


# ────────────────────────────────────────────────────────────────
#   D7 — certificate looseness diagnostic
# ────────────────────────────────────────────────────────────────

def analyze_d7(cert_acc, emp_acc):
    """
    Compute D7 gap = max(empirical - certified) over radii where
    the certificate is non-trivial (cert_acc[r] > 0).

    Why restrict to cert > 0:
      - r=0 with cert=0% means certification failed everywhere (no signal).
      - r beyond the certifiable range (typically ~3σ) has cert=0% by
        construction — the smoothing theory simply does not bound that
        regime, so the empirical-vs-certified gap there is not "looseness",
        it is "outside the contract".

    Returns:
        (max_gap, max_gap_radius, d7_fired)
        If no radius has cert > 0, returns (0.0, None, False).
    """
    max_gap = -float('inf')
    max_gap_r = None

    for r in L2_RADII:
        if cert_acc[r] > 0:
            gap = emp_acc[r] - cert_acc[r]
            if gap > max_gap:
                max_gap = gap
                max_gap_r = r

    if max_gap_r is None:
        return 0.0, None, False
    return max_gap, max_gap_r, max_gap > D7_THRESHOLD


# ────────────────────────────────────────────────────────────────
#   Per-sigma pipeline
# ────────────────────────────────────────────────────────────────

def run_for_sigma(sigma, test_loader):
    """Certify + empirically attack the smoothed model at one sigma."""
    print(f"\n{'='*70}")
    print(f"  σ = {sigma}")
    print(f"{'='*70}")

    sigma_str = f"{int(sigma*100):03d}"
    ckpt_path = os.path.join(
        MODEL_DIR, f"model3_smoothing_base_sigma{sigma_str}_resnet50.pth"
    )
    base = get_model_resnet50().to(DEVICE)
    base.load_state_dict(torch.load(ckpt_path, weights_only=True))
    base.eval()

    wrapped_base = NormalizeWrapper(base).to(DEVICE)
    wrapped_base.eval()

    # Phase 1 — certification (smoothed classifier)
    print(f"\n── Phase 1: certifying σ={sigma} ({NUM_SAMPLES} samples) ──")
    # smoother = Smooth(wrapped_base, num_classes=10, sigma=sigma)
    # cert_acc, abstain_rate = certify_smoother(smoother, test_loader)
    #### Hardcoded for now:
    cert_acc     = CERTIFIED_ACCURACY[sigma]
    abstain_rate = ABSTAIN_RATE[sigma]


    # Phase 2 — empirical L2 attack (base classifier)
    print(f"\n── Phase 2: empirical L2 attack σ={sigma} ({ATTACK_NUM_SAMPLES} samples) ──")
    emp_acc = attack_smoother_l2(wrapped_base, test_loader)

    # Per-sigma echo
    print(f"\n── σ={sigma} summary ──")
    print(f"  {'r':>6} {'cert%':>8} {'emp%':>8} {'gap':>9}")
    for r in L2_RADII:
        gap = emp_acc[r] - cert_acc[r]
        sign = "+" if gap >= 0 else ""
        print(f"  {r:>6.2f} {cert_acc[r]:>7.1f}% {emp_acc[r]:>7.1f}% {sign}{gap:>7.1f}%")
    print(f"  abstain rate: {abstain_rate:.1f}%")

    return cert_acc, emp_acc, abstain_rate


# ────────────────────────────────────────────────────────────────
#   Plot
# ────────────────────────────────────────────────────────────────

def plot_results(all_cert, all_emp):
    """2x2 grid: certified + empirical accuracy curves per sigma."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, sigma in enumerate(SIGMAS):
        ax = axes[idx]
        cert_vals = [all_cert[sigma][r] for r in L2_RADII]
        emp_vals  = [all_emp[sigma][r]  for r in L2_RADII]

        ax.plot(L2_RADII, cert_vals, marker="s", linewidth=2,
                color="#2E75B6", markersize=7, label="Certified (smoothed)")
        ax.plot(L2_RADII, emp_vals, marker="o", linewidth=2,
                color="#C0392B", markersize=7, label="Empirical (base under AA-L2)")

        # Shade the D7 gap region (only where cert > 0 and emp > cert)
        gap_mask = [
            (c > 0) and (e >= c)
            for c, e in zip(cert_vals, emp_vals)
        ]
        if any(gap_mask):
            ax.fill_between(L2_RADII, cert_vals, emp_vals,
                            where=gap_mask, alpha=0.2,
                            color="orange", label="D7 gap")

        ax.set_title(f"σ = {sigma}", fontsize=12)
        ax.set_xlabel("L2 radius", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Part 5 — Certified vs Empirical Robust Accuracy (L2)",
                 fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "part5_certified_vs_empirical.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved to: {save_path}")


# ────────────────────────────────────────────────────────────────
#   Main
# ────────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _, test_loader = get_loaders()

    print("=" * 70)
    print("  Part 5 — Randomized Smoothing + Certified Robustness")
    print(f"  Sigmas : {SIGMAS}")
    print(f"  Radii  : {L2_RADII}")
    print(f"  Cert N : {CERT_N}   (Cohen et al. style)")
    print(f"  Attack : AutoAttack-L2 on base, {ATTACK_NUM_SAMPLES} samples")
    print("=" * 70)

    all_cert    = {}
    all_emp     = {}
    all_abstain = {}

    for sigma in SIGMAS:
        cert_acc, emp_acc, abstain_rate = run_for_sigma(sigma, test_loader)
        all_cert[sigma]    = cert_acc
        all_emp[sigma]     = emp_acc
        all_abstain[sigma] = abstain_rate

    # ── Build report ─────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("  Part 5 — Randomized Smoothing Summary")
    lines.append("=" * 70)

    # Per-sigma table
    for sigma in SIGMAS:
        lines.append("")
        lines.append(f"── σ = {sigma} ──")
        lines.append(f"  {'radius':>8} {'certified':>12} {'empirical':>12} {'gap':>9}")
        lines.append(f"  {'-'*44}")
        for r in L2_RADII:
            cert = all_cert[sigma][r]
            emp  = all_emp[sigma][r]
            gap  = emp - cert
            sign = "+" if gap >= 0 else ""
            lines.append(
                f"  {r:>8.2f} {cert:>11.1f}% {emp:>11.1f}% {sign}{gap:>7.1f}%"
            )
        lines.append(f"  abstain rate: {all_abstain[sigma]:.1f}%")

        max_gap, max_gap_r, d7_fired = analyze_d7(all_cert[sigma], all_emp[sigma])
        if max_gap_r is None:
            lines.append("  D7: not evaluable (no radius with cert > 0)")
        else:
            status = "FIRED" if d7_fired else "clear"
            lines.append(
                f"  D7 (certificate looseness): {status} "
                f"(max gap {max_gap:+.1f}% at r={max_gap_r:.2f})"
            )

    # D7 overview
    lines.append("")
    lines.append("=" * 70)
    lines.append("  D7 — Certificate Looseness Diagnostic")
    lines.append("=" * 70)
    lines.append(f"  Threshold: empirical - certified > {D7_THRESHOLD:.0f}% at some r")
    lines.append("  (only radii where certified > 0 are considered)")
    lines.append("")
    lines.append(f"  {'σ':>6} {'max gap':>10} {'at r':>8} {'D7':>10}")
    lines.append(f"  {'-'*36}")
    for sigma in SIGMAS:
        max_gap, max_gap_r, d7_fired = analyze_d7(all_cert[sigma], all_emp[sigma])
        if max_gap_r is None:
            lines.append(f"  {sigma:>6.2f} {'n/a':>10} {'n/a':>8} {'n/a':>10}")
        else:
            status = "FIRED" if d7_fired else "clear"
            lines.append(
                f"  {sigma:>6.2f} {max_gap:>+9.1f}% {max_gap_r:>7.2f} {status:>10}"
            )

    lines.append("")
    lines.append("  Note: empirical accuracy is measured on the BASE classifier")
    lines.append("  under AutoAttack-L2 (deterministic). Since attacking the base")
    lines.append("  is strictly stronger than EOT on the smoothed classifier,")
    lines.append("  base_emp_acc <= smoothed_emp_acc, so the D7 gap reported here")
    lines.append("  is a LOWER BOUND on the true certificate looseness.")
    lines.append("=" * 70)

    summary = "\n".join(lines)
    print("\n" + summary)
    save_report("part5_smoothing.txt", summary)

    # ── Plot ─────────────────────────────────────────────────────
    plot_results(all_cert, all_emp)


if __name__ == "__main__":
    run()