import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.resnet_model import get_model_resnet50
from models.robust_resnet_model import get_robust_model
from data.cifar10_loader import get_loaders
from attacks.transfer_attack import generate_transfer_examples
from configs.settings import DEVICE, MODEL_DIR, OUTPUT_DIR, scale_epsilon
from configs.save_report import save_report


# ────────────────────────────────────────────────────────────────
#   Config
# ────────────────────────────────────────────────────────────────

SMOKE_TEST = False  # flip to False for the full run

NUM_SAMPLES        = 1000
EPSILONS           = [2/255, 4/255, 8/255]
KAPPA              = 20.0
STEPS              = 100
ATTACK_BATCH_SIZE  = 128

# D3 thresholds — pre-data guesses, recalibrate after the first full run
D3_SUCCESS_FLOOR       = 0.30   # need ≥30% success to evaluate (else "not evaluable")
D3_CONFIDENCE_THRESHOLD = 0.8   # mean softmax on successes > this = overconfident


# ────────────────────────────────────────────────────────────────
#   CW attack runner (per model, per epsilon)
# ────────────────────────────────────────────────────────────────

def run_cw_attack(model, loader, epsilon_raw, num_samples,
                  kappa=KAPPA, steps=STEPS, batch_size=ATTACK_BATCH_SIZE):
    """
    Filter to correctly-classified samples, run CW-Linf at the given raw
    epsilon and kappa, return per-sample success + softmax confidence.

    The CW attack reused from attacks/transfer_attack.py operates in
    normalized image space. epsilon_raw is in [0,1] pixel space and we
    scale it by 1/min(std) so the perturbation budget in raw pixel space
    matches the intent. Same convention as the rest of the project.
    """
    model.eval()

    # ── Filter pass: correctly-classified samples ────────────────
    clean_imgs   = []
    clean_lbls   = []
    seen_total   = 0
    seen_correct = 0

    with torch.no_grad():
        for images, labels in loader:
            if seen_correct >= num_samples:
                break

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            mask    = preds.eq(labels)
            seen_total += labels.size(0)

            correct_imgs = images[mask]
            correct_lbls = labels[mask]

            remaining = num_samples - seen_correct
            if correct_imgs.size(0) > remaining:
                correct_imgs = correct_imgs[:remaining]
                correct_lbls = correct_lbls[:remaining]

            if correct_imgs.size(0) > 0:
                clean_imgs.append(correct_imgs)
                clean_lbls.append(correct_lbls)
                seen_correct += correct_imgs.size(0)

    images_clean = torch.cat(clean_imgs, dim=0)
    labels_clean = torch.cat(clean_lbls, dim=0)

    print(f"  Collected {images_clean.size(0)} correctly-classified samples "
          f"(first-pass selection rate: {100*seen_correct/seen_total:.1f}%)")

    # ── Run CW in chunks ─────────────────────────────────────────
    epsilon_scaled = scale_epsilon(epsilon_raw)

    adv_chunks = []
    total = images_clean.size(0)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        x_chunk = images_clean[i:end]
        y_chunk = labels_clean[i:end]
        adv_chunk = generate_transfer_examples(
            model, x_chunk, y_chunk,
            epsilon=epsilon_scaled,
            kappa=kappa,
            steps=steps,
        )
        adv_chunks.append(adv_chunk)
        print(f"  CW: {end}/{total} processed")
    adv_images = torch.cat(adv_chunks, dim=0)

    # ── Evaluate confidence + success ────────────────────────────
    with torch.no_grad():
        adv_outputs = model(adv_images)
        probs = F.softmax(adv_outputs, dim=1)
        confidences, adv_preds = probs.max(dim=1)
        success = adv_preds.ne(labels_clean)

    return {
        "n_total":     labels_clean.size(0),
        "success":     success.cpu(),
        "confidences": confidences.cpu(),
        "adv_preds":   adv_preds.cpu(),
        "true_labels": labels_clean.cpu(),
    }


# ────────────────────────────────────────────────────────────────
#   D3 aggregation
# ────────────────────────────────────────────────────────────────

def analyze_d3(results,
               success_floor=D3_SUCCESS_FLOOR,
               conf_thr=D3_CONFIDENCE_THRESHOLD):
    """
    D3 fires when the model can be made confidently wrong:
      1. Attack success rate >= success_floor (else: not evaluable)
      2. Mean softmax confidence on successful adversarials > conf_thr

    Why the success floor: with very few successes, mean confidence is
    drawn from a handful of outlier samples and is not statistically
    representative. "Not evaluable" is itself a meaningful state — it
    means the model is robust enough at this budget that D3 cannot
    gather the data it needs.
    """
    n_total     = results["n_total"]
    success     = results["success"]
    confidences = results["confidences"]

    n_success    = success.sum().item()
    success_rate = n_success / n_total if n_total > 0 else 0.0

    if success_rate < success_floor:
        return {
            "fired":     False,
            "evaluable": False,
            "reason":   (f"success rate {success_rate*100:.1f}% < "
                         f"{success_floor*100:.0f}% floor"),
            "n_total":   n_total,
            "n_success": n_success,
            "success_rate": success_rate,
            "mean_confidence_success":   None,
            "median_confidence_success": None,
            "n_confidently_wrong":   0,
            "frac_confidently_wrong": 0.0,
        }

    conf_succ        = confidences[success]
    n_confident_wrong = (conf_succ > conf_thr).sum().item()
    mean_conf        = conf_succ.mean().item()
    median_conf      = conf_succ.median().item()

    return {
        "fired":     mean_conf > conf_thr,
        "evaluable": True,
        "n_total":   n_total,
        "n_success": n_success,
        "success_rate":             success_rate,
        "mean_confidence_success":  mean_conf,
        "median_confidence_success": median_conf,
        "n_confidently_wrong":      n_confident_wrong,
        "frac_confidently_wrong":   n_confident_wrong / n_success,
    }


# ────────────────────────────────────────────────────────────────
#   Printing
# ────────────────────────────────────────────────────────────────

def _print_d3_block(title, d3):
    print(f"\n  ── {title} ──")
    print(f"    Samples              : {d3['n_total']}")
    print(f"    Attack successes     : {d3['n_success']} "
          f"({d3['success_rate']*100:.1f}%)")
    if not d3.get("evaluable", False):
        print(f"    D3 not evaluable     : {d3['reason']}")
        return
    print(f"    Mean conf (succ)     : {d3['mean_confidence_success']:.4f}")
    print(f"    Median conf (succ)   : {d3['median_confidence_success']:.4f}")
    print(f"    Confidently wrong    : {d3['n_confidently_wrong']}/{d3['n_success']} "
          f"({d3['frac_confidently_wrong']*100:.1f}%)")
    print(f"    D3 status            : {'FIRED' if d3['fired'] else 'clear'}")


# ────────────────────────────────────────────────────────────────
#   Smoke test
# ────────────────────────────────────────────────────────────────

def run_smoke_test():
    """
    Smoke test: Model 1 only, ε=2/255, n=100.
    Expected: very high success rate (Model 1 has no robustness at 2/255),
    high mean confidence on successful adversarials (Tramèr §6 overconfidence
    claim), D3 should FIRE.
    """
    print("=" * 70)
    print("  D3 SMOKE TEST (CW-based) — Model 1, eps=2/255, kappa=20, n=100")
    print("=" * 70)

    model = get_model_resnet50().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "model1_standard_resnet50.pth"),
        weights_only=True
    ))
    model.eval()

    _, test_loader = get_loaders()

    results = run_cw_attack(
        model, test_loader,
        epsilon_raw = 2/255,
        num_samples = 100,
        kappa       = 20.0,
        steps       = 100,
    )
    d3 = analyze_d3(results)

    _print_d3_block("Smoke test — Model 1 @ eps=2/255", d3)
    print("=" * 70)


# ────────────────────────────────────────────────────────────────
#   Report
# ────────────────────────────────────────────────────────────────

def _report_lines(all_results):
    lines = []
    lines.append("=" * 70)
    lines.append("  Part 6 — D3 Boundary Overconfidence (CW-based)")
    lines.append("=" * 70)
    lines.append(f"  CW Linf, kappa={KAPPA}, steps={STEPS}, n={NUM_SAMPLES}")
    lines.append(f"  D3 thresholds: success_floor={D3_SUCCESS_FLOOR*100:.0f}%, "
                 f"mean conf > {D3_CONFIDENCE_THRESHOLD}")
    lines.append("")
    lines.append(
        f"  {'eps':<8} {'model':<10} {'success%':>10} {'mean_conf':>11} "
        f"{'conf_wrong%':>13} {'D3':>16}"
    )
    lines.append(f"  {'-'*70}")

    for eps in sorted(all_results.keys()):
        eps_label = f"{round(eps*255)}/255"
        for model_name, (_, d3) in all_results[eps].items():
            if not d3.get("evaluable", False):
                lines.append(
                    f"  {eps_label:<8} {model_name:<10} "
                    f"{d3['success_rate']*100:>9.1f}% "
                    f"{'n/a':>11} {'n/a':>13} "
                    f"{'not evaluable':>16}"
                )
            else:
                status = "FIRED" if d3["fired"] else "clear"
                lines.append(
                    f"  {eps_label:<8} {model_name:<10} "
                    f"{d3['success_rate']*100:>9.1f}% "
                    f"{d3['mean_confidence_success']:>11.4f} "
                    f"{d3['frac_confidently_wrong']*100:>12.1f}% "
                    f"{status:>16}"
                )
    lines.append("")
    lines.append("=" * 70)
    return lines


# ────────────────────────────────────────────────────────────────
#   Plot
# ────────────────────────────────────────────────────────────────

def plot_d3(all_results, save_path):
    """
    Grid: rows = epsilons, cols = Model 1 / Model 2.
    Each cell: confidence histogram on successful adversarials, with
    D3 threshold + mean lines overlaid.
    """
    n_eps = len(all_results)
    fig, axes = plt.subplots(n_eps, 2, figsize=(12, 4 * n_eps),
                             squeeze=False)

    sorted_eps = sorted(all_results.keys())

    for row, eps in enumerate(sorted_eps):
        eps_label = f"{round(eps*255)}/255"
        for col, (model_name, (results, d3)) in enumerate(
            all_results[eps].items()
        ):
            ax = axes[row, col]
            success     = results["success"]
            confidences = results["confidences"]

            if success.sum().item() == 0:
                ax.text(0.5, 0.5, "No successful adversarials",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{model_name} @ eps={eps_label}")
                continue

            conf_succ = confidences[success].numpy()

            ax.hist(conf_succ, bins=30, range=(0, 1),
                    color="#2E75B6", alpha=0.8, edgecolor="white")
            ax.axvline(D3_CONFIDENCE_THRESHOLD, color="red", linestyle="--",
                       label=f"D3 thr. = {D3_CONFIDENCE_THRESHOLD}")

            status = ""
            if d3.get("evaluable", False):
                status = " — FIRED" if d3["fired"] else " — clear"
                ax.axvline(d3["mean_confidence_success"], color="green",
                           linestyle=":",
                           label=f"mean = {d3['mean_confidence_success']:.3f}")
            else:
                status = " — not evaluable"

            ax.set_title(f"{model_name} @ eps={eps_label}{status}")
            ax.set_xlabel("Softmax confidence on successful adversarial")
            ax.set_ylabel("Count")
            ax.set_xlim(0, 1)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle("D3 — Boundary Overconfidence (CW)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved to: {save_path}")


# ────────────────────────────────────────────────────────────────
#   Full run
# ────────────────────────────────────────────────────────────────

def run_full():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _, test_loader = get_loaders()

    eps_labels = [f"{round(e*255)}/255" for e in EPSILONS]
    print("=" * 70)
    print(f"  Part 6 — D3 Boundary Overconfidence (CW-based)")
    print(f"  Samples : {NUM_SAMPLES}")
    print(f"  Kappa   : {KAPPA}")
    print(f"  Steps   : {STEPS}")
    print(f"  Epsilons: {eps_labels}")
    print(f"  D3 thresholds: success_floor={D3_SUCCESS_FLOOR*100:.0f}%, "
          f"mean conf > {D3_CONFIDENCE_THRESHOLD}")
    print("=" * 70)

    # Load models once and reuse across epsilons
    model1 = get_model_resnet50().to(DEVICE)
    model1.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "model1_standard_resnet50.pth"),
        weights_only=True
    ))
    model1.eval()

    model2 = get_robust_model()
    model2.eval()

    all_results = {}

    for eps in EPSILONS:
        eps_label = f"{round(eps*255)}/255"
        print(f"\n{'='*70}")
        print(f"  Epsilon = {eps_label}")
        print(f"{'='*70}")

        print(f"\n── Model 1 ──")
        r1   = run_cw_attack(model1, test_loader, eps, NUM_SAMPLES)
        d3_1 = analyze_d3(r1)
        _print_d3_block(f"Model 1 @ eps={eps_label}", d3_1)

        print(f"\n── Model 2 ──")
        r2   = run_cw_attack(model2, test_loader, eps, NUM_SAMPLES)
        d3_2 = analyze_d3(r2)
        _print_d3_block(f"Model 2 @ eps={eps_label}", d3_2)

        all_results[eps] = {
            "Model 1": (r1, d3_1),
            "Model 2": (r2, d3_2),
        }

    # ── Final summary ────────────────────────────────────────────
    summary = "\n".join(_report_lines(all_results))
    print("\n" + summary)
    save_report("part6_d3.txt", summary)

    # ── Plot ─────────────────────────────────────────────────────
    plot_path = os.path.join(OUTPUT_DIR, "part6_d3.png")
    plot_d3(all_results, plot_path)


# ────────────────────────────────────────────────────────────────
#   Entrypoint
# ────────────────────────────────────────────────────────────────

def run():
    if SMOKE_TEST:
        run_smoke_test()
    else:
        run_full()


if __name__ == "__main__":
    run()