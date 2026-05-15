import torch
from models.resnet_model import get_model_resnet50
from models.smoothing_base import NormalizeWrapper
from models.smoothing import Smooth
from configs.settings import DEVICE
from data.cifar10_loader import CIFAR10_MEAN, CIFAR10_STD
from data.cifar10_loader import get_loaders
import os
from configs.settings import MODEL_DIR
from attacks.autoattack_eval import run_autoattack

# r = σ · Φ⁻¹(p_A_lower)

# ------ sigmas to evaluate ----------------
SIGMAS = [0.25, 0.5, 0.75, 1.00]

# ------ L2 radii to evaluate --------------
L2_RADII = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.5]

# ------- Hyperparameters (Cohen et. al style) --------
NUM_SAMPLES   = 1000     # number of test images to evaluate
CERT_N0       = 100      # selection phase Monte Carlo samples
CERT_N        = 10000    # estimation phase Monte Carlo samples
CERT_ALPHA    = 0.001    # certification failure probability
CERT_BATCH    = 400      # forward-pass batch size during MC sampling


# ------ Empirical L2 attack hyperparameters ------------------------
ATTACK_NUM_SAMPLES = 1000
ATTACK_BATCH = 128

# ── D7 diagnostic threshold ───────────────────────────────────
# Fires if empirical - certified > D7_THRESHOLD at any radius within the certifiable range.
# Calibration: pending after the first tests.
D7_THRESHOLD = 20.0

def attack_smoother_l2(wrapped_base, test_loader, num_samples=ATTACK_NUM_SAMPLES):
    """
    Run L2 AutoAttack on the wrapped base classifier at each non-zero
    radius in L2_RADII. Returns empirical accuracy at each radius.

    Why we attack the base (not the smoothed classifier directly):
      AutoAttack needs gradients. The smoothed classifier is stochastic
      (fresh noise per forward pass), so its gradient is unreliable.
      Attacking the base gives an UPPER BOUND on smoothed robustness —
      strictly stronger than attacking the smoothed model with EOT would.

    The wrapped_base accepts [0,1] inputs and normalizes internally,
    matching what run_autoattack expects when model_expects_raw=True.

    Returns:
        emp_accuracy : dict mapping radius r → empirical accuracy (%)
                       (r=0.0 entry = clean accuracy, no attack needed)
    """

    emp_accuracy = {}

    for r in L2_RADII:
        if r == 0.0:
            # Clean accuracy — no attack
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
    """
    Helper: clean accuracy of the wrapped base on num_samples images.
    Used as the r=0.0 entry of the empirical curve.
    """
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




def certify_smoother(smoother, test_loader, num_samples=NUM_SAMPLES):
    """
    Certify num_samples test images and return certified accuracy at each radius in L2_RADII.

    For each test image:
      1. Convert from normalized → [0,1] pixel space (smoother expects [0,1]).
      2. Call smoother.certify() to get (predicted_class, certified_radius).
      3. For each r in L2_RADII, count as "certified-correct" if
         prediction == true_label AND certified_radius >= r.

    Returns:
        cert_accuracy : dict mapping radius r → certified accuracy (%)
        abstain_rate  : % of samples that returned ABSTAIN (no certification possible)
    """

    # Counters for the correctly classify images that are certified at radius >=r
    correct_at_radius = { r:0 for r in L2_RADII}
    abstain_count = 0
    total = 0

    # CIFAR-10 normalization tensors for the conversion
    mean3 = torch.tensor(CIFAR10_MEAN).view(3, 1, 1).to(DEVICE)
    std3  = torch.tensor(CIFAR10_STD).view(3, 1, 1).to(DEVICE)

    for images, labels in test_loader:
        if total >= num_samples:  # per batch check
            break

        for i in range(images.size(0)):
            if total >= num_samples:   # total number of images check
                break

            img_normalized = images[i].to(DEVICE)
            label = labels[i].item()
            
            # Undo normalization → [0,1] space
            img_01 = torch.clamp(
                img_normalized * std3 + mean3, 0, 1
            ).unsqueeze(0)


            # certify
            pred, radius = smoother.certify(
                img_01,
                CERT_N0,
                CERT_N,
                CERT_ALPHA,
                CERT_BATCH
            )

            if pred == Smooth.ABSTAIN:
                abstain_count += 1
            elif pred == label:
                for r in L2_RADII:
                    if r <= radius:
                        correct_at_radius[r] += 1

            total += 1
            # Progress indicator — certification is slow
            if total % 50 == 0:
                print(f"  certified {total}/{num_samples} images...")

    # Convert counts → percentages
    cert_accuracy = {r: 100.0 * c / total for r, c in correct_at_radius.items()}
    abstain_rate  = 100.0 * abstain_count / total
    
    return cert_accuracy, abstain_rate



def run():

    _, test_loader = get_loaders()

     # ── SMOKE TEST: σ=0.25, 3 radii, 200 images ──────────────
    sigma = 0.25
    sigma_str = f"{int(sigma*100):03d}"
    ckpt_path = os.path.join(
        MODEL_DIR, f"model3_smoothing_base_sigma{sigma_str}_resnet50.pth"
    )
    base = get_model_resnet50().to(DEVICE)
    base.load_state_dict(torch.load(ckpt_path, weights_only=True))
    base.eval()
    wrapped_base = NormalizeWrapper(base).to(DEVICE)
    wrapped_base.eval()

    # Temporarily restrict radii and samples for smoke test
    original_radii = L2_RADII.copy()
    L2_RADII[:] = [0.0, 0.25, 0.50, 1.00]   # in-place modify for the function

    emp_acc = attack_smoother_l2(wrapped_base, test_loader, num_samples=200)

    L2_RADII[:] = original_radii  # restore

    print("\n── L2 attack smoke test (σ=0.25, n=200) ──")
    for r, acc in emp_acc.items():
        cert = "n/a"  # we don't have certified for these specific images
        print(f"  r={r:.2f}  →  empirical {acc:.1f}%")

    return  # stop here for smoke test
    print(f"\n{'='*60}")
    print(f"  Part 5 — sigma = {SIGMAS}")
    print(f"{'='*60}")

    for sigma in SIGMAS:
        base = get_model_resnet50().to(DEVICE)
        sigma_str = f"{int(sigma*100):03d}" 
        ckpt_path = os.path.join(
            MODEL_DIR, f"model3_smoothing_base_sigma{sigma_str}_resnet50.pth"
        )
        base.load_state_dict(torch.load(ckpt_path, weights_only = True))
        base.eval()

        wrapped_base = NormalizeWrapper(base).to(DEVICE)
        wrapped_base.eval()

        smoother = Smooth(wrapped_base, num_classes=10, sigma=sigma)

        cert_acc, abstain_rate = certify_smoother(smoother, test_loader)


        print(f"\n── σ={sigma} (n={NUM_SAMPLES}) ────────")
        for r in L2_RADII:
            print(f"  r={r:.2f}  →  {cert_acc[r]:.1f}% certified-correct")
        print(f"  abstain rate: {abstain_rate:.1f}%")



if __name__ == "__main__":
    run()
