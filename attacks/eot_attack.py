import torch
import torch.nn as nn

from configs.settings import DEVICE, PGD_STEPS, PGD_RESTARTS
from attacks.pgd import CIFAR10_MIN, CIFAR10_MAX


# ─────────────────────────────────────────────────────────────────────────────
#   Expectation-Over-Transformations (EOT) PGD attack.
#
#   Reference: Tramèr et al. 2020 §15, "On Adaptive Attacks to Adversarial
#   Example Defenses", which uses EOT to break ME-Net by averaging
#   gradients over many stochastic forward passes per attack step.
#
#   Standard PGD breaks on stochastic preprocessing defenses because the
#   gradient at any single forward pass is noisy — each mask draw points
#   in a different direction in input space. Averaging K independent
#   gradient draws (K=40 in the Tramèr protocol) recovers an estimate of
#   E[∇L] that's stable enough for PGD to make consistent progress.
#
#   What this attack is and isn't:
#     - IS: a strict generalization of pgd_attack — set k_samples=1
#       and you get vanilla PGD.
#     - IS: the right tool for ME-Net, randomized smoothing attacked
#       directly, and any other defense with stochastic inference.
#     - IS NOT: BPDA. If the defense has a non-differentiable component
#       (e.g. quantization), EOT alone is not enough. ME-Net's mask is
#       non-differentiable but the SVD reconstruction is differentiable
#       enough that EOT works. For genuinely non-differentiable cases,
#       BPDA would substitute identity in the backward pass — out of
#       scope here.
# ─────────────────────────────────────────────────────────────────────────────


def eot_pgd_attack(model, images, labels, epsilon,
                   steps=PGD_STEPS,
                   step_size=None,
                   restarts=PGD_RESTARTS,
                   k_samples=40):
    """
    PGD with EOT gradient averaging.

    Args:
        model     : the model under attack (in eval mode). The model's
                    forward pass MUST be stochastic for EOT to matter —
                    otherwise k_samples extra passes are pure waste and
                    you should use pgd_attack instead.
        images    : clean inputs, normalized, shape (N, 3, 32, 32)
        labels    : true labels, shape (N,)
        epsilon   : perturbation budget in NORMALIZED space (use
                    scale_epsilon() on a raw-pixel epsilon before
                    passing in, same convention as pgd_attack)
        steps     : PGD iterations
        step_size : per-step delta size (default epsilon/10)
        restarts  : random restarts; the best per-sample perturbation
                    across restarts is kept
        k_samples : number of stochastic forward passes to average
                    gradients over at each PGD step. Tramèr §15 uses 40.

    Returns:
        best_adv : adversarial examples, one per input, chosen as the
                   restart with the highest cross-entropy loss against
                   the true label (averaged across noise samples for
                   the loss evaluation, to match the attack objective).
    """
    if epsilon == 0:
        return images.clone()

    if step_size is None:
        step_size = epsilon / 10

    criterion = nn.CrossEntropyLoss()
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # Track the best adversarial example per sample across restarts.
    best_adv  = images.clone()
    best_loss = torch.full(
        (images.size(0),), fill_value=-float("inf"), device=DEVICE
    )

    for _ in range(restarts):
        # Random start inside the epsilon ball.
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        delta = torch.clamp(images + delta, CIFAR10_MIN, CIFAR10_MAX) - images
        delta.requires_grad_(True)

        for _ in range(steps):
            # ── EOT gradient average ────────────────────────────
            # Accumulate gradients from k_samples independent
            # stochastic forward passes. Each forward triggers a
            # fresh mask draw inside ME-Net (or fresh noise for any
            # other stochastic defense).
            grad_accum = torch.zeros_like(delta)
            for _ in range(k_samples):
                if delta.grad is not None:
                    delta.grad.zero_()

                outputs = model(images + delta)
                loss    = criterion(outputs, labels)
                loss.backward()
                grad_accum = grad_accum + delta.grad.detach()

            mean_grad = grad_accum / k_samples

            # ── PGD step with the averaged gradient ─────────────
            with torch.no_grad():
                delta.data = delta.data + step_size * mean_grad.sign()
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                delta.data = torch.clamp(
                    images + delta.data, CIFAR10_MIN, CIFAR10_MAX
                ) - images

            # Zero the live grad in preparation for the next step's
            # accumulation. The detached grad_accum is unaffected.
            if delta.grad is not None:
                delta.grad.zero_()

        # ── Restart selection: keep the strongest adversarial ──
        # Loss is evaluated as the mean over k_samples noise draws to
        # match the attack objective (and to give a more stable
        # ranking than a single noisy evaluation would).
        with torch.no_grad():
            adv_images = images + delta
            mean_per_sample_loss = torch.zeros(
                adv_images.size(0), device=DEVICE
            )
            for _ in range(k_samples):
                outputs = model(adv_images)
                mean_per_sample_loss = (
                    mean_per_sample_loss
                    + nn.CrossEntropyLoss(reduction="none")(outputs, labels)
                )
            mean_per_sample_loss = mean_per_sample_loss / k_samples

            improved = mean_per_sample_loss > best_loss
            best_loss = torch.where(improved, mean_per_sample_loss, best_loss)
            best_adv[improved] = adv_images[improved]

    return best_adv


# ─────────────────────────────────────────────────────────────────────────────
#   Robust-accuracy evaluator under EOT-PGD.
#
#   Mirrors evaluate_robust() in part1_baseline.py but uses eot_pgd_attack
#   and evaluates the adversarial under MAJORITY vote across n_eval noise
#   samples — the per-pass eval would be too noisy to trust on a
#   stochastic model.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_eot_robust(model, loader, epsilon,
                        num_samples,
                        steps=PGD_STEPS,
                        restarts=PGD_RESTARTS,
                        k_samples=40,
                        n_eval=20):
    """
    Robust accuracy under EOT-PGD. The model is stochastic, so we
    classify the adversarial example by majority vote across n_eval
    independent forward passes — a single forward could flip the
    prediction purely by mask draw.

    Args:
        model       : stochastic model under attack (eval mode)
        loader      : data loader of normalized images
        epsilon     : NORMALIZED-space epsilon
        num_samples : total images to evaluate
        steps       : EOT-PGD steps
        restarts    : EOT-PGD restarts
        k_samples   : K for the EOT gradient average inside the attack
        n_eval      : forward passes for majority-vote evaluation

    Returns:
        accuracy (%) under EOT-PGD with majority-vote evaluation.
    """
    model.eval()
    correct = 0
    total   = 0

    for images, labels in loader:
        if total >= num_samples:
            break

        remaining = num_samples - total
        images = images[:remaining].to(DEVICE)
        labels = labels[:remaining].to(DEVICE)

        if epsilon == 0:
            adv = images
        else:
            adv = eot_pgd_attack(
                model, images, labels, epsilon,
                steps=steps, restarts=restarts, k_samples=k_samples
            )

        # ── Majority-vote evaluation ─────────────────────────
        # Accumulate per-class vote counts across n_eval passes.
        with torch.no_grad():
            num_classes = 10  # CIFAR-10; pull from settings if generalized
            votes = torch.zeros(
                adv.size(0), num_classes, device=DEVICE, dtype=torch.long
            )
            for _ in range(n_eval):
                outputs = model(adv)
                preds   = outputs.argmax(dim=1)
                votes.scatter_add_(
                    1, preds.unsqueeze(1),
                    torch.ones_like(preds.unsqueeze(1))
                )
            final_preds = votes.argmax(dim=1)
            correct += final_preds.eq(labels).sum().item()
            total   += labels.size(0)

    return 100.0 * correct / total