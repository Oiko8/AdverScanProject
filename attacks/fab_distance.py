import torch
import torch.nn.functional as F
from autoattack.fab_pt import FABAttack_PT

from configs.settings import DEVICE
from data.cifar10_loader import CIFAR10_MEAN, CIFAR10_STD
from models.smoothing_base import NormalizeWrapper


def run_fab_distances(model, loader,
                     eps_cap=1.0,
                     num_samples=1000,
                     model_expects_raw=False,
                     n_iter=100,
                     n_restarts=1,
                     batch_size=64):
    """
    Run FAB-L2 (untargeted) on correctly-classified samples and return
    per-sample (distance, confidence, success). This is the raw material
    for the D3 boundary-overconfidence diagnostic — aggregation and
    thresholding happen in part6_d3.py.

    Why untargeted FAB:
      D3 asks "how far is the nearest boundary?" — we want the closest
      adversarial in any direction, not class-conditional minimum
      distances. Untargeted FAB does this in one search instead of nine.

    Why correctly-classified only:
      D3 is a calibration check on the model's confident region.
      Already-misclassified samples have no defined boundary distance —
      the attacker doesn't need to move them anywhere.

    Why L2 (not Linf):
      L2 distances spread out across samples; Linf distances cluster
      near the budget cap, which kills the diagnostic's resolution.

    Args:
        model            : model under test (eval mode set internally)
        loader           : test loader (produces NORMALIZED images)
        eps_cap          : max L2 budget for FAB search. Samples whose
                          true boundary lies beyond this cap come back
                          with success=False and distance ≈ eps_cap.
        num_samples      : target count of correctly-classified samples
        model_expects_raw: True for RobustBench base models that accept
                          [0,1] input directly; False for our standard
                          ResNet-50 trained on normalized input.
        n_iter           : FAB iterations per restart
        n_restarts       : FAB restarts (1 is usually enough for
                          minimum-distortion; more helps escape bad
                          local optima)
        batch_size       : chunk size for FAB.perturb (memory budget)

    Returns dict:
        distances   : Tensor (N,) — L2(adv - clean) in [0,1] pixel space
        confidences : Tensor (N,) — max softmax probability on adv
        success     : BoolTensor (N,) — did FAB flip the prediction
                      away from the true label
        true_labels : Tensor (N,) — original labels for the kept samples
        adv_preds   : Tensor (N,) — model's prediction on the adv example
        clean_acc   : float — clean accuracy measured on the filter pass
                      (informational; the loader is not exhausted past
                      what was needed to reach num_samples)
    """
    # ── Resolve the model the attack runs against ────────────────
    # FAB will feed it [0,1] inputs. RobustBench base models accept
    # that directly; our standard models expect normalized input and
    # need to be wrapped.
    if model_expects_raw:
        eval_model = model
    else:
        eval_model = NormalizeWrapper(model).to(DEVICE)
    eval_model.eval()

    # Buffers for re-normalizing loader output back to [0,1]
    mean = torch.tensor(CIFAR10_MEAN, device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  device=DEVICE).view(1, 3, 1, 1)

    # ── Filter pass: collect correctly-classified samples ───────
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

            # Denormalize loader output → [0,1] for eval_model
            images_01 = torch.clamp(images * std + mean, 0, 1)

            outputs = eval_model(images_01)
            preds = outputs.argmax(dim=1)
            mask = preds.eq(labels)
            seen_total += labels.size(0)

            correct_imgs = images_01[mask]
            correct_lbls = labels[mask]

            # Trim to exactly num_samples on the last batch
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
    clean_acc    = 100.0 * seen_correct / seen_total if seen_total > 0 else 0.0

    print(f"  Clean accuracy on filter pass: {clean_acc:.1f}% "
          f"({seen_correct}/{seen_total})")
    print(f"  Running FAB-L2 on {images_clean.size(0)} correctly-classified "
          f"samples  (eps_cap={eps_cap})")

    # ── Build FAB attacker ──────────────────────────────────────
    # Untargeted, L2, minimum-distortion. Hyperparameters are the
    # AutoAttack defaults — they're tuned for finding tight boundary
    # distances and shouldn't be touched without reason.
    fab = FABAttack_PT(
        predict    = eval_model,
        norm       = 'L2',
        n_restarts = n_restarts,
        n_iter     = n_iter,
        eps        = eps_cap,
        alpha_max  = 0.1,
        eta        = 1.05,
        beta       = 0.9,
        verbose    = False,
        device     = DEVICE,
        seed       = 0,
    )

    # ── Chunked perturb ─────────────────────────────────────────
    # FAB.perturb runs the full batch through every iteration in
    # parallel; chunking keeps peak memory bounded.
    adv_chunks = []
    total = images_clean.size(0)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        x_chunk = images_clean[i:end]
        y_chunk = labels_clean[i:end]
        adv_chunk = fab.perturb(x_chunk, y_chunk)
        adv_chunks.append(adv_chunk)
        print(f"  FAB: {end}/{total} processed")
    adv_images = torch.cat(adv_chunks, dim=0)

    # ── Per-sample distance and confidence ──────────────────────
    with torch.no_grad():
        # L2 distance per sample, flatten spatial+channel dims first
        diffs = (adv_images - images_clean).flatten(start_dim=1)
        distances = diffs.norm(p=2, dim=1)

        # Softmax confidence on the adversarial example
        adv_outputs = eval_model(adv_images)
        probs = F.softmax(adv_outputs, dim=1)
        confidences, adv_preds = probs.max(dim=1)

        # Success = FAB actually moved us off the true class
        success = adv_preds.ne(labels_clean)

    print(f"  FAB success rate: {success.float().mean().item()*100:.1f}% "
          f"(samples where prediction was flipped within eps_cap)")
    print(f"  Mean L2 distance (successful samples only): "
          f"{distances[success].mean().item():.4f}"
          if success.any() else "  No successful attacks within eps_cap.")

    return {
        "distances":   distances.cpu(),
        "confidences": confidences.cpu(),
        "success":     success.cpu(),
        "true_labels": labels_clean.cpu(),
        "adv_preds":   adv_preds.cpu(),
        "clean_acc":   clean_acc,
    }