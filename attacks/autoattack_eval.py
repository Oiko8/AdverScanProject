import io
import sys
import torch
import autoattack
from configs.settings import DEVICE, EVAL_NUM_SAMPLES


def run_autoattack(model, loader, epsilon, norm = "Linf",
                   num_samples=EVAL_NUM_SAMPLES,
                   model_expects_raw=False):
    """
    Runs AutoAttack ensemble on the model at a given epsilon.

    Args:
        model            : the model to attack
        loader           : data loader (produces normalized images)
        epsilon          : perturbation budget in raw [0,1] space
        num_samples      : number of test images to evaluate
        model_expects_raw: if True, model expects [0,1] input directly
                           (e.g. RobustBench base models).
                           if False, model expects normalized input
                           (e.g. our standard ResNet-18).

    Returns:
        accuracy  : robust accuracy under AutoAttack (%)
        components: dict with per-component accuracy breakdown
    """
    model.eval()

    # ── Collect num_samples images ────────────────────────────────
    all_images = []
    all_labels = []
    total      = 0

    for images, labels in loader:
        remaining = num_samples - total
        all_images.append(images[:remaining])
        all_labels.append(labels[:remaining])
        total += images[:remaining].size(0)
        if total >= num_samples:
            break

    images = torch.cat(all_images, dim=0).to(DEVICE)
    labels = torch.cat(all_labels, dim=0).to(DEVICE)

    # ── Denormalize to [0,1] for AutoAttack ───────────────────────
    mean = torch.tensor([0.4914, 0.4822, 0.4465],
                        device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010],
                        device=DEVICE).view(1, 3, 1, 1)

    images_01 = torch.clamp(images * std + mean, 0, 1)

    # ── Wrap model if it expects normalized input ─────────────────
    if model_expects_raw:
        # Model already expects [0,1] — use directly, no wrapping
        eval_model = model
    else:
        # Model expects normalized input — wrap so AutoAttack can
        # feed it [0,1] and the wrapper normalizes internally
        class NormalizedModel(torch.nn.Module):
            def __init__(self, model, mean, std):
                super().__init__()
                self.model = model
                self.mean  = mean
                self.std   = std

            def forward(self, x):
                return self.model((x - self.mean) / self.std)

        eval_model = NormalizedModel(model, mean, std).to(DEVICE)

    eval_model.eval()

    # ── Run AutoAttack ─────────────────────────────────────────────
    adversary = autoattack.AutoAttack(
        eval_model,
        norm=norm,
        eps=epsilon,
        version="standard",
        verbose=True
    )

    captured   = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = captured

    adv_images = adversary.run_standard_evaluation(
        images_01, labels, bs=128
    )

    sys.stdout = sys_stdout
    log = captured.getvalue()
    print(log)

    # ── Parse per-component accuracies ────────────────────────────
    components    = {}
    component_map = {
        "apgd-ce": "APGD-CE",
        "apgd-t" : "APGD-DLR",
        "fab-t"  : "FAB",
        "square" : "Square",
    }

    for line in log.splitlines():
        line_lower = line.lower()
        for key, name in component_map.items():
            if f"robust accuracy after {key}" in line_lower:
                try:
                    acc = float(line.split(":")[1].strip().split("%")[0])
                    components[name] = acc
                except (IndexError, ValueError):
                    pass

    # ── Final robust accuracy ──────────────────────────────────────
    with torch.no_grad():
        outputs      = eval_model(adv_images)
        _, predicted = outputs.max(1)
        correct      = predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / num_samples
    return accuracy, components