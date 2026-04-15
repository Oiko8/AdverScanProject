import io
import sys
import torch
import autoattack
from configs.settings import DEVICE, EVAL_NUM_SAMPLES


def run_autoattack(model, loader, epsilon, num_samples=EVAL_NUM_SAMPLES):
    """
    Runs AutoAttack ensemble on the model at a given epsilon.

    Returns:
        accuracy      : robust accuracy under AutoAttack (%)
        components    : dict with per-component accuracy breakdown
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

    # ── Wrap model to handle [0,1] input ──────────────────────────
    class NormalizedModel(torch.nn.Module):
        def __init__(self, model, mean, std):
            super().__init__()
            self.model = model
            self.mean  = mean
            self.std   = std

        def forward(self, x):
            return self.model((x - self.mean) / self.std)

    wrapped = NormalizedModel(model, mean, std).to(DEVICE)
    wrapped.eval()

    # ── Run AutoAttack, capture verbose output ─────────────────────
    adversary = autoattack.AutoAttack(
        wrapped,
        norm="Linf",
        eps=epsilon,
        version="standard",
        verbose=True
    )

    # Capture stdout so we can parse component results
    captured = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = captured

    adv_images = adversary.run_standard_evaluation(
        images_01, labels, bs=128
    )

    sys.stdout = sys_stdout
    log = captured.getvalue()

    # Print the log so the user still sees it
    print(log)

    # ── Parse per-component accuracies from log ───────────────────
    components = {}
    component_map = {
        "apgd-ce" : "APGD-CE",
        "apgd-t"  : "APGD-DLR",
        "fab-t"   : "FAB",
        "square"  : "Square",
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
        outputs      = wrapped(adv_images)
        _, predicted = outputs.max(1)
        correct      = predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / num_samples
    return accuracy, components