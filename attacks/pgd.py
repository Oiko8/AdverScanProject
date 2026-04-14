import torch
import torch.nn as nn
from configs.settings import DEVICE, PGD_STEPS, PGD_STEP_SIZE, PGD_RESTARTS

# Valid pixel range after CIFAR-10 normalization
CIFAR10_MIN = -2.4291
CIFAR10_MAX =  2.7537


def pgd_attack(model, images, labels, epsilon, steps=PGD_STEPS,
               step_size=PGD_STEP_SIZE, restarts=PGD_RESTARTS):
    """
    PGD L-inf attack.

    Args:
        model   : the model to attack (in eval mode)
        images  : clean input batch, shape (N, 3, 32, 32), normalized
        labels  : true labels, shape (N,)
        epsilon : perturbation budget in normalized space
        steps   : number of PGD iterations
        step_size: size of each gradient step (defaults to epsilon / 10)
        restarts : number of random restarts

    Returns:
        best_adv : adversarial examples that maximize loss across all restarts
    """
    if epsilon == 0:
        return images.clone()

    if step_size is None:
        step_size = epsilon / 10

    criterion = nn.CrossEntropyLoss()
    images    = images.to(DEVICE)
    labels    = labels.to(DEVICE)

    # Track the best adversarial example per sample (highest loss)
    best_adv  = images.clone()
    best_loss = torch.full((images.size(0),), fill_value=-float("inf"),
                           device=DEVICE)

    for _ in range(restarts):

        # Random start within the epsilon ball
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        # FIX: clamp to normalized range, not [0, 1]
        delta = torch.clamp(images + delta, CIFAR10_MIN, CIFAR10_MAX) - images
        delta.requires_grad_(True)

        for _ in range(steps):
            outputs = model(images + delta)
            loss    = criterion(outputs, labels)
            loss.backward()

            with torch.no_grad():
                # Gradient sign step
                delta.data = delta.data + step_size * delta.grad.sign()
                # Project back into epsilon ball
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                # FIX: clamp to normalized range, not [0, 1]
                delta.data = torch.clamp(images + delta.data,
                                         CIFAR10_MIN, CIFAR10_MAX) - images

            delta.grad.zero_()

        # Evaluate this restart — keep samples with highest loss
        with torch.no_grad():
            adv_images = images + delta
            outputs    = model(adv_images)
            per_sample_loss = nn.CrossEntropyLoss(reduction="none")(outputs, labels)

            improved           = per_sample_loss > best_loss
            best_loss          = torch.where(improved, per_sample_loss, best_loss)
            best_adv[improved] = adv_images[improved]

    return best_adv