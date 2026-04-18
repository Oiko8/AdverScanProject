import torch
import torch.nn as nn
from configs.settings import DEVICE
from attacks.pgd import CIFAR10_MIN, CIFAR10_MAX


def cw_loss(outputs, labels, kappa=20.0):
    """
    Carlini-Wagner margin loss for gradient ascent.

    Returns a loss that when MAXIMIZED pushes the model toward
    high-confidence misclassification.

    Loss = wrong_logit - correct_logit
    Maximizing this increases the best wrong class score
    relative to the correct class score.
    kappa ensures we keep pushing even after initial misclassification.
    """
    batch_size  = outputs.size(0)
    num_classes = outputs.size(1)

    correct_logits = outputs[torch.arange(batch_size), labels]

    mask = torch.ones_like(outputs, dtype=torch.bool)
    mask[torch.arange(batch_size), labels] = False
    wrong_logits = outputs[mask].view(batch_size, num_classes - 1).max(dim=1).values

    # Maximize: wrong - correct + kappa
    # Positive when wrong < correct + kappa (not yet confident enough)
    # Gradient ascent on this pushes wrong up and correct down
    loss = torch.clamp(wrong_logits - correct_logits + kappa, min=0)
    return loss.mean()


def generate_transfer_examples(surrogate, images, labels,
                                epsilon, kappa=20.0,
                                steps=100, step_size=None):
    """
    Generates high-confidence adversarial examples on the surrogate
    model using the CW margin loss.

    High confidence (kappa=20) improves transferability to other
    models as shown in Carlini & Wagner Section VIII-D.

    Args:
        surrogate  : surrogate model (normalized input)
        images     : clean images, normalized, shape (N, 3, 32, 32)
        labels     : true labels, shape (N,)
        epsilon    : perturbation budget (already scaled for norm. space)
        kappa      : confidence margin for CW loss
        steps      : PGD iterations
        step_size  : step size (default epsilon/10)

    Returns:
        adv_images : high-confidence adversarial examples
    """
    if step_size is None:
        step_size = epsilon / 10

    surrogate.eval()
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, CIFAR10_MIN, CIFAR10_MAX) - images
    delta.requires_grad_(True)

    for _ in range(steps):
        outputs = surrogate(images + delta)
        loss    = cw_loss(outputs, labels, kappa=kappa)
        loss.backward()

        with torch.no_grad():
            delta.data = delta.data + step_size * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(
                images + delta.data, CIFAR10_MIN, CIFAR10_MAX
            ) - images

        delta.grad.zero_()

    return (images + delta).detach()


def evaluate_transfer(target, adv_images, labels):
    """
    Evaluates transfer success rate of adversarial examples
    on the target model.

    Returns:
        transfer_success_rate : % of adversarial examples that
                                fool the target model
        transfer_accuracy     : % of images target classifies correctly
    """
    target.eval()
    adv_images = adv_images.to(DEVICE)
    labels     = labels.to(DEVICE)

    with torch.no_grad():
        outputs      = target(adv_images)
        _, predicted = outputs.max(1)
        correct      = predicted.eq(labels).sum().item()
        total        = labels.size(0)

    accuracy             = 100.0 * correct / total
    transfer_success_rate = 100.0 - accuracy
    return transfer_success_rate, accuracy