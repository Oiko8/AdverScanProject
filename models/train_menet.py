import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from models.resnet_model import get_model_resnet50
from models.menet import MENetWrapper
from data.cifar10_loader import get_loaders
from configs.settings import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, MODEL_DIR
)


# ─────────────────────────────────────────────────────────────────────────────
#   Train Model 4 — ResNet-50 with ME-Net preprocessing as augmentation.
#
#   Why a dedicated training script (vs. reusing models/train.py):
#     The base CNN must see ME-Net-processed inputs at every training step
#     so it learns to classify the reconstructed distribution. Just bolting
#     ME-Net onto a pre-trained ResNet-50 at inference time gives ~10%
#     clean accuracy because the CNN was never exposed to the artifacts
#     of the mask+SVD step.
#
#   Implementation detail:
#     We train the full MENetWrapper end-to-end. The wrapper's forward
#     pass already does normalized → [0,1] → ME-Net → normalized → CNN,
#     so training data goes through the exact same pipeline that
#     inference will use. Only the base CNN's weights are updated;
#     ME-Net itself has no trainable parameters.
# ─────────────────────────────────────────────────────────────────────────────


# ── Defaults matching the ME-Net paper's standard config ─────────────────────

DEFAULT_P_KEEP = 0.5
DEFAULT_RANK   = 10


def train_menet(p_keep=DEFAULT_P_KEEP, rank=DEFAULT_RANK,
                save_name=None):
    """
    Train ResNet-50 with ME-Net preprocessing.

    Args:
        p_keep    : mask keep probability (fraction of pixels retained)
        rank      : USVT rank for matrix completion
        save_name : checkpoint filename (auto-generated if None)
    """
    if save_name is None:
        p_tag    = f"p{int(p_keep*100):02d}"
        r_tag    = f"r{rank:02d}"
        save_name = f"model4_menet_base_{p_tag}_{r_tag}_resnet50.pth"

    save_path = os.path.join(MODEL_DIR, save_name)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("  Training Model 4 — ResNet-50 + ME-Net preprocessing")
    print(f"  p_keep    : {p_keep}")
    print(f"  rank      : {rank}")
    print(f"  Save path : {save_path}")
    print("=" * 60)

    # ── Build the full wrapper, train end-to-end ─────────────────────
    base  = get_model_resnet50().to(DEVICE)
    model = MENetWrapper(base, p_keep=p_keep, rank=rank).to(DEVICE)

    # The data loader produces normalized images. MENetWrapper
    # handles the [0,1] round-trip internally — exactly the same as
    # inference, which is the point.
    train_loader, test_loader = get_loaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        # Only train the base classifier's parameters.
        model.base.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    best_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Training pass ─────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

        scheduler.step()

        train_acc  = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        # ── Eval pass ─────────────────────────────────────────────
        # Note: this is "ME-Net-on, fresh mask per batch" — i.e. the
        # actual inference distribution. Numbers will fluctuate a bit
        # between epochs purely from the random mask draw; the
        # trend (not the per-epoch number) is what matters.
        model.eval()
        test_correct = 0
        test_total   = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total   += labels.size(0)
        test_acc = 100.0 * test_correct / test_total

        if test_acc > best_acc:
            best_acc = test_acc
            # Save only the base CNN's state_dict. The ME-Net wrapper
            # is reconstructed at inference time from p_keep + rank.
            torch.save(model.base.state_dict(), save_path)
            marker = " <-- best"
        else:
            marker = ""

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.1f}% | "
            f"ME-Net test acc: {test_acc:.1f}%"
            f"{marker}"
        )

    print(f"\nTraining complete. Best ME-Net test accuracy: {best_acc:.1f}%")
    print(f"Base CNN checkpoint saved to: {save_path}")
    print(f"To load: get_menet_model(ckpt_path='{save_path}', "
          f"p_keep={p_keep}, rank={rank})")
    return save_path


if __name__ == "__main__":
    # CLI: python -m models.train_menet [p_keep] [rank]
    p_keep = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_P_KEEP
    rank   = int(sys.argv[2])   if len(sys.argv) > 2 else DEFAULT_RANK
    train_menet(p_keep=p_keep, rank=rank)