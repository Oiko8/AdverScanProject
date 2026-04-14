import os
import torch
import torch.nn as nn
import torch.optim as optim
from configs.settings import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, MODEL_DIR
)


def evaluate(model, loader):
    """Returns accuracy on the given loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train(model, train_loader, test_loader, save_name="model1_standard.pth"):
    """
    Trains model for NUM_EPOCHS with:
      - SGD + momentum + weight decay
      - Cosine annealing LR schedule
      - Saves the best checkpoint by test accuracy
    """
    save_path = os.path.join(MODEL_DIR, save_name)
    os.makedirs(MODEL_DIR, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    best_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Training pass ────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_acc  = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)
        test_acc   = evaluate(model, test_loader)

        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            marker = " <-- best"
        else:
            marker = ""

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.1f}% | "
            f"Test acc: {test_acc:.1f}%"
            f"{marker}"
        )

    print(f"\nTraining complete. Best test accuracy: {best_acc:.1f}%")
    print(f"Checkpoint saved to: {save_path}")
    return save_path