import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# ── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET         = "CIFAR10"
NUM_CLASSES     = 10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── Training (Model 1 — standard ResNet-18) ───────────────────────────────────
BATCH_SIZE  = 128
NUM_EPOCHS  = 30
LEARNING_RATE = 0.1
MOMENTUM    = 0.9
WEIGHT_DECAY = 5e-4

# ── Attack (PGD) ─────────────────────────────────────────────────────────────
# Epsilon values to sweep for the accuracy-vs-epsilon curve
EPSILONS = [0/255, 2/255, 4/255, 8/255, 16/255]

PGD_STEPS    = 100     # number of PGD iterations
PGD_STEP_SIZE = None   # None means auto: epsilon / 10
PGD_RESTARTS = 3       # random restarts per sample

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_BATCH_SIZE  = 128
EVAL_NUM_SAMPLES = 1000   # run attacks on 1000 test images (standard in literature)