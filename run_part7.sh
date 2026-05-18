#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#   Train Model 4 (ME-Net base ResNet-50) and run Part 7 evaluation.
#
#   Usage:
#     ./run_part7.sh                  # defaults: p_keep=0.50, rank=10
#     ./run_part7.sh 0.40             # p_keep=0.40, rank=10
#     ./run_part7.sh 0.50 8           # p_keep=0.50, rank=8
#
#   The same p_keep and rank are passed to both stages so the
#   experiment driver loads the checkpoint training just produced.
# ─────────────────────────────────────────────────────────────────

set -e

P_KEEP=${1:-0.50}
RANK=${2:-10}

mkdir -p logs

# Tag for the log filenames — strip the decimal point from p_keep
# so the filename is shell-safe.
P_TAG=$(printf "p%02d" $(echo "$P_KEEP * 100" | bc | cut -d. -f1))
R_TAG=$(printf "r%02d" "$RANK")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="logs/train_menet_${P_TAG}_${R_TAG}_${TIMESTAMP}.log"
PART7_LOG="logs/part7_menet_${P_TAG}_${R_TAG}_${TIMESTAMP}.log"

echo "============================================================"
echo "  Part 7 pipeline: train Model 4 + EOT evaluation"
echo "============================================================"
echo "  p_keep        : $P_KEEP"
echo "  rank          : $RANK"
echo "  Train log     : $TRAIN_LOG"
echo "  Part 7 log    : $PART7_LOG"
echo "============================================================"

# Record GPU info for reproducibility (no-op if no GPU).
if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    echo "── GPU ─────────────────────────────────────────────────────"
    nvidia-smi --query-gpu=name,driver_version,memory.total \
               --format=csv,noheader
fi

# ── Stage A: train Model 4 ────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Stage A — training Model 4 base CNN with ME-Net augmentation"
echo "============================================================"
python -m models.train_menet "$P_KEEP" "$RANK" 2>&1 | tee "$TRAIN_LOG"

# ── Stage B: run Part 7 (Stages 1, 2, D1/D4 diagnostics, Stage 5 EOT) ──
echo ""
echo "============================================================"
echo "  Stage B — Part 7 full pipeline on Model 4"
echo "============================================================"
python -m experiments.part7_menet "$P_KEEP" "$RANK" 2>&1 | tee "$PART7_LOG"

echo ""
echo "============================================================"
echo "  Done."
echo "  Report : report/part7_menet.txt"
echo "  Plots  : outputs/part7_eot_gap.png"
echo "           outputs/part7_d4_trajectory.png"
echo "  Logs   : $TRAIN_LOG"
echo "           $PART7_LOG"
echo "============================================================"