#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#   AdverScan — full pipeline rerun
#   Regenerates all reports and plots for Parts 1–4.
#
#   Assumes:
#     - Model 1 checkpoint already exists at:
#         models/model1_standard_resnet50.pth
#       (train it with: python -m models.train  or your training script)
#     - Model 2 (RobustBench Engstrom2019) downloads automatically on first run.
#
#   Outputs:
#     - report/  ← .txt diagnostic summaries
#     - outputs/ ← .png plots
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on any error

echo "============================================================"
echo "  AdverScan — Full Pipeline Rerun"
echo "============================================================"

# ── Sanity check: Model 1 checkpoint must exist ──────────────────
if [ ! -f "models/model1_standard_resnet50.pth" ]; then
    echo ""
    echo "ERROR: Model 1 checkpoint not found at:"
    echo "       models/model1_standard_resnet50.pth"
    echo ""
    echo "Train it first before running this script."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────
#   Part 1 — Model 1: PGD baseline + D1/D4 diagnostics
# ─────────────────────────────────────────────────────────────────
# echo ""
# echo "── Part 1: Model 1 — PGD baseline (Stage 1) ────────────────"
# python -m experiments.part1_baseline

# echo ""
# echo "── Part 1: Model 1 — D1 (FGSM vs PGD) + D4 (loss trajectory) ──"
# python -m diagnostics.model1_pgd

# # ─────────────────────────────────────────────────────────────────
# #   Part 2 — Model 1: AutoAttack (Stage 2) + D1 secondary + D6
# # ─────────────────────────────────────────────────────────────────
# echo ""
# echo "── Part 2: Model 1 — AutoAttack (Stage 2) ───────────────────"
# python -m experiments.part2_autoattack

# # ─────────────────────────────────────────────────────────────────
# #   Part 3 — Model 2: PGD (Stage 1) + AutoAttack (Stage 2)
# # ─────────────────────────────────────────────────────────────────
# echo ""
# echo "── Part 3: Model 2 — PGD (Stage 1) ──────────────────────────"
# python -m experiments.part3_model2_stage1

# echo ""
# echo "── Part 3: Model 2 — D1 + D4 diagnostics ────────────────────"
# python -m diagnostics.model2_pgd

echo ""
echo "── Part 3: Model 2 — AutoAttack (Stage 2) ───────────────────"
python -m experiments.part3_model2_stage2

# ─────────────────────────────────────────────────────────────────
#   Part 4 — Transfer attack (Model 1 → Model 2) + kappa sweep
# ─────────────────────────────────────────────────────────────────
echo ""
echo "── Part 4: Transfer attack + kappa sweep ────────────────────"
python -m experiments.part4_transfer

# ─────────────────────────────────────────────────────────────────
#   Done
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  All stages complete."
echo ""
echo "  Reports : ./report/"
echo "  Plots   : ./outputs/"
echo "============================================================"