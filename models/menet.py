import torch
import torch.nn as nn
from data.cifar10_loader import CIFAR10_MEAN, CIFAR10_STD


# ─────────────────────────────────────────────────────────────────────────────
#   ME-Net preprocessing (Yang et al. 2019, "ME-Net: Towards Effective
#   Adversarial Robustness with Matrix Estimation", ICML 2019)
#
#   The defense:
#     1. Randomly mask a fraction (1 - p_keep) of pixels per image.
#     2. Reconstruct the masked image via low-rank matrix estimation.
#     3. Forward the reconstructed image through a CNN.
#
#   Steps 1+2 are stochastic and non-smooth — gradient masking is the
#   intended (or accidental, depending on whom you ask) consequence.
#   Tramèr et al. 2020 §15 showed it can be bypassed with EOT.
#
#   This implementation is from scratch, not the released 2018 code, for
#   the same reasons the project's Smooth class is from scratch: cleaner
#   integration with the rest of the pipeline, modern PyTorch compatibility,
#   and the algorithm itself is short enough that a reimplementation is
#   less work than patching the original.
# ─────────────────────────────────────────────────────────────────────────────


# ── Matrix completion: USVT ──────────────────────────────────────────────────

def usvt_reconstruct(x_masked, mask, rank):
    """
    Universal Singular Value Thresholding (Chatterjee 2015).
    Per-channel rank-`rank` truncated SVD reconstruction.

    Args:
        x_masked : masked image batch, shape (B, C, H, W), values in [0,1]
        mask     : binary keep-mask, shape (B, 1, H, W), broadcast over channels
        rank     : number of top singular values to retain per channel

    Returns:
        x_reconstructed : shape (B, C, H, W), values in [0,1]

    The 1/p_keep rescaling compensates for the energy lost to masking,
    so the reconstructed values stay on roughly the same scale as the
    clean input. Without it the reconstructed image would be dimmer
    by a factor of p_keep.
    """
    B, C, H, W = x_masked.shape

    # Per-image observed fraction (per-image because the random mask
    # has small batch-level variance; computing per-image is cleaner).
    # Shape after squeeze + view: (B,)
    p_keep = mask.view(B, -1).mean(dim=1).clamp(min=1e-6)
    scale = (1.0 / p_keep).view(B, 1, 1, 1)

    # Rescale observed entries before SVD. This is the standard USVT
    # estimator for the underlying matrix.
    x_scaled = x_masked * scale

    # Reshape to (B*C, H, W) for batched SVD.
    x_flat = x_scaled.reshape(B * C, H, W)

    # Batched SVD — torch.linalg.svd handles arbitrary leading dims.
    # full_matrices=False keeps the output compact: shapes are
    #   U: (B*C, H, k_full), S: (B*C, k_full), Vh: (B*C, k_full, W)
    # where k_full = min(H, W).
    U, S, Vh = torch.linalg.svd(x_flat, full_matrices=False)

    # Truncate to top-`rank` singular values.
    r = min(rank, S.shape[-1])
    U_r  = U[..., :r]                 # (B*C, H, r)
    S_r  = S[..., :r]                 # (B*C, r)
    Vh_r = Vh[..., :r, :]             # (B*C, r, W)

    # Reconstruct: X = U_r diag(S_r) Vh_r
    x_recon_flat = U_r * S_r.unsqueeze(-2) @ Vh_r   # (B*C, H, W)
    x_reconstructed = x_recon_flat.reshape(B, C, H, W)

    # Clip back to valid pixel range. USVT can produce slightly
    # negative or >1 values from the rank truncation.
    return x_reconstructed.clamp(0.0, 1.0)


# ── ME-Net preprocessing module ──────────────────────────────────────────────

class MENetPreprocessing(nn.Module):
    """
    Stochastic preprocessing:
        x in [0,1]  →  mask  →  USVT reconstruction  →  x' in [0,1]

    Used as both:
      - training-time data augmentation (a fresh mask per forward pass
        during training teaches the CNN to handle reconstructed inputs).
      - inference-time defense (same procedure at eval — this is what
        makes the model stochastic and EOT-attackable).

    The mask is sampled fresh on every forward call. This is essential:
      - constant mask → trivially differentiable → not a defense
      - fresh per call → gradient at any one forward pass is noisy
        → vanilla PGD gets confused, EOT recovers the true gradient
    """

    def __init__(self, p_keep=0.5, rank=10):
        super().__init__()
        self.p_keep = p_keep
        self.rank   = rank

    def forward(self, x):
        # x: (B, C, H, W), values in [0,1]
        B, _, H, W = x.shape

        # Sample a fresh mask. Shape (B, 1, H, W), broadcasts over channels.
        # bernoulli expects a probability tensor, so build one of the right
        # shape on the same device/dtype as x.
        mask = torch.bernoulli(
            torch.full((B, 1, H, W), self.p_keep,
                       device=x.device, dtype=x.dtype)
        )

        x_masked = x * mask
        x_recon  = usvt_reconstruct(x_masked, mask, rank=self.rank)
        return x_recon


# ── End-to-end wrapper: loader → ME-Net → CNN ────────────────────────────────

class MENetWrapper(nn.Module):
    """
    The full Model 4 inference pipeline.

    Input  : NORMALIZED image (from the project's data loader)
    Output : class logits

    Internal flow:
        normalized in   →   denormalize to [0,1]   →   ME-Net preprocessing
                       →   renormalize             →   base CNN

    The CNN was trained with this same wrapper around it (see
    models/train_menet.py), so it sees a consistent distribution
    of ME-Net-processed inputs at train and test time.

    Why route through [0,1]: the random mask + SVD reconstruction are
    only meaningful in the natural pixel space. Applying them to
    normalized values would distort the mean-shift across channels.
    """

    def __init__(self, base_classifier, p_keep=0.5, rank=10):
        super().__init__()
        self.base   = base_classifier
        self.menet  = MENetPreprocessing(p_keep=p_keep, rank=rank)

        # Buffers so .to(device) moves them correctly.
        self.register_buffer(
            "mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        )

    def forward(self, x_normalized):
        # normalized → [0,1]
        x_01 = torch.clamp(x_normalized * self.std + self.mean, 0.0, 1.0)

        # stochastic preprocessing
        x_processed = self.menet(x_01)

        # [0,1] → normalized
        x_renorm = (x_processed - self.mean) / self.std

        return self.base(x_renorm)


# ── Convenience loader ───────────────────────────────────────────────────────

def get_menet_model(ckpt_path, p_keep=0.5, rank=10):
    """
    Load Model 4: base ResNet-50 trained with ME-Net augmentation,
    wrapped so it accepts normalized input.

    Args:
        ckpt_path : path to the trained base classifier checkpoint
        p_keep    : ME-Net mask keep probability (must match training)
        rank      : ME-Net USVT rank (must match training)

    Returns:
        eval-mode MENetWrapper on DEVICE.
    """
    from models.resnet_model import get_model_resnet50
    from configs.settings import DEVICE

    base = get_model_resnet50().to(DEVICE)
    base.load_state_dict(torch.load(ckpt_path, weights_only=True))
    base.eval()

    model = MENetWrapper(base, p_keep=p_keep, rank=rank).to(DEVICE)
    model.eval()
    return model


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick shape + range sanity check, no model needed.
    torch.manual_seed(0)
    x = torch.rand(4, 3, 32, 32)

    pre = MENetPreprocessing(p_keep=0.5, rank=10)
    y = pre(x)

    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    assert (y >= 0).all() and (y <= 1).all(), "output out of [0,1]"

    # Two passes must differ — confirms statefulness of the mask.
    y1 = pre(x)
    y2 = pre(x)
    diff = (y1 - y2).abs().mean().item()
    assert diff > 0.01, f"forward passes too similar ({diff}) — mask not random"

    print(f"ME-Net preprocessing smoke test passed.")
    print(f"  input  range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  output range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  mean diff between two stochastic passes: {diff:.4f}")