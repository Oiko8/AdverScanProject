## D7 — Certificate Looseness

**Applies to**: certified models (randomized smoothing, PixelDP-style defenses).

**Definition.** D7 fires when, at some L2 radius within the certificate's claimed range, empirical robust accuracy exceeds certified accuracy by more than a threshold τ percentage points.

```
D7 fires if, for any radius r ≤ r_max:
    empirical_accuracy(r) - certified_accuracy(r) > τ
```

Default τ = 20 percentage points (to be calibrated after first run).

**What it measures.** The certified accuracy from randomized smoothing is provably a *lower bound* on the smoothed classifier's true robust accuracy at every radius r. The empirical accuracy under a strong L2 attack is an *upper bound* on the same quantity (with caveats — see "limitations" below). The gap between them measures how loose the certificate is.

**Interpretation.**

- **Small gap (< τ)**: the certificate is tight — most of the true robust accuracy is captured by the proof. The defender can trust the certified numbers as a close approximation to the deployment-time robustness.
- **Large gap (≥ τ, D7 FIRED)**: the certificate is loose — there is substantial robust accuracy the certificate fails to capture. This is *not* a defect of the model. It is a defect of the analysis. The defender's model is genuinely more robust than the certificate proves. Possible causes:
  - The certification budget (n in Monte Carlo sampling) is too small, giving overly conservative `p_A_lower` bounds.
  - The certified radius formula `r = σ · Φ⁻¹(p_A_lower)` is a worst-case bound, not the best achievable.
  - The smoothing distribution choice (Gaussian) may not be optimal for the input data distribution.
- **Gap is negative (empirical < certified)**: indicates a bug. The certificate is a mathematical guarantee — empirical accuracy cannot legitimately fall below it. If this fires, the attack is misconfigured, the σ used during certification doesn't match the σ used during training, or the normalization wrapper is wrong.

**Limitations.**

- D7 depends on the strength of the empirical attack. A weak attack will overstate empirical accuracy and inflate the gap artificially. The empirical number must come from a state-of-the-art L2 attack (AutoAttack-L2 or PGD-L2 with adaptive step sizes).
- When attacking the base classifier rather than the smoothed classifier directly (the standard practice; see Part 5 design notes on EOT), the empirical accuracy is an upper bound on what attacking the deployed smoothed model could achieve. The D7 gap is therefore an upper bound on the true certificate looseness.
- D7 only applies within the certifiable radius range (typically r ≤ 3σ). Outside that range, the certificate is uninformative (always 0%) and the gap is not meaningful.

**Remediation when D7 fires.**

1. Increase the certification Monte Carlo budget (n from 10,000 to 100,000) to tighten `p_A_lower`. This is the cheapest fix and often sufficient.
2. Retrain the base classifier with stronger noise augmentation matched to the inference-time σ.
3. Consider alternative smoothing distributions (e.g. uniform, Laplacian) if the input distribution suggests Gaussian is suboptimal.
4. Accept the gap as an honest reporting of the limit of current certified-defense theory. The model is robust; the proof is what's loose.

**Why D7 is novel.** The randomized smoothing literature reports certified and empirical accuracy separately, but does not name the gap as a structured diagnostic. AdverScan introduces D7 to make the gap a first-class signal alongside D1–D6, with an explicit threshold and remediation list. This is the kind of diagnostic the survey paper (Wang et al. 2023, §V-B-2) calls out as missing from current robustness evaluation practice.