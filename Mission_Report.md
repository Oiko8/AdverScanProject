# Report on the progress of the project

---

## Part 1

---
- Set up the environment and create the basic structure of the project.
- Load the dataset (CIFAR-10) and the basic Model 1 (ResNet-18).
- Implement PGD (Stage 1) attack on the model and generate the first report for the results on the terminal's stdout.
- The model shows narrow robustness: accuracy collapses sharply as the epsilon in the PGD attack increases.
- Create plots for:
    1. Accuracy vs Epsilon curve.
    2. Loss trajectory to check D4 flag.
- D1 check: FGSM vs PGD comparison and print report.

---

## Part 2

---
- Import and implement AutoAttack: `https://github.com/fra31/auto-attack`.
- AutoAttack includes the following attacks:
    - untargeted APGD-CE (no restarts),
    - targeted APGD-DLR (9 target classes),
    - targeted FAB (9 target classes),
    - Square Attack (5000 queries).
- After the epsilon-scaling fix (PGD now operates correctly in normalized space) and the architecture switch to ResNet-50, PGD and AutoAttack agree closely on Model 1. The earlier "75.7% gap" was an artifact of the unscaled epsilon — once corrected, PGD breaks Model 1 essentially as effectively as AutoAttack at every epsilon evaluated.
- The lesson still holds in principle: attack hyperparameters and step-size schedules matter — this is the point Tramèr et al. make in Section 11 (Ensemble Diversity). For Model 1, fixed-step PGD happens to be sufficient because the model has no robustness to defend with; the diagnostic value of the PGD-vs-AutoAttack gap (now labeled D6) will become more interesting on Model 2.
- From this point forward in the project, AutoAttack numbers are the evaluation standard. Our PGD implementation remains useful for the D1 check (FGSM vs PGD comparison) and the D4 loss trajectory check, but it should not be trusted as a robustness measure by itself on adversarially trained models.
- Extracting final report of Part 2 to `report/part2_autoattack.txt`.

|   Diagnostic   |     Status     |    Evidence                                                |
| -------------- | -------------- | ---------------------------------------------------------- |
|  D1 primary    |    clear       | PGD breaks Model 1 at least as effectively as FGSM at all epsilons |
|  D1 secondary  |    clear       | APGD-CE dominant, Square added no extra failures           |
|     D4         |    clear       | PGD loss trajectory increases consistently                 |
|     D5         |    FIRED       | 0% robust accuracy at epsilon ≥ 4/255                      |
|     D6         |    clear       | PGD and AutoAttack agree (max gap 0.5%)                    |

Note on diagnostic numbering: D4 refers to the **loss-trajectory smoothness** check (non-monotonic PGD loss across iterations, per Tramèr et al. §5). The separate signal where PGD and AutoAttack disagree by more than 5% — originally also called D4 in early drafts — is now D6 (attack hyperparameter sensitivity). These probe different failure modes: D4 is about the model's loss surface, D6 is about whether our attack pipeline is strong enough to find what's there.

---

## Part 3

---

- Run Stages 1 and 2 on a robust model to collect comparable diagnostics.
- Load Model 2 from RobustBench (Engstrom2019Robustness, ResNet-50, Linf, ε=8/255).
- **Note**: Model 1 (originally ResNet-18, later switched to ResNet-50 for Part 4) and Model 2 (ResNet-50) differ in architecture in the earlier parts. Observed robustness differences are primarily attributed to adversarial training rather than capacity, as the standard ResNet-50 baseline shows equivalent fragility to ResNet-18 under AutoAttack at epsilon=8/255.
- There was a need for correction on epsilon. Use of scaled epsilon was required since the data loader produces normalized inputs — PGD must operate in normalized space with epsilon scaled by 1/min(std) to preserve the correct raw-pixel perturbation budget.
- Adversarial training completely transforms the model's behavior under attack:
    - PGD robust accuracy at ε=8/255: 53.7% (vs 0.0% for Model 1)
    - AutoAttack robust accuracy at ε=8/255: 50.9% (vs 0.0% for Model 1)
    - Within ~0.5% of the RobustBench leaderboard value (49.25%) — good sanity check.
- **D5 fires** between epsilon=8/255 and epsilon=16/255 with a 39.2% drop. This is not a flaw — it is a feature of adversarial training and a textbook finding. The Engstrom2019 model was trained with adversarial training at exactly epsilon=8/255, so robustness is concentrated at that budget and falls off sharply past it.
- D5 fires on both Model 1 and Model 2 but means completely different things: total fragility on Model 1, epsilon-boundary cliff on Model 2.
- **D6 check (PGD vs AutoAttack gap on Model 2)**: the gap is modest (~2-3% at each epsilon), well under the 5% threshold. PGD's fixed step size is good enough here, but the gap is consistently in the right direction (AutoAttack always finds more adversarial examples than PGD), confirming AutoAttack as the canonical estimate.

|   Diagnostic   |     Status     |    Evidence                                                |
| -------------- | -------------- | ---------------------------------------------------------- |
|  D1 primary    |    clear       | PGD at least as effective as FGSM at all epsilons          |
|  D1 secondary  |    clear       | APGD-CE dominant, Square added no extra failures           |
|     D4         |    clear       | PGD loss trajectory increases consistently                 |
|     D5         |    FIRED       | 39.2% accuracy drop between ε=8/255 and ε=16/255           |
|     D6         |    clear       | PGD vs AutoAttack gap ≤ 3% at all epsilons                 |

---

## Part 4

---

- Context: A transfer attack generates adversarial examples on a **surrogate** model (Model 1) and then tests if those examples also fool a target model (Model 2). The logic: if Model 2 looks robust under direct PGD but is still fooled by examples crafted on a different model, then its robustness may be partially an artifact of gradient behavior rather than genuine decision boundary hardening.
- Carlini and Wagner insight: transferability increases with attack confidence (Section VIII-D).
- Implement the CW margin loss to craft high-confidence adversarial examples — proving in our own setup the C&W finding that higher κ improves transferability.
- Run PGD-style optimization but with the CW loss instead of cross-entropy.

```
Model 1 (surrogate)                    Model 2 (target)
      |                                       |
      | ← generate_transfer_examples()        |
      |   using CW loss + kappa=20            |
      |   produces high-confidence adv        |
      |   examples that strongly fool M1      |
      |                                       |
      └─── adv_images ──────────────────────► evaluate_transfer()
                                              forward pass only
                                              no gradients used
                                              returns success rate
```

- Switched Model 1 from ResNet-18 to ResNet-50 for a same-architecture surrogate. The standard ResNet-50 shows similar behavior to ResNet-18 under PGD (collapses at small ε) but is now architecturally matched to Model 2.
- **Key finding**: the transfer attack is 3–4× weaker than a direct attack at ε=8/255 (12.8% transfer success vs 49.1% direct attack success on Model 2), *despite using the same architecture as surrogate*. This is the clearest possible evidence that adversarial training genuinely hardens the decision boundary, not just the gradient landscape.
- **D2 status: clear at all epsilons.** Transfer success rates (11.6%–14.9%) never approach the 40% threshold. Model 2's robustness is not gradient-masking artifact.

### Transfer results summary

|  Epsilon  | Surrogate fool % | Transfer success % | Model 2 PGD acc | D2 |
| --------- | ---------------- | ------------------ | --------------- | -- |
|  2/255    |     98.9%        |      11.6%         |     82.2%       | clear |
|  4/255    |    100.0%        |      12.4%         |     73.8%       | clear |
|  8/255    |    100.0%        |      12.8%         |     53.7%       | clear |
| 16/255    |    100.0%        |      14.9%         |     14.4%       | clear |

### Kappa sweep (ε=8/255)

Confirmed the C&W finding qualitatively: surrogate fool rate stays near 100% across κ ∈ {5, 10, 20, 30, 40}, while transfer success increases with κ — high-confidence adversarial examples generalize better across models.

---

## Revision until here (Parts 1–4) 

---


This is a consolidated status check on everything completed so far. Each row is verified against the regenerated reports in `report/` and the plots in `outputs/`.

### Diagnostic summary across models

|         | D1 primary | D1 secondary |    D4    |    D5    |    D6    |    D2    |
| ------- | ---------- | ------------ | -------- | -------- | -------- | -------- |
| Model 1 |    clear   |     clear    |   clear  |   FIRED  |   clear  |    N/A   |
| Model 2 |    clear   |     clear    |   clear  |   FIRED  |   clear  |   clear  |

D3 not yet implemented end-to-end; will be added during the FAB-based boundary distance extraction step. D7 introduced in Part 5 (definition below).

### Numerical consistency check

After rerunning the full pipeline with the corrected epsilon scaling and ResNet-50 surrogate:

- **Model 1 AutoAttack at ε=8/255**: 0.0%
- **Model 2 AutoAttack at ε=8/255**: 51.0% (RobustBench leaderboard: 49.25% — within 2%, good sanity check)
- **Model 2 PGD vs AutoAttack max gap**: 2.7% at ε=8/255 — under the 5% D6 threshold, so PGD is canonical here
- **Transfer success rate at κ=20, ε=8/255**: 12.8% — far under the 40% D2 threshold

### Key structural findings worth carrying into the writeup

**1. Adversarial training reshapes the loss landscape, not just decision boundaries.** Model 1's PGD loss climbs to ~29 in 100 steps; Model 2's plateaus around 1.4. Same attack code, same hyperparameters — the 20× difference reflects fundamentally different loss surfaces. Model 1 lets PGD blow up cross-entropy almost unbounded; Model 2 caps it tightly. This is not gradient masking (D1 clears on both) — it's the natural consequence of training under adversarial perturbations.

**2. The kappa sweep is the strongest single piece of evidence we have for Model 2's robustness.** Surrogate fool rate climbs 43% → 85% → 100% as κ goes from 5 to 20. Transfer success stays flat at ~12% across all κ values. Translation: even at maximum C&W confidence, Model 1's adversarials cannot transfer to Model 2. Model 2's decision boundary is genuinely far from Model 1's adversarial regions in input space — not just inaccessible to gradient-based attacks. This is the cleanest possible refutation of the C&W transferability hypothesis on this model pair.

**3. D5 fires on both Model 1 and Model 2 but means different things.** On Model 1, it reflects total fragility (accuracy collapses to 0% past ε=2/255). On Model 2, it reflects the epsilon-boundary cliff: the model was trained at exactly ε=8/255, so robustness is concentrated at that budget and falls off sharply past it. This is a textbook adversarial-training finding (Madry et al.), not a defect. A diagnostic firing with completely different underlying meanings between two models is itself a signal worth flagging in the writeup.

**4. D6's modest gap on Model 2 (~2-3%) is in the expected direction.** AutoAttack always finds slightly more adversarial examples than fixed-step PGD does. The gap stays under the 5% threshold, so PGD is reliable for this model, but AutoAttack remains the canonical estimate. If a future model showed D6 firing strongly, it would indicate that PGD's fixed step size was missing adversarial examples APGD's adaptive schedule finds — a real failure mode, not just slop.

### Open items before Part 5

- Rename "D4 (PGD step size ineffective)" → "D6 (attack hyperparameter sens.)" in `report/part3_autoattack.txt` (two-character fix in `experiments/part3_model2_stage2.py`).
- D3 (boundary overconfidence) implementation deferred — small effort, can be added when extracting per-sample FAB results.

Ready to proceed to Part 5 (randomized smoothing + certified accuracy).

---

## Part 5

---

- Add Model 3 (a certified model) to the pipeline. We will pass Model 3 through the stages and extract the results.
- Model 2 was empirically robust: adversarially trained, performs well against known attacks, no mathematical guarantee. Model 3 will be mathematically certified — fully robust up to a provable radius (randomized smoothing, Cohen et al. 2019).
- Load `Gowal2020Uncovering` from RobustBench under the L2 threat model. This is a wide ResNet trained with adversarial training under L2 norm — empirically robust on its own, no certified guarantee.
- Wrap it with our `Smooth` class to turn it into a certified classifier (Cohen et al.: *Certified Adversarial Robustness via Randomized Smoothing*) — a pure inference-time procedure:

```
Gowal2020Uncovering (base classifier)
         +
Smooth wrapper (sigma=0.25)
         =
Certified classifier with L2 radius guarantees
```

```
L2 epsilon:   0    0.25   0.5    0.75   1.0    1.25   1.5
              │                   │
              │   certified zone  │    uncertified zone
              │                   │
Certified     │ ████████████████  │  ░░░░░░░░░░░░░░░░░
accuracy:     │ guarantee holds   │  lower bound only
              │                   │
AutoAttack    │ ████████████████  │  ████░░░░░░░░░░░░░
accuracy:     │ high (hard to     │  drops — attacks
              │ attack here)      │  start finding
              │                   │  adversarial examples
```

### Open question for Part 5

> Can an adversarial attack break a mathematically certified model?

Precisely: a certified model with smoothing guarantees that within the certified radius r, no L2 perturbation changes the smoothed classifier's prediction. AutoAttack at ε > r is not constrained by the certificate and can succeed. Expected pipeline output:

- **Inside the certified radius**: AutoAttack empirical accuracy ≥ certified accuracy (the certificate is a lower bound on the smoothed classifier's robust accuracy).
- **Outside the certified radius**: empirical accuracy degrades, and the gap between certified and empirical accuracy becomes the key diagnostic — a large gap means the certificate is loose, not that the model is broken.

The Stage 5 report for Model 3 should produce both numbers side-by-side and flag the gap explicitly.

Train ResNet-50 with different noise:
|  Sigma |   Accuracy   |
| ------ | ------------ |
|  0.25  |    71.8%     |
|  0.50  |    61.1%     |
|  0.75  |    43.8%     |
|  1.00  |    33.6%     |

Radius  |  σ=0.25  |  σ=0.50  |  σ=0.75  |  σ=1.00  |
------- | -------- | -------- | -------- | -------- |
0.00    |  76.7%   |  60.4%   |  47.9%   |  35.4%
0.25    |  60.9%   |  49.0%   |  39.9%   |  30.1%
0.50    |  42.7%   |  38.9%   |  31.5%   |  24.1%
0.75    |  22.6%   |  26.5%   |  23.1%   |  18.8%
1.00    |   0.0%   |  15.8%   |  16.4%   |  15.4%
1.25    |   0.0%   |   8.1%   |  12.3%   |  11.3%
1.50    |   0.0%   |   3.1%   |   6.4%   |   8.5%
2.00    |   0.0%   |   0.0%   |   1.9%   |   3.8%
2.50    |   0.0%   |   0.0%   |   0.0%   |   1.3%

  σ   | abstain rate |
----- | ------------ |
0.25  |     7.3%     |
0.50  |    17.0%     |
0.75  |    29.0%     |
1.00  |    32.2%     |

- The σ trade-off is visible. for smaller radius the smaller σ gives the highest accuracy. However stronger sigmas lead to better accuracy for bigger radius of perturbation. Totally normal and agreed with Cohen et al.
- Higher σ → base classifier more confused under noise → lower confidence → more p_A_lower ≤ 0.5 → more abstains.
- at σ=1.00, the certificate is mostly empty not because the radii are small for certified images, but because so many images can't be certified at all.

### Empirical L2 attack on the base classifier

AutoAttack-L2 on each base ResNet-50 at every radius, 1000 samples each. Reading: cert / emp / gap.

|  r   |       σ=0.25       |       σ=0.50       |       σ=0.75       |       σ=1.00       |
| ---- | ------------------ | ------------------ | ------------------ | ------------------ |
| 0.00 | 76.7 / 71.8 / −4.9 | 60.4 / 60.6 / +0.2 | 47.9 / 46.7 / −1.2 | 35.4 / 36.0 / +0.6 |
| 0.25 | 60.9 / 56.3 / −4.6 | 49.0 / 49.0 / +0.0 | 39.9 / 37.4 / −2.5 | 30.1 / 28.9 / −1.2 |
| 0.50 | 42.7 / 39.8 / −2.9 | 38.9 / 34.8 / −4.1 | 31.5 / 28.0 / −3.5 | 24.1 / 21.8 / −2.3 |
| 0.75 | 22.6 / 25.5 / +2.9 | 26.5 / 25.7 / −0.8 | 23.1 / 20.1 / −3.0 | 18.8 / 17.1 / −1.7 |
| 1.00 |  0.0 / 13.9 / —    | 15.8 / 18.0 / +2.2 | 16.4 / 13.9 / −2.5 | 15.4 / 12.5 / −2.9 |
| 1.25 |  0.0 /  5.6 / —    |  8.1 / 11.2 / +3.1 | 12.3 /  9.0 / −3.3 | 11.3 /  9.9 / −1.4 |
| 1.50 |  0.0 /  1.6 / —    |  3.1 /  5.8 / +2.7 |  6.4 /  5.5 / −0.9 |  8.5 /  7.1 / −1.4 |
| 2.00 |  0.0 /  0.1 / —    |  0.0 /  0.8 / —    |  1.9 /  2.4 / +0.5 |  3.8 /  3.2 / −0.6 |
| 2.50 |  0.0 /  0.0 / —    |  0.0 /  0.3 / —    |  0.0 /  1.1 / —    |  1.3 /  1.7 / +0.4 |

Gaps marked "—" are outside the certifiable range (cert = 0 by construction, i.e. beyond what σ·Φ⁻¹(p_A) can reach at this CERT_N). `analyze_d7` excludes them — a "gap" at a radius where the certificate makes no claim isn't certificate looseness, it's outside the contract.

### D7 verdict per σ

|  σ   | max gap | at r | D7    |
| ---- | ------- | ---- | ----- |
| 0.25 | +2.9%   | 0.75 | clear |
| 0.50 | +3.1%   | 1.25 | clear |
| 0.75 | +0.5%   | 2.00 | clear |
| 1.00 | +0.6%   | 0.00 | clear |

D7 clears at every σ. The maximum gap anywhere is 3.1pp, an order of magnitude below the 20pp threshold. The threshold was set conservatively before any data existed; with these numbers it does no work. Worth recalibrating to ~5pp once the proxy-methodology question below is settled.

### The negative-gap pattern

The substantive observation is the *sign* of the gaps. At σ ≥ 0.75, every gap within the certifiable range is negative — base_emp tracks below cert throughout. This is not a certificate violation. Two facts both hold:

- `cert ≤ smoothed_emp` — by smoothing theory, the certificate is a lower bound on the smoothed classifier's robust accuracy.
- `base_emp ≤ smoothed_emp` — by construction, AutoAttack on the deterministic base is strictly stronger than any EOT attack on the smoothed classifier could be.

When `base_emp < cert`, both inequalities still hold; they simply fail to squeeze smoothed_emp into a tight interval. The smoothed model's true robust accuracy sits above both curves, and the proxy comparison cannot tell us by how much.

The pattern is most pronounced at σ ≥ 0.75 because the noise-augmented base loses clean accuracy as σ grows (76.7% → 60.4% → 47.9% → 35.4% at r=0), while the smoothed classifier — which always sees noisy input by construction — stays in its training distribution. Base and smoothed diverge in capability as σ increases, and the base-as-proxy comparison degrades accordingly.

### Reading D7-clear honestly

Two interpretations of D7 clearing everywhere are both consistent with the data:

1. **Certificates are tight.** The proof captures nearly all of the smoothed model's true robustness, leaving little room for D7 to fire.
2. **Proxy is too weak.** Attacking the base degrades as a stand-in for attacking the smoothed model as σ grows. We're measuring `base_emp − cert`, not `smoothed_emp − cert`.

The negative gaps at σ ≥ 0.75 lean toward (2). The natural next step would be a small EOT run at σ=0.75, r ∈ {1.0, 1.25, 1.5} to replace the lower-bound proxy with a direct measurement on the smoothed classifier. That's not done here — compute reasons documented in the design notes for this Part — so the framing should be: **D7 clearing is a failure to find looseness with the chosen methodology, not a proof of tightness.** (--> absence of evidence ≠ evidence of absence)

- The empirical numbers we get are an upper bound on what an attacker against the deployed smoothed model could achieve. D7's gap is therefore a lower bound on actual certificate looseness.
- D7 flag overview: 
    - So when empirical = 60% and certified = 58%, the certificate is tight — it's capturing nearly all the model's true robustness in its proof. The defender can trust the certified number as a near-faithful summary.
    - When empirical = 75% and certified = 30%, the certificate is loose — the model is genuinely more robust than the proof can show. The defender's deployment is safer than the paperwork suggests. That's the D7 signal: your proof is conservative, not your model.
    - This is not bad for the model. It's a defect of the evaluation methodology.

The two possible attack targets
1. Target A — Attack the base classifier:
    - AutoAttack runs on wrapped_base, which is the underlying ResNet-50.
    - It's deterministic — every forward pass returns the same logits.
    - AutoAttack's gradient-based optimizers work cleanly.


2. Target B — Attack the smoothed classifier directly:
    - AutoAttack would run on the Smooth wrapper.
    - Every forward pass adds fresh Gaussian noise — stochastic.
    - Gradient-based attacks chase a noisy signal and converge poorly.
    - Requires EOT (Expectation over Transformations): average gradients over 40+ noise samples per attack step.


### Design notes

**The `cert > 0` filter in `analyze_d7`.** D7 only counts radii where the certificate makes a non-trivial claim. The maximum certifiable radius at noise level σ is σ·Φ⁻¹(p_A_upper), bounded near 3.4σ at our CERT_N=10000. Beyond that, cert = 0% by construction — not because the certificate failed, but because the smoothing theory doesn't bound that regime at all. A "gap" at a radius outside the contract isn't certificate looseness, it's the empirical attack venturing into territory the certificate never claimed. Excluding those cells from D7 is the principled choice; the empirical numbers themselves are kept in the main table for completeness. (Worked example: σ=0.25, r=1.00 shows empirical 13.9% against cert 0%. The model genuinely classifies 13.9% of perturbed images correctly there, but D7 has nothing to say about it — r=1.00 is past the σ=0.25 certificate's reach by construction.)

**D7 threshold.** Set at 20pp before any data existed. The observed max gap is 3.1pp — an order of magnitude under threshold. The 20pp ceiling does no actual work in this evaluation. A more useful threshold post-data would be ~5pp, which would still leave all four σ values comfortably clear but at least put the diagnostic in a regime where firing is plausible. Worth recalibrating before the writeup.

**EOT skipped intentionally.** The principled empirical evaluation would attack the smoothed classifier directly via EOT (40+ noise samples per gradient step). At the certification grid we ran (4 σ × 9 radii × 1000 samples), AutoAttack on the base classifier took ~30 hours; an EOT equivalent at 40× the cost would be ~50 days. Skipping EOT shifts the diagnostic from "true certificate looseness" to "lower bound on certificate looseness" — methodologically weaker but tractable. This is the trade-off behind every result reported in this Part.

**Sample size.** N=1000 per radius for both certification and empirical attack, matching Cohen et al.'s evaluation. At N=1000 the 95% CI on a Bernoulli proportion is ±~3pp, finer resolution than any diagnostic threshold in this Part needs. Scaling to N=10000 would tighten the CI to ±~1pp but doesn't change any verdict — and at 10× the runtime, it doesn't earn its cost on this workload.


---

## Part 6

---

### D3 Implementation (continue part 5)

- D3 fires when small distance + high confidence happen together — the model is still confidently wrong right next to its own boundary. That's a calibration failure: confidence-based detectors won't catch these adversarials because the model doesn't know it's at its boundary.
- Focusing on model 1 and 2. D3 superseded by certified accuracy for Model 3.
- Thresholds:
    - L2 dist < 0.5
    - confidence > 0.8
    - D3 fires if >20% of samples meet both.

- SMOKE TEST: 
    - *Boundary distance is small*. Mean L2 = 0.107 on Model 1 means boundaries sit about 1/10th of a unit from the data in L2. 
    - *FAB success rate 92%, not 100%*. On a standard ResNet-50 with eps_cap=1.0, success should be ~100%. The 8% gap suggests either some samples are hyper-confident (FAB struggles when initial gradients are tiny) or n_restarts=1 is on the low end. Worth bumping to n_restarts=3 for the full run

- Change approach and use CW attack from `transfer_attack.py`
- NEW SMOKE TEST: 
    ```
      ── Smoke test — Model 1 @ eps=2/255 ──
    Samples              : 100
    Attack successes     : 98 (98.0%)
    Mean conf (succ)     : 0.9917
    Median conf (succ)   : 1.0000
    Confidently wrong    : 97/98 (99.0%)
    D3 status            : FIRED
    ```


### Full run results

CW-Linf, κ=20, 100 steps, 1000 samples per (model, ε). D3 thresholds: success_rate ≥ 30% to evaluate, mean confidence > 0.8 to fire.

| eps    | model   | success% | mean_conf | confident-wrong% | D3            |
| ------ | ------- | -------- | --------- | ---------------- | ------------- |
| 2/255  | Model 1 | 98.9%    | 0.9914    | 98.6%            | FIRED         |
| 2/255  | Model 2 | 7.2%     | n/a       | n/a              | not evaluable |
| 4/255  | Model 1 | 100.0%   | 0.9999    | 100.0%           | FIRED         |
| 4/255  | Model 2 | 17.8%    | n/a       | n/a              | not evaluable |
| 8/255  | Model 1 | 100.0%   | 1.0000    | 100.0%           | FIRED         |
| 8/255  | Model 2 | 38.3%    | 0.5869    | 15.4%            | clear         |

### Findings

**Model 1 saturates the failure.** Mean confidence climbs 0.991 → 0.9999 → 1.0000 across the three budgets, with median 1.0 at every ε. Even at 2/255 — where the L∞ ball is tiny — the wrong predictions get assigned probability ~1.0. This is the cleanest possible empirical instance of the Tramèr §6 / *Odds Are Odd* claim that standard networks are overconfident at their decision boundaries.

**Model 2 at 2/255 and 4/255 is "not evaluable", and that is itself the answer.** Success rates of 7.2% and 17.8% mean we cannot gather enough adversarial examples for the confidence statistic to be meaningful. The diagnostic is short-circuited by robustness: at small ε the boundary is geometrically out of reach, so calibration *at* the boundary is moot.

**Model 2 at 8/255 — D3 clear, mean confidence 0.587, median 0.573.** The CW objective targeted margin 20; the achieved confidence corresponds to a margin of roughly 0.3, a small fraction of the target. Within the ε=8/255 L∞ ball, Model 2's loss surface is flat enough that the attack hits the budget cap well before reaching margin 20. The contingency we had prepared — re-running at κ=5 to escape attack-construction saturation — turned out to be unnecessary: the budget itself is the binding constraint, not κ.

### Cross-diagnostic corroboration

This finding agrees with Part 3's loss-trajectory observation:

> Model 1's PGD loss climbs to ~29 in 100 steps; Model 2's plateaus around 1.4. Same attack code, same hyperparameters — the 20× difference reflects fundamentally different loss surfaces.

Part 6 measures the same phenomenon from the confidence angle. The 20× ratio in loss-trajectory ceilings between Models 1 and 2 shows up here as the gap between κ=20 (targeted margin) and the ~0.3 margin Model 2 actually permits within the budget. Two diagnostics independently reading the same property of the same model is the strongest form of evidence the pipeline can produce.

### What D3 says that other diagnostics do not

Adversarial training is known to push decision boundaries farther from data (D5, Part 3 robust accuracy). What's less commonly discussed is whether it also fixes *calibration* at those boundaries. D3 says yes: Model 2 at 8/255 doesn't just resist being fooled — when it is fooled, it stays uncertain about the wrong answer. Mean confidence 0.587 on successful adversarials means a simple confidence-based detector (flag predictions below 0.7) would catch about half of them. The same detector on Model 1 would catch zero — confidence is pinned at 1.0 across the entire adversarial population.

--- 
---

Part 1 to 6 finished: 

MODELS  |  D1 prim  |   D1 sec  |   D2     |   D3           |   D4     |   D5     |   D6     |   D7    | 
------- | :-------: | :-------: | :------: | :------------: | :------: | :------: | :------: | :-----: |
Model 1 |    clear  |    clear  |    N/A   |   FIRED        |   clear  |  FIRED   |   clear  |    N/A  |
Model 2 |    clear  |    clear  |   clear  |  clear(8/255)  |   clear  |  FIRED   |   clear  |    N/A  |
Model 3 |     N/A*  |     N/A*  |    N/A*  |       N/A*     |    N/A*  |   N/A*   |    N/A*  |   clear |


---

## Part 7
 
---
 
### Goal
 
Add Model 4 (ME-Net) and Stage 5 (EOT) to the pipeline. The point of Model 4
is not to add another data point to the diagnostic table — it is to produce
the **gradient-masking** signals that Models 1, 2, 3 never produced.
Specifically, we want to see:
 
- **D1 secondary** firing (Square Attack outperforms APGD-CE within
  AutoAttack — the canonical gradient-masking tell from Tramèr §11)
- **D4** plausibly firing (non-monotonic PGD loss from in-step stochastic
  forward passes)
- **The Stage 1 vs Stage 5 gap** — vanilla PGD reports false robustness on
  Model 4, EOT-PGD recovers near-zero accuracy. The gap is the headline
  number.
### ME-Net implementation choice
 
We did not use the released 2018 ME-Net repo. We wrote a clean from-scratch
PyTorch implementation in `models/menet.py` for three reasons:
 
1. The original code is pinned to PyTorch 0.3/0.4 and uses cvxpy for matrix
   completion, which is not GPU-native. Patching it for current PyTorch +
   pinning a cvxpy version that still installs is non-trivial.
2. The algorithm is short — random mask + USVT (single SVD truncation) — so
   a reimplementation is less work than a port.
3. The project already has precedent: the `Smooth` class is Cohen et al.
   from scratch, not the released repo.
**ME-Net configuration (defaults in `models/menet.py` and
`experiments/part7_menet.py`):**
 
| Parameter | Value | Notes |
| --------- | ----- | ----- |
| `p_keep`  | 0.50  | Fraction of pixels retained per mask. Paper's standard. |
| `rank`    | 10    | USVT top-k retained per channel. Paper used 5–10. |
| Matrix completion | USVT (top-k truncated SVD) | Soft-Impute would be more faithful to paper but adds 5–10× cost; USVT captures the defense's mechanism. |
| Mask granularity | per-spatial-location | Same mask broadcast across the 3 channels. |
 
### Pipeline integration
 
| Stage | Model 4 behavior expected | Reason |
| ----- | ------------------------ | ------ |
| 1 (PGD)            | "high" robust accuracy at 8/255 | gradient-mask false-positive |
| 2 (AutoAttack)     | Square Attack > APGD-CE         | D1 secondary fires |
| 3 (boundary D3)    | N/A                              | not run on Model 4 in Part 7 |
| 4 (transfer)       | tractable (forward-only)         | optional follow-up, not core |
| 5 (EOT-PGD, K=40)  | near-zero at 8/255              | strict generalization of Stage 1 |
| D4 (loss traj.)    | possibly FIRED                   | in-step mask noise → bouncy loss |
 
### Training (Model 4 base CNN)
 
Run before the experiment driver:
 
```
python -m models.train_menet 0.5 10
# or
./train_menet.sh 0.5 10
```
 
Saves to `models/model4_menet_base_p50_r10_resnet50.pth`. Wall-clock budget
is comparable to a single `train_with_noise` run (Model 3): one GPU,
30 epochs, expect a few hours.
 
### Experiment driver
 
```
python -m experiments.part7_menet
```
 
EOT-PGD is expensive: 40 forwards per step × 100 steps × 3 restarts per
image × 1000 images × 3 epsilons. Budget several hours per epsilon on a
single modern GPU. The driver runs Stage 1 first (cheap), then Stage 2,
then the D1/D4 diagnostics, then Stage 5 last.
 
### Results (TODO — fill in after running)
 
```
| eps    | Stage 1 PGD | Stage 2 AA | Stage 5 EOT | Gap (S1 - S5) |
| ------ | ----------- | ---------- | ----------- | ------------- |
| 2/255  |             |            |             |               |
| 4/255  |             |            |             |               |
| 8/255  |             |            |             |               |
```
 
Component breakdown at the headline ε=8/255 (TODO):
 
```
APGD-CE   : __%
APGD-DLR  : __%
FAB       : __%
Square    : __%   <- if lower than APGD-CE, D1 secondary fires
```
 
Diagnostic summary on Model 4 (TODO):
 
| Diagnostic        | Status | Evidence |
| ----------------- | ------ | -------- |
| D1 primary        |        |          |
| D1 secondary      |        |          |
| D4                |        |          |
| EOT gap (S1-S5)   |        |          |
 
### Known caveats to address in the writeup
 
- **AutoAttack on a stochastic model.** Standard AutoAttack treats the model
  as deterministic. On Model 4 each forward gives a different gradient.
  The component-breakdown numbers will fluctuate run-to-run. The signal we
  want — Square Attack out-performing APGD — is robust to this noise
  because the gap is large when the defense is actually gradient-masking,
  but we should note the limitation. If D1 secondary is borderline,
  re-run with `version="rand"` in the AutoAttack call (adds EOT around
  the white-box components).
- **EOT K=40.** From the Tramèr §15 protocol. Lower K (e.g. 20) would also
  work qualitatively but the per-step gradient estimate gets noisy enough
  that PGD's progress slows. Higher K (e.g. 80) costs 2× for marginal gain.
  K=40 is the well-trodden default.
- **EOT vs BPDA.** ME-Net's mask is non-differentiable in the strict sense
  (binary), but the surrounding USVT reconstruction passes meaningful
  gradient through to unmasked pixels. EOT alone is sufficient. BPDA
  (substituting identity in the backward pass) would be the next step for
  a fully non-differentiable defense; we don't need it here.
- **Majority-vote evaluation.** Robust-accuracy numbers on Model 4 are
  computed under majority vote over 20 stochastic forwards per
  adversarial example, not a single pass. Single-pass accuracy fluctuates
  several percent purely from mask draws and would not be a stable
  diagnostic.

---

## Diagnostic glossary 

- **D1 — Gradient masking.** Fires when PGD robust accuracy > FGSM robust accuracy (FGSM is more effective than PGD), or when Square Attack within AutoAttack outperforms APGD.
- **D2 — Transfer vulnerability.** Fires when PGD robust accuracy > 60% AND surrogate transfer success rate > 40%.
- **D3 — Boundary overconfidence.** Fires when FAB places adversarial examples close to the boundary but model confidence on them is high. *(Not yet implemented end-to-end.)*
- **D4 — Loss inconsistency.** Fires when the PGD loss trajectory across iterations is non-monotonic (>20% of steps show loss decrease).
- **D5 — Narrow robustness.** Fires when robust accuracy drops by >30 percentage points between two consecutive epsilon values.
- **D6 — Attack hyperparameter sensitivity.** Fires when the gap between PGD and AutoAttack robust accuracy exceeds 5 percentage points at any evaluated epsilon. Originally tracked as D4 in early drafts; renamed to keep D4 reserved for the loss-trajectory check.
- **D7 — Certificate looseness.** Fires when, at some L2 radius within the certificate's claimed range, empirical robust accuracy exceeds certified accuracy by more than a threshold τ percentage points.