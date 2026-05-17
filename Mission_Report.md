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

--- 
## Diagnostic glossary 

- **D1 — Gradient masking.** Fires when PGD robust accuracy > FGSM robust accuracy (FGSM is more effective than PGD), or when Square Attack within AutoAttack outperforms APGD.
- **D2 — Transfer vulnerability.** Fires when PGD robust accuracy > 60% AND surrogate transfer success rate > 40%.
- **D3 — Boundary overconfidence.** Fires when FAB places adversarial examples close to the boundary but model confidence on them is high. *(Not yet implemented end-to-end.)*
- **D4 — Loss inconsistency.** Fires when the PGD loss trajectory across iterations is non-monotonic (>20% of steps show loss decrease).
- **D5 — Narrow robustness.** Fires when robust accuracy drops by >30 percentage points between two consecutive epsilon values.
- **D6 — Attack hyperparameter sensitivity.** Fires when the gap between PGD and AutoAttack robust accuracy exceeds 5 percentage points at any evaluated epsilon. Originally tracked as D4 in early drafts; renamed to keep D4 reserved for the loss-trajectory check.
- **D7 — Certificate looseness.** Fires when, at some L2 radius within the certificate's claimed range, empirical robust accuracy exceeds certified accuracy by more than a threshold τ percentage points.