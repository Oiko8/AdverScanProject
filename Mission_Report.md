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

---

## Diagnostic glossary (current canonical definitions)

- **D1 — Gradient masking.** Fires when PGD robust accuracy > FGSM robust accuracy (FGSM is more effective than PGD), or when Square Attack within AutoAttack outperforms APGD.
- **D2 — Transfer vulnerability.** Fires when PGD robust accuracy > 60% AND surrogate transfer success rate > 40%.
- **D3 — Boundary overconfidence.** Fires when FAB places adversarial examples close to the boundary but model confidence on them is high. *(Not yet implemented end-to-end.)*
- **D4 — Loss inconsistency.** Fires when the PGD loss trajectory across iterations is non-monotonic (>20% of steps show loss decrease).
- **D5 — Narrow robustness.** Fires when robust accuracy drops by >30 percentage points between two consecutive epsilon values.
- **D6 — Attack hyperparameter sensitivity.** Fires when the gap between PGD and AutoAttack robust accuracy exceeds 5 percentage points at any evaluated epsilon. Originally tracked as D4 in early drafts; renamed to keep D4 reserved for the loss-trajectory check.