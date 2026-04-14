# AdverScan — A Systematic Adversarial Robustness Testing Pipeline

---

## Part 1 — Project Statement and Motivation

The central problem this project addresses is the gap between **apparent robustness** and **actual robustness** in neural network classifiers. Existing evaluation tools tell you whether a model breaks under attack. They do not tell you *why* it appeared not to break, or where a developer should focus improvement effort. This pipeline fills that gap.

The core insight, drawn from Tramèr et al. (2020), is that almost all robustness evaluation failures are **methodological, not technical**. The attacks needed to reveal fragility already exist. What is missing is a structured workflow that applies them in a complementary sequence, interprets the pattern of results as a diagnostic, and converts that diagnostic into actionable guidance.

The project is grounded in three explicit constraints from the literature:

**First**, adaptive attacks cannot be fully automated. The pipeline therefore does not claim to be a complete robustness oracle. It is a structured preliminary audit that automates the first three or four diagnostic hypotheses a researcher would form by hand. For custom or novel defenses, the pipeline outputs hypotheses for further manual investigation, not final verdicts.

**Second**, attack diversity matters more than attack quantity. Running BIM, PGD, and MIM together is redundant — they are too similar to reveal qualitatively different failure modes. The pipeline selects attacks that are genuinely complementary: each stage probes a different dimension of the model's robustness.

**Third**, robustness is not a binary property. A model can be genuinely robust to L∞ perturbations of ε = 4/255, brittle at ε = 8/255, and completely fooled by transfer examples regardless of ε. The pipeline captures this by reporting accuracy as a continuous function of perturbation budget, not as a single pass/fail number.

---

## Part 2 — Scope and Boundaries

**In scope:**
- Image classification on CIFAR-10
- Four model types representing distinct robustness claims
- Five diagnostic categories derived from the literature
- A before/after hardening demonstration
- A developer-facing report with prioritized remediation guidance

**Explicitly out of scope:**
- ImageNet (not feasible in the two-month timeline)
- Novel attack implementations (all attacks used from established libraries)
- Certified robustness verification beyond what PixelDP or randomized smoothing already provides
- Defenses against non-norm-bounded attacks (geometric transforms, semantic perturbations)
- Custom or novel defenses — the pipeline is validated against known defense types only

---

## Part 3 — The Four Model Types

Each model is chosen to represent a distinct class of robustness claim, so that the pipeline's diagnostic stages produce meaningfully different outputs across models.

**Model 1 — Standard ResNet-18 (weak baseline).** Trained normally on CIFAR-10. Expected to fail quickly under PGD at small epsilon. This model exists to calibrate the pipeline: all stages should find it easy to break, and the diagnostics should produce clean overconfidence and loss inconsistency signals.

**Model 2 — Adversarially trained ResNet (empirical defense).** Loaded from RobustBench, trained with PGD-based adversarial training. This is the primary comparison target. Expected to resist PGD but potentially show boundary proximity issues under CW-style attacks within AutoAttack.

**Model 3 — Randomized smoothing or PixelDP-style certified model.** This model has a mathematical guarantee attached to its robustness claim, not just an empirical one. Including it forces the pipeline to handle the distinction between certified accuracy and empirical accuracy under attack — a distinction the Lecuyer et al. paper makes central. For this model, the pipeline reports both the certified lower bound and the empirical accuracy under attack, and flags when the gap between them is large.

**Model 4 — Preprocessing or stochastic defense.** A model with a randomized or non-differentiable component in its inference pipeline. This is the model type most susceptible to gradient masking and EOT bypass. Including it activates the conditional fifth stage of the pipeline and tests the EOT diagnostic path.

---

## Part 4 — The Five Pipeline Stages

Each stage is not just an attack — it is an attack plus a diagnostic check that produces a structured output consumed by the next stage.

### Stage 1 — First-Order Robustness Baseline (PGD)

Run PGD-L∞ with 100 steps, step size α = ε/10, three random restarts, across ε ∈ {2/255, 4/255, 8/255, 16/255}.

**Primary output:** robust accuracy vs. epsilon curve.

**Diagnostic checks at this stage:**
- If robust accuracy drops to near-zero at ε = 2/255, the model has no meaningful robustness.
- If FGSM accuracy exceeds PGD accuracy, this is the classical gradient masking signal and should be flagged immediately.
- If the accuracy vs. epsilon curve shows a sudden cliff rather than gradual degradation, this signals that the model's robustness is narrow and brittle.
- Monitor the PGD loss trajectory per iteration. If the loss fluctuates wildly rather than increasing monotonically, this indicates a non-smooth loss landscape — the k-WTA failure mode from Tramèr et al. Section 5.

### Stage 2 — Strong Standardized Evaluation (AutoAttack)

Run AutoAttack's full ensemble: APGD-CE, APGD-DLR, FAB, and Square Attack. Use fixed hyperparameters as in the standard RobustBench evaluation. Run at ε = 8/255 as the primary operating point.

**Primary output:** AutoAttack robust accuracy, comparison with Stage 1 PGD accuracy.

**Diagnostic checks at this stage:**
- If PGD accuracy and AutoAttack accuracy are close (within 3–5%), the model's loss surface is well-behaved and gradient-based attacks are saturating it correctly.
- If there is a large gap (AutoAttack significantly lower than PGD), this confirms that PGD was failing to optimize effectively — typically caused by a non-smooth or inconsistent loss function, which is Theme T4 from Tramèr et al.
- Log which component of AutoAttack (APGD-CE, APGD-DLR, FAB, or Square) finds the most adversarial examples. If Square Attack (the black-box component) dominates, this is a strong signal of gradient masking.

### Stage 3 — Boundary Proximity Check (FAB results from AutoAttack)

AutoAttack already runs FAB internally. Extract and analyze the FAB component's results separately at this stage rather than running CW independently.

**Primary output:** estimated distance to the decision boundary for correctly classified examples.

**Diagnostic checks at this stage:**
- Compare average boundary distance for correctly classified examples against the epsilon budget. If the average boundary distance is much smaller than ε, the model's robustness is real but limited.
- Flag overconfidence near the boundary: if a model has high softmax confidence on examples that FAB places close to the decision boundary, this is the calibration failure described in Tramèr et al. Section 6 (*The Odds are Odd*). Such a model will fool confidence-based detectors.

### Stage 4 — Transfer Diagnostic (Surrogate Model)

Train a standard ResNet-18 from scratch as a surrogate (or use the Stage 1 standard model as the surrogate when testing the other three models). Generate adversarial examples on the surrogate using PGD at high confidence (κ = 20 in C&W notation). Evaluate their transfer success rate on the target model.

**Primary output:** transfer success rate from surrogate to target.

**Diagnostic checks at this stage:**
- If PGD robustness (Stage 1) is high but transfer success rate is also high (above 40%), this is strong evidence of brittle gradients or masking — the model's direct gradient is misleading, but the underlying decision boundary is still exploitable.
- If transfer success rate is low and PGD robustness is high, the model has genuine first-order robustness.
- This is the diagnostic that Carlini & Wagner Section VIII-D formalizes: a defense must demonstrate that transferability fails, not just that direct attacks fail.

### Stage 5 — Conditional Adaptive Evaluation (EOT, BPDA)

This stage runs only if Model 4 (stochastic or preprocessing defense) is being evaluated, or if Stage 1 produces the gradient masking signal.

Run EOT by averaging gradients over 40 noise samples per PGD step (following the ME-Net evaluation in Tramèr et al. Section 15). If the defense has a non-differentiable preprocessing component, implement BPDA by substituting the identity function in the backward pass.

**Primary output:** EOT/BPDA robust accuracy, comparison with Stage 1 PGD accuracy.

**Diagnostic checks at this stage:**
- If EOT significantly reduces accuracy compared to vanilla PGD, the defense's stochasticity was providing false robustness rather than genuine protection.
- If BPDA significantly reduces accuracy, the non-differentiable component was masking gradients and the underlying classifier has less robustness than reported.

---

## Part 5 — The Five Diagnostic Categories

Each diagnostic is a measurable condition derived from the stage outputs. The pipeline maps combinations of stage results onto these categories automatically.

- **D1 — Gradient masking:** Triggered when FGSM accuracy > PGD accuracy, or when Square Attack (black-box) within AutoAttack outperforms APGD (white-box). The model is obfuscating its gradients, giving a false sense of robustness to gradient-based attacks.

- **D2 — Transfer vulnerability:** Triggered when PGD robust accuracy > 60% but surrogate transfer success rate > 40%. The model's direct gradient is unhelpful for an attacker, but the decision boundary is still exploitable via a surrogate.

- **D3 — Boundary overconfidence:** Triggered when FAB finds adversarial examples close to the boundary (average L2 distance < ε/2) but the model's softmax confidence on those examples is high (> 0.8). The model is poorly calibrated near its decision boundary.

- **D4 — Loss inconsistency:** Triggered when the PGD loss trajectory shows non-monotonic behavior across iterations. The loss surface is too non-smooth for gradient-based attacks to navigate reliably — this usually means the attack results are understating the model's vulnerability, not that the model is robust.

- **D5 — Narrow robustness:** Triggered when robust accuracy drops sharply between two consecutive epsilon values (more than 30 percentage points between ε = 4/255 and ε = 8/255). The model has genuine robustness at small perturbation budgets but no robustness at the standard evaluation budget.

---

## Part 6 — Developer-Facing Report Structure

The pipeline's final output for each model is a one-page structured report containing:

- Robust accuracy table across all epsilon values and all attack stages
- Accuracy vs. epsilon curve (visual)
- Diagnostic flags triggered, with plain-language explanations of what each flag means
- A prioritized list of remediation recommendations based on which diagnostics fired

The remediation recommendations follow a fixed priority ordering:

1. **If D1 fires:** your attack evaluation is understating real vulnerability. Re-evaluate with black-box attacks before claiming any robustness.
2. **If D4 fires:** your loss function is non-smooth. Ensure the loss used for attack evaluation is consistent — see the objective function analysis in Carlini & Wagner Section V.
3. **If D2 fires:** your model's gradients are not representative of its true boundary. Consider ensemble adversarial training with diverse surrogate models.
4. **If D3 fires:** add calibration regularization or temperature scaling to align confidence with actual boundary distance.
5. **If D5 fires:** increase adversarial training epsilon or diversify the attack distribution used during training.

---

## Part 7 — Hardening Demonstration

The hardening demo is the pipeline's most important deliverable for communicating its value.

**Starting point:** Model 1 (standard ResNet-18). Run the full five-stage pipeline. Expected result: D1 not triggered (standard model has clean gradients), D3 triggered (overconfidence), D5 triggered (no robustness at any epsilon), D2 partially triggered.

**Intervention:** Apply PGD adversarial training using Madry et al. protocol, ε = 8/255, 10 PGD steps during training.

**After training:** Run the full pipeline again on the hardened model.

**Expected change:** D5 disappears, D3 partially resolves, D2 reduces. Potentially D4 appears if training introduced any instability.

The report for this demo is a side-by-side comparison: which diagnostics were present before, which remain after, and what the accuracy improvement looks like on the epsilon curve. This is the concrete output that demonstrates the pipeline's usefulness to a developer.

---

## Part 8 — Technical Stack

All components chosen for stability, reproducibility, and minimal implementation overhead.

| Component | Details |
|---|---|
| **Framework** | PyTorch |
| **Models** | RobustBench model zoo for Models 2, 3, and 4; standard PyTorch training for Model 1 and the surrogate |
| **AutoAttack** | `autoattack` library, version-pinned; fixed hyperparameters, no tuning |
| **Foolbox** | PGD implementation and transfer attack generation |
| **RobustBench** | Loading pre-trained robust models and comparing results against the public leaderboard |
| **Dataset** | CIFAR-10 via `torchvision.datasets` |
| **Reporting** | `matplotlib` for curves, JSON for structured diagnostic output, a simple Python class that accumulates stage results and renders the final report |

---

## Part 9 — Eight-Week Implementation Timeline

### Week 1 — Environment and Baseline
Set up the repository structure and dependency pinning. Load CIFAR-10. Train or load Model 1 (standard ResNet-18). Implement Stage 1 (PGD) and confirm it produces a clean accuracy-vs-epsilon curve. Implement the D1 gradient masking check (FGSM vs PGD comparison).

**Milestone:** working Stage 1 with D1 diagnostic on Model 1.

### Week 2 — AutoAttack Integration and Stage 2
Integrate AutoAttack. Run Stages 1 and 2 on Model 1. Implement the component-level analysis (which AutoAttack sub-attack finds the most examples). Implement the D4 loss trajectory check by logging PGD loss per iteration.

**Milestone:** Stages 1 and 2 working, D1 and D4 diagnostics implemented.

### Week 3 — Robust Model Loading and Boundary Checks
Load Model 2 from RobustBench. Run Stages 1 and 2 on Model 2. Extract FAB results from AutoAttack and implement the boundary proximity calculation. Implement D3 (boundary overconfidence) and D5 (narrow robustness) diagnostics.

**Milestone:** all five diagnostics implemented, pipeline running on two models.

### Week 4 — Transfer Attacks and Stage 4
Implement the surrogate attack workflow. Generate high-confidence adversarial examples from Model 1 as surrogate. Evaluate transfer to Model 2. Implement D2 (transfer vulnerability) diagnostic.

**Milestone:** Stage 4 complete, all five diagnostics working end-to-end on Models 1 and 2.

### Week 5 — Certified Model and Stochastic Defense
Load or train Model 3 (certified/smoothing model). Load Model 4 (stochastic defense). Implement Stage 5 (EOT) for Model 4. Handle the certified accuracy vs. empirical accuracy reporting for Model 3.

**Milestone:** all four models running through all applicable stages.

### Week 6 — Report Generation
Implement the structured report class. Generate the accuracy-vs-epsilon plots. Produce the diagnostic flag output with plain-language explanations. Generate the prioritized remediation list from the diagnostic combination.

**Milestone:** full pipeline producing a structured report for each model.

### Week 7 — Hardening Demonstration
Apply adversarial training to Model 1. Run the full pipeline before and after. Produce the side-by-side diagnostic comparison report. Validate that the diagnostics change as expected.

**Milestone:** hardening demo complete.

### Week 8 — Validation, Documentation, and Write-Up
Cross-check all results against RobustBench leaderboard numbers for sanity. Write the project report grounding each design decision in the literature. Prepare the final presentation. Address any implementation issues found during validation.

**Milestone:** submission-ready project.

---

## Part 10 — Key Risks and Mitigations

**Risk 1 — Model 4 is not available pre-trained.**
Mitigation: use the ME-Net defense from Tramèr et al. Section 15, for which the authors released code, or use randomized smoothing from the smoothing library as a substitute stochastic defense. Both are known to be evaluable with EOT.

**Risk 2 — AutoAttack runtime is too slow on available hardware.**
Mitigation: run AutoAttack on 1,000 test images (standard in the literature, per the PixelDP paper) rather than the full 10,000-image test set. This is explicitly acceptable practice.

**Risk 3 — The transfer attack success rate is too low to be diagnostic.**
Mitigation: generate high-confidence adversarial examples (κ = 20) rather than minimum-distortion examples. As Carlini & Wagner Section VIII-D shows, confidence controls transferability directly, and this addresses the concern noted in Tramèr et al. Section 5 that transfer attacks typically succeed less than half the time.

**Risk 4 — Weeks 5 and 6 compress.**
Mitigation: Model 3 and Model 4 can share Stage 5 infrastructure. If time pressure is severe, drop the conditional BPDA path and keep only EOT — this still covers the stochastic defense case fully.