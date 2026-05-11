# Report on the progress of the project

---

## Part 1

---
- Set up the environment and creating the basic structure of the project.
- Load the dataset(CIFER-10) and the basic model 1(Res-Net-18).
- Implement PGD (stage 1) attack on the model and generate the first report for the results on the terminal's sdout.
- The model shows narrow robustness as the accuracy falls deep when the epsilon in the PGD attacks increases.
- Create plots for:  
    1. Accuracy vs Epsilon curve.
    2. Loss trajectory to check D4 flag.
- D1 check: PGD vs FGSM and print report.

---

## Part 2

---
- Import and implement autoattack: `https://github.com/fra31/auto-attack`.
- Autoattack includes the following attacks: 
    - untargeted APGD-CE (no restarts),
    = targeted APGD-DLR (9 target classes),
    - targeted FAB (9 target classes),
    - Square Attack (5000 queries).
- Findings for the lower epsilon: 
```
apgd-ce alone at epsilon=2/255  →  robust accuracy: 5.90%
our PGD at epsilon=2/255        →  robust accuracy: 80.2%
```
- Findings shows that PGD attack was a weak evaluator and autoattack that uses *adaptive step size schedule*, different *loss functions* and *targeted attacks* is way more efficient. This is a direct demonstration of why attack hyperparameters matter as much as the attack algorithm itself — the lesson Tramèr et al. make in Section 11 (Ensemble Diversity) when they write that simply increasing iterations and step size reduces accuracy from 48% to 10%.
- From this point forward in the project, AutoAttack numbers are the evaluation standard. Our PGD implementation remains useful for the D1 check (FGSM vs PGD comparison) and the D4 loss trajectory check, but it should not be trusted as a robustness measure by itself.
- Extracting final report of week 2 to report/week2_autoattack.txt

|   Diagnostic   |     Status     |    Evidence                         |
| -------------- | -------------- | ----------------------------------- |
|  D1 primary    |    clear       | PGD stronger than FGSM at all epsilons |
|  D1 secondary  |    clear       | APGD-CE dominant, Square not needed    |
|     D4         |    FIRED       |75.7% gap between PGD and AutoAttack at epsilon=2/255 |
|     D5         |    FIRED       |0% robust accuracy at epsilon ≥ 4/255 |

  
---

## Part 3

--- 

- Run the stages 1 and 2 on a robust model to take different diagnostics.
- Load model 2 from RobustBench.
- **Note**: Model 1 (ResNet-18) and Model 2 (ResNet-50) differ in architecture. Observed robustness differences are primarily attributed to adversarial training rather than capacity, as standard ResNet-50 baselines show equivalent fragility to ResNet-18 under AutoAttack at epsilon=8/255.
- There was a need for correction on epsilon. Usage of scaled epsilon since I used normalized inputs.
- Adversarial training completely transforms the model's behavior under attack.
- D5 fires between epsilon=8/255 and epsilon=16/255 with a 39.2% drop. This is not a flaw — it is a feature of adversarial training and a textbook finding. The Engstrom2019 model was trained with adversarial training at exactly epsilon=8/255.
- D5 fires on both but means completely different things — total fragility on Model 1, epsilon-boundary cliff on Model 2.

---

## Part 4

---

- Context: A transfer attack generates adversarial examples on a **surrogate** model (model 1 in this case) and then tests if those examples also effectively fool a target model (Model 2). The logic is: if Model 2 looks robust under direct PGD but is still fooled by examples from a completely different model, then its robustness may be partially an artifact of gradient behavior rather than genuine decision boundary hardening.
- Carlini and Wagner insight : transferability increases with confidence.
- Calculating CW loss for high confident adversarial examples to prove Carlini and Wagner point that transferability increases with hig confidence.
- Run PGD attack but with using CW loss and not cross entropy. 

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

- Change model 1 from resnet-18 to resnet-50 for better comparison with the adv. trained resnet-50. In the first parts the standard resnet-50 has similar behaviour on the pgd attack like the resnet-18, but breaks more easily.
- The transfer attack is 3-4 times weaker than a direct attack despite using the same architecture as surrogate. This is the clearest possible evidence that adversarial training genuinely hardens the decision boundary — not just the gradient landscape.


--- 

## Part 5

---

- Add model3 (a certified model) in the game! We pass the model 3 from the pipeline and we extract the results.
- Model 2 was emprically robust. It was adversarially trained and performed well against known attacks. In the other hand the Model 3 is mathematically certified and fully robust up to a level (Pixel-DP style).
- We load Gowal2020Uncovering from RobustBench under the L2 threat model. This is a wide ResNet model trained with adversarial training under L2 norm. No mathematical guarantees -> empirically robust.
- The idea is to wrap it with our `Smooth` class to turn it into certified classifier.( Cohen et al.: Certified Adversarial Robustness via Randomized Smoothing) — it is a pure inference-time procedure.
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
- Can an adversarial attack break a mathematically certified model?