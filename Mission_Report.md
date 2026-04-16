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