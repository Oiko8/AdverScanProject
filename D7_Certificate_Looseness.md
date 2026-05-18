## D7 — Certificate Looseness

**Applies to**: certified models (randomized smoothing, PixelDP-style defenses).

**Definition.** D7 fires when, at some L2 radius within the certificate's claimed range, empirical robust accuracy exceeds certified accuracy by more than a threshold τ percentage points.

```
D7 fires if, for any radius r ≤ r_max:
    empirical_accuracy(r) - certified_accuracy(r) > τ
```

Default τ = 20 percentage points (to be calibrated after first run).

**Why D7 is novel.** The randomized smoothing literature reports certified and empirical accuracy separately, but does not name the gap as a structured diagnostic. AdverScan introduces D7 to make the gap a first-class signal alongside D1–D6, with an explicit threshold and remediation list. This is the kind of diagnostic the survey paper (Wang et al. 2023, §V-B-2) calls out as missing from current robustness evaluation practice.