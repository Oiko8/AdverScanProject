import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm, binomtest
from statsmodels.stats.proportion import proportion_confint
from configs.settings import DEVICE


class Smooth:
    """
    Randomized smoothing wrapper around a base classifier.
    Implements Cohen et al. (2019) — Certified Adversarial Robustness
    via Randomized Smoothing.

    The smoothed classifier g(x) returns the class that the base
    classifier f most often predicts when x is corrupted with
    Gaussian noise N(0, sigma^2 * I).

    Mathematical guarantee:
        If g predicts class c_A with high enough confidence,
        then no L2 perturbation smaller than r can change g's
        prediction, where:
            r = (sigma/2) * (Phi_inv(p_A) - Phi_inv(p_B))
    """

    # Abstain token — returned when certification fails
    ABSTAIN = -1

    def __init__(self, base_classifier, num_classes, sigma):
        """
        Args:
            base_classifier : the underlying model f (expects [0,1] input)
            num_classes     : number of output classes (10 for CIFAR-10)
            sigma           : noise level for Gaussian smoothing
        """
        self.base_classifier = base_classifier
        self.num_classes     = num_classes
        self.sigma           = sigma

    def predict(self, x, n, alpha, batch_size):
        """
        Predict the class of x using majority vote over n noisy samples.
        Returns ABSTAIN if the top class cannot be certified at
        confidence level 1-alpha.

        Args:
            x          : input image, shape (3, 32, 32), in [0, 1]
            n          : number of Monte Carlo samples
            alpha      : failure probability (e.g. 0.001)
            batch_size : how many noisy copies to evaluate at once

        Returns:
            predicted class (int) or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2   = counts.argsort()[::-1][:2]

        count1 = counts[top2[0]]
        count2 = counts[top2[1]]

        # One-sided binomial test: is count1 significantly > n/2?
        if binomtest(count1, n=n, p=0.5, alternative='greater').pvalue > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def certify(self, x, n0, n, alpha, batch_size):
        """
        Certify the prediction for x and return the certified L2 radius.

        Two-phase procedure:
          Phase 1 (n0 samples): find the top class c_A
          Phase 2 (n  samples): lower-bound p_A using Clopper-Pearson interval

        Args:
            x          : input image, shape (3, 32, 32), in [0, 1]
            n0         : samples for selection phase (e.g. 100)
            n          : samples for estimation phase (e.g. 1000)
            alpha      : total failure probability (e.g. 0.001)
            batch_size : forward pass batch size

        Returns:
            (predicted_class, certified_radius)
            predicted_class = ABSTAIN if certification fails
            certified_radius = 0.0 if abstaining
        """
        self.base_classifier.eval()

        # Phase 1 — find top class
        counts0 = self._sample_noise(x, n0, batch_size)
        c_A     = counts0.argmax().item()

        # Phase 2 — estimate p_A with lower confidence bound
        counts  = self._sample_noise(x, n, batch_size)
        count_A = counts[c_A].item()

        # Clopper-Pearson lower bound on p_A at confidence 1 - alpha
        p_A_lower = self._lower_confidence_bound(count_A, n, alpha)

        if p_A_lower < 0.5:
            # Cannot certify — p_A not significantly above 0.5
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(p_A_lower)
            return c_A, radius

    def _sample_noise(self, x, num, batch_size):
        """
        Run the base classifier on num noisy copies of x.
        Returns a count array of shape (num_classes,).
        """
        counts = np.zeros(self.num_classes, dtype=int)

        with torch.no_grad():
            for _ in range(int(np.ceil(num / batch_size))):
                this_batch = min(batch_size, num - counts.sum())
                if this_batch <= 0:
                    break

                # Add Gaussian noise
                batch = x.repeat(this_batch, 1, 1, 1)
                noise = torch.randn_like(batch) * self.sigma
                noisy = torch.clamp(batch + noise, 0, 1)

                # Forward pass
                outputs     = self.base_classifier(noisy.to(DEVICE))
                predictions = outputs.argmax(dim=1)

                # Accumulate votes
                for pred in predictions.cpu().numpy():
                    counts[pred] += 1

        return counts

    def _lower_confidence_bound(self, count_A, n, alpha):
        """
        Clopper-Pearson lower bound on binomial proportion.
        Returns the lower end of a one-sided (1-alpha) confidence interval.
        """
        return proportion_confint(count_A, n, alpha=2*alpha, method="beta")[0]