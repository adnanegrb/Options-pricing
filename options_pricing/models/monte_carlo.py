"""
Monte Carlo pricing under geometric Brownian motion.

The idea: simulate many possible terminal stock prices S_T under the
risk-neutral measure, compute the payoff for each one, average them,
and discount back to today. As the number of paths grows, this average
converges to the true risk-neutral expectation (law of large numbers).

Two variance reduction tricks are used here, both because a raw Monte
Carlo estimate is noisy and slow to converge (error shrinks like
1/sqrt(n_paths), so cutting the error in half needs 4x the paths):

1. Antithetic variates: for every random draw Z, we also use -Z. Since
   Z and -Z are negatively correlated, averaging the payoff from both
   reduces the variance of the estimator without needing extra random
   draws — we effectively get two paths' worth of signal from one
   random number.

2. Control variates: not used here directly (the closed-form BS price
   IS the control target we validate against in tests), but worth
   knowing the general idea — subtract off a correlated quantity whose
   expectation you already know exactly, which cancels out shared noise.
"""

import numpy as np

from options_pricing.models.base import OptionParams


class MonteCarlo:
    def __init__(self, params: OptionParams, n_paths: int = 100_000, seed: int = 42):
        self.p = params
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def _simulate_terminal_prices(self) -> np.ndarray:
        p = self.p

        # Antithetic variates: draw half the random numbers, then mirror
        # them. Z and -Z together give n_paths total simulated prices.
        half = self.n_paths // 2
        z = self.rng.standard_normal(half)
        z_antithetic = np.concatenate([z, -z])

        drift = (p.r - p.q - 0.5 * p.sigma ** 2) * p.T
        diffusion = p.sigma * np.sqrt(p.T) * z_antithetic

        return p.S * np.exp(drift + diffusion)

    def price(self) -> float:
        p = self.p
        s_t = self._simulate_terminal_prices()

        if p.option_type == "call":
            payoffs = np.maximum(s_t - p.K, 0.0)
        else:
            payoffs = np.maximum(p.K - s_t, 0.0)

        return np.exp(-p.r * p.T) * payoffs.mean()

    def price_with_confidence_interval(self, confidence: float = 0.95) -> tuple[float, float, float]:
        """
        Returns (price, lower_bound, upper_bound) for the given confidence
        level, using the standard error of the discounted payoffs.
        """
        p = self.p
        s_t = self._simulate_terminal_prices()

        if p.option_type == "call":
            payoffs = np.maximum(s_t - p.K, 0.0)
        else:
            payoffs = np.maximum(p.K - s_t, 0.0)

        discounted = np.exp(-p.r * p.T) * payoffs
        estimate = discounted.mean()
        std_error = discounted.std(ddof=1) / np.sqrt(len(discounted))

        from scipy.stats import norm
        z_score = norm.ppf(0.5 + confidence / 2)

        return estimate, estimate - z_score * std_error, estimate + z_score * std_error
