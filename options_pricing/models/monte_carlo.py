import numpy as np

from options_pricing.models.base import OptionParams


class MonteCarlo:
    def __init__(self, params: OptionParams, n_paths: int = 100_000, seed: int = 42):
        self.p = params
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def _simulate_terminal_prices(self) -> np.ndarray:
        p = self.p

        
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
