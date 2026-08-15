from math import log, sqrt, exp
from scipy.stats import norm

from options_pricing.models.base import OptionParams


class BlackScholes:
    def __init__(self, params: OptionParams):
        self.p = params
        self._d1, self._d2 = self._compute_d1_d2()

    def _compute_d1_d2(self):
        p = self.p
        
        d1 = (log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma ** 2) * p.T) / (p.sigma * sqrt(p.T))
        d2 = d1 - p.sigma * sqrt(p.T)
        return d1, d2

    def price(self) -> float:
        p = self.p
        discounted_spot = p.S * exp(-p.q * p.T)
        discounted_strike = p.K * exp(-p.r * p.T)

        if p.option_type == "call":
            return discounted_spot * norm.cdf(self._d1) - discounted_strike * norm.cdf(self._d2)
        else:
            
            return discounted_strike * norm.cdf(-self._d2) - discounted_spot * norm.cdf(-self._d1)

    def d1(self) -> float:
        return self._d1

    def d2(self) -> float:
        return self._d2
