"""
Black-Scholes closed-form pricing.

This is the model everything else in the library gets checked against.
The core idea: under the risk-neutral measure, the discounted stock price
is a martingale, and the option price is the discounted expected payoff.
For a call/put on lognormal GBM, that expectation has a closed form —
we don't need simulation or a PDE solver, just N(d1) and N(d2).

With a continuous dividend yield q, the forward price of the stock is
S * exp((r - q) * T) instead of S * exp(r * T), which is why q shows up
subtracted from r everywhere below.
"""

from math import log, sqrt, exp
from scipy.stats import norm

from options_pricing.models.base import OptionParams


class BlackScholes:
    def __init__(self, params: OptionParams):
        self.p = params
        self._d1, self._d2 = self._compute_d1_d2()

    def _compute_d1_d2(self):
        p = self.p
        # d1 measures, roughly, how far in-the-money the option is expected
        # to be at expiry, scaled by volatility. d2 = d1 - sigma*sqrt(T) is
        # the same quantity but for the risk-neutral probability of the
        # option finishing in the money (used in N(d2) below).
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
            # Put price via the same d1/d2, using N(-d) instead of N(d).
            # This is equivalent to deriving it from put-call parity:
            # C - P = discounted_spot - discounted_strike.
            return discounted_strike * norm.cdf(-self._d2) - discounted_spot * norm.cdf(-self._d1)

    def d1(self) -> float:
        return self._d1

    def d2(self) -> float:
        return self._d2
