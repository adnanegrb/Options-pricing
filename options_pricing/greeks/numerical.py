"""
Numerical Greeks via bump-and-reprice.

The idea: instead of differentiating the pricing formula by hand, just
perturb an input by a small amount h, reprice, and take the finite
difference. This is slower and less precise than the analytical
formulas, but it's a completely independent check — if a sign got
flipped or a term got dropped in the analytical derivation, this
should catch it, which is exactly what the test suite uses it for.

Central difference (V(x+h) - V(x-h)) / (2h) is used instead of forward
difference (V(x+h) - V(x)) / h because it's second-order accurate in h
rather than first-order — the error shrinks like h^2 instead of h,
so a small bump gives a much closer answer to the true derivative.
"""

import copy
from options_pricing.models.black_scholes import BlackScholes
from options_pricing.models.base import OptionParams


class NumericalGreeks:
    def __init__(self, params: OptionParams, h: float = 1e-4):
        self.p = params
        self.h = h

    def _price_with(self, **overrides) -> float:
        bumped = copy.copy(self.p)
        for key, value in overrides.items():
            setattr(bumped, key, value)
        return BlackScholes(bumped).price()

    def delta(self) -> float:
        p, h = self.p, self.h
        up = self._price_with(S=p.S + h)
        down = self._price_with(S=p.S - h)
        return (up - down) / (2 * h)

    def gamma(self) -> float:
        p, h = self.p, self.h
        up = self._price_with(S=p.S + h)
        mid = BlackScholes(p).price()
        down = self._price_with(S=p.S - h)
        # Second derivative: central difference of the first derivative,
        # which reduces to this three-point formula.
        return (up - 2 * mid + down) / (h ** 2)

    def vega(self) -> float:
        p, h = self.p, self.h
        up = self._price_with(sigma=p.sigma + h)
        down = self._price_with(sigma=p.sigma - h)
        return (up - down) / (2 * h)

    def theta(self) -> float:
        p, h = self.p, self.h
        # Theta is decay as time passes, i.e. as T decreases, so this is
        # the negative of the derivative with respect to T.
        up = self._price_with(T=p.T + h)
        down = self._price_with(T=p.T - h)
        return -(up - down) / (2 * h)

    def rho(self) -> float:
        p, h = self.p, self.h
        up = self._price_with(r=p.r + h)
        down = self._price_with(r=p.r - h)
        return (up - down) / (2 * h)

    def all_greeks(self) -> dict:
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "theta": self.theta(),
            "vega": self.vega(),
            "rho": self.rho(),
        }
