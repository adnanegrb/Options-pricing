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
        
        return (up - 2 * mid + down) / (h ** 2)

    def vega(self) -> float:
        p, h = self.p, self.h
        up = self._price_with(sigma=p.sigma + h)
        down = self._price_with(sigma=p.sigma - h)
        return (up - down) / (2 * h)

    def theta(self) -> float:
        p, h = self.p, self.h
        
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
