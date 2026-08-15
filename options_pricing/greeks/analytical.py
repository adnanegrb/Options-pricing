from math import exp, sqrt
from scipy.stats import norm

from options_pricing.models.base import OptionParams
from options_pricing.models.black_scholes import BlackScholes


class AnalyticalGreeks:
    def __init__(self, params: OptionParams):
        self.p = params
        self.bs = BlackScholes(params)
        self.d1 = self.bs.d1()
        self.d2 = self.bs.d2()

    def delta(self) -> float:
        p = self.p
        
        if p.option_type == "call":
            return exp(-p.q * p.T) * norm.cdf(self.d1)
        else:
            return -exp(-p.q * p.T) * norm.cdf(-self.d1)

    def gamma(self) -> float:
        p = self.p
        
        return exp(-p.q * p.T) * norm.pdf(self.d1) / (p.S * p.sigma * sqrt(p.T))

    def theta(self) -> float:
        p = self.p
       
        term1 = -(p.S * exp(-p.q * p.T) * norm.pdf(self.d1) * p.sigma) / (2 * sqrt(p.T))
        if p.option_type == "call":
            term2 = -p.r * p.K * exp(-p.r * p.T) * norm.cdf(self.d2)
            term3 = p.q * p.S * exp(-p.q * p.T) * norm.cdf(self.d1)
        else:
            term2 = p.r * p.K * exp(-p.r * p.T) * norm.cdf(-self.d2)
            term3 = -p.q * p.S * exp(-p.q * p.T) * norm.cdf(-self.d1)
        return term1 + term2 + term3

    def vega(self) -> float:
        p = self.p
        
        return p.S * exp(-p.q * p.T) * norm.pdf(self.d1) * sqrt(p.T)

    def rho(self) -> float:
        p = self.p
        
        if p.option_type == "call":
            return p.K * p.T * exp(-p.r * p.T) * norm.cdf(self.d2)
        else:
            return -p.K * p.T * exp(-p.r * p.T) * norm.cdf(-self.d2)

    def all_greeks(self) -> dict:
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "theta": self.theta(),
            "vega": self.vega(),
            "rho": self.rho(),
        }
