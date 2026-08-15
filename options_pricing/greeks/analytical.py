"""
Analytical Greeks: delta, gamma, theta, vega, rho.

Each Greek is a partial derivative of the Black-Scholes price with
respect to one input. These have closed forms because Black-Scholes
itself has a closed form — differentiate C = S*N(d1) - K*e^(-rT)*N(d2)
with respect to whatever you want, and (after the dust settles, using
that d(N(d1))/dS terms cancel against d(N(d2))/dS terms) you get the
formulas below.

Delta is the one you should be able to derive on a whiteboard without
hesitation: dC/dS = N(d1). The intuition — delta is the number of
shares you'd hold to replicate the option's short-term price movement,
which is exactly the hedge ratio in the Black-Scholes replication
argument.
"""

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
        # Sensitivity of price to a $1 move in the underlying.
        if p.option_type == "call":
            return exp(-p.q * p.T) * norm.cdf(self.d1)
        else:
            return -exp(-p.q * p.T) * norm.cdf(-self.d1)

    def gamma(self) -> float:
        p = self.p
        # Sensitivity of delta to a $1 move in the underlying. Same
        # formula for calls and puts — gamma doesn't depend on option_type
        # because delta_call - delta_put = e^(-qT) is constant in S.
        return exp(-p.q * p.T) * norm.pdf(self.d1) / (p.S * p.sigma * sqrt(p.T))

    def theta(self) -> float:
        p = self.p
        # Sensitivity of price to the passage of time (time decay).
        # Expressed here per year; divide by 365 for a "per day" figure,
        # which is the convention most desks actually quote.
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
        # Sensitivity of price to a 1-unit (i.e. 100 vol point) move in
        # sigma. Same formula for calls and puts — vega doesn't depend on
        # option_type since put-call parity has no sigma dependence.
        return p.S * exp(-p.q * p.T) * norm.pdf(self.d1) * sqrt(p.T)

    def rho(self) -> float:
        p = self.p
        # Sensitivity of price to a 1-unit move in the risk-free rate.
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
