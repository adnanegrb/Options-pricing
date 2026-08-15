"""
Shared parameters for every pricing model in this library.

Every model below (Black-Scholes, Monte Carlo, Binomial) takes the same
five numbers as input: spot, strike, time to maturity, risk-free rate,
volatility. Bundling them in one dataclass means we only define the
option contract once, and every model just reads from it.
"""

from dataclasses import dataclass


@dataclass
class OptionParams:
    S: float          # spot price of the underlying today
    K: float           # strike price
    T: float           # time to maturity, in years
    r: float           # risk-free rate (continuously compounded)
    sigma: float        # volatility of the underlying (annualized)
    q: float = 0.0       # continuous dividend yield, defaults to 0
    option_type: str = "call"  # "call" or "put"

    def __post_init__(self):
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
