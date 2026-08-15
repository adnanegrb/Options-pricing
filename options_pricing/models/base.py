from dataclasses import dataclass


@dataclass
class OptionParams:
    S: float          # spot price of the underlying today
    K: float           # strike price
    T: float           # time to maturity, in years
    r: float           # risk-free rate (continuously compounded)
    sigma: float        # volatility of the underlying (annualized)
    q: float = 0.0       
    option_type: str = "call"  

    def __post_init__(self):
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
