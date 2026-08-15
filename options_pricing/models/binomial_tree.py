import numpy as np

from options_pricing.models.base import OptionParams


class BinomialTree:
    def __init__(self, params: OptionParams, n_steps: int = 500, american: bool = False):
        self.p = params
        self.n_steps = n_steps
        self.american = american

    def price(self) -> float:
        p = self.p
        n = self.n_steps
        dt = p.T / n

        
        u = np.exp(p.sigma * np.sqrt(dt))
        d = 1.0 / u
        growth = np.exp((p.r - p.q) * dt)
        prob_up = (growth - d) / (u - d)

        if not (0.0 < prob_up < 1.0):
            raise ValueError(
                f"risk-neutral probability {prob_up:.4f} is out of (0,1) — "
                "n_steps is too small relative to T and sigma, arbitrage "
                "appears in the discretized tree"
            )

        
        j = np.arange(n + 1)
        terminal_prices = p.S * (u ** j) * (d ** (n - j))

        if p.option_type == "call":
            values = np.maximum(terminal_prices - p.K, 0.0)
        else:
            values = np.maximum(p.K - terminal_prices, 0.0)

        discount = np.exp(-p.r * dt)

        
        for step in range(n - 1, -1, -1):
            values = discount * (prob_up * values[1:] + (1 - prob_up) * values[:-1])

            if self.american:
                
                j = np.arange(step + 1)
                stock_at_step = p.S * (u ** j) * (d ** (step - j))
                if p.option_type == "call":
                    exercise_value = np.maximum(stock_at_step - p.K, 0.0)
                else:
                    exercise_value = np.maximum(p.K - stock_at_step, 0.0)
                values = np.maximum(values, exercise_value)

        return float(values[0])
