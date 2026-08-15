"""
Cox-Ross-Rubinstein (CRR) binomial tree.

The idea: chop time to maturity into n small steps. At each step, the
stock either moves up by a factor u or down by a factor d. This turns
the continuous GBM process into a discrete tree, on which the option
price can be computed by backward induction: start at the known payoff
at maturity, and work backwards, at each node taking the discounted,
risk-neutral-weighted average of the two possible next nodes.

Why this matters beyond "another way to get a number": as n -> infinity
(step size -> 0), the binomial price converges to the Black-Scholes
price. That's not a coincidence — CRR is a discretization of the same
underlying GBM assumption, and this convergence is exactly what you'd
derive if asked "why does discrete hedging converge to Black-Scholes?"
in an interview.

The binomial tree also handles something Black-Scholes closed-form
can't: American-style early exercise. At every node we compare
"exercise now" vs "hold the option," and take the max.
"""

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

        # Up/down factors and risk-neutral probability. u = exp(sigma*sqrt(dt))
        # is chosen so that the tree's variance matches the GBM variance
        # sigma^2 * dt over each small step — that's the condition that
        # makes this converge to Black-Scholes as n grows.
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

        # Stock prices at maturity: after n steps, j up-moves and (n-j)
        # down-moves gives S * u^j * d^(n-j), for j = 0..n.
        j = np.arange(n + 1)
        terminal_prices = p.S * (u ** j) * (d ** (n - j))

        if p.option_type == "call":
            values = np.maximum(terminal_prices - p.K, 0.0)
        else:
            values = np.maximum(p.K - terminal_prices, 0.0)

        discount = np.exp(-p.r * dt)

        # Backward induction: at each step, collapse the tree by one layer.
        # values[i] holds the option value at each node of the current layer.
        for step in range(n - 1, -1, -1):
            values = discount * (prob_up * values[1:] + (1 - prob_up) * values[:-1])

            if self.american:
                # Recompute the stock price at each node of this layer and
                # compare immediate exercise against holding.
                j = np.arange(step + 1)
                stock_at_step = p.S * (u ** j) * (d ** (step - j))
                if p.option_type == "call":
                    exercise_value = np.maximum(stock_at_step - p.K, 0.0)
                else:
                    exercise_value = np.maximum(p.K - stock_at_step, 0.0)
                values = np.maximum(values, exercise_value)

        return float(values[0])
