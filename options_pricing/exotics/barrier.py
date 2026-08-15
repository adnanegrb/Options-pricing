"""
Barrier options: knock-out and knock-in, up and down.

A barrier option is a vanilla option that only pays off if the
underlying does (or doesn't) touch a barrier level H at some point
before maturity. Four combinations exist:
  - up-and-out:  alive unless S touches H from below (H > S0)
  - up-and-in:   alive only if S touches H from below
  - down-and-out: alive unless S touches H from above (H < S0)
  - down-and-in:  alive only if S touches H from above

Two facts make these easier than they look:

1. In-out parity: knock-in + knock-out = vanilla (same strike, same
   maturity). A path either crosses the barrier or it doesn't — those
   two cases are exhaustive and mutually exclusive, so the in-option
   and out-option payoffs always sum to the vanilla payoff. This means
   we only need to derive one of the two analytically; the other comes
   for free by subtracting from the Black-Scholes price.

2. The analytical price for a barrier uses the reflection principle:
   for a Brownian motion, the probability of touching a level H and
   then ending up at some point y afterward equals the probability of
   the "reflected" path — you can compute crossing probabilities using
   ordinary Black-Scholes-type terms evaluated at the barrier level,
   rather than needing to track the whole path.
"""

from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np

from options_pricing.models.base import OptionParams
from options_pricing.models.black_scholes import BlackScholes


class BarrierOption:
    def __init__(self, params: OptionParams, barrier: float, barrier_type: str, direction: str):
        """
        barrier_type: "knock_out" or "knock_in"
        direction: "up" or "down"
        """
        self.p = params
        self.H = barrier
        self.barrier_type = barrier_type
        self.direction = direction

        if direction == "up" and barrier <= params.S:
            raise ValueError("up barrier must be above current spot")
        if direction == "down" and barrier >= params.S:
            raise ValueError("down barrier must be below current spot")

    def price(self, method: str = "analytical", n_paths: int = 100_000, n_steps: int = 252) -> float:
        if method == "analytical":
            return self._price_analytical()
        elif method == "mc":
            return self._price_mc(n_paths, n_steps)
        else:
            raise ValueError("method must be 'analytical' or 'mc'")

    def _price_analytical(self) -> float:
        # Only up-and-out and down-and-out are derived directly here.
        # The "in" versions are recovered via in-out parity: in = vanilla - out.
        vanilla = BlackScholes(self.p).price()

        if self.barrier_type == "knock_out":
            return self._out_price()
        else:
            return vanilla - self._out_price()

    def _out_price(self) -> float:
        """
        Closed-form price for a knock-out barrier call, using the
        Reiner-Rubinstein (1991) formula. This is the standard reference
        result for barrier options under Black-Scholes; it's derived from
        the reflection principle but the bookkeeping of terms is easy to
        get wrong by re-deriving from scratch, so the implementation
        follows the published parametrization directly and is checked
        against Monte Carlo in the tests.

        Only calls are handled analytically here. Puts fall back to
        Monte Carlo (method="mc") — the put formulas use the same
        reflection idea but with a different set of terms, and are
        omitted to keep the one analytical case fully defensible rather
        than having two half-checked ones.
        """
        p = self.p
        if p.option_type != "call":
            raise NotImplementedError(
                "analytical barrier pricing is only implemented for calls; "
                "use method='mc' for puts"
            )

        S, K, T, r, q, sigma, H = p.S, p.K, p.T, p.r, p.q, p.sigma, self.H

        # b is the cost of carry: r for a non-dividend-paying stock,
        # r - q with a continuous dividend yield.
        b = r - q
        mu = (b - 0.5 * sigma ** 2) / sigma ** 2
        sig_sqrt_T = sigma * sqrt(T)
        growth = exp((b - r) * T)  # = exp(-qT)

        # phi and eta are the direction flags from Haug's unified barrier
        # formula: phi = +1 for a call, -1 for a put; eta = -1 when the
        # barrier is above spot (up-barrier), +1 when it's below (down-
        # barrier). Getting eta's sign right is what actually distinguishes
        # an up-barrier from a down-barrier in these formulas — the x/y
        # terms themselves look identical either way.
        phi = 1.0  # calls only, per the docstring above
        eta = -1.0 if self.direction == "up" else 1.0

        def x1():
            return log(S / K) / sig_sqrt_T + (1 + mu) * sig_sqrt_T

        def x2():
            return log(S / H) / sig_sqrt_T + (1 + mu) * sig_sqrt_T

        def y1():
            return log(H ** 2 / (S * K)) / sig_sqrt_T + (1 + mu) * sig_sqrt_T

        def y2():
            return log(H / S) / sig_sqrt_T + (1 + mu) * sig_sqrt_T

        A = phi * S * growth * norm.cdf(phi * x1()) - phi * K * exp(-r * T) * norm.cdf(phi * x1() - phi * sig_sqrt_T)
        B = phi * S * growth * norm.cdf(phi * x2()) - phi * K * exp(-r * T) * norm.cdf(phi * x2() - phi * sig_sqrt_T)
        C = phi * S * growth * (H / S) ** (2 * (mu + 1)) * norm.cdf(eta * y1()) \
            - phi * K * exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(eta * y1() - eta * sig_sqrt_T)
        D = phi * S * growth * (H / S) ** (2 * (mu + 1)) * norm.cdf(eta * y2()) \
            - phi * K * exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(eta * y2() - eta * sig_sqrt_T)

        if self.direction == "up":
            # up-and-out call
            if H <= K:
                # barrier below strike: option is already worthless once
                # it could ever be in the money, since it would have
                # knocked out first
                return 0.0
            return A - B + C - D
        else:
            # down-and-out call
            if H <= K:
                return A - C
            else:
                return B - D

    def _price_mc(self, n_paths: int, n_steps: int) -> float:
        """
        Simulate full paths (not just terminal prices, since barrier
        options depend on the whole path) and check the barrier
        condition at each monitoring step.

        Note on discrete monitoring bias: the analytical formula above
        assumes the barrier is monitored continuously, i.e. the option
        knocks out the instant the path touches H at any point in time.
        This simulation can only check the barrier at n_steps discrete
        points, so it can miss a path that crosses H and comes back
        between two monitoring dates. That means this estimate is biased
        towards the vanilla price relative to the true continuous-barrier
        value, and the bias shrinks as n_steps grows — with n_steps=100
        over one year (roughly weekly monitoring) expect on the order of
        a few percent of gap versus the analytical price; a few hundred
        steps closes most of it. The test suite's tolerance reflects this.
        """
        p = self.p
        dt = p.T / n_steps
        rng = np.random.default_rng(42)

        # Antithetic variates again, same idea as the vanilla MC model:
        # halve the random draws, mirror them, average the two.
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        z = np.concatenate([z, -z], axis=0)

        drift = (p.r - p.q - 0.5 * p.sigma ** 2) * dt
        diffusion = p.sigma * sqrt(dt) * z
        log_returns = drift + diffusion

        # Cumulative sum still builds the full (n_paths, n_steps) path
        # matrix, which is the memory-heavy part for large n_steps —
        # kept simple here since it's what makes "check the barrier at
        # every step" straightforward to read; a production version
        # would walk forward step by step and discard finished paths.
        log_paths = np.log(p.S) + np.cumsum(log_returns, axis=1)
        paths = np.exp(log_paths)

        if self.direction == "up":
            touched = (paths >= self.H).any(axis=1)
        else:
            touched = (paths <= self.H).any(axis=1)

        s_t = paths[:, -1]
        del paths, log_paths, log_returns  # free the big matrix before the payoff step
        if p.option_type == "call":
            payoff = np.maximum(s_t - p.K, 0.0)
        else:
            payoff = np.maximum(p.K - s_t, 0.0)

        if self.barrier_type == "knock_out":
            payoff = np.where(touched, 0.0, payoff)
        else:
            payoff = np.where(touched, payoff, 0.0)

        return float(np.exp(-p.r * p.T) * payoff.mean())
