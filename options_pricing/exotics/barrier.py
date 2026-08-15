
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
        
        vanilla = BlackScholes(self.p).price()

        if self.barrier_type == "knock_out":
            return self._out_price()
        else:
            return vanilla - self._out_price()

    def _out_price(self) -> float:
        
        p = self.p
        if p.option_type != "call":
            raise NotImplementedError(
                "analytical barrier pricing is only implemented for calls; "
                "use method='mc' for puts"
            )

        S, K, T, r, q, sigma, H = p.S, p.K, p.T, p.r, p.q, p.sigma, self.H

        
        b = r - q
        mu = (b - 0.5 * sigma ** 2) / sigma ** 2
        sig_sqrt_T = sigma * sqrt(T)
        growth = exp((b - r) * T)  # = exp(-qT)

        
        phi = 1.0  
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
        
        p = self.p
        dt = p.T / n_steps
        rng = np.random.default_rng(42)

        
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        z = np.concatenate([z, -z], axis=0)

        drift = (p.r - p.q - 0.5 * p.sigma ** 2) * dt
        diffusion = p.sigma * sqrt(dt) * z
        log_returns = drift + diffusion

        
        log_paths = np.log(p.S) + np.cumsum(log_returns, axis=1)
        paths = np.exp(log_paths)

        if self.direction == "up":
            touched = (paths >= self.H).any(axis=1)
        else:
            touched = (paths <= self.H).any(axis=1)

        s_t = paths[:, -1]
        del paths, log_paths, log_returns  
        if p.option_type == "call":
            payoff = np.maximum(s_t - p.K, 0.0)
        else:
            payoff = np.maximum(p.K - s_t, 0.0)

        if self.barrier_type == "knock_out":
            payoff = np.where(touched, 0.0, payoff)
        else:
            payoff = np.where(touched, payoff, 0.0)

        return float(np.exp(-p.r * p.T) * payoff.mean())
