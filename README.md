# Options Pricing Library

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

A small options pricing library covering the models most likely to come up in a quant interview: Black-Scholes, Monte Carlo, a binomial tree, the five standard Greeks, and one exotic payoff (barrier options).

I built this to go deep rather than wide. Every model here is cross checked against at least one other independent method, and every number the tests assert is something I can actually derive on paper, not just something that happened to come out when I ran the code. If you want the math behind each model, `NOTES.md` has it, including a real bug I hit (and fixed) while building the barrier pricer.

## What's here

**Black-Scholes.** Closed-form, with a continuous dividend yield.

**Monte Carlo.** GBM simulation with antithetic variates for variance reduction, plus a confidence interval on the price estimate.

**Binomial tree (CRR).** European and American exercise, used to show convergence to Black-Scholes as the number of steps grows.

**Greeks.** Delta, gamma, theta, vega, rho, computed both analytically and via bump and reprice, cross checked against each other.

**Barrier options.** Up and down, knock in and knock out, priced both by a closed-form formula (calls only) and by Monte Carlo.

## Quick start

```bash
pip install -r requirements.txt
```

```python
from options_pricing.models.base import OptionParams
from options_pricing.models.black_scholes import BlackScholes
from options_pricing.models.monte_carlo import MonteCarlo
from options_pricing.greeks.analytical import AnalyticalGreeks

p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")

BlackScholes(p).price()
MonteCarlo(p, n_paths=100_000).price()
AnalyticalGreeks(p).all_greeks()
```

Barrier option:

```python
from options_pricing.exotics.barrier import BarrierOption

barrier = BarrierOption(p, barrier=120, barrier_type="knock_out", direction="up")
barrier.price("analytical")
barrier.price("mc")
```

## Tests

```bash
pytest tests/ -v
```

21 tests, mostly structured around checking models against each other rather than checking fixed numbers. Put-call parity, Monte Carlo and binomial convergence to Black-Scholes, analytical Greeks against numerical Greeks, and barrier in-out parity (knock-in plus knock-out equals vanilla).

## Structure

```
options_pricing/
├── models/
│   ├── base.py               shared OptionParams
│   ├── black_scholes.py      closed-form pricing
│   ├── monte_carlo.py        GBM simulation, antithetic variates
│   └── binomial_tree.py      CRR tree, European and American
├── greeks/
│   ├── analytical.py         closed-form Greeks
│   └── numerical.py          bump-and-reprice Greeks
└── exotics/
    └── barrier.py            barrier options, analytical + MC
```

## A known limitation, on purpose

The analytical barrier formula only covers calls. Puts are priced via Monte Carlo instead. I could've coded up the put formula too, it's a similar structure with different terms, but I'd rather have one case I can fully explain than two I'd have to half explain under pressure. `NOTES.md` goes into why.

## License

MIT
