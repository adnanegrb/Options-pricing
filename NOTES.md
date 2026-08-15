# Notes — derivations to know cold

This file is not documentation of the code. It's the math you should be
able to reproduce on a whiteboard without looking anything up, because
this is what an interviewer will actually probe. Read it *before* you
read the code, not after — understanding the "why" first makes the code
obvious; reading the code first just gives you syntax to memorize.

---

## 1. Black-Scholes PDE → closed form

Start from the replication argument: hold Δ shares of stock and borrow
cash so that the portfolio Δ·S − B replicates the option over the next
instant. Requiring the portfolio to be riskless (its return must equal
r, or there's arbitrage) gives the Black-Scholes PDE:

    ∂V/∂t + ½σ²S²·∂²V/∂S² + rS·∂V/∂S − rV = 0

Solving this PDE with the terminal condition V(S,T) = max(S−K, 0) gives
the closed form:

    C = S·N(d1) − K·e^(−rT)·N(d2)

**Why N(d1) and N(d2) show up, intuitively:** N(d2) is the risk-neutral
probability the option finishes in the money — it's the "probability of
paying K" weight on the discounted strike. N(d1) is a probability-like
term too, but under the *stock-as-numeraire* measure rather than the
cash measure — think of it as "the expected fraction of S you keep,
weighted by the paths that finish in the money."

**Delta derivation you should be able to do live:**
∂C/∂S = N(d1) + S·φ(d1)·∂d1/∂S − K·e^(−rT)·φ(d2)·∂d2/∂S

The two φ terms cancel exactly (this is the non-obvious step — it
follows from S·φ(d1) = K·e^(−rT)·φ(d2), which itself follows from the
definitions of d1 and d2), leaving Δ = N(d1). If an interviewer asks
"why do the correction terms vanish," that cancellation is the answer.

---

## 2. Why antithetic variates reduce variance

Monte Carlo error scales like σ_estimator / √n — to halve the error you
need 4x the paths. Antithetic variates is a cheap way to cut the
variance without more random draws.

For each standard normal Z, also compute the payoff using −Z. If f is
the discounted payoff function, we're averaging f(Z) and f(−Z) instead
of two independent draws. The variance of the *average* of two random
variables X and Y is:

    Var((X+Y)/2) = ¼[Var(X) + Var(Y) + 2·Cov(X,Y)]

If f is monotonic in Z (true for a call or put payoff, since S_T is
monotonic in Z), then f(Z) and f(−Z) are *negatively* correlated —
when one is high the other tends to be low. That makes Cov(X,Y) < 0,
which directly shrinks the variance of the average versus using two
independent samples. That's the whole mechanism — no more, no less.

---

## 3. Binomial tree → Black-Scholes convergence

The CRR tree picks u = e^(σ√Δt), d = 1/u so that the tree's one-step
log-return has variance σ²Δt — matching the variance of GBM's log-return
over the same interval. As n → ∞ (Δt → 0), the discrete binomial walk
converges (in distribution, by the Central Limit Theorem — many small
independent up/down steps summing to something Gaussian) to the same
lognormal terminal distribution that Black-Scholes assumes. That's why
the binomial price converges to Black-Scholes: they're two ways of
representing the same limiting stochastic process.

This is exactly the answer to "why does discrete hedging converge to
continuous Black-Scholes hedging as you rebalance more frequently."

---

## 4. Barrier options: in-out parity and reflection

**In-out parity** (should be instant, not derived): every path either
touches the barrier or it doesn't — mutually exclusive and exhaustive.
So knock-in payoff + knock-out payoff = vanilla payoff, for every
single path, which means it holds for the discounted expectations too:

    C_in + C_out = C_vanilla

This is a *model-free* identity — true under any dynamics, not just
GBM — because it only uses "the path did or didn't cross," not any
specific distributional assumption.

**Reflection principle**, informally: for a driftless Brownian motion
started at 0, the probability that it hits a level H and is at some
point y afterward equals the probability of a "mirrored" path — reflect
everything after the hitting time through H. This lets you replace "the
probability of touching H and ending up at y" with an ordinary
(un-reflected) Brownian motion probability, evaluated at a mirrored
endpoint (like H²/S instead of S). That's the mechanical trick behind
every term in the closed-form barrier formulas: they're vanilla
Black-Scholes-shaped expressions, but some evaluated at the real spot S
and some evaluated at the "reflected" level H²/S.

**Why the code's analytical barrier price uses this specific
Reiner-Rubinstein A/B/C/D parametrization** rather than deriving from
scratch: the full derivation with a nonzero drift (r − q ≠ 0, which is
the realistic case) requires a Girsanov change of measure to remove the
drift before applying reflection, then transforming back — doable, but
long enough that re-deriving it from first principles under interview
time pressure is unrealistic. What you should be able to say instead:
"the price is built from vanilla Black-Scholes terms evaluated at the
real strike/spot and at the barrier-reflected strike/spot, weighted by
a power of (H/S) that comes from the Girsanov drift adjustment" — that
sentence demonstrates you understand *why* the formula has this shape,
without requiring you to reproduce four pages of algebra live.

---

## 5. Debugging story worth having ready

When this barrier formula was first implemented, it priced an
up-and-out call as **negative** — a clear sign something was wrong,
since option prices can't be negative. Two real bugs were found by
comparing against an independently-coded Monte Carlo simulation:

1. The cost-of-carry term b = r − q wasn't applied consistently to
   every S-dependent term in the formula.
2. The formula has a direction flag (η, "eta") that flips sign depending
   on whether the barrier is above or below spot — up-barrier and
   down-barrier cases share the same-looking terms but need opposite
   signs in a few places, and that flag was missing entirely.

The fix was verified by checking the down-and-out case (a cleaner
diagnostic than up-and-out, since it doesn't hit the H ≤ K edge case)
against an independent Monte Carlo — they now agree to within simulation
noise. There's also a residual, *expected* small gap between the
analytical price and the Monte Carlo price: the analytical formula
assumes continuous barrier monitoring, while the simulation can only
check the barrier at discrete time steps, so it slightly underprices
knock-outs unless the step count is large. That gap and why it exists
is documented directly in `barrier.py`.

This is a good story to tell in an interview if asked "walk me through
a bug you found" — it shows you validate numerically, not just trust a
formula, and that you know the difference between a coding bug and an
expected discretization effect.
