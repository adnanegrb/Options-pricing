# Notes: derivations to know cold

This file isn't documentation of the code. It's the math I should be able to reproduce on a whiteboard without looking anything up, because that's what an interviewer will actually probe. Read this before reading the code, not after. Understanding the why first makes the code obvious. Reading the code first just gives you syntax to memorize.

## 1. Black-Scholes PDE to closed form

Start from the replication argument: hold $\Delta$ shares of stock and borrow cash so that the portfolio $\Delta S - B$ replicates the option over the next instant. Requiring the portfolio to be riskless (its return must equal $r$, or there's arbitrage) gives the Black-Scholes PDE:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

Solving this PDE with the terminal condition $V(S,T) = \max(S-K, 0)$ gives the closed form:

$$C = S N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

Why $N(d_1)$ and $N(d_2)$ show up, intuitively: $N(d_2)$ is the risk neutral probability the option finishes in the money, it's the probability of paying $K$ weight on the discounted strike. $N(d_1)$ is a probability-like term too, but under the stock-as-numeraire measure rather than the cash measure. Think of it as the expected fraction of $S$ you keep, weighted by the paths that finish in the money.

Delta derivation I should be able to do live:

$$\frac{\partial C}{\partial S} = N(d_1) + S\phi(d_1)\frac{\partial d_1}{\partial S} - Ke^{-rT}\phi(d_2)\frac{\partial d_2}{\partial S}$$

The two $\phi$ terms cancel exactly. That's the non-obvious step, it follows from $S\phi(d_1) = Ke^{-rT}\phi(d_2)$, which itself follows from the definitions of $d_1$ and $d_2$. Leaves $\Delta = N(d_1)$. If an interviewer asks why the correction terms vanish, that cancellation is the answer.

## 2. Why antithetic variates reduce variance

Monte Carlo error scales like $\sigma_{\text{estimator}} / \sqrt{n}$, to halve the error you need 4x the paths. Antithetic variates is a cheap way to cut the variance without more random draws.

For each standard normal $Z$, also compute the payoff using $-Z$. If $f$ is the discounted payoff function, we're averaging $f(Z)$ and $f(-Z)$ instead of two independent draws. The variance of the average of two random variables $X$ and $Y$ is:

$$\text{Var}\left(\frac{X+Y}{2}\right) = \frac{1}{4}\left[\text{Var}(X) + \text{Var}(Y) + 2\,\text{Cov}(X,Y)\right]$$

If $f$ is monotonic in $Z$ (true for a call or put payoff, since $S_T$ is monotonic in $Z$), then $f(Z)$ and $f(-Z)$ are negatively correlated, when one is high the other tends to be low. That makes $\text{Cov}(X,Y) < 0$, which directly shrinks the variance of the average versus using two independent samples. That's the whole mechanism, no more, no less.

## 3. Binomial tree convergence to Black-Scholes

The CRR tree picks

$$u = e^{\sigma\sqrt{\Delta t}}, \qquad d = \frac{1}{u}$$

so that the tree's one-step log return has variance $\sigma^2 \Delta t$, matching the variance of GBM's log return over the same interval. As $n \to \infty$ ($\Delta t \to 0$), the discrete binomial walk converges (in distribution, by the Central Limit Theorem, many small independent up/down steps summing to something Gaussian) to the same lognormal terminal distribution that Black-Scholes assumes. That's why the binomial price converges to Black-Scholes, they're two ways of representing the same limiting stochastic process.

This is exactly the answer to "why does discrete hedging converge to continuous Black-Scholes hedging as you rebalance more frequently."

## 4. Barrier options: in-out parity and reflection

In-out parity should be instant, not derived. Every path either touches the barrier or it doesn't, mutually exclusive and exhaustive. So knock-in payoff plus knock-out payoff equals vanilla payoff, for every single path, which means it holds for the discounted expectations too:

$$C_{\text{in}} + C_{\text{out}} = C_{\text{vanilla}}$$

This is a model-free identity, true under any dynamics, not just GBM, because it only uses "the path did or didn't cross," not any specific distributional assumption.

Reflection principle, informally: for a driftless Brownian motion started at 0, the probability that it hits a level $H$ and is at some point $y$ afterward equals the probability of a mirrored path, reflect everything after the hitting time through $H$. This lets you replace "the probability of touching $H$ and ending up at $y$" with an ordinary (un-reflected) Brownian motion probability, evaluated at a mirrored endpoint like $H^2/S$ instead of $S$. That's the mechanical trick behind every term in the closed-form barrier formulas: they're vanilla Black-Scholes shaped expressions, but some evaluated at the real spot $S$ and some evaluated at the reflected level $H^2/S$.

The library's analytical barrier price uses the Reiner-Rubinstein parametrization, built from four terms:

$$A = \phi S e^{-qT} N(\phi x_1) - \phi K e^{-rT} N(\phi x_1 - \phi\sigma\sqrt{T})$$

$$C = \phi S e^{-qT}\left(\frac{H}{S}\right)^{2(\mu+1)} N(\eta y_1) - \phi K e^{-rT}\left(\frac{H}{S}\right)^{2\mu} N(\eta y_1 - \eta\sigma\sqrt{T})$$

where $\mu = (r - q - \sigma^2/2)/\sigma^2$, $\phi = 1$ for a call, and $\eta = -1$ for an up barrier, $+1$ for a down barrier. Why this specific shape rather than deriving from scratch: the full derivation with a nonzero drift ($r - q \neq 0$, the realistic case) requires a Girsanov change of measure to remove the drift before applying reflection, then transforming back. Doable, but long enough that re-deriving it from first principles under interview time pressure isn't realistic. What I should be able to say instead: the price is built from vanilla Black-Scholes terms evaluated at the real strike and spot and at the barrier-reflected strike and spot, weighted by a power of $H/S$ that comes from the Girsanov drift adjustment. That sentence demonstrates understanding of why the formula has this shape, without requiring reproducing four pages of algebra live.

## 5. Debugging story worth having ready

When this barrier formula was first implemented, it priced an up-and-out call as negative, a clear sign something was wrong since option prices can't be negative. Two real bugs were found by comparing against an independently coded Monte Carlo simulation.

First, the cost of carry term $b = r - q$ wasn't applied consistently to every $S$-dependent term in the formula. Second, the formula has a direction flag $\eta$ that flips sign depending on whether the barrier is above or below spot. Up-barrier and down-barrier cases share the same-looking terms but need opposite signs in a few places, and that flag was missing entirely.

The fix was verified by checking the down-and-out case (a cleaner diagnostic than up-and-out, since it doesn't hit the $H \leq K$ edge case) against an independent Monte Carlo, they now agree to within simulation noise. There's also a residual, expected small gap between the analytical price and the Monte Carlo price: the analytical formula assumes continuous barrier monitoring, while the simulation can only check the barrier at discrete time steps, so it slightly underprices knock-outs unless the step count is large. That gap and why it exists is documented directly in `barrier.py`.

This is a good story to tell in an interview if asked to walk through a bug I found. It shows I validate numerically, not just trust a formula, and that I know the difference between a coding bug and an expected discretization effect.
