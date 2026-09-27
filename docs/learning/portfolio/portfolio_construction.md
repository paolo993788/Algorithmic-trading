# Portfolio construction under estimation error: covariance shrinkage, risk budgeting and walk-forward rebalancing

Learning note for:

- [`backtest_engine.covariance`](../../../scripts/backtest_engine/backtest_engine/covariance.py): sample, EWMA and Ledoit-Wolf covariance;
- [`backtest_engine.allocation`](../../../scripts/backtest_engine/backtest_engine/allocation.py): allocation rules and Euler risk decomposition;
- [`backtest_engine.rebalance`](../../../scripts/backtest_engine/backtest_engine/rebalance.py): walk-forward rebalancing with drift, lag, costs and a volatility target;
- [`backtest_engine.universes`](../../../scripts/backtest_engine/backtest_engine/universes.py): the multi-asset universe.

The research note that applies them is [`notebooks/portfolio_construction/portfolio_construction_under_estimation_error.ipynb`](../../../notebooks/portfolio_construction/portfolio_construction_under_estimation_error.ipynb).

**Where this fits.** This is step 3 of the systematic-trading track ([curriculum](../README.md)).

- Step 1 made returns tradable.
- Step 2 made inference about them honest.
- This step combines assets, or strategies, into one portfolio. It explains why the textbook answer (mean-variance optimisation) fails with estimated inputs, and what practitioners do instead.

**Prerequisites** (section 15 has a self-check):

- matrix algebra: inverse, eigenvalues, positive definiteness;
- Lagrange multipliers;
- the variance of a sum of random variables;
- the Sharpe ratio;
- the bootstrap tests of step 2.

---

## 1. Intuition

You must split your money between a few assets. You know roughly how risky each is and how they move together. You know almost nothing about which will earn more.

Mean-variance optimisation asks for both. It then acts on its inputs with complete confidence. An asset whose expected return you overestimated by one percentage point gets a much larger weight; an asset whose risk you underestimated gets levered. The optimiser does not know which of its inputs are noise, so it concentrates on the noise. Michaud (1989) called optimisers "estimation-error maximisers".

There are three responses, and this note covers all three.

1. **Ignore the inputs you cannot estimate.**
   - 1/N ignores everything.
   - Inverse volatility, equal risk contribution (ERC) and minimum variance use risks only.
   - Maximum diversification uses risks and correlations.

   Each of these is the optimal portfolio under an implicit assumption about expected returns (section 4.3). Choosing a rule means choosing that assumption.
2. **Estimate the inputs better.**
   - Shrink the covariance towards a structured target (Ledoit-Wolf).
   - Shrink the means towards a common value (Bayes-Stein).

   Both trade a little bias for a large reduction in variance.
3. **Judge the result out of sample, after costs, against 1/N, with multiple-testing corrections.** An allocation rule is a strategy like any other, and the tools of step 2 apply to it.

**Risk is decomposed as well as measured.** For every portfolio you want to know where its risk comes from. Euler's theorem splits volatility or expected shortfall exactly into contributions by asset. That tells you, for example, that an "equal-weight" portfolio of 11 assets is mostly a bet against the US dollar (section 9).

---

## 2. A concrete numerical example

Three assets:

| Asset | Volatility | Expected excess return | Sharpe ratio |
|---|---|---|---|
| A (equities) | 16% | 5% | 0.31 |
| B (bonds) | 6% | 2% | 0.33 |
| C (commodities) | 20% | 3% | 0.15 |

Correlations: A-B $-0.2$, A-C $0.3$, B-C $0$.

The rules give these weights. The risk-contribution column gives each asset's share of portfolio volatility (Euler, section 4.5).

| Rule | Weights A / B / C (%) | Volatility | Risk contributions A / B / C (%) | Diversification ratio | Sharpe |
|---|---|---|---|---|---|
| 1/N | 33.3 / 33.3 / 33.3 | 9.69% | 39.4 / 2.0 / 58.7 | 1.44 | 0.34 |
| Inverse vol | 22.4 / 59.7 / 17.9 | 6.41% | 34.4 / 25.0 / 40.6 | 1.68 | 0.45 |
| ERC | 21.3 / 63.8 / 15.0 | 6.01% | 33.3 / 33.3 / 33.3 | 1.70 | 0.46 |
| Minimum variance | 15.4 / 81.7 / 2.9 | 5.14% | 15.4 / 81.7 / 2.9 | 1.55 | 0.48 |
| Maximum diversification | 21.5 / 67.0 / 11.5 | 5.70% | 35.3 / 41.2 / 23.5 | 1.71 | 0.48 |
| Maximum Sharpe | 25.5 / 72.6 / 1.8 | 5.44% | 45.8 / 52.2 / 2.0 | 1.62 | 0.51 |

Things to notice (all computed with `backtest_engine.allocation`):

- **1/N is not equal risk.** Commodities (C) are a third of the capital and 59% of the risk; bonds a third of the capital and 2% of the risk.
- **ERC equalises the risk contributions exactly**, at 33.3% each.
- **Minimum variance has risk contributions equal to its weights.** This is not a coincidence: at the unconstrained minimum, $\Sigma w$ is proportional to $\mathbf 1$ (section 4.1), so $w_i(\Sigma w)_i/\sigma_p \propto w_i$.
- **Maximum diversification maximises the diversification ratio** $w'\sigma/\sigma_p$ (1.71) by construction.
- **Maximum Sharpe is the best by construction (0.51)** if the means are right. It barely holds C, because C has the lowest Sharpe ratio.
- **All the rules except 1/N are close in Sharpe ratio** (0.45-0.51) because bonds and equities have similar Sharpe ratios here. Their implicit assumptions (section 4.3) are nearly right in this example.

**Sensitivity to the means.**

- Raise A's expected return from 5% to 6%: the maximum-Sharpe weights move to 28.5 / 70.8 / 0.7.
- Raise B's from 2% to 3% instead: they move to 21.1 / 77.9 / 0.9.

A one-point error in a mean, far smaller than the standard error of any mean estimated from history (section 4.8), moves weights by 3-5 points. With 11 assets and two years of data the movements are an order of magnitude larger.

**Leverage.** To reach 10% volatility, the minimum-variance portfolio must be levered 1.94 times and 1/N only 1.03 times. Comparing the rules at equal risk means accepting leverage (section 4.9).

---

## 3. Formulation

**Notation.**

- $N$ assets, returns $r_t\in\mathbb R^N$ in excess of cash, $t=1,\dots,T$.
- Mean $\mu$, covariance $\Sigma$ (positive definite), volatilities $\sigma_i=\sqrt{\Sigma_{ii}}$, and $D=\operatorname{diag}(\sigma)$.
- Correlation matrix $C=D^{-1}\Sigma D^{-1}$.
- Weights $w$, the fraction of capital held long ($w_i<0$ is short). Portfolio volatility $\sigma_p(w)=\sqrt{w'\Sigma w}$.
- Sample estimates $\hat\mu=\frac1T\sum_t r_t$ and $S=\frac1T\sum_t(r_t-\hat\mu)(r_t-\hat\mu)'$. The unbiased version divides by $T-1$.

**Allocation rules** (fully invested: $\mathbf 1'w=1$; long only unless stated):

| Rule | Problem |
|---|---|
| 1/N | $w_i=1/N$ |
| Inverse volatility | $w_i\propto 1/\sigma_i$ |
| Risk budgeting (ERC if $b_i=1/N$) | $w_i(\Sigma w)_i/\sigma_p=b_i\sigma_p$, $w\ge0$ |
| Minimum variance | $\min_w w'\Sigma w$ |
| Maximum diversification | $\max_w w'\sigma/\sigma_p(w)$ |
| Mean-variance | $\max_w \mu'w-\frac\gamma2 w'\Sigma w$ |
| Maximum Sharpe (tangency) | $\max_w \mu'w/\sigma_p(w)$ |

**Euler risk contributions.** For a risk measure $R(w)$ that is positively homogeneous of degree one ($R(\lambda w)=\lambda R(w)$ for $\lambda>0$):

$$R(w)=\sum_i w_i\frac{\partial R}{\partial w_i},\qquad RC_i=w_i\frac{\partial R}{\partial w_i}.$$

- **Volatility:** $RC_i=w_i(\Sigma w)_i/\sigma_p$.
- **Expected shortfall at level $\alpha$:** $RC_i=-E[w_ir_i\mid w'r\le -\mathrm{VaR}_\alpha]$ (Tasche, 2000).

**Covariance shrinkage.** $\hat\Sigma=\delta F+(1-\delta)S$, with $F$ a structured target and $\delta\in[0,1]$. Two targets are implemented:

- **scaled identity**, $F=\bar m I$ with $\bar m=\operatorname{tr}(S)/N$ (Ledoit and Wolf, 2004a);
- **constant correlation**, $F_{ii}=S_{ii}$ and $F_{ij}=\bar\rho\sqrt{S_{ii}S_{jj}}$, with $\bar\rho$ the average sample correlation (Ledoit and Wolf, 2004b).

**Bayes-Stein means** (Jorion, 1986):

$$\hat\mu_{BS}=(1-\phi)\hat\mu+\phi\,\mu_0\mathbf 1,\qquad \phi=\frac{N+2}{(N+2)+T(\hat\mu-\mu_0\mathbf 1)'S^{-1}(\hat\mu-\mu_0\mathbf 1)},$$

with $\mu_0$ the mean return of the minimum-variance portfolio.

**Walk-forward backtest** (`rebalance.walk_forward_allocation`).

- At each decision date $d$, the rule receives the last $L$ returns up to $d$ and returns targets $w^*_d$.
- With a volatility target $\tau$, the targets are scaled by $\min(\tau/\hat\sigma_p(w^*_d),\,\ell_{\max})$, where $\hat\sigma_p$ is annualised and $\ell_{\max}$ is the leverage cap.
- The trade executes at the close of $d+\text{lag}$. Its cost is $\sum_i c_i|w^*_i-w^{\text{drift}}_i|$.
- Between trades the weights drift:

$$w_{i,t}=\frac{w_{i,t-1}(1+r_{i,t})}{1+w_{t-1}'r_t}.$$

---

## 4. Derivations

### 4.1 Minimum variance

Minimise $w'\Sigma w$ subject to $\mathbf 1'w=1$. The Lagrangian is $w'\Sigma w-\lambda(\mathbf 1'w-1)$, and setting its gradient to zero gives $2\Sigma w=\lambda\mathbf 1$. So

$$w_{mv}=\frac{\Sigma^{-1}\mathbf 1}{\mathbf 1'\Sigma^{-1}\mathbf 1},\qquad \sigma^2_{mv}=\frac1{\mathbf 1'\Sigma^{-1}\mathbf 1}.$$

Two consequences:

- **Equal marginal risks.** At the optimum, $(\Sigma w)_i=\sigma^2_{mv}$ for every asset: every held asset has the same marginal contribution to variance. So $RC_i=w_i\sigma^2_{mv}/\sigma_{mv}=w_i\sigma_{mv}$, which is the weight times the total, as in section 2.
- **Long-only case.** With $w\ge0$ the problem is a quadratic programme. The Karush-Kuhn-Tucker conditions say that $(\Sigma w)_i$ equals a common value $\lambda/2$ where $w_i>0$, and is at least $\lambda/2$ where $w_i=0$: an excluded asset would add variance at the margin. `test_minimum_variance_closed_form_and_long_only_kkt` checks both conditions.

### 4.2 Maximum Sharpe (tangency) and why a levered investor holds it

The Sharpe ratio $\mu'w/\sqrt{w'\Sigma w}$ is homogeneous of degree zero in $w$: scaling $w$ does not change it. At the maximum, the gradient is zero:

$$\frac{\mu}{\sigma_p}-\frac{(\mu'w)\Sigma w}{\sigma_p^3}=0\ \Rightarrow\ \Sigma w\propto\mu\ \Rightarrow\ w_{tan}=\frac{\Sigma^{-1}\mu}{\mathbf 1'\Sigma^{-1}\mu}.$$

The normalisation needs $\mathbf 1'\Sigma^{-1}\mu>0$. Otherwise $\Sigma^{-1}\mu$ is the optimal position, but it has negative net exposure, and dividing by a negative sum flips its sign. The code raises an error in that case.

**Why a levered investor holds the tangency portfolio.** A mean-variance investor who can borrow and lend at the risk-free rate chooses $x=\Sigma^{-1}\mu/\gamma$: tangency weights times a scalar, whatever $\gamma$. Only the leverage depends on risk aversion. With futures, or with a volatility target, the budget constraint $\mathbf 1'w=1$ is irrelevant. That is why the research note compares **maximum Sharpe** rather than a mean-variance rule with an arbitrary $\gamma$.

**Long only: an exact reduction to non-negative least squares.** With $w\ge0$ the Sharpe ratio is not concave in $w$, but the problem can be rewritten.

1. Substitute $z=Dy$ with $y\ge0$ unnormalised and $s=D^{-1}\mu$ (the Sharpe ratios). Then $\text{Sharpe}(y)=s'z/\sqrt{z'Cz}$.
2. For a direction $u\ge0$ with $s'u>0$, write $f(u)=u'Cu/(s'u)^2=1/\text{Sharpe}(u)^2$ and consider $z=au$:
$$\min_{a}\ a^2u'Cu+(a\,s'u-1)^2=\frac{f(u)}{1+f(u)}.$$
3. This is increasing in $f$. So minimising $\|L'z\|^2+(s'z-1)^2$ over $z\ge0$, with $C=LL'$ the Cholesky factorisation, gives exactly the direction that maximises the Sharpe ratio.
4. That problem is non-negative least squares:
$$\min_{z\ge0}\left\|\begin{pmatrix}L'\\ s'\end{pmatrix}z-\begin{pmatrix}0\\1\end{pmatrix}\right\|^2.$$
The Lawson-Hanson (1974) active-set algorithm solves it in finitely many steps (`scipy.optimize.nnls`).
5. Then $w\propto z/\sigma$.

The first implementation used a general sequential quadratic programming solver (SLSQP). It failed on real two-year windows ("positive directional derivative in line search"). The reduction is exact and has no tolerance to tune.

The KKT test (`test_maximum_sharpe`) checks the solution directly. The derivative of the Sharpe ratio, $\partial S/\partial w_i=\big(\mu_i-S(\Sigma w)_i/\sigma_p\big)/\sigma_p$, must be zero for held assets and non-positive for excluded ones. The test also checks the result against 50,000 random long-only portfolios.

### 4.3 The implicit views of risk-based rules

Each risk-based rule is the tangency portfolio for some implicit vector of expected returns. Substituting into $w\propto\Sigma^{-1}\mu$:

| Rule | Tangency if | Proof |
|---|---|---|
| Minimum variance | $\mu=c\mathbf 1$ (equal expected returns) | $\Sigma^{-1}\mu=c\Sigma^{-1}\mathbf 1$ |
| Maximum diversification | $\mu=c\sigma$ (equal Sharpe ratios) | the objective is the Sharpe ratio with $\mu=\sigma$ |
| Inverse volatility | equal Sharpe ratios and constant correlation $\rho$ | $\Sigma^{-1}\mu=cD^{-1}C^{-1}\mathbf 1$, and $C^{-1}\mathbf 1=\mathbf 1/(1+(N-1)\rho)$ |
| ERC | as inverse volatility | with constant correlation, ERC equals inverse volatility (Maillard, Roncalli and Teiletche, 2010) |
| 1/N | equal Sharpe ratios, constant correlation and equal volatilities | |

**Minimum variance assumes that low-volatility assets have the highest Sharpe ratios.** That is the "low-volatility anomaly" (Frazzini and Pedersen, 2014): it holds for stocks within a market, but it is not a law across asset classes. In the research note, minimum variance holds 2-year Treasuries at 10x leverage. It wins when bonds rally and loses 45% when they do not.

**ERC assumes equal Sharpe ratios.** Asness, Frazzini and Pedersen (2012) argue that leverage-averse investors bid up high-volatility assets, so that low-volatility assets do offer higher risk-adjusted returns. That argument justifies levering bonds, and it is also what makes risk parity vulnerable to a bond sell-off.

### 4.4 Risk budgeting: existence, uniqueness and the solver

We want $w_i(\Sigma w)_i=b_i\,w'\Sigma w$ with $b_i>0$ and $\sum b_i=1$. Spinu (2013) observes that the strictly convex problem

$$\min_{y>0}\ \tfrac12y'\Sigma y-\sum_ib_i\ln y_i$$

has first-order condition $(\Sigma y)_i=b_i/y_i$, that is $y_i(\Sigma y)_i=b_i$. Summing over $i$ gives $y'\Sigma y=1$, so $y_i(\Sigma y)_i=b_i\,y'\Sigma y$: exactly the risk-budget condition. The weights are $w=y/\mathbf 1'y$.

- **Existence and uniqueness.** The objective is strictly convex (a positive definite quadratic minus a sum of logarithms) and tends to infinity at the boundary $y_i\to0$ and as $\|y\|\to\infty$. So a unique minimiser exists and it is interior (long only).
- **Newton's method.** The gradient is $g=\Sigma y-b/y$ and the Hessian $H=\Sigma+\operatorname{diag}(b/y^2)$. Step $y\leftarrow y-tH^{-1}g$, with $t$ halved until $y$ stays positive and the objective falls enough (Armijo condition).
- **Numerical lessons from the real data.**
  1. **Scale.** For a daily covariance, the unknowns are of order $1/\sigma_i$ (hundreds), and round-off stalls the iteration near $10^{-10}$. Risk budgeting is scale-equivariant: if $z$ solves the problem for $C$, then $y=D^{-1}z$ solves it for $\Sigma=DCD$, because $y_i(\Sigma y)_i=z_i(Cz)_i$. The solver therefore works on the correlation matrix.
  2. **Line search near the optimum.** The predicted decrease $g'H^{-1}g$ falls below what double precision resolves in the objective (about $10^{-16}$ of its size). The Armijo test then fails although the step is good. Once the squared Newton decrement is below $10^{-10}$, the solver takes pure Newton steps, which converge quadratically (Boyd and Vandenberghe, 2004, §9.5).
  3. **Result.** 1,732 solves on the research note's windows (three window lengths, sample and shrunk covariances) converge to a spread of risk contributions below $10^{-12}$, in 0.6 s in total.

### 4.5 Euler's theorem and the risk contributions

If $R(\lambda w)=\lambda R(w)$, differentiate with respect to $\lambda$ at $\lambda=1$: $\sum_iw_i\partial_iR(w)=R(w)$.

- **Volatility.** $\partial_i\sigma_p=(\Sigma w)_i/\sigma_p$, so $RC_i=w_i(\Sigma w)_i/\sigma_p$. The contributions can be negative for a hedge (an asset negatively correlated with the rest of the portfolio).
- **Expected shortfall.** Write $L=-w'r$ for the loss. For continuous distributions,
$$\mathrm{ES}_\alpha(w)=E[L\mid L\ge \mathrm{VaR}_\alpha],\qquad \partial_i\mathrm{ES}_\alpha=E[-r_i\mid L\ge\mathrm{VaR}_\alpha]$$
(Tasche, 2000). The derivative of the conditioning event contributes nothing, because at the VaR quantile the density term cancels.
- **Estimation from scenarios.** Take the worst $k=\lceil(1-\alpha)n\rceil$ scenarios; the contributions are the averages of $-w_ir_i$ over them, and they sum to the ES exactly. For Gaussian returns the ES shares equal the volatility shares, because $E[r\mid w'r]$ is linear in $w'r$ with slope $\Sigma w/\sigma_p^2$. The test checks this, and $\mathrm{ES}_{97.5\%}=\sigma_p\varphi(1.96)/0.025\approx2.338\,\sigma_p$.
- **Realised decomposition.** In the research note the scenarios are the realised daily P&Ls of each asset, $w_{i,t-1}r_{i,t}$. They add up to the portfolio's gross return, so the realised ES splits exactly by asset class.

### 4.6 Ledoit-Wolf: the optimal shrinkage intensity

Minimise the expected Frobenius loss $E\|\delta F+(1-\delta)S-\Sigma\|^2$ over $\delta$. It is a quadratic in $\delta$:

$$\delta^*=\frac{E\|S-\Sigma\|^2-E\langle F-\Sigma,\,S-\Sigma\rangle}{E\|F-S\|^2}.$$

- **Numerator.** The first term is the total variance of the sample covariance, $\sum_{ij}\mathrm{Var}(\sqrt T s_{ij})/T=\pi/T$. The second is the covariance between the target and the sample, $\rho/T$. For the identity target it is negligible, because $F$ depends on $S$ only through $\operatorname{tr}(S)/N$, an average of $N$ variances whose error is small relative to that of $S$ itself.
- **Denominator.** It is dominated by the misspecification of the target, $\gamma=\|F-\Sigma\|^2$, which does not shrink with $T$.
- **Result.** $\delta^*\approx(\pi-\rho)/(\gamma T)$. Ledoit and Wolf estimate $\pi$, $\rho$ and $\gamma$ consistently from the data; the formulas are in the docstring of `covariance.py` and transcribed loop by loop in `test_constant_correlation_target_matches_the_paper_formulas`.

Interpretation:

- $\delta$ falls like $1/T$. With four years of daily data the research note finds $\delta\approx0.03$: the sample covariance of 11 assets is already precise.
- Shrinkage matters when $N/T$ is not small: hundreds of stocks, monthly data, or covariance matrices of strategies with short histories.
- The target must suit the truth. The scaled identity imposes equal variances, which is badly wrong for assets whose volatilities differ 15:1. That is why the constant-correlation target is the default, and why the identity target is tested on a near-spherical truth.

### 4.7 Why shrinking means works: James-Stein

For $N\ge3$ independent normal means observed with unit variance, $X\sim N(\theta,I)$, the estimator $\big(1-(N-2)/\|X\|^2\big)X$ has lower total mean squared error than $X$ itself, for **every** $\theta$ (James and Stein, 1961). The sample mean is inadmissible.

The intuition: $E\|X\|^2=\|\theta\|^2+N$. The observations are on average too far from the origin, and pulling them back reduces the error more than the bias it adds.

Jorion (1986) applies this to portfolio means. He shrinks towards the minimum-variance portfolio's mean (the grand mean that is least affected by noise), with the intensity $\phi$ of section 3.

- $\phi$ is large when the dispersion of the sample means is small relative to what noise alone would produce, $N/T$ in the metric $S^{-1}$.
- With 11 assets and 504 days, $T(\hat\mu-\mu_0\mathbf 1)'S^{-1}(\hat\mu-\mu_0\mathbf 1)$ is typically of order $N$. On the research note's windows, $\phi$ has a median of 0.59 and ranges from 0.35 to 0.85: more than half of the dispersion of the sample means is treated as noise.

### 4.8 How noisy is the sample tangency portfolio?

The in-sample maximum squared Sharpe ratio is $\hat\theta^2=\hat\mu'S^{-1}\hat\mu$. Treat $S$ as known and let $\hat\mu\sim N(\mu,\Sigma/T)$. Then

$$E[\hat\mu'\Sigma^{-1}\hat\mu]=\mu'\Sigma^{-1}\mu+\operatorname{tr}(\Sigma^{-1}\Sigma/T)=\theta^2+\frac NT.$$

With $T$ measured in years, the bias is $N/T$ in annual squared-Sharpe units.

**Example.** With 11 assets and a two-year window, the bias is $11/2=5.5$. Even if the true maximum Sharpe ratio were 0.5 ($\theta^2=0.25$), the in-sample estimate would be around $\sqrt{0.25+5.5}\approx2.4$. The noise is twenty times the signal.

Accounting for the estimated covariance (with the unbiased divisor $T-1$), the exact null expectation is $E[\hat\theta^2]=\frac{(T-1)N}{(T-N-2)T}$. It follows from Hotelling's $T^2=T\hat\theta^2$, with $\frac{T-N}{(T-1)N}T^2\sim F(N,T-N)$ under the null. A simulation with $N=11$ and $T=504$ (2,000 draws, seed 1) gives 0.0226 against the exact 0.0224 and the approximation 0.0218 (daily units).

This is why sample-mean rules turn over so much (8-11 times a year in the research note), and why their out-of-sample Sharpe ratios depend on the setting. Kan and Zhou (2007) derive the expected out-of-sample loss of the plug-in rule, and show that combining it with the minimum-variance portfolio reduces the loss.

### 4.9 Volatility targeting and leverage

**Scaling.** Scaling the weights by $\tau/\hat\sigma_p$ makes the ex-ante volatility equal to $\tau$ at each rebalance (checked to $10^{-10}$). The realised volatility is not equal to $\tau$ for two reasons:

1. **the estimate lags:** a 504-day window is slow to see a new regime;
2. **volatility itself is persistent but mean-reverting.**

**It changes the Sharpe ratio.** For a strategy with returns $R_t=\ell_{t-1}X_t$ and leverage $\ell$ chosen from past data:

$$E[R]=E[\ell]E[X]+\operatorname{Cov}(\ell_{t-1},X_t).$$

If high predicted volatility is not followed by proportionally higher returns (Moreira and Muir, 2017), cutting leverage when volatility is high raises the Sharpe ratio. If volatility spikes arrive after long calm periods, the leverage is highest just before the spike. The research note finds both: volatility targeting lowers the Sharpe ratio of 1/N (0.31 to 0.25) and raises that of maximum Sharpe (0.31 to 0.49).

**The cap.** With a 2-year Treasury at 0.5% volatility, a 10% target asks for 20x. A cap is a risk limit. It also makes the comparison unequal, because a capped portfolio runs at less than the target.

### 4.10 The break-even cost multiple

Let $\bar g$ and $\bar c$ be the mean daily gross return and mean daily cost of a rule. With costs scaled by $k$, the mean net return is $\bar g-k\bar c$. The rule's net mean equals the benchmark's when

$$k^*=\frac{\bar g_s-\bar g_b}{\bar c_s-\bar c_b}.$$

- $k^*>1$: the advantage survives the assumed costs.
- $k^*<0$: the rule is behind before costs, so cheaper trading cannot help.
- Undefined if the rule trades less than the benchmark.

$k^*$ is a first-order statement about means. The Sharpe ratio also depends on the volatility of costs, which is negligible here. The research note reports the net Sharpe ratio at $k=0,1,5,20$ as well.

---

## 5. Assumptions

- **Stationary means and covariances within each window.** Every rule assumes that the next month resembles the last two years. Correlation regime shifts (2021-2022) violate this.
- **Returns of positions, not of contracts.** The universe uses constant-maturity bond returns, price indices and interbank carry as proxies for futures and forwards. Financing is embedded in the excess return; margin is ignored.
- **Linear costs.** Cost is proportional to notional traded, with no market impact that grows with size. Adequate for a small investor in liquid futures, not for a large fund (see step 4).
- **Frictionless leverage up to the cap,** with no margin calls and no forced deleveraging.
- **Daily closes as execution prices,** one day after the decision.
- **Constant-correlation target.** Ledoit-Wolf assumes independent and identically distributed returns for its optimal intensity. With volatility clustering the estimates of $\pi$ and $\rho$ are noisier, but the estimator remains consistent.

---

## 6. Implementation mapped to code

| Concept | Code | Notes |
| --- | --- | --- |
| Sample and EWMA covariance | `covariance.sample_covariance`, `ewma_covariance` | EWMA with half-life, weights sum to one |
| Ledoit-Wolf shrinkage | `covariance.ledoit_wolf(x, target)` | returns `(cov, delta)`; identity equals scikit-learn |
| 1/N, inverse vol | `allocation.equal_weight`, `inverse_volatility` | |
| Risk budgeting and ERC | `allocation.risk_budget(cov, budgets)` | Spinu's problem on the correlation matrix, damped then pure Newton |
| Minimum variance | `allocation.minimum_variance(cov, long_only, max_weight)` | closed form, or SLSQP when bounds bind |
| Maximum diversification | `allocation.maximum_diversification` | min-var on the correlation matrix, rescaled by $1/\sigma$ |
| Mean-variance | `allocation.mean_variance(mu, cov, gamma, ...)` | closed form with the budget multiplier, or SLSQP |
| Maximum Sharpe | `allocation.maximum_sharpe(mu, cov, ...)` | closed form; long only by exact NNLS; cash if no $\mu_i>0$ |
| Bayes-Stein means | `allocation.bayes_stein_means(x, cov)` | returns `(means, phi)` |
| Euler contributions | `allocation.risk_contributions`, `expected_shortfall_contributions` | exact sums |
| Diversification ratio | `allocation.diversification_ratio` | |
| Rebalancing calendar | `rebalance.rebalance_dates(index, frequency)` | last trading day of each complete period |
| Walk-forward backtest | `rebalance.walk_forward_allocation(...)` | lag, drift, per-asset costs, volatility target and cap |
| Break-even costs | `rebalance.break_even_multiplier` | |
| Universe | `universes.load_multi_asset_excess_returns`, `synthetic_multi_asset` | FRED sources; synthetic for offline runs |

Tests (`python -m pytest tests/backtest_engine -k "allocation or covariance or rebalance"`):

- `test_allocation.py`:
  - min-var closed form and KKT;
  - ERC: equal contributions, budgets, special cases, a daily-scale covariance, and the ordering of volatilities;
  - maximum diversification against the other rules;
  - maximum Sharpe: closed form, scale invariance, KKT, random portfolios, weight caps, cash;
  - mean-variance and Bayes-Stein limits;
  - Euler sums and the Gaussian ES.
- `test_covariance.py`:
  - scikit-learn equality for the identity target;
  - the paper transcription for the constant-correlation target;
  - the oracle loss by simulation for both targets;
  - intensity vanishing with $T$;
  - EWMA limits.
- `test_rebalance.py`:
  - timing, lag and windows, including the incomplete last month;
  - drift, turnover and per-asset costs;
  - identical assets never trade again;
  - no look-ahead under a future shock;
  - exact ex-ante volatility, the leverage cap and the break-even multiple.

---

## 7. Python or C++?

**Measured.** The research notebook runs 96 walk-forward backtests: 8 rules, 12 settings, about 270 rebalances each. That takes 34 seconds, in Python.

The time goes to the solvers:

- minimum variance with the long-only bound binding uses SLSQP, about 1.5 s per backtest;
- everything else takes 0.1-0.4 s per backtest;
- 1,732 ERC solves take 0.6 s;
- the bootstrap tests reuse the C++ engine of step 2 and take about 1 s.

Moving the allocation layer to C++ would save seconds on a task that runs once. It would also cost the use of well-tested solvers (LAPACK through NumPy, Lawson-Hanson NNLS, SLSQP). C++ becomes worthwhile for:

- thousands of assets (covariance of size $10^6$, where the $O(N^3)$ solves and the memory layout matter);
- intraday rebalancing;
- a bootstrap of the whole backtest (resampling returns and re-running every rule), which multiplies the cost by the number of replicates.

---

## 8. Statistical interpretation

- **Differences in Sharpe ratios are hard to detect.**
  - At equal volatility, the mean difference between two rules is about $\sigma(\text{SR}_1-\text{SR}_2)$. Its standard error depends on the volatility of the difference, which is small when the rules hold similar portfolios.
  - Maximum diversification minus 1/N has a Sharpe ratio of 0.43 over 22 years: $t\approx0.43\sqrt{22}\approx2.0$, and the bootstrap gives 2.19. That is significant alone, but not after the seven comparisons (SPA $p=0.06$).
- **The number of configurations matters even when they are correlated.** The 84 rule-settings of the notebook amount to about 8 independent trials (step 2, section 4.5). The best configuration's naive $p=0.01$ becomes an SPA $p=0.15$.
- **Episode dependence** tells you whether an edge is broad or concentrated. Removing the 20 best days of the difference (0.4% of the sample) cuts its Sharpe ratio from 0.43 to 0.16.
- **Covariance estimation error is small with daily data**, and mean estimation error is enormous (section 4.8). That asymmetry is the statistical case for risk-based rules.
- **Walk-forward is out of sample for the data, not for the researcher.** The rules were chosen knowing the literature, and 2008 and 2022 are known episodes.

---

## 9. Finance interpretation

- **Risk parity is a leveraged bond position with an equity and currency overlay.** In the research note, ERC puts 27% of its risk in three Treasury maturities and holds 1-4.5 times capital in them. Industry risk-parity funds (bonds, equities, commodities, inflation-linked bonds) have the same structure, and they suffered large losses in 2022, when bonds and equities fell together.
- **Equal weights by asset are arbitrary.** 1/N on this universe is 55% currency risk, because six of eleven assets are currencies. The unit of diversification should be the risk source: rates, equity, currency, commodity. That is a reason for risk budgeting by asset class ($b_i$ per class, not per asset).
- **Minimum variance is a bet on the low-volatility anomaly**, here across asset classes. It holds the lowest-volatility asset at the leverage cap, and it earns that asset's Sharpe ratio with its tail risk levered tenfold.
- **Maximum diversification** assumes equal Sharpe ratios but uses correlations. In this sample it put more risk in equities than ERC did, which paid in the 2009-2019 bull market.
- **Maximum Sharpe with sample means** is partly a trend-following rule. A two-year trailing mean is a slow momentum signal, and the long-only rule holds cash when every trailing mean is negative (mid-2022 to 2024).
- **Costs are not the constraint** for monthly rebalancing of liquid futures (1-8.5 bp a year at the assumed costs). They become one for daily rebalancing, less liquid instruments, or large size (step 4).

---

## 10. Failure modes (checklist for any allocation backtest)

1. **Correlation regime change.** Does the rule depend on a hedge (negative stock-bond correlation) that can disappear? Stress it with the correlation sign flipped.
2. **Leverage on low-volatility assets.** What is the maximum leverage the rule requests, and how often does the cap bind? Low realised volatility is when leverage is highest and the next shock is least expected.
3. **Hidden concentration.** Decompose risk by source, not by asset. 1/N and ERC by asset inherit the composition of the universe.
4. **Noisy means.** Report turnover and the stability of weights. A rule that jumps between concentrated portfolios is fitting noise (section 4.8).
5. **Slow risk estimates.** Compare realised with ex-ante volatility month by month, not only on average.
6. **Unequal comparisons.** Rules compared unlevered differ in volatility; rules compared at a volatility target differ in leverage timing. Report both.
7. **Survivorship in the universe.** The assets were chosen because they exist today and are liquid. Their history favours assets that did well.
8. **The benchmark is not the only baseline.** Compare with 1/N and with cash, and report the break-even cost multiple against both.

---

## 11. Alternatives

- **Hierarchical risk parity** (Lopez de Prado, 2016): clusters the correlation matrix and allocates by recursive bisection. No matrix inversion, so it is robust to near-singular covariances.
- **Nonlinear shrinkage** (Ledoit and Wolf, 2017): shrinks each eigenvalue differently. Better than linear shrinkage when $N$ is large.
- **Factor-model covariance:** $\Sigma=BFB'+\Delta$ with a few factors. It has far fewer parameters, but it is biased if the factors are wrong. A statistical (PCA) factor model with too few factors misstates the correlations it leaves in the residual, which is why it is not used here.
- **Constraints as shrinkage** (Jagannathan and Ma, 2003): a long-only constraint on minimum variance is equivalent to shrinking the large covariances. That explains why "wrong" constraints help.
- **Black-Litterman** (Black and Litterman, 1992): starts from the market-equilibrium means and blends in views with explicit confidence. It is the Bayesian answer to section 4.8.
- **Combining rules** (Kan and Zhou, 2007; DeMiguel, Garlappi and Uppal, 2009): an optimal mix of 1/N or minimum variance with the tangency portfolio, with the weights chosen to minimise the expected estimation loss.
- **Turnover-aware optimisation** (Garleanu and Pedersen, 2013): trades partially towards the target, with the trading rate set by costs and signal decay.
- **Risk budgeting by class** with a stressed covariance (correlations set to their crisis values): addresses failure modes 1 and 3 directly.

---

## 12. Interview questions with answers

### Basic

1. **Write the minimum-variance portfolio with a budget constraint.**
   $w=\Sigma^{-1}\mathbf 1/(\mathbf 1'\Sigma^{-1}\mathbf 1)$, from the Lagrangian of $w'\Sigma w$ with $\mathbf 1'w=1$.
2. **What is risk parity?**
   Weights such that every asset contributes the same share of portfolio volatility, $w_i(\Sigma w)_i=\sigma_p^2/N$. With two assets, or with equal correlations, it is inverse volatility.
3. **Why does 1/N perform well out of sample?**
   It has no estimation error. Optimised rules must gain more from using the data than they lose from estimating it, which takes very long samples when means are involved (DeMiguel, Garlappi and Uppal: about 3,000 months for 25 assets).
4. **What is the Euler decomposition, and why do the contributions sum to the total?**
   For a risk measure that is homogeneous of degree one, $R(w)=\sum_i w_i\partial R/\partial w_i$ by Euler's theorem. Volatility and expected shortfall are homogeneous of degree one.
5. **Why shrink a covariance matrix?**
   The sample covariance has $N(N+1)/2$ parameters. When $N/T$ is not small, its extreme eigenvalues are biased (large ones up, small ones down), and optimisers load on the directions whose risk is most underestimated. Shrinkage towards a structured target reduces the variance of the estimate at the cost of some bias.

### Intermediate

6. **Show that maximum diversification is the tangency portfolio when all Sharpe ratios are equal.**
   With $\mu=c\sigma$, the Sharpe ratio is $c\,w'\sigma/\sigma_p$, which is $c$ times the diversification ratio.
7. **Why does a levered investor hold the tangency portfolio regardless of risk aversion?**
   Without a budget constraint, the mean-variance solution is $\Sigma^{-1}\mu/\gamma$: risk aversion only scales it. Leverage separates portfolio choice from risk choice (two-fund separation).
8. **How is the Ledoit-Wolf intensity chosen, and how does it scale with $T$?**
   It minimises the expected Frobenius loss. The optimal value is (variance of the sample estimate minus its covariance with the target) divided by the target's squared misspecification. It is of order $1/T$.
9. **Your risk-parity backtest has a Sharpe ratio of 0.8 over 1990-2020. What would you check?**
   - Leverage and the bond share of risk.
   - The stock-bond correlation over the sample (negative almost throughout).
   - Performance when rates rise (1994, 2013, 2022).
   - Financing costs and margin.
   - Whether the result survives a positive-correlation stress.
10. **Why is the expected shortfall contribution of an asset $-E[w_ir_i\mid \text{tail}]$?**
    Differentiate $\mathrm{ES}(w)=E[-w'r\mid -w'r\ge\mathrm{VaR}]$ with respect to $w_i$. The derivative of the conditioning event vanishes at the quantile (Tasche, 2000), leaving the conditional expectation of $-r_i$. Multiply by $w_i$.

### Advanced

11. **How biased is the in-sample maximum Sharpe ratio?**
    $E[\hat\theta^2]\approx\theta^2+N/T$ (with $T$ in the units of the Sharpe ratio). With 11 assets and two years of data, the bias is 5.5 in annual squared Sharpe. Exactly, under the null, $E[\hat\theta^2]=(T-1)N/((T-N-2)T)$ (Hotelling).
12. **Long-only maximum Sharpe is not a concave problem in $w$. How do you solve it exactly?**
    Homogeneity: fix the numerator $\mu'y=1$ and minimise $y'\Sigma y$ over $y\ge0$, a convex quadratic programme. Or, as implemented, reduce it to NNLS on $\big(\begin{smallmatrix}L'\\ s'\end{smallmatrix}\big)z\approx\big(\begin{smallmatrix}0\\1\end{smallmatrix}\big)$, whose solution direction maximises the Sharpe ratio exactly (section 4.2).
13. **Prove that the risk-budgeting solution exists and is unique.**
    Spinu's objective $\frac12y'\Sigma y-\sum b_i\ln y_i$ is strictly convex on $y>0$ and coercive: it tends to infinity at the boundary and at infinity. So it has a unique minimiser, whose first-order conditions are the budget conditions after normalisation.
14. **When does volatility targeting raise the Sharpe ratio?**
    When realised returns per unit of predicted volatility fall as predicted volatility rises, i.e. when volatility is predictable but the risk premium does not rise proportionally. $E[\ell X]=E[\ell]E[X]+\operatorname{Cov}(\ell,X)$: the covariance term must be positive.
15. **Design a test of whether allocation rule A beats 1/N.**
    - Walk-forward, net of costs, at equal ex-ante risk and on the same days.
    - Test the daily difference with a block bootstrap; if several rules or settings were tried, use SPA and Romano-Wolf over all of them.
    - Report episode dependence, subperiods and the break-even cost multiple.

---

## 13. Exercises (answers not provided)

**Level 1: mechanics.**

1. For two assets with volatilities $\sigma_1,\sigma_2$ and correlation $\rho$, derive the minimum-variance weight in closed form. When is it negative?
2. Verify numerically that ERC equals inverse volatility for any 2x2 covariance matrix, using `allocation.risk_budget`.
3. Compute the risk contributions of 1/N for the universe of the research note on its last window. Which asset contributes most, and why?

**Level 2: understanding the estimators.**

4. Show that, with the identity target, the covariance term $E\langle F-\Sigma,S-\Sigma\rangle$ is of lower order than $E\|S-\Sigma\|^2$ as $N$ grows.
5. Simulate $\hat\theta^2$ for $N=5$ and $N=50$ with $T=250$ under the null, and compare with $(T-1)N/((T-N-2)T)$. What happens as $N\to T$?
6. Prove that the Bayes-Stein intensity $\phi$ is 1 when all sample means are equal, and tends to 0 as $T\to\infty$ with fixed distinct true means.

**Level 3: derivations and design.**

7. Derive the expected shortfall contributions of a Gaussian portfolio in closed form and show that their shares equal the volatility shares.
8. Extend `risk_budget` to budgets by asset class: each class gets a budget, split equally among its assets. Formulate it as a Spinu problem and implement it with tests.
9. Implement Kan and Zhou's (2007) three-fund rule (tangency, minimum variance, cash) and compare it with the rules of the research note using the same SPA test.

**Level 4: research.**

10. Replace the constant-maturity Treasury proxies with futures returns built with `backtest_engine.futures` (once contract data are available), and repeat the research note. Which conclusions change?
11. Build a stressed covariance with the stock-bond correlation set to $+0.3$. Compare the ERC and maximum-diversification weights and their 2022 losses with those obtained from the historical covariance.
12. Apply the allocation layer to the strategies of this repository (momentum, carry, pairs) instead of assets. How do the implicit views of section 4.3 translate when the "assets" are strategies with short histories?

---

## 14. Cumulative curriculum

1. [Futures, rolls and carry](../futures/futures_rolls_and_carry.md): tradable returns.
2. [Data snooping and multiple testing](../statistics/data_snooping_and_multiple_testing.md): honest inference about those returns.
3. **Portfolio construction under estimation error** (this note): combining assets or strategies, measuring where the risk sits, and testing the combination against 1/N.
4. Execution and market impact (planned): when costs stop being linear.

## 15. Fundamentals first: self-check

- Can you invert a 2x2 covariance matrix by hand and state when a covariance matrix is positive definite?
- Do you know how to solve an equality-constrained quadratic problem with a Lagrange multiplier, and what the KKT conditions add for inequality constraints?
- Can you compute the variance of $w'r$ and the correlation of two portfolios?
- Do you know what an eigenvalue decomposition says about the directions of largest and smallest variance?
- Can you explain the bias-variance trade-off of an estimator, and why a biased estimator can have lower mean squared error?
- Can you state Euler's theorem for homogeneous functions?

## 16. Levels of understanding

- **Minimum (use it correctly):**
  - compare every rule with 1/N out of sample, after costs, at equal ex-ante risk;
  - report turnover, leverage, the risk decomposition by source and the crisis behaviour;
  - count and test all configurations tried.
- **Quantitative (defend the numbers):**
  - derive the closed forms of minimum variance and tangency;
  - state the implicit views of the risk-based rules;
  - explain the Ledoit-Wolf intensity and its $1/T$ scaling;
  - quantify the noise in sample tangency portfolios ($N/T$).
- **Deep (design with it):**
  - reduce non-convex allocation problems to convex ones;
  - make solvers robust to scale and round-off;
  - choose the unit of diversification (asset, class, factor);
  - design stress tests for the assumptions each rule relies on.

## 17. References

- Asness, C. S., Frazzini, A. and Pedersen, L. H. (2012). Leverage aversion and risk parity. *Financial Analysts Journal*, 68(1), 47-59.
- Black, F. and Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28-43.
- Boyd, S. and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Choueifaty, Y. and Coignard, Y. (2008). Toward maximum diversification. *Journal of Portfolio Management*, 35(1), 40-51.
- Chopra, V. K. and Ziemba, W. T. (1993). The effect of errors in means, variances, and covariances on optimal portfolio choice. *Journal of Portfolio Management*, 19(2), 6-11.
- DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915-1953.
- Frazzini, A. and Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1-25.
- Garleanu, N. and Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *Journal of Finance*, 68(6), 2309-2340.
- Jagannathan, R. and Ma, T. (2003). Risk reduction in large portfolios: why imposing the wrong constraints helps. *Journal of Finance*, 58(4), 1651-1684.
- James, W. and Stein, C. (1961). Estimation with quadratic loss. *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 361-379.
- Jorion, P. (1986). Bayes-Stein estimation for portfolio analysis. *Journal of Financial and Quantitative Analysis*, 21(3), 279-292.
- Kan, R. and Zhou, G. (2007). Optimal portfolio choice with parameter uncertainty. *Journal of Financial and Quantitative Analysis*, 42(3), 621-656.
- Lawson, C. L. and Hanson, R. J. (1974). *Solving Least Squares Problems*. Prentice-Hall.
- Ledoit, O. and Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. *Journal of Empirical Finance*, 10(5), 603-621.
- Ledoit, O. and Wolf, M. (2004a). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.
- Ledoit, O. and Wolf, M. (2004b). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management*, 30(4), 110-119.
- Ledoit, O. and Wolf, M. (2017). Nonlinear shrinkage of the covariance matrix for portfolio selection: Markowitz meets Goldilocks. *Review of Financial Studies*, 30(12), 4349-4388.
- Lopez de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *Journal of Portfolio Management*, 42(4), 59-69.
- Maillard, S., Roncalli, T. and Teiletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management*, 36(4), 60-70.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77-91.
- Michaud, R. O. (1989). The Markowitz optimization enigma: is 'optimized' optimal? *Financial Analysts Journal*, 45(1), 31-42.
- Moreira, A. and Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611-1644.
- Roncalli, T. (2013). *Introduction to Risk Parity and Budgeting*. Chapman and Hall/CRC.
- Spinu, F. (2013). An algorithm for computing risk parity weights. SSRN working paper 2297383.
- Tasche, D. (2000). Risk contributions and performance measurement. Working paper, Technische Universität München.
