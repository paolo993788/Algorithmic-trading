# Data snooping and multiple testing: inference after a search

Learning note for [`backtest_engine.validation`](../../../scripts/backtest_engine/backtest_engine/validation.py), [`backtest_engine.cross_validation`](../../../scripts/backtest_engine/backtest_engine/cross_validation.py) and the joint stationary bootstrap in [`cpp/overfitting.hpp`](../../../scripts/backtest_engine/cpp/overfitting.hpp).

The research note that applies them is [`notebooks/strategy_validation/data_snooping_and_multiple_testing.ipynb`](../../../notebooks/strategy_validation/data_snooping_and_multiple_testing.ipynb).

**Where this fits.** This is step 2 of the systematic-trading track ([curriculum](../README.md)). Step 1 made returns tradable; this step makes the *evidence* about those returns honest. Every strategy note that follows reports its results through these tests.

**Prerequisites** (section 15 has a self-check):

- $t$-statistics and $p$-values;
- the normal distribution and its tails;
- serial correlation;
- the idea of the bootstrap.

---

## 1. Intuition

Ask 100 people to flip a fair coin ten times. Somebody will get nine heads. That person has not found a biased coin; the result is guaranteed by the number of people.

A parameter grid does the same thing. Each configuration is a "person", its backtest is the sequence of flips, and the best Sharpe ratio is the nine heads. Reporting the winner with an ordinary $t$-test answers the question "if I had tried only this one, would it look significant?". But you did not try only this one.

Two facts make the problem worse and better at the same time:

- **Worse:** the maximum of $K$ noisy estimates grows with $K$, roughly like $\sqrt{2\ln K}$ standard errors, even when every true mean is zero.
- **Better:** configurations of one grid are highly correlated (a 230-day and a 240-day momentum rule trade almost the same positions). A grid of 104 configurations contains the selection opportunity of about six independent strategies, not 104.

A good test must charge for the search, but only for the search that actually happened.

The tools in this module answer three questions:

1. **Does *anything* in the grid work?** The Reality Check and the SPA test.
2. **Which configurations work,** with a controlled probability of any false discovery? The Romano-Wolf stepdown.
3. **How many trials did I really run?** The effective number of trials, used by the deflated Sharpe ratio.

Two robustness diagnostics complete the module:

- **Is performance stable over time?** The sup-Wald break test.
- **Does it come from a handful of days?** Episode dependence.

## 2. A concrete numerical example

**Independent strategies with no skill.** Take 20 independent strategies, none with a true edge.

| Quantity | Value |
| --- | --- |
| Probability that at least one has a naive one-sided $t > 1.645$ | $1 - 0.95^{20} = 64\%$ |
| Expected largest $t$-statistic | 1.87 (the mean of the maximum of 20 standard normals) |
| Bonferroni threshold that keeps the family-wise error at 5% | $z_{1-0.05/20} = 2.81$ |

**Correlated strategies.** Suppose instead the 20 strategies share 95% of their variance. They are then almost one strategy:

- the expected maximum falls to about $\sqrt{0.05}\times 1.87 \approx 0.4$;
- a threshold near 2.0 already controls the family-wise error;
- Bonferroni, at 2.81, is far too strict and throws away power.

In `test_romano_wolf_controls_familywise_error_and_beats_bonferroni_under_correlation`, 10 such datasets with a modest true edge give 84 Romano-Wolf rejections against 20 Bonferroni rejections, with no false discoveries in the null simulations above the nominal rate.

**The repository's momentum grid** (104 configurations, 2000-2025, 5 bp costs):

- The best configuration has $t = 1.37$, so a naive $p$-value of 0.085, close to "significant at 10%".
- The SPA bootstrap says that the best of 104 correlated configurations exceeds 1.37 by chance with probability 0.38. The notebook's first chart shows the gap between the two curves.
- With the 12-month rule, fixed in advance, as the benchmark, the best configuration's $t$ is −0.03: the search added nothing.

## 3. Formulation

**Symbols:**

| Symbol | Meaning |
| --- | --- |
| $K$, $T$ | number of strategies (configurations) and of periods |
| $r_{t,k}$, $b_t$ | return of strategy $k$ and of the benchmark in period $t$ (benchmark 0 by default) |
| $d_{t,k} = r_{t,k} - b_t$ | relative performance; $\bar d_k$ its sample mean |
| $\omega_k^2$ | variance of $\sqrt T\,\bar d_k$ (long-run variance); $\hat\omega_k$ its estimate |
| $t_k = \sqrt T\,\bar d_k/\hat\omega_k$ | studentised performance |
| $\bar d^{*}_{b,k}$ | mean of strategy $k$ in bootstrap replicate $b = 1,\dots,B$ |
| $q = 1/L$ | probability of starting a new block in the stationary bootstrap; $L$ the mean block length |

**Hypotheses:**

$$H_0:\ \max_k E[d_{k}] \le 0 \qquad\text{vs}\qquad H_1:\ \max_k E[d_k] > 0 .$$

$H_0$ is *composite*: it holds for every vector of means that are all $\le 0$. The rejection probability must be at most $\alpha$ for all of them, and the hardest case ("least favourable") is when all means equal 0.

**The statistics:**

| Procedure | Statistic | Bootstrap null |
| --- | --- | --- |
| Reality Check | $V = \max_k \sqrt T\,\bar d_k$ | $V^*_b = \max_k \sqrt T(\bar d^*_{b,k} - \bar d_k)$ |
| SPA | $T_{SPA} = \max(0, \max_k t_k)$ | $T^*_b = \max\big(0, \max_k \sqrt T(\bar d^*_{b,k} - g(\bar d_k))/\hat\omega_k\big)$ |
| Romano-Wolf | stepdown on the sorted $t_{(1)} \ge \dots \ge t_{(K)}$ | $t^*_{b,k} = \sqrt T(\bar d^*_{b,k} - \bar d_k)/\hat\omega_k$ |

The $p$-values are the shares of bootstrap statistics at least as large as the observed one.

**The SPA recentring functions** decide where each strategy's null mean sits:

- $g_u(x) = x$: every strategy at the boundary;
- $g_l(x) = \max(x,0)$: every strategy as bad as estimated;
- $g_c(x) = x\,\mathbb 1\{x > -\hat\omega_k\sqrt{2\ln\ln T}/\sqrt T\}$: the consistent choice.

**Romano-Wolf adjusted $p$-values** (Romano and Wolf, 2016):

$$\tilde p_{(j)} = \max_{i\le j}\ \frac1B\#\Big\{b:\ \max_{l\ge i} t^*_{b,(l)} \ge t_{(i)}\Big\}.$$

Reject $H_{(j)}: E[d_{(j)}] \le 0$ when $\tilde p_{(j)} \le \alpha$.

**Effective number of trials:** the $N_{eff}$ such that

$$E[\max \text{ of } N_{eff} \text{ i.i.d. } N(0,1)] = E\big[\max_k t^*_k\big].$$

**Break test** (Andrews, 1993): with $\hat\sigma^2_{LR}$ the Newey-West long-run variance,

$$\sup_{m \in [0.15T,\,0.85T]} W(m), \qquad W(m) = \frac{(\bar d_{1:m} - \bar d_{m+1:T})^2}{\hat\sigma^2_{LR}\,(1/m + 1/(T-m))}.$$

## 4. Derivations

### 4.1 Why the maximum grows like $\sqrt{2\ln K}$

For independent $Z_k \sim N(0,1)$, the union bound gives

$$P(\max_k Z_k > x) \le K\,\bar\Phi(x) \approx K\,\frac{\varphi(x)}{x}.$$

This is of order 1 when $\varphi(x) \approx x/K$, i.e. when $x^2/2 \approx \ln K$, so $x \approx \sqrt{2\ln K}$.

For $K = 20$ that is 2.45, an overestimate of the mean of the maximum (1.87) but the right growth rate. The formula used by the deflated Sharpe ratio refines it:

$$E[\max_k Z_k] \approx (1-\gamma)\,\Phi^{-1}(1 - 1/K) + \gamma\,\Phi^{-1}(1 - 1/(Ke)),$$

where $\gamma$ is the Euler-Mascheroni constant (Bailey and Lopez de Prado, 2014). It is within 2% of the exact 2.043 for $K = 30$ (tested).

### 4.2 Why the Reality Check loses power and how SPA fixes it

The Reality Check centres every bootstrap mean at the sample mean, $V^*_b = \max_k\sqrt T(\bar d^*_{b,k} - \bar d_k)$. This imitates the case where every strategy has true mean exactly 0.

Now add 50 strategies that are clearly *worse* than the benchmark, with large variance. They cannot produce the observed maximum in reality, because their means are negative. In the bootstrap, however, their centred draws are as likely as anyone's to be the maximum. $V^*$ grows, the critical value grows, and the $p$-value of the genuinely good strategy rises. `test_spa_power_and_robustness_to_poor_strategies` shows the RC $p$-value going from below 0.05 to above 0.5 when 50 poor strategies are added.

Hansen (2005) makes two changes:

1. **Studentise.** Dividing by $\hat\omega_k$ puts strategies with different volatilities on the same scale, so a volatile strategy does not dominate the maximum.
2. **Recentre.** A strategy whose mean is significantly negative is left at its negative mean in the bootstrap, so it rarely produces the maximum.

The threshold $\sqrt{2\ln\ln T}$ comes from the law of the iterated logarithm. It grows slowly enough that strategies with mean exactly 0 are, with probability tending to one, never classified as negative, which keeps the size. It also grows fast enough that strategies with strictly negative means are eventually excluded, which gives consistency.

The three $p$-values bracket the answer:

- $p_l$ (lower): liberal;
- $p_c$ (consistent): the one to report;
- $p_u$ (upper): conservative, equal to a studentised Reality Check.

### 4.3 The stationary bootstrap and its variance

Returns are serially dependent: volatility clusters, and positions persist. Resampling single observations would destroy that dependence and understate the variance of a mean.

The stationary bootstrap (Politis and Romano, 1994) resamples blocks whose lengths are geometric with mean $L$, starting at uniform positions, wrapping around the end of the sample. Random lengths make the resampled series stationary, which a fixed block length does not.

Two observations $i$ lags apart land in the same block with probability about $(1-q)^i$. That gives the expected bootstrap variance of $\sqrt T\,\bar d^*$:

$$\hat\omega^2 = \hat\gamma_0 + 2\sum_{i=1}^{T-1}\kappa(T,i)\,\hat\gamma_i,\qquad \kappa(T,i) = \Big(1-\frac iT\Big)(1-q)^i + \frac iT(1-q)^{T-i}.$$

The second term of $\kappa$ accounts for the wrap-around. This is a kernel estimator of the long-run variance with a data-driven kernel. `test_bootstrap_variance_formula_matches_simulation` checks it against 20,000 C++ replicates within 5%, and it reduces to the i.i.d. variance when $q = 1$.

**Choosing $L$** (Politis and White, 2004; corrected by Patton, Politis and White, 2009). The mean-squared error of the variance estimate is minimised by

$$L^* = \Big(\frac{2G^2}{D}\Big)^{1/3}T^{1/3}, \qquad G = \sum_k |k|\gamma_k, \qquad D = 2\Big(\sum_k\gamma_k\Big)^2 .$$

Both sums are estimated with a flat-top lag window whose bandwidth is chosen from the autocorrelations.

**Worked check for an AR(1)** with coefficient $\rho$, where $\gamma_k = \sigma_x^2\rho^{|k|}$:

$$G = 2\sigma_x^2\frac{\rho}{(1-\rho)^2}, \qquad \sum_k\gamma_k = \sigma_x^2\frac{1+\rho}{1-\rho},$$

so $G/\sum_k\gamma_k = 2\rho/(1-\rho^2)$ and $L^* = (2\rho/(1-\rho^2))^{2/3}T^{1/3}$. For $\rho = 0.5$ and $T = 20{,}000$ that is 32.9. The estimator gives 29.7 in the test.

### 4.4 Why the Romano-Wolf stepdown controls the family-wise error

**First step.** Let $c_1$ be the $(1-\alpha)$ quantile of $\max_k t^*_k$ over *all* strategies. Suppose the true nulls are the set $I_0$. A false rejection in the first step needs $\max_{k\in I_0} t_k > c_1$. Under $H_0$ that has probability at most $P(\max_{k\in I_0} t^*_k > c_1) \le P(\max_k t^*_k > c_1) = \alpha$, because the maximum over a subset is smaller.

**Later steps.** After removing the rejected hypotheses, the procedure repeats with the maximum over the remaining set. Its quantile is smaller, so more rejections become possible without breaking the argument. The closure principle and monotonicity of the critical values give family-wise error control across all steps (Romano and Wolf, 2005).

**The key point.** The joint bootstrap distribution uses the correlation between strategies. When they are highly correlated, $\max_k t^*_k$ is barely larger than a single $t^*$, so the threshold is close to the single-test one. Bonferroni ignores this and uses $z_{1-\alpha/K}$ whatever the dependence.

### 4.5 The effective number of trials

The deflated Sharpe ratio compares the selected Sharpe ratio with the expected maximum of $N$ independent null trials. What matters is therefore the expected maximum of the actual, correlated trials. `SnoopingTest.effective_trials` computes the average of $\max_k t^*_{b,k}$ over bootstrap replicates, which keeps autocorrelation and fat tails. It then solves for the $N_{eff}$ that gives the same expected maximum under the formula of 4.1.

The eigenvalue "participation ratio" $(\sum\lambda)^2/\sum\lambda^2$ is also available. It measures the dimension of the *variance* and is much smaller when one factor dominates: for the momentum grid it gives 2.4 against 5.5. That is why it is not used for the DSR.

**Checks.** Identical trials give 1, independent ones give $K$, and $m$ independent groups of identical trials give $m$. All three are tested.

### 4.6 The sup-Wald test and its null distribution

Let $S_m = \sum_{t\le m}d_t$. Under a constant mean, $S_{\lfloor \pi T\rfloor}/(\sigma_{LR}\sqrt T)$ converges to a Brownian motion $B(\pi)$. The Wald statistic for a break at $m = \pi T$ is algebraically

$$W(m) = \frac{T\,(S_m - \pi S_T)^2}{\hat\sigma^2_{LR}\,m(T-m)} \;\Rightarrow\; \frac{(B(\pi) - \pi B(1))^2}{\pi(1-\pi)},$$

the squared standardised Brownian bridge.

The supremum over $\pi \in [0.15, 0.85]$ has no closed-form distribution. The code simulates it: 20,000 Brownian paths on 2,000 steps. Its quantiles, 7.16, 8.74 and 12.26, match Andrews' (1993) tables (7.12, 8.68, 12.16) within simulation and discretisation error.

Trimming is essential: without it the supremum diverges near the ends of the sample.

### 4.7 Purged cross-validation

A label that uses returns from $t_0$ to $t_1$ contains information from that whole interval. If a training observation's interval overlaps the test period, the model has seen test information, and its test score is optimistic.

**Purging** removes every training observation with $[t_0, t_1]$ overlapping $[\min t_0^{test}, \max t_1^{test}]$. **Embargo** also removes the observations just after the test block, whose features may be serially correlated with the test labels.

**Combinatorial purged CV** uses every choice of $k$ of $N$ groups as the test set. Each group is tested $\binom{N-1}{k-1}$ times, and the splits recombine into $\phi = k\binom Nk/N$ full backtest paths (Lopez de Prado, 2018). The tests verify the absence of overlap, the embargo, the partition and these counts. This machinery matters for machine-learning models with multi-period labels; the backtests above have one-period P&L and no label overlap.

## 5. Assumptions

- **Stationarity.** The $d_t$ are strictly stationary and weakly dependent (mixing). The bootstrap reproduces dependence up to the block length.
- **Stability.** The dependence structure and the means are stable over the sample. Regime shifts violate this; the break test has low power against them.
- **The grid is the complete search.** Unreported trials make every test optimistic.
- **Break test.** It assumes a single break in the mean with trimming at 15%. Multiple breaks need sequential procedures (Bai and Perron, 1998).
- **Effective trials.** Normality is assumed for the Gaussian version; the bootstrap version relaxes it.

## 6. Implementation mapped to code

| Concept | Code | Notes |
| --- | --- | --- |
| Joint stationary bootstrap of means | `_core.stationary_bootstrap_means`, `bt::StationaryIndices` (C++) | Stream $(seed, b)$ per replicate: bitwise identical for any thread count |
| Bootstrap index stream (for references and tests) | `_core.stationary_bootstrap_indices` | Same generator as the bootstrap of the Sharpe ratio |
| $\hat\omega^2$ (bootstrap variance) | `stationary_bootstrap_variance` | FFT autocovariances, exact formula above |
| Block length $L^*$ | `optimal_block_length` | Politis-White with the 2009 correction |
| RC, SPA, Romano-Wolf, naive | `SnoopingTest.reality_check`, `.spa`, `.romano_wolf`, `.naive`, `.summary` | One set of draws for all; constant strategies dropped and listed |
| Null distributions (for charts) | `SnoopingTest.null_max_t(recentring)` | `upper` also gives the Romano-Wolf first-step null |
| Effective trials | `SnoopingTest.effective_trials`, `effective_number_of_trials(method)` | Bootstrap and Gaussian estimates |
| DSR with effective trials | `metrics.deflated_sharpe_ratio(..., n_trials=)` | Report both counts |
| Break test | `sup_wald_break` | Newey-West long-run variance, simulated $p$-value |
| Episode dependence | `episode_dependence` | Sharpe ratio after removing the best or worst periods |
| Purged and combinatorial CV | `cross_validation.purged_kfold`, `combinatorial_purged_splits`, `number_of_paths` | Integer positions or timestamps |

## 7. Python or C++?

**Measured.** The bootstrap is the only heavy step: $B \times T \times K$ additions. For the momentum grid ($B$ = 5,000, $T$ = 6,692, $K$ = 104, that is $3.5\times10^9$ additions) it takes about 0.8 s in C++ on four threads. A NumPy version that gathers rows by index would take about 4 s; that was measured on 200 replicates and scaled.

The C++ loop reads each resampled row contiguously and accumulates in time order. That is why the results are bitwise identical across thread counts, a property a parallel NumPy reduction would not guarantee.

Everything else (variances by FFT, recentring, stepdown, block length) is vectorised NumPy on arrays of size $B\times K$ and costs milliseconds. The whole research notebook runs in about 20 seconds.

## 8. Statistical interpretation

- **Size and power.** The Monte Carlo checks in the tests and the notebook reproduce the momentum grid's correlation matrix with no skill:
  - the naive test rejects in 27-38% of samples;
  - the Reality Check, SPA and Romano-Wolf reject in 0-5%.

  With a genuine edge ($t \approx 4.5$), SPA rejects in over 90% of samples. A test with correct size and no power would be useless; both are checked.
- **Family-wise error (FWER) vs false discovery rate (FDR).** Romano-Wolf controls the probability of *any* false discovery. When hundreds of hypotheses are tested and a few false discoveries are acceptable, controlling the expected *share* of false discoveries (Benjamini and Hochberg, 1995) is more powerful. In strategy selection, where one false discovery can receive capital, FWER is the prudent choice.
- **"Not significant" is not "no edge".** With a true annual Sharpe ratio of 0.3, even a single pre-specified strategy needs about $(2/0.3)^2 \approx 44$ years of data to reach $t = 2$. The tests can say "no evidence"; they cannot say "no skill".
- **Statistical vs economic significance.** A configuration can pass SPA and still be worthless after realistic costs and capacity. The notebook reports how the conclusions change with costs.

## 9. Finance interpretation

- **A $t$-statistic of 2 is no longer a discovery.** After decades of data mining on the same data, Harvey, Liu and Zhu (2016) argue for $t > 3$ for new factors. The tests here are the per-study version of that argument.
- **The grid's value lies in robustness, not selection.** The momentum grid shows that performance is smooth across look-backs (robust), and that the in-sample winner is not better than the rule fixed in advance. That is a reason to trade the a-priori rule and ignore the winner, not to reject momentum as an idea.
- **Episode dependence is a risk statement.** A pairs strategy whose Sharpe ratio turns negative when the 5 best days are removed is a short-option-like bet on rare dislocations, and its capital should be sized for their absence.
- **For an allocator,** the relevant questions are, in order:
  1. How many variants were tried (including unreported ones)?
  2. What is the SPA or Romano-Wolf result?
  3. What is the deflated Sharpe ratio with an honest effective count?
  4. How much comes from a few episodes?

## 10. Failure modes

1. **Testing only the reported grid** when many more variants were tried (other universes, cost assumptions, data windows). The committee study counts 184 research trials.
2. **Choosing the benchmark after seeing the results.** A benchmark chosen to make the best strategy look good is data snooping at a higher level.
3. **Using the Reality Check with many poor strategies.** Its power collapses (section 4.2); report SPA.
4. **Studentising with i.i.d. standard errors** on serially correlated P&L. Positions that persist make daily P&L autocorrelated; use the bootstrap (or HAC) variance.
5. **Trusting the DSR alone.** It assumes i.i.d. returns and uses only the selected strategy's moments. In the notebook the best pairs configuration has a DSR of 0.96 with effective trials, but an SPA $p$-value of 0.20.
6. **Too short a block length** for strongly persistent series. It understates variances and inflates significance; the notebook shows the results do not depend on it here.
7. **Ignoring non-stationarity.** A strategy that worked until 2008 and lost afterwards can pass a full-sample test; look at the break test and at subsamples.
8. **Ordinary k-fold on overlapping labels.** Leakage makes validation scores optimistic; use purging and an embargo.

## 11. Alternatives

- **Bonferroni and Holm (1979).** Simple FWER control, but conservative under dependence.
- **Benjamini-Hochberg (1995) and Storey's $q$-values.** FDR control for many hypotheses.
- **The model confidence set** (Hansen, Lunde and Nason, 2011). The set of models not significantly worse than the best, rather than a test against a benchmark.
- **CSCV and PBO** (already in the repository). They measure how often the in-sample winner underperforms out of sample: a complementary question about the *selection procedure* rather than the existence of an edge.
- **Haircut Sharpe ratios** (Harvey and Liu, 2015). An adjusted Sharpe ratio after a multiple-testing correction.
- **Out-of-sample hold-out or paper trading.** The cleanest test, but costly in time; the stepdown tells you which candidates deserve that time.
- **Bayesian shrinkage** of performance estimates towards zero, with a prior calibrated on the distribution of trial results.

## 12. Interview questions with answers

### Basic

1. **You test 20 independent strategies with no skill at the 5% level. What is the chance that at least one looks significant?**
   $1-0.95^{20} \approx 64\%$.
2. **What is the family-wise error rate?**
   The probability of at least one false rejection among all the hypotheses tested.
3. **Why do we need a block bootstrap for returns?**
   Returns and P&L are serially dependent (volatility clustering, persistent positions). Resampling single observations destroys that dependence and understates the variance of means and Sharpe ratios.
4. **What does the Bonferroni correction do, and when is it too conservative?**
   It tests each hypothesis at $\alpha/K$. It is too conservative when the tests are positively correlated, as grid configurations are.
5. **What does PBO measure that SPA does not?**
   How often the in-sample winner falls below the median out of sample, i.e. whether the *selection process* generalises. SPA tests whether *any* configuration beats the benchmark.

### Intermediate

6. **Explain the difference between White's Reality Check and Hansen's SPA test.**
   SPA studentises (so volatile strategies do not dominate) and recentres (so strategies significantly worse than the benchmark are not treated as if at the boundary). The Reality Check loses power as irrelevant poor strategies are added; SPA does not.
7. **How does the Romano-Wolf procedure use the correlation between strategies?**
   Its critical value is a quantile of the bootstrap distribution of the *maximum* $t^*$ over the remaining strategies. Under strong correlation that maximum is close to a single $t^*$, so the threshold is lower than Bonferroni's.
8. **How would you choose the block length?**
   Automatically, by Politis-White (with the Patton-Politis-White correction), and check that conclusions are robust to halving and doubling it.
9. **What is the effective number of trials, and why does it matter for the deflated Sharpe ratio?**
   The number of independent trials whose expected maximum matches that of the correlated trials. The DSR benchmark is an expected maximum, so counting correlated configurations as independent overstates the selection bias.
10. **What is purged cross-validation and when do you need it?**
    It removes training observations whose label intervals overlap the test period, and embargoes those just after it. You need it when labels span several periods (for example, 20-day forward returns), because ordinary k-fold leaks information.

### Advanced

11. **Why does SPA use the threshold $\sqrt{2\ln\ln T}$?**
    By the law of the iterated logarithm, $\sqrt T\,\bar d_k/\omega_k$ for a strategy with mean exactly zero stays above $-\sqrt{2\ln\ln T}$ with probability tending to one, so it is kept at the boundary (correct size). A strictly negative mean pushes the statistic to $-\infty$ at rate $\sqrt T$, so it is eventually excluded (consistency).
12. **Derive the expected variance of the stationary-bootstrap mean.**
    Two observations $i$ lags apart are in the same block with probability about $(1-q)^i$, adjusted for wrap-around. The variance is $\gamma_0 + 2\sum_i\kappa(T,i)\gamma_i$ with $\kappa(T,i) = (1-i/T)(1-q)^i + (i/T)(1-q)^{T-i}$.
13. **What is the asymptotic distribution of the sup-Wald statistic for a break in the mean, and why must the sample be trimmed?**
    The supremum over $\pi\in[\pi_0,1-\pi_0]$ of $(B(\pi)-\pi B(1))^2/(\pi(1-\pi))$. Without trimming, the normalised Brownian bridge is unbounded near 0 and 1, so the supremum diverges.
14. **A strategy has a naive $t$ of 2.4 in sample, SPA $p$ = 0.15 over the grid, and a DSR of 0.96. Out of sample its $t$ is 1.4. What do you conclude, and what would you do?**
    - The naive and DSR results overstate the evidence. The DSR ignores serial dependence and the joint grid; the SPA result is the relevant one, and the out-of-sample drop is consistent with selection bias.
    - Do not allocate on this evidence.
    - If the economic rationale is strong, pre-register one configuration and paper-trade it, count the trial, and check episode dependence and costs.
15. **How would you test a whole research programme (several grids, several universes)?**
    Stack every configuration ever tried into one matrix over a common sample (or relative to a common benchmark) and apply SPA and Romano-Wolf jointly. Count unreported trials, and report the effective number of trials of the union.

## 13. Exercises (answers not provided)

**Level 1: mechanics.**

1. Compute the Bonferroni threshold for 54 one-sided tests at 5%, and the probability that at least one of 54 independent null strategies passes a naive 5% test.
2. With `SnoopingTest`, reproduce the momentum grid's naive, RC and SPA $p$-values from the notebook with a different seed. How much do they move?
3. Use `episode_dependence` on the momentum baseline: how many days must be removed to halve its Sharpe ratio?

**Level 2: understanding the procedures.**

4. Show that the SPA upper $p$-value equals a studentised Reality Check.
5. Construct an example (on paper) where Holm rejects a hypothesis that Romano-Wolf does not. Hint: think about negative dependence.
6. Show that $\kappa(T,i)\to 1$ as $q\to 0$. Interpret the resulting variance estimate, and explain why it is zero.

**Level 3: derivations and design.**

7. Derive the expected maximum of $K$ equicorrelated standard normals with correlation $\rho$ in terms of the independent case. Use the result to approximate the effective number of trials of such a grid.
8. Design a Monte Carlo that estimates the power of SPA to detect one configuration with annual Sharpe 0.5 among 100 correlated nulls with 15 years of daily data. Which parameters matter most?
9. Implement a Benjamini-Hochberg version of the stepdown on the bootstrap $p$-values. Compare the discoveries with Romano-Wolf on simulated grids with 10% true edges.

**Level 4: research.**

10. Stack the three grids of the notebook on their common monthly sample (aggregate the daily ones) and test the union jointly. Compare with the three separate tests.
11. Estimate the effective number of trials of the entire research history of this repository (all notebooks), and recompute the committee's deflated Sharpe ratio with it.
12. Rebuild the momentum grid on tradable futures returns (`backtest_engine.futures`) and repeat the analysis. Does the conclusion depend on the spot proxy?

## 14. Cumulative curriculum

1. [Futures, rolls and carry](../futures/futures_rolls_and_carry.md): tradable returns.
2. **Data snooping and multiple testing** (this note): honest inference about those returns.
3. [Portfolio construction under estimation error](../portfolio/portfolio_construction.md): combining assets or strategies, and testing the combination against 1/N with the tools of this note.
4. Execution and market impact (planned).

## 15. Fundamentals first: self-check

- Can you compute a one-sided $p$-value from a $t$-statistic and state what it is the probability *of*?
- Do you know why the variance of a mean of autocorrelated data is the long-run variance $\sum_k\gamma_k$ divided by $T$, and how Newey-West estimates it?
- Can you explain the bootstrap principle (the empirical distribution as a stand-in for the population) and why it needs the data to be (approximately) exchangeable, or resampled in blocks?
- Can you state the difference between a simple and a composite null hypothesis?
- Do you know what a Brownian motion and a Brownian bridge are?

## 16. Levels of understanding

- **Minimum (use it correctly):**
  - report the number of configurations tried;
  - run `SnoopingTest(...).summary()` and report the SPA consistent $p$-value and the Romano-Wolf discoveries;
  - never present a naive $t$-statistic of a selected strategy as evidence.
- **Quantitative (defend the numbers):**
  - explain why RC and SPA differ;
  - derive the growth of the expected maximum;
  - interpret the effective number of trials and the DSR with it;
  - check size and power by simulation.
- **Deep (design research with it):**
  - explain the recentring threshold, the stepdown's family-wise error proof, and the bootstrap variance formula;
  - choose between FWER and FDR for a given decision problem;
  - build a research process in which every trial is logged and tested jointly.

## 17. References

- Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821-856.
- Bai, J. and Perron, P. (1998). Estimating and testing linear models with multiple structural changes. *Econometrica*, 66(1), 47-78.
- Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*, 40(5), 94-107.
- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B*, 57(1), 289-300.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business and Economic Statistics*, 23(4), 365-380.
- Hansen, P. R., Lunde, A. and Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453-497.
- Harvey, C. R. and Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13-28.
- Harvey, C. R., Liu, Y. and Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5-68.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Lo, A. W. and MacKinlay, A. C. (1990). Data-snooping biases in tests of financial asset pricing models. *Review of Financial Studies*, 3(3), 431-467.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley (chapters 7, 11 and 12).
- Newey, W. K. and West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4), 631-653.
- Patton, A., Politis, D. N. and White, H. (2009). Correction to "Automatic block-length selection for the dependent bootstrap". *Econometric Reviews*, 28(4), 372-375.
- Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.
- Politis, D. N. and White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews*, 23(1), 53-70.
- Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*, 73(4), 1237-1282.
- Romano, J. P. and Wolf, M. (2016). Efficient computation of adjusted p-values for resampling-based stepdown multiple testing. *Statistics and Probability Letters*, 113, 38-40.
- Sullivan, R., Timmermann, A. and White, H. (1999). Data-snooping, technical trading rule performance, and the bootstrap. *Journal of Finance*, 54(5), 1647-1691.
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097-1126.
