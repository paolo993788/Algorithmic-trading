# Futures, rolls and carry: from contract prices to tradable returns

Learning note for [`backtest_engine.futures`](../../../scripts/backtest_engine/backtest_engine/futures.py) and [`backtest_engine.schwartz_smith`](../../../scripts/backtest_engine/backtest_engine/schwartz_smith.py). It explains why a spot price is not a tradable return, how rolls, continuous series and carry are defined, and how the code is validated against a model whose answers are known in closed form.

**Where this fits.** This is the first note of the systematic-trading track ([curriculum](../README.md)). Every later module (carry and trend strategies, portfolio construction, risk, execution) needs returns that a real position could have earned, and in commodities that means futures.

**Prerequisites** (section 15 lists them with self-checks):

- simple and log returns;
- present value and continuous compounding;
- the definition of a forward contract;
- Brownian motion and the Ornstein-Uhlenbeck process;
- the idea of a risk-neutral measure.

---

## 1. Intuition

**You cannot own the price of oil.** A barrel stored in Cushing, Oklahoma costs money to store and insure, and nobody running a fund holds barrels. The published "spot price" is a quote for immediate delivery. A fund trades futures instead: an agreement to buy or sell a fixed quantity, for delivery in a fixed month, at a price agreed today and settled every day.

A futures contract **expires**. To keep exposure to oil you must sell the contract you hold before it expires and buy a later one. That is the **roll**. If the later contract is more expensive (**contango**), you sell cheap and buy dear at every roll. The price of a *given* contract drifts up towards the spot price as delivery approaches, but a chart of "the front-month price" jumps down at every roll. Neither the jump nor the spot move is what your position earned.

Your P&L is exactly the sum of the price changes of the contracts you actually held, each one over the days you held it. That sentence is the whole module. The rest is careful bookkeeping of *which* contract was held *when*, plus a model to check the bookkeeping.

**Carry** is what you would earn if nothing changed: if the spot price stayed where it is, a contract in backwardation (later contracts cheaper) would rise towards the spot price, and one in contango would fall. The slope of the futures curve therefore measures the "return if nothing happens". Whether that return is actually earned on average is an empirical question about **risk premia**, not an accounting identity. That distinction is the most common interview trap in this area.

## 2. A concrete numerical example

A fund is long 10 WTI crude-oil contracts (NYMEX CL, 1,000 barrels each) in the May contract, and plans to roll into June today.

| | Yesterday | Today (roll day) | Tomorrow |
| --- | --- | --- | --- |
| May contract (CLK) | 59.40 | 60.00 | (not held) |
| June contract (CLM) | 60.55 | 61.20 | 61.50 |
| Contract held after the close | May | **June** (rolled at the close) | June |

- **P&L today:** $10 \times 1{,}000 \times (60.00 - 59.40) = 6{,}000$ USD, earned on the May contract that was held since yesterday's close.
- **The roll itself costs nothing in P&L.** Selling 10 May at 60.00 and buying 10 June at 61.20 are both at market prices. It costs only transaction costs: 20 contracts traded, or 10 calendar spreads.
- **P&L tomorrow:** $10 \times 1{,}000 \times (61.50 - 61.20) = 3{,}000$ USD, earned on June.
- **The unadjusted "front-month" chart** goes 59.40, 61.20, 61.50. Its first change, +1.80, is the true move (+0.60) plus the **roll gap** (+1.20) that nobody earned. A backtest on that chart overstates today's return by 2 percentage points.
- **Roll ratio:** $\rho = 61.20/60.00 = 1.02$. The realised **roll yield** of this roll is $-\ln 1.02 = -1.98\%$.
- **Annualised carry** from the two nearest contracts, one month apart: $(\ln 60.00 - \ln 61.20)\times 12 = -23.8\%$ per year. If the spot price stayed at 60 for a year and the curve kept this shape, rolling monthly would lose about 21% ($1 - e^{-0.238}$).

**The same logic, in April 2020.** The May 2020 WTI contract (CLK2020) settled at **−37.63 USD** on 20 April 2020, one day before its last trade date, 21 April. The EIA Cushing spot series printed −36.98 the same day; it is used by this repository's spot-based backtests and is in the FRED cache.

A position that rolled out of May five business days before expiry had left on 14 April. That is what `roll_dates(calendar, days_before=5)` gives. Such a position never experienced the negative price. A backtest on the spot series books that move as if a fund could have traded it.

## 3. Formulation

**Symbols:**

| Symbol | Meaning |
| --- | --- |
| $t$ | trading date; prices are settlements at the close of $t$ |
| $c$ | a contract, identified by root and delivery month (`CLK2020`) |
| $F^c_t$ | settlement price of contract $c$ at $t$ (currency per unit of the underlying) |
| $M$ | contract multiplier (units per contract: 1,000 barrels for CL) |
| $T_c$ | last trade date of $c$; $\tau = T_c - t$ its time to maturity in years (calendar days / 365.25) |
| $c(t)$ | contract held **after** the close of $t$ (the roll schedule) |
| $n_t$ | number of contracts held after the close of $t$ |
| $s$ | a roll date: $c(s) \neq c(s-1)$ |
| $\rho_s = F^{c(s)}_s / F^{c(s-1)}_s$ | roll ratio |
| $g_s = F^{c(s)}_s - F^{c(s-1)}_s$ | roll gap |
| $P_t = F^{c(t)}_t$ | unadjusted (spliced) continuous price |

**Daily results** of the position held since the previous close:

$$\Delta F_t = F^{c(t-1)}_t - F^{c(t-1)}_{t-1}, \qquad r_t = \frac{\Delta F_t}{F^{c(t-1)}_{t-1}}, \qquad \text{P\&L}_t = n_{t-1}\, M\, \Delta F_t - \text{costs}_t .$$

- $r_t$ is undefined when the previous price is not positive (April 2020); the code returns NaN, while $\Delta F_t$ and the P&L stay well defined.
- **Costs:** $\text{costs}_t = k \sum_c |n_{t,c} - n_{t-1,c}|$, with $k$ the cost per contract. A roll of $n$ contracts trades $2|n|$ contracts.

**Continuous series:**

$$A_t = P_t + \sum_{s>t} g_s \quad(\text{difference-adjusted}), \qquad R_t = P_t \prod_{s>t} \rho_s \quad(\text{ratio-adjusted}),$$

where the sums and products run over roll dates after $t$. That is the "end anchor": the last segment equals market prices. The "start anchor" instead uses $A_t = P_t - \sum_{s \le t} g_s$ and $R_t = P_t / \prod_{s\le t}\rho_s$.

**Carry** between the near-th and far-th nearby contracts, with delivery months $m_{near} < m_{far}$ counted in years:

$$\text{carry}_t = \frac{\ln F^{near}_t - \ln F^{far}_t}{m_{far} - m_{near}} .$$

Carry is positive in **backwardation** ($F^{near} > F^{far}$) and negative in **contango**.

**Expiry rules** (NYMEX, used by `last_trade_date`):

- **CL:** 3 business days before the 25th calendar day of the month before delivery. If the 25th is not a business day, count from the last business day before it.
- **HO and RB:** the last business day of the month before delivery.
- **NG:** the third-last business day of the month before delivery.

## 4. Derivations

### 4.1 Why a futures position earns an excess return

A futures contract costs nothing to enter; only margin, a performance bond, is posted. Every day the exchange credits or debits the price change, $n M \Delta F_t$ (variation margin).

A fund that keeps the full notional $n M F$ in cash as collateral earns interest on that cash **plus** the futures P&L. So the futures return $r_t$ is an *excess* return over the collateral rate. A "total return" futures index is excess return plus T-bill interest.

Daily settlement also makes futures prices differ slightly from forward prices when interest rates are random and correlated with the underlying (Cox, Ingersoll and Ross, 1981). This module works with futures settlements directly, so the distinction never enters.

### 4.2 The continuous-series identities

**Difference adjustment.** Take a roll date $s$ with no other roll between $s-1$ and $s$:

$$A_s - A_{s-1} = \Big(F^{new}_s + \sum_{u>s} g_u\Big) - \Big(F^{old}_{s-1} + g_s + \sum_{u>s} g_u\Big) = F^{new}_s - g_s - F^{old}_{s-1} = F^{old}_s - F^{old}_{s-1} = \Delta F_s .$$

On days without a roll the adjustment terms are equal on both days and cancel. So **changes** of $A$ are exactly the P&L per unit of the contract held.

**Ratio adjustment.** With products instead of sums:

$$\frac{R_s}{R_{s-1}} = \frac{F^{new}_s \prod_{u>s}\rho_u}{F^{old}_{s-1}\rho_s \prod_{u>s}\rho_u} = \frac{F^{new}_s}{F^{old}_{s-1}}\cdot\frac{F^{old}_s}{F^{new}_s} = \frac{F^{old}_s}{F^{old}_{s-1}} = 1 + r_s .$$

So **ratios** of $R$ are exactly the returns of the contract held. Neither identity depends on the anchor: moving the anchor multiplies $R$, or shifts $A$, by a constant within each segment boundary.

**What each series is good for:**

- Use $R$ for anything built on returns: momentum, volatility, correlations.
- Use $A$ for P&L per contract.
- Never use $P$ for either.

`test_continuous_series_identities_and_anchors` checks both identities to $10^{-13}$ on simulated data, and checks that $P$ differs from the true returns only on roll dates.

### 4.3 Decomposition into price move and roll yield

Take logs of the ratio identity and use $P_s = F^{new}_s = \rho_s F^{old}_s$:

$$\ln(1+r_t) = \ln\frac{F^{old}_t}{F^{old}_{t-1}} = \ln\frac{P_t}{P_{t-1}} - \ln\rho_t, \qquad \rho_t \equiv 1 \text{ when there is no roll}.$$

Summed over a period, the log excess return equals the log change of the unadjusted series plus the **realised roll yield** $-\sum_s \ln\rho_s$. Roll yield is positive when the new contract is cheaper than the old one (backwardation).

This is an accounting identity (`return_decomposition`). It does *not* say that roll yield is a free return: in contango the spot price may be *expected* to rise and offset it (section 9).

### 4.4 Cost of carry and the theory of storage

For a storable commodity, no-arbitrage between buying now and storing, and buying forward, gives

$$F(t,T) = S_t\, e^{(r + u - y)(T - t)},$$

where:

- $r$ is the interest rate;
- $u$ is the storage cost rate;
- $y$ is the **convenience yield**, the benefit of holding the physical good (keeping a refinery running, avoiding stock-outs).

The curve is therefore in contango when $r + u > y$ and in backwardation when $y > r + u$. The theory of storage (Kaldor, 1939; Working, 1949) predicts that the convenience yield is high when inventories are low. Low stocks therefore go with backwardation, and abundant stocks (for example, Cushing tanks nearly full in April 2020) with steep contango.

### 4.5 Expected futures returns are risk premia: the Schwartz-Smith model

The model (Schwartz and Smith, 2000) writes the log spot price as $\ln S_t = \chi_t + \xi_t$. The two factors evolve under the physical measure $P$ as

$$d\chi_t = -\kappa\chi_t\,dt + \sigma_\chi dW^\chi_t, \qquad d\xi_t = \mu_\xi\,dt + \sigma_\xi dW^\xi_t, \qquad dW^\chi dW^\xi = \rho\,dt .$$

- $\chi_t$ is a short-term deviation that mean-reverts to zero, for example after a refinery outage.
- $\xi_t$ is the long-run equilibrium level, a random walk that moves with technology and depletion.

**Futures prices.** The market prices each risk with a constant premium, $\lambda_\chi$ and $\lambda_\xi$. Under the pricing measure $Q$ the drifts become $-\kappa\chi_t - \lambda_\chi$ and $\mu^*_\xi = \mu_\xi - \lambda_\xi$. The futures price is the $Q$-expected spot at delivery. Under $Q$, $\ln S_T$ is Gaussian with

- mean $m = e^{-\kappa\tau}\chi_t + \xi_t - \tfrac{\lambda_\chi}{\kappa}(1-e^{-\kappa\tau}) + \mu^*_\xi\tau$;
- variance $v = \tfrac{\sigma_\chi^2}{2\kappa}(1-e^{-2\kappa\tau}) + \sigma_\xi^2\tau + \tfrac{2\rho\sigma_\chi\sigma_\xi}{\kappa}(1-e^{-\kappa\tau})$.

So $F(t,T) = E_Q[S_T] = e^{m + v/2}$:

$$\ln F(t,T) = e^{-\kappa\tau}\chi_t + \xi_t + A(\tau), \qquad A(\tau) = \mu^*_\xi\tau - \frac{\lambda_\chi}{\kappa}(1-e^{-\kappa\tau}) + \tfrac12 v(\tau).$$

`test_futures_price_is_the_risk_neutral_expectation_of_the_spot` simulates the $Q$ dynamics and recovers $F(0,T)$ within four Monte Carlo standard errors.

**Returns.** Under $Q$ a futures price is a martingale, because entering the contract costs nothing:

$$\frac{dF}{F} = e^{-\kappa\tau}\sigma_\chi\, dW^{\chi,Q} + \sigma_\xi\, dW^{\xi,Q}.$$

By Girsanov's theorem, $dW^{\chi,Q} = dW^\chi + (\lambda_\chi/\sigma_\chi)\,dt$ and $dW^{\xi,Q} = dW^\xi + (\lambda_\xi/\sigma_\xi)\,dt$. Substituting gives the drift under $P$:

$$\mu_F(\tau) = E_P\Big[\frac{dF}{F}\Big]\Big/dt = e^{-\kappa\tau}\lambda_\chi + \lambda_\xi .$$

**This is the key result.** The expected return of a futures position is the risk premium, and nothing else. The spot drift $\mu_\xi$ does not appear: with $\lambda_\chi = \lambda_\xi = 0$ a futures position earns zero on average however fast the spot is expected to rise. `test_futures_earn_nothing_without_risk_premia_even_if_spot_drifts` shows this exactly, with zero volatility, through the whole pipeline of calendar, roll schedule and returns.

**Finite horizon.** Drift and volatility depend only on $\tau$, so over $[t, t+h]$ the futures price is lognormal with deterministic coefficients, and

$$E_P\Big[\frac{F(t+h,T)}{F(t,T)}\Big] = \exp\Big(\int_t^{t+h}\mu_F(T-u)\,du\Big) = \exp\Big(\lambda_\xi h + \lambda_\chi\frac{e^{-\kappa(\tau-h)}(1-e^{-\kappa h})}{\kappa}\Big).$$

`expected_gross_return` implements this formula. Two tests check it:

- a direct Monte Carlo check;
- `test_pipeline_earns_the_model_risk_premium`. That test simulates 200 paths of four years. It builds every contract from the real CL calendar, rolls the second nearby five business days before expiry, and compares the average tradable return with the average of this formula over the contracts actually held. The difference is within four standard errors, while the premium itself is more than eight standard errors from zero, so the test would catch a wrong contract or a misplaced roll.

### 4.6 The Samuelson effect

From $dF/F$ above, the instantaneous volatility of a contract is

$$\sigma_F(\tau)^2 = e^{-2\kappa\tau}\sigma_\chi^2 + \sigma_\xi^2 + 2e^{-\kappa\tau}\rho\sigma_\chi\sigma_\xi .$$

Short-term shocks $\chi$ die out before distant delivery dates, so near contracts are more volatile than far ones, and $\sigma_F(\tau) \to \sigma_\xi$ as $\tau \to \infty$ (Samuelson, 1965). With the test parameters ($\kappa = 1.5$, $\sigma_\chi = 0.35$, $\sigma_\xi = 0.20$, $\rho = 0.3$), $\sigma_F$ is 41.5% one month from expiry, 29.5% at six months and 20.6% at two years.

## 5. Assumptions

- **Settlement prices are tradable.** A fund can buy and sell at the daily settlement plus a cost. In reality, trading at settlement uses TAS (trade at settlement) orders or incurs slippage.
- **The expiry calendar is known in advance and correct.** The holiday calendar approximates the exchange: NYSE rules, with unscheduled closures passed explicitly. On real data, misalignment is detected with `roll_alignment_report`.
- **Liquidity is sufficient at the chosen nearby** and roll date. Liquidity migrates from the front to the second contract in the days before expiry; rolling too late (April 2020) or into a thin back month is costly.
- **Costs are linear in contracts traded.** There is no market impact; the market-impact model comes later in the roadmap.
- **Collateral** earns the risk-free rate, which is omitted from excess returns.
- **For the Schwartz-Smith validation:**
  - constant parameters and risk premia;
  - Gaussian factors;
  - maturity measured to the last trade date;
  - no seasonality (natural gas and gasoline are strongly seasonal; the model is a laboratory, not a description of those markets).

## 6. Implementation mapped to code

| Concept | Function | Notes |
| --- | --- | --- |
| Contract specifications | `SPECS`, `ContractSpec` | Multiplier, tick and tick value, expiry rule (CME Group specifications) |
| Exchange holidays, Good Friday | `us_exchange_holidays`, `easter_sunday`, `trading_days` | NYSE rules; `extra` for unscheduled closures |
| Last trade date | `last_trade_date`, `expiry_calendar` | NumPy business-day arithmetic (`np.busday_offset`) |
| k-th nearby (EIA convention) | `nearby_contracts` | A contract stays nearby up to and including its last trade date |
| Roll schedule | `roll_dates`, `roll_schedule` | Uses only the calendar, so it cannot look ahead |
| Nearby panel to contract panel | `nearby_to_contract_panel` | Needed for EIA "Contract 1..4" data |
| Returns, gaps, ratios | `held_contract_returns` | NaN return when the base price is not positive |
| Continuous series | `continuous_price(method, anchor)` | `none`, `difference` or `ratio`; `end` or `start` anchor |
| Roll-yield decomposition | `return_decomposition` | Exact identity |
| Carry | `carry(near, far)` | Annualised log slope by delivery month |
| Positions and P&L | `contract_positions`, `futures_pnl` | Integer contracts, daily mark to market, 2 trades per contract rolled |
| Sizing and margin | `contracts_for_risk`, `margin_utilisation` | Margin per contract is an input (exchanges change it) |
| Data validation | `validate_contract_panel`, `roll_alignment_report` | Structural checks; pooled t-statistic for calendar alignment |
| Data | `load_eia_nearby`, `load_contract_csv`, `parse_eia_v2` | Key from `EIA_API_KEY`; data cached, never committed |
| Validation model | `SchwartzSmith`, `contract_panel` | Closed-form futures, premia, volatility; exact simulation |

**The timing convention,** stated once and used everywhere:

1. The contract held after the close of $t$ is decided at the close of $t$.
2. The P&L of day $t$ belongs to the contract held after the close of $t-1$.
3. On a roll date, the old contract earns that day's move and the new one starts earning the next day.

`test_returns_use_only_past_prices` changes every price after a date and checks that nothing before it moves.

## 7. Python or C++?

The futures layer is Python (NumPy and pandas), on purpose:

- Every operation is a vectorised pass over dates, or a `searchsorted` over at most a few hundred contracts.
- Building one simulated four-year path, the 72-contract panel and its tradable returns takes a few milliseconds. The 200-path validation test runs in about one second.
- Calendar logic (holidays, rules, contract codes) is exactly the kind of code that is clearer and safer in Python.

C++ is used where it is measured to matter: parameter grids and resampling over thousands of configurations. The existing C++ momentum engine accepts any price panel, so it runs unchanged on ratio-adjusted futures series. By the identity in 4.2, ratio-adjusted series produce exactly the tradable returns.

## 8. Statistical interpretation

**The validation tests are hypothesis tests.** For example:

- Null hypothesis $H_0$: the pipeline's mean tradable return equals the model's expected return.
- Statistic: $z = (\bar r - \bar\mu)/\widehat{SE}$ with $\widehat{SE} = s/\sqrt{N}$.
- Decision: reject if $|z| > 4$.

Under $H_0$ and approximate normality, a Type I error (a correct implementation failing) has probability about $6\times10^{-5}$ per test. That is a deliberately conservative threshold, so the suite is not flaky.

**A Type II error** (a bug passing) is the real danger. That is why each test also asserts **power**: the pipeline test requires the premium itself to be more than eight standard errors from zero. A bug that mixed up contracts would then produce an error comparable to the premium and be caught. Always ask a Monte Carlo test two questions: "Could it fail by chance?" and "Would it fail if the code were wrong?"

**Detecting a premium in real data is hard.** In the validation model, a second-nearby position has an expected excess return of about 14% a year with 38% volatility: a Sharpe ratio of 0.36. The $t$-statistic of a mean return grows like $SR\sqrt{\text{years}}$. Reaching $t = 2$ needs $(2/0.36)^2 \approx 31$ years of data for one contract, with parameters chosen to make the premium *large*.

This is why commodity risk premia are studied in cross-sections and long samples (Gorton and Rouwenhorst, 2006, use 1959-2004). It is also why a single commodity's backtest is weak evidence either way.

**Statistical versus economic significance:**

- A carry signal can be statistically significant across 30 commodities and still be economically useless after roll costs and market impact.
- A large average return on one contract can be economically tempting yet statistically indistinguishable from zero.

Report both.

## 9. Finance interpretation

- **Contango is not a guaranteed loss.** Under the expectations hypothesis, futures prices equal expected spot prices. A contango curve then means the spot is expected to rise, and the negative roll yield is on average offset by that rise, leaving zero expected excess return. Only the **risk premium** is earned on average (4.5).
- **Keynes' normal backwardation (1930):** producers hedge by selling futures and pay speculators a premium, so futures prices sit below expected spot prices and longs earn a positive expected return.
- **Hedging pressure:** the premium varies with who needs to hedge (Bessembinder, 1992; de Roon, Nijman and Veld, 2000).
- **Carry as a signal:** across asset classes, carry predicts returns (Koijen, Moskowitz, Pedersen and Vrugt, 2018). In commodities, the basis carries information about both spot and term premia (Szymanowska, de Roon, Nijman and van den Goorbergh, 2014). Whether this holds net of costs in a small universe must be tested out of sample, with the number of variants tried reported.
- **April 2020.** Storage at Cushing was nearly full, and the expiring May contract required taking physical delivery. Holders who could not take barrels had to sell at any price the day before expiry. The event was a failure of the *expiring* contract under a physical constraint. It is not a return that a rolled futures strategy, let alone a spot position, could earn. A backtest must treat it exactly as the contracts held did.
- **Spot proxies in this repository.** The existing momentum and pairs notebooks use EIA spot prices. The audit flagged this; the notebooks will be rebuilt on futures returns once contract data are available.

## 10. Failure modes

1. **Signals or returns on the unadjusted front series.** Momentum sees fake trends from roll gaps, and volatility estimates are inflated on roll days.
2. **Level-based signals on end-anchored adjusted series.** The level at date $t$ was not known at $t$ (later rolls rewrote it), so a rule such as "buy below 50 USD" becomes a look-ahead. Use start-anchored series or returns (`test_point_in_time_property_of_the_anchor`).
3. **Ratio adjustment through non-positive prices.** Undefined: the code raises.
4. **Difference-adjusted series used for returns.** They can be negative and do not preserve percentage changes.
5. **Rolling on the last trade date** of physically delivered contracts. This exposes the position to delivery squeezes; `days_before` exists for a reason.
6. **A wrong holiday calendar or expiry rule.** Rolls land on the wrong day and positions are held into expiry. Detect it with `roll_alignment_report` and published expiry dates.
7. **Missing prices for the held contract.** P&L cannot be computed; the code refuses (`futures_pnl` raises) rather than silently assuming zero.
8. **Roll costs forgotten.** Each roll is two trades, or one calendar spread; at monthly rolls these add up.
9. **Sizing with fractional contracts** in small accounts. Rounding error can be a large share of the intended exposure (`contracts_for_risk`).
10. **Survivorship and data vintage.** Contract lists and settlement corrections must come from the same point-in-time source.
11. **Interpreting roll yield as alpha.** It is an accounting component, not an expected return (section 9).
12. **Monte Carlo tests without power analysis.** They pass even when the code is wrong (section 8).

## 11. Alternatives

- **Roll rules:**
  - fixed days before expiry (implemented);
  - on a fixed business day of the month, like the S&P GSCI roll over its 5th to 9th business days;
  - volume- or open-interest-based: roll when the next contract becomes more liquid, using only the previous day's volume;
  - optimised curve position: hold the contract with the best carry net of costs, as in "enhanced" commodity indices.
- **Constant-maturity series:** interpolate between contracts to a fixed $\tau$, for example 60 days. This is good for research on term structure but not directly tradable, because it needs two contracts in proportions that change daily.
- **Calendar-spread execution of rolls:** usually tighter than two outright trades.
- **Richer models:**
  - Gibson and Schwartz (1990): stochastic convenience yield;
  - Schwartz (1997): three factors, with stochastic rates;
  - models with seasonality for natural gas and gasoline;
  - estimation of Schwartz-Smith on a futures panel by the Kalman filter, as in the original paper, a natural next exercise.

## 12. Interview questions with answers

### Basic

1. **What is the P&L of being long 10 CL contracts when the price rises by 1 USD?**
   $10 \times 1{,}000 \times 1 = 10{,}000$ USD. One tick (0.01) on one contract is 10 USD.
2. **Define contango and backwardation.**
   Contango: later delivery months are more expensive ($F^{far} > F^{near}$). Backwardation: they are cheaper. Carry is negative in contango and positive in backwardation.
3. **Why is a futures return an excess return?**
   Entering a futures contract costs nothing (margin is a deposit, not a purchase), so the daily P&L is earned on top of whatever the collateral earns. A collateralised position returns the futures excess return plus the cash rate.
4. **What is wrong with backtesting on a "continuous front-month" chart?**
   It jumps by the roll gap at every roll; nobody earns those jumps. Returns, volatility and momentum signals are all contaminated.
5. **What is a roll?**
   Closing the contract you hold before it expires and opening a later one, to keep exposure without taking delivery.

### Intermediate

6. **Back-adjusted (difference) or ratio-adjusted series for a momentum signal?**
   Ratio-adjusted. Its percentage changes equal the returns earned, whereas differences preserve dollar P&L per contract but distort percentages and can go negative.
7. **Does contango mean a long futures position loses money?**
   Not on average. Futures prices embed expected spot changes plus a risk premium. Contango implies negative roll yield, but if it reflects an expected spot rise, the two offset. The expected return is the risk premium (Schwartz-Smith: $e^{-\kappa\tau}\lambda_\chi + \lambda_\xi$), whatever the slope.
8. **How would you choose the roll date for WTI?**
   Before the last trade date and before liquidity leaves the expiring contract, avoiding delivery and squeeze risk, typically a few business days to two weeks before expiry. Validate with volume and open interest, account for roll costs (calendar spreads), and keep the rule fixed in advance: choosing roll dates by backtest performance is overfitting.
9. **What is the Samuelson effect, and which model feature produces it?**
   Near contracts are more volatile than far ones. Mean-reverting short-term shocks ($\kappa > 0$) are forgotten by the time distant contracts deliver, so $\sigma_F(\tau)$ decreases towards the volatility of the long-run factor.
10. **Why can end-anchored back-adjusted prices create look-ahead?**
    Each new roll rewrites all earlier levels. Any rule using price *levels* (thresholds, support lines, price divided by a fixed number) uses information from the future. Returns and ratios are unaffected.

### Advanced

11. **Derive the expected excess return of a futures contract in the Schwartz-Smith model.**
    $F$ is a $Q$-martingale with $dF/F = e^{-\kappa\tau}\sigma_\chi dW^{\chi,Q} + \sigma_\xi dW^{\xi,Q}$. The change of measure $dW^{Q} = dW^{P} + (\lambda/\sigma)\,dt$ adds the drift $e^{-\kappa\tau}\lambda_\chi + \lambda_\xi$ under $P$. Over a finite horizon, integrate the drift: $E[F_{t+h}/F_t] = \exp(\lambda_\xi h + \lambda_\chi e^{-\kappa(\tau-h)}(1-e^{-\kappa h})/\kappa)$.
12. **How would you test whether carry predicts commodity futures returns?**
    - Build point-in-time carry from the curve at $t$ and returns from $t$ to $t+1$ on the contracts actually held, net of costs.
    - Use non-overlapping or HAC-adjusted inference, and cross-sectional portfolios, with Fama-MacBeth or time-series regressions.
    - Control for the number of commodities and variants tried (Reality Check, SPA, Romano-Wolf).
    - Report out-of-sample results and the dependence on episodes such as 2008 and 2020.
13. **What happened on 20 April 2020, and how should a backtest treat it?**
    The May WTI contract settled at −37.63 USD the day before expiry, because holders who could not take delivery at a nearly full Cushing had to sell. A backtest should book the P&L only for positions in that contract on that day: typically none for a strategy rolling days before expiry. It needs code that handles negative prices (P&L defined, percentage return undefined) instead of crashing or silently dropping the day.
14. **How would you estimate the Schwartz-Smith model?**
    As a state-space model:
    - measurement equation: $\ln F_{k,t} = e^{-\kappa\tau_{k,t}}\chi_t + \xi_t + A(\tau_{k,t}) + \varepsilon_{k,t}$ for the contracts observed on each date;
    - transition: the exact discrete-time version of the factor dynamics.

    Estimate by Kalman-filter maximum likelihood (Schwartz and Smith, 2000). The risk premia $\lambda$ are notoriously imprecise (section 8); $\kappa$ and the volatilities are much better identified by the cross-section of maturities.
15. **A strategy's backtest on spot prices shows a Sharpe ratio of 0.8. What do you ask before believing it?**
    - Which contracts would have been traded, with what roll rule?
    - What is the roll yield of that market (contango erodes returns that the spot series never shows)?
    - Do signal and returns come from the same tradable series?
    - What are the costs per roll?
    - Is the result driven by expiry events that the traded contracts would not have experienced?

## 13. Exercises (answers not provided)

**Level 1: mechanics.**

1. Compute the last trade date of CLN2020 (July 2020 delivery) by hand from the rule, and check it with `last_trade_date`.
2. A curve shows 70.00 (front) and 69.30 (second, one month later). Compute the annualised carry, and say whether it is contango or backwardation.
3. You hold 3 NG contracts (10,000 MMBtu each) and the price moves from 2.500 to 2.463. Compute the P&L in USD and in ticks.

**Level 2: identities and code.**

4. Prove that the start-anchored difference-adjusted series has the same daily changes as the end-anchored one.
5. Construct a three-day contract panel in which ratio adjustment fails, and explain which assumption breaks.
6. Using `roll_schedule` with `nearby=2`, explain why a roll date of the *first* contract can still change the contract held.

**Level 3: derivations and design.**

7. Derive $E_P[\ln F(t+h,T) - \ln F(t,T)]$ in the Schwartz-Smith model. Explain why it differs from the log of the expected gross return.
8. Design a volume-based roll rule that provably uses no future information. Write the test that would prove it (in the style of `test_returns_use_only_past_prices`).
9. For a contract with annual excess-return volatility of 30%, how many years of data are needed to detect a 4% annual premium with a one-sided 5% test and 80% power? Show the computation.

**Level 4: research.**

10. Once contract data are available, reproduce the WTI time-series momentum study on ratio-adjusted futures instead of spot prices. Decompose the difference in performance into roll yield and signal changes, and report how many variants you tried.
11. Estimate the Schwartz-Smith model by Kalman filter on CL contracts 1-4. Compare the implied volatility term structure with realised volatilities by maturity, and test whether $\lambda_\chi$ and $\lambda_\xi$ are distinguishable from zero.
12. Test whether the carry of CL, HO, RB and NG predicts next-month returns. Control for multiple testing across the four commodities and several roll rules, and discuss why four assets are too few for a cross-sectional strategy.

## 14. Cumulative curriculum

This note is step 1 of the systematic-trading track in [`docs/learning/README.md`](../README.md):

1. **Futures, rolls and carry** (this note): tradable returns.
2. Statistical validation of strategies: Reality Check, SPA, Romano-Wolf, purged cross-validation (planned).
3. Portfolio construction and risk: covariance estimation, risk parity, constraints, costs (planned).
4. Execution and market impact (planned).

Each later note assumes the timing convention and return definitions established here.

## 15. Fundamentals first: self-check before reading on

- Can you convert between simple and log returns, and explain why log returns add over time?
- Can you price a forward on a non-dividend asset by no-arbitrage, $F = S e^{r\tau}$, and extend it to storage costs and convenience yield?
- Do you know the distribution of an Ornstein-Uhlenbeck process at a future date?
- Can you state Girsanov's theorem for a constant change of drift?
- Can you compute the standard error of a sample mean, and explain why daily returns make it large relative to annual premia?

If any answer is no, review Hull (futures pricing, chapters on forwards and futures and on commodities) and any stochastic-calculus primer on the Ornstein-Uhlenbeck process before sections 4.5 and 4.6.

## 16. Levels of understanding

- **Minimum (you can use the code correctly):**
  - you know which contract is held on each day and why the unadjusted series is wrong;
  - you use ratio-adjusted series for returns and signals;
  - you roll before expiry and count roll costs.
- **Quantitative (you can defend the numbers):**
  - you can prove the adjustment identities and the roll-yield decomposition;
  - you can compute carry and interpret it as an expected return only under stated assumptions;
  - you can read a Monte Carlo test by its size and power.
- **Deep (you can do research with it):**
  - you can derive futures prices and risk premia in a factor model and connect them to the theory of storage and hedging pressure;
  - you can explain why premia are hard to detect;
  - you can design tests of carry and momentum on tradable contracts that survive multiple-testing corrections and the scrutiny of a portfolio manager.

## 17. References

- Bessembinder, H. (1992). Systematic risk, hedging pressure, and risk premiums in futures markets. *Review of Financial Studies*, 5(4), 637-667.
- CME Group. Contract specifications: Light Sweet Crude Oil (CL), NY Harbor ULSD (HO), RBOB Gasoline (RB), Henry Hub Natural Gas (NG). https://www.cmegroup.com
- Cox, J. C., Ingersoll, J. E. and Ross, S. A. (1981). The relation between forward prices and futures prices. *Journal of Financial Economics*, 9(4), 321-346.
- de Roon, F. A., Nijman, T. E. and Veld, C. (2000). Hedging pressure effects in futures markets. *Journal of Finance*, 55(3), 1437-1456.
- Erb, C. B. and Harvey, C. R. (2006). The strategic and tactical value of commodity futures. *Financial Analysts Journal*, 62(2), 69-97.
- Gibson, R. and Schwartz, E. S. (1990). Stochastic convenience yield and the pricing of oil contingent claims. *Journal of Finance*, 45(3), 959-976.
- Gorton, G. and Rouwenhorst, K. G. (2006). Facts and fantasies about commodity futures. *Financial Analysts Journal*, 62(2), 47-68.
- Hull, J. C. *Options, Futures, and Other Derivatives*. Pearson (chapters on forward and futures prices and on commodity futures).
- Kaldor, N. (1939). Speculation and economic stability. *Review of Economic Studies*, 7(1), 1-27.
- Keynes, J. M. (1930). *A Treatise on Money*. Macmillan.
- Koijen, R. S. J., Moskowitz, T. J., Pedersen, L. H. and Vrugt, E. B. (2018). Carry. *Journal of Financial Economics*, 127(2), 197-225.
- Meeus, J. (1991). *Astronomical Algorithms*. Willmann-Bell (Gregorian Easter algorithm).
- Moskowitz, T. J., Ooi, Y. H. and Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228-250.
- Samuelson, P. A. (1965). Proof that properly anticipated prices fluctuate randomly. *Industrial Management Review*, 6(2), 41-49.
- Schwartz, E. S. (1997). The stochastic behavior of commodity prices: implications for valuation and hedging. *Journal of Finance*, 52(3), 923-973.
- Schwartz, E. S. and Smith, J. E. (2000). Short-term variations and long-term dynamics in commodity prices. *Management Science*, 46(7), 893-911.
- Szymanowska, M., de Roon, F., Nijman, T. and van den Goorbergh, R. (2014). An anatomy of commodity futures risk premia. *Journal of Finance*, 69(1), 453-482.
- Working, H. (1949). The theory of price of storage. *American Economic Review*, 39(6), 1254-1262.
