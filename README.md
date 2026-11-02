# indquant — systematic trading research for Indian equities (NSE)

A research, backtesting, signal-generation and paper-trading framework for the Indian stock
market, built with the methodology systematic asset managers use: cross-sectional factor
models, walk-forward machine learning, realistic Indian transaction costs, risk-based
portfolio construction, account-level circuit breakers, and — most importantly —
**statistical tests that tell you when a result is probably luck**.

> **Read this first.** No model can guarantee profits, and this one does not claim to.
> Large funds (BlackRock's systematic equity team, AQR, etc.) get their edge from the
> *method* **and** from paid point-in-time data, execution infrastructure and large research
> teams. This repo gives you the method. It is designed to say *"this does not beat the
> index after costs"* when that is the truth — and on free data it often will be.
> It is research software, not investment advice; the author is not a SEBI-registered
> investment adviser. You can lose money.

---

## What it does

| Sleeve | Style | Holding period | Default allocation |
|---|---|---|---|
| **Core** | Long-only multi-factor portfolio of ~25 stocks from the NIFTY 500, monthly rebalance | months | 70% |
| **Swing** | Daily long entries on the 200 most liquid stocks with ATR stop / target brackets | 1–15 days | 30% |

Both sleeves share one data layer, one cost model, one event-driven engine, one risk governor
and one reporting stack. The same code path produces backtests, today's signals and the
paper-trading ledger — so live signals are exactly what the backtest would have done.

### The pipeline

```
NSE universe (NIFTY 500 list, point-in-time membership if you have it)
  └─ liquidity filter (price, 63-day ADV ≥ ₹5 cr, ≥300 days history)
      ├─ CORE: factors → cross-sectional ranks → composite score
      │        (+ walk-forward LightGBM ranker, purged & embargoed)
      │        → top-N with rank buffer → inverse-vol weights, 8% name cap, 25% sector cap
      │        → vol target 15%, market-regime filter, risk governor
      └─ SWING: candidate = model's expected R-multiple after costs ≥ 0.10
               (LightGBM trained on triple-barrier outcomes matching the exact trade rules)
               → 12 veto filters ("reasons NOT to trade") → risk-per-trade sizing
               → 2-ATR stop / 3-ATR target / 15-day time stop / break-even stop
  └─ engine: orders decided at close t, filled at OPEN t+1, integer shares,
             STT + stamp + exchange + SEBI + GST + DP charges + square-root impact slippage
  └─ report: net-of-cost metrics, benchmark, deflated Sharpe, null test, holdout, tax estimate
```

---

## Built to avoid the ways trading models fool people

Most backtests look great and fail live because of a handful of well-known errors. Each one
has a specific defence here:

| Pitfall | Defence in this repo |
|---|---|
| **Look-ahead bias** (using tomorrow's data today) | All features are causal; trades fill at the *next* open. A test corrupts every price after date *t* and asserts that every score, trade and equity value up to *t* is unchanged (`tests/test_no_lookahead.py`). |
| **Leaky ML validation** (overlapping labels) | Walk-forward expanding window with **purging + embargo** (López de Prado). The model used on any day was trained only on outcomes fully known before its fit date. |
| **Pattern-mining without cause and effect** | Every factor must have a registered economic hypothesis (mechanism, expected sign, academic reference) in `research/hypotheses.py`. `indquant research` labels each factor *supported / not significant / CONTRADICTED / NO HYPOTHESIS – data mining*. |
| **Selection bias** ("try 50 variants, show the best") | Every backtest is written to a **trial log**; the **Deflated Sharpe Ratio** automatically uses the true number of configurations you have tried on that data. |
| **Luck** | Probabilistic & deflated Sharpe, block-bootstrap Sharpe confidence interval, alpha t-stat, and a **random-signal null test** (`--null-test 19`): the same rules with random stock picks — if random picks do as well, your signal adds nothing. |
| **Overfitting** | Results are split into *development* and an untouched final *holdout* period; the report warns when the holdout degrades. `indquant sweep` shows parameter sensitivity — prefer flat regions. |
| **Survivorship bias** | Supports a point-in-time constituents file. Without one, every report carries a prominent warning that history is overstated. |
| **Ignoring costs** | Indian delivery charges and market impact are modelled on every fill; reports show cost drag and turnover. |
| **Pure-noise sanity** | Integration tests run the full pipeline on a synthetic **random walk** and assert that it does *not* find significant alpha, and on a market with planted momentum and assert that it does. |

## Reasons *not* to trade (swing veto filters)

A candidate only becomes an order if it passes every check; every rejection is counted and
today's decisions are printed in the signal report ("why / why not"):

`stop_too_tight` · `stop_too_wide` · `chasing_big_move` (near circuit) · `overextended` ·
`erratic_range` (spread proxy) · liquidity cap (≤2% of ADV) · `reentry_cooldown` after a losing
exit · `sector_crowded` · `correlated_with_open_position` · `portfolio_heat` (total open risk) ·
`event_blackout` (results dates) · `negative_news` · market regime · risk-governor halt.

## Account-level circuit breakers (risk governor)

Sits above every strategy (`portfolio/governor.py`, `governor:` in the config):

* **Drawdown kill switch** — at 25% drawdown from peak, stop new entries and cut core exposure;
  resume only after recovery to 15% (hysteresis prevents flip-flopping).
* **Profit lock ("trap doors")** — once equity is +20% above its base, a trailing floor
  (peak − 12%) is armed and ratchets up. Breaking it de-risks for 21 sessions; the lock then
  re-bases and must be re-earned (so it can't trap you permanently in cash).
* **Loss-streak brake** — after 4 consecutive losing trades, position sizes are halved for 10 sessions.

## News sentiment

`indquant news` (or `sentiment.live_fetch: true`) pulls headlines from Indian business RSS
feeds, matches them to companies, scores them with a finance word list and **appends them
to a history file with timestamps**. Sentiment is used as a veto for swing entries.
Backtests only use sentiment from a point-in-time file — old news you never recorded is
never "remembered". After a few months of collected history you can test whether sentiment
actually predicts returns before giving it more weight.

---

## Install

```bash
git clone <this repo> && cd Trading
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                                  # 43 tests, ~30 s
```

Python ≥ 3.10. Dependencies: numpy, pandas, scipy, scikit-learn, lightgbm, yfinance, pyyaml,
matplotlib, pyarrow.

## Quick start

**1. See the whole pipeline on synthetic data (no internet needed):**

```bash
indquant --source synthetic backtest --with-research --null-test 19
# -> output/backtest/<timestamp>/report.html
```

**2. Real NSE data (Yahoo Finance, free):**

```bash
indquant fetch                                   # NIFTY 500 list from NSE + prices (cached)
indquant healthcheck                             # data quality
indquant research                                # factor ICs vs. hypotheses
indquant backtest --with-research --null-test 19 # full report
indquant sweep --param core.top_n --values 15,25,40   # sensitivity (counts as trials)
```

Faster first run on a smaller universe: `--set data.universe=nifty100`.

**3. Paper trade before risking money:**

```bash
indquant paper-init --variant champion --capital 1000000
indquant paper-init --variant concentrated --variant-set core.top_n=15   # a challenger
indquant paper-update        # run after every market close (see cron below)
indquant paper-report        # league table, promotion advice, trade journal
indquant signals             # today's orders (dry run) for the champion
```

### Daily routine (after 4 pm IST)

```cron
# Mon–Fri 16:30 IST: update paper book, write signals; non-zero exit = data problem
30 16 * * 1-5  cd /path/to/Trading && .venv/bin/indquant paper-update >> logs/cron.log 2>&1
```

Each run writes `output/signals/<date>/<variant>/summary.md` and `orders.csv`: what to buy or
sell at the next open, share counts, stop/target levels, holdings, the governor state and the
swing "why / why not" table. If the data looks stale, partial or corrupted the run **refuses
to produce signals** (override with `--force` only if you know why). During market hours
the still-forming daily bar is dropped automatically.

Review `indquant paper-report` weekly; change rules rarely (months, not days), and treat
every change as a new trial.

### Champion / challenger

Run several variants on paper in parallel. `paper-report` compares each challenger with the
champion using a deflated test on their daily return difference, and only *recommends*
promotion after `paper.min_days_before_promotion` (≈6 months) with ≥95% confidence.
Promotion is always a manual `indquant paper-promote --variant NAME`.

---

## A sensible path from research to real money

1. `indquant research` — factors you rely on should be *supported*, none *contradicted*.
2. `indquant backtest --null-test 19` on real data — require: positive excess CAGR after
   costs, deflated Sharpe ≥ 0.95 with your real trial count, null-test p < 0.05, and a
   holdout that is not dramatically worse than development.
3. Get survivorship-free data (point-in-time constituents) and re-check — free-data
   backtests are optimistic.
4. Paper trade ≥ 6 months. Live paper results should be in the range the backtest implied.
5. Only then trade small real size, placing orders manually from `orders.csv`
   (e.g. as AMO orders). Keep paper-trading the same variant alongside as the reference.

If a step fails, the honest conclusion is to buy a low-cost index fund instead.

---

## Configuration

Everything is in `config/default.yaml` (commented). Override with your own YAML
(`--config my.yaml`, merged over defaults) or inline `--set section.key=value`.
Unknown keys are rejected so typos never silently fall back to defaults.

Key sections: `data`, `universe_filter`, `costs`, `allocation`, `core`, `swing`
(+ `swing.filters`), `risk`, `governor`, `sentiment`, `events`, `healthcheck`, `ml`,
`validation`, `tax`, `paper`, `logging`.

### Optional data files

| File | Format | Purpose |
|---|---|---|
| `data.membership_file` | `date,symbol` snapshots; each date lists the full index membership from that date | removes survivorship bias |
| `data.fundamentals_file` | `symbol,date,roe,debt_to_equity,earnings_yield,book_to_price` where `date` = **announcement** date | enables `quality` / `value` factors |
| `sentiment.file` | `symbol,timestamp,score` (publication time, IST) | point-in-time sentiment for backtests |
| `events.file` | `symbol,date[,type]` | results-date blackout for swing entries |
| `data.source=csv` + `data.csv_dir` | one `SYMBOL.csv` per stock: `Date,Open,High,Low,Close,Volume` (adjusted) | paid / self-built data |

## India-specific modelling

* **Costs** (delivery/CNC): STT 0.1% buy & sell, stamp duty 0.015% on buys, NSE transaction
  charge 0.00297%, SEBI ₹10/crore, GST 18% on brokerage + exchange + SEBI fees, DP charge per
  sell, brokerage configurable. Plus slippage = half-spread + square-root market impact.
* **Taxes**: approximate STCG (20%) / LTCG (12.5% above ₹1.25 lakh) per financial year from
  FIFO lots, giving an approximate post-tax CAGR.
* **Benchmark**: NIFTYBEES (Nifty 50 ETF, dividend-adjusted) — an investable alternative.
* **Execution**: decisions at the close, orders for the next open (AMO-friendly), integer
  shares, positions in stocks that stop trading are closed after 10 sessions.
* **Regulation**: this project never places orders. SEBI's 2025 framework for retail
  algorithmic trading routes automated API order placement through broker-approved,
  registered algos; read it before automating anything.

## Project layout

```
config/default.yaml          all parameters
src/indquant/
  data/        yahoo, csv, synthetic market, universe & liquidity, fundamentals, sentiment, events
  features/    factors (momentum, residual momentum, low-vol, beta, reversal, trend, 52w-high), technicals, preprocessing
  research/    factor IC / quantile analysis, hypotheses registry, trial log, random-signal null test
  models/      labels (forward-rank, triple-barrier), purged walk-forward validation, LightGBM ranker
  portfolio/   selection & caps, covariance / vol target / regime, risk governor
  strategies/  factor_portfolio (core), swing
  backtest/    costs, account, engine, metrics, runner, HTML report
  live/        healthcheck, signal generation, SQLite paper book (champion/challenger)
  cli.py       `indquant` command
tests/         unit, look-ahead, integration, live/paper/CLI tests
```

## Known limitations

* Free Yahoo data has gaps, occasional bad ticks (basic cleaning is applied) and no delisted
  stocks. Results on it are optimistic.
* Fundamentals are only usable from a point-in-time file you supply.
* The news word list is simple and headline-to-company matching is heuristic; treat
  sentiment as a veto until your own collected history proves it predicts returns.
* Sleeves are simulated with separate cash; there is no periodic re-balancing between them.
* Tax estimates ignore loss carry-forward, grandfathering and surcharge.

## License

MIT. No warranty. Not investment advice.
