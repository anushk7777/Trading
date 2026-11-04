# Beginner's guide: from zero to your first trade

This guide takes you from an empty computer to placing your first real trade with
indquant, one step at a time. No programming experience is needed: you will copy commands
into a terminal and click buttons in your browser.

> **Read this before you start.** indquant is a research and decision-support tool, not a
> money machine and not investment advice. It may tell you, honestly, that its strategy does
> **not** beat a simple NIFTY index fund. If that happens, believe it. Only ever trade with
> money you can afford to lose completely. The path below deliberately puts **months of
> testing and paper trading** before any real money.

**The road map**

| Stage | What you do | Time |
|---|---|---|
| 1. [Set up your computer](#1-set-up-your-computer) | Install Python and Git, download indquant | 30 min |
| 2. [Your settings](#2-your-settings-file) | Copy one file and set your capital | 5 min |
| 3. [Practice on simulated data](#3-practice-run-on-simulated-data-no-internet-needed) | Run a backtest, learn to read the report | 15 min |
| 4. [Test on real NSE data](#4-test-the-strategy-on-real-nse-data) | Download history, check data, run the real backtest | 1–2 hours |
| 5. [Decide: go or no-go](#5-go-or-no-go-reading-the-results-honestly) | Compare the results with a checklist | 30 min |
| 6. [Paper trade](#6-paper-trade-for-at-least-6-months) | Fake money: the console from tomorrow, plus the automatic book | **6+ months** |
| 7. [Prepare for real money](#7-prepare-for-real-money) | Broker account, capital, optional AI reviewer | 1 day |
| 8. [Your first real trade](#8-your-first-real-trade-step-by-step) | Console → approve → broker app → stop-loss | 2 days |
| 9. [Daily and weekly routine](#9-daily-and-weekly-routine) | 10 minutes each evening | ongoing |
| [Troubleshooting](#troubleshooting) · [Glossary](#glossary) | | |

---

## 1. Set up your computer

You need a Windows, macOS or Linux computer with internet. Everything is free.

### 1.1 Install Python (version 3.10 or newer)

* **Windows:** download Python from <https://www.python.org/downloads/>. In the installer,
  **tick "Add python.exe to PATH"** at the bottom of the first screen, then click *Install Now*.
* **macOS:** download the macOS installer from the same page, or run `brew install python`
  if you use Homebrew.
* **Linux (Ubuntu/Debian):** `sudo apt install python3 python3-venv python3-pip git`

### 1.2 Install Git

* **Windows:** <https://git-scm.com/download/win>. Click *Next* through the installer and keep
  the defaults.
* **macOS:** run `git --version` in Terminal; if Git is missing, macOS offers to install it.
* **Linux:** installed by the command above.

### 1.3 Open a terminal

* **Windows:** press the Start key, type **PowerShell**, open it.
* **macOS:** press Cmd + Space, type **Terminal**, open it.

Check that both tools work. Each command should print a version number:

```bash
python --version      # macOS/Linux may need: python3 --version
git --version
```

### 1.4 Download indquant

```bash
git clone -b ccr-59b7c76d-j90rbr https://github.com/anushk7777/Trading.git
cd Trading
```

(If the work has been merged into the main branch, you can leave out `-b ccr-59b7c76d-j90rbr`.)

**From now on, always run commands from inside this `Trading` folder.**

### 1.5 Create a private Python environment and install

This keeps indquant's packages separate from the rest of your computer.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,ai]"
```
If PowerShell says *"running scripts is disabled"*, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ai]"
```

Your prompt now starts with `(.venv)`. **Every time you open a new terminal**, go to the
`Trading` folder and run the activate line again (`.venv\Scripts\Activate.ps1` on Windows,
`source .venv/bin/activate` on Mac/Linux).

### 1.6 Check the installation

```bash
pytest
```

You should see all tests pass after about a minute. If something fails, see
[Troubleshooting](#troubleshooting).

---

## 2. Your settings file

```bash
# Windows
copy config\example.yaml config\my.yaml
# macOS / Linux
cp config/example.yaml config/my.yaml
```

Open `config/my.yaml` in any text editor (Notepad is fine) and set:

* `initial_capital`: the amount you would trade with, in rupees.
* `core.top_n`: how many stocks the main portfolio holds. Use the table below.

indquant reads `config/my.yaml` automatically. Every other setting is explained in
`config/default.yaml`; don't change those until you understand them.

**How much capital makes sense?** Every sale costs a fixed ~₹16 depository (DP) charge per
stock, and you can only buy whole shares. Very small positions therefore lose a noticeable
share of their value to charges.

| Capital you plan to trade | Suggested setting |
|---|---|
| Under ₹2 lakh | Paper trade only for now; a NIFTY index fund is a sensible home for the money meanwhile |
| ₹2–5 lakh | `core.top_n: 10`; consider `allocation: {core: 1.0, swing: 0.0}` to keep positions large enough |
| ₹5–15 lakh | the example file as it is (`top_n: 20`, 70/30 split) |
| Above ₹15 lakh | defaults (`top_n: 25`) |

---

## 3. Practice run on simulated data (no internet needed)

Run a complete backtest on a made-up market. This shows you what everything looks like
without downloading anything:

```bash
indquant --source synthetic backtest --with-research
```

It takes about a minute and ends with a line like `Report: output/backtest/2026…/report.html`.
Open that file in your browser by double-clicking it in the `output/backtest` folder.

**How to read the report**

* **"Read this first" (yellow boxes):** honest warnings. With simulated data you will see
  "SYNTHETIC DATA" and probably "NOT STATISTICALLY SIGNIFICANT". That is expected: these
  numbers mean nothing about the real market.
* **Performance summary:**
  * **CAGR:** average yearly growth.
  * **Max drawdown:** the worst fall from a peak, i.e. how much pain to expect.
  * **Sharpe:** return per unit of risk; higher is better.
  * **CAGR vs benchmark:** how much better or worse than simply holding NIFTY.
  * **Deflated Sharpe:** the probability the result isn't luck; you want ≥ 0.95.
  * **Alpha after factors:** the return left over after removing well-known factor effects.
* **Charts:** growth of your money against the benchmark, drawdowns, and how invested
  each strategy was.
* **Factor attribution / stress tests:** what drives the returns, and how the portfolio did
  in past crashes (COVID 2020, 2022 …).

---

## 4. Test the strategy on real NSE data

### 4.1 Download history (one time, then quick daily updates)

```bash
indquant fetch
```

Because `config/my.yaml` says `source: nse`, this downloads NSE's official daily files
("bhavcopies") from 2014 onwards, including companies that were later delisted. Studying only
today's survivors would make history look better than it was. **The first download takes
roughly 30–60 minutes.** It downloads slowly on purpose to be polite to NSE, and you can stop
and restart it at any time.

### 4.2 Check the data

```bash
indquant data-audit
```

This compares the NSE data with Yahoo Finance for 100 large stocks and lists any day where
they disagree. A few mismatches are normal (dividends). If a stock shows a big mismatch on
one date, a split or bonus was probably handled wrongly. Add a line for it to a
corporate-actions file (see `nse.corporate_actions_file` in `config/default.yaml`).

### 4.3 Check the ideas, then run the real backtest

```bash
indquant research
indquant backtest --with-research --null-test 19
```

`research` shows which factors actually predicted returns in Indian data. The backtest
takes several minutes (it re-trains the machine-learning models through history the way
they would have been trained at the time) and writes a report like the one in step 3.

### 4.4 Optional: smart-money data (bulk deals and insider buying)

indquant can follow public "smart money" disclosures: bulk deals (large trades, with the
buyer's name) and promoters or directors buying their own company's shares. It only follows
bulk-deal buyers whose past record survives a statistical luck test.

```bash
indquant disclosures fetch --since 2018-01-01   # tries NSE's website; may be blocked
indquant smart-money                             # who passes the luck test, recent insider buying
```

If the fetch is blocked, download the CSV files yourself from nseindia.com (bulk deals, and
insider trading under "Corporate filings") and import them:
`indquant disclosures import --bulk bulk.csv --insider insider.csv`. Then re-run step 4.3:
`research` tells you whether these signals actually worked in Indian data.

---

## 5. Go or no-go: reading the results honestly

Open the new report. **Only continue to paper trading if most of these are true:**

| Check | Where in the report | Good sign |
|---|---|---|
| Beats the index after all costs | "CAGR vs benchmark" | clearly positive |
| Not luck | "Deflated Sharpe" | ≥ 0.95 |
| Real skill, not just market exposure | "Alpha t-stat" | ≥ 2 |
| Not just a factor ETF in disguise | "Factor-adjusted alpha t-stat" | positive, ideally ≥ 2 |
| Better than random stock picks | "random-signal test" warning | p < 0.05 |
| Still works in recent years | "Holdout" column vs "Development" | not much worse |
| You can live with the losses | "Max drawdown" | a fall you would not panic-sell in |

If you tried several settings (`indquant sweep …`), also check the **probability of
backtest overfitting** it prints: below 30% is reassuring, 50% or more means the "best"
setting was luck.

**If the checks fail:** stop here. Keeping your money in a low-cost NIFTY 50 or NIFTY 500 index
fund is the honest conclusion, and that is a good outcome, because the tool has just saved you money.

---

## 6. Paper trade for at least 6 months

Paper trading runs the strategy every day on real prices with fake money. It is the only
way to see how the system behaves *after* the backtest ended. There are two ways; you can
run both at the same time.

### 6.1 Practise with the real buttons, using fake money (you can start tomorrow)

This uses the same Trade Desk console you will use for real money. Instead of you placing
orders at a broker, indquant fills them **automatically against the next day's real NSE
prices**, with realistic costs and its stop-loss/target rules.

**Tonight (after about 7 pm):**
```bash
indquant fetch                                   # bring prices up to date
indquant --set execution.broker=paper console    # browser opens; the header says "PAPER - fake money"
```
Click **Refresh plan**, read the plan, and **Approve** the trades you like. They show
"Fills automatically at the next session's open".

**Tomorrow evening:** run the same two commands and click **Refresh plan** again. You will see:
* *"Paper fill: BUY 19 TCS @ ₹4,012.60"* when the open was within your limit, or *"not filled:
  limit not reached"* when the stock opened above it (gap-up), just like a real limit order;
* new positions with their paper stop-loss and target, which are checked automatically every day;
* *"Paper GTT fired …"* when a stop or target was hit.

**Approving during market hours** (say, at 1 pm, using yesterday's prices) is allowed, but
the order fills at the **next** session's open, never at a price that already printed this
morning, exactly like a real order placed at 1 pm. For the most up-to-date plan, refresh and
approve in the evening.

The fake-money book lives in its own file (`state/desk_paper.db`) and never mixes with real
trades. To make it permanent, set `execution: {broker: paper}` in `config/my.yaml`.

> **Be honest with yourself:** tomorrow tests the *workflow*, not the strategy. One day, or
> even one month, of results is mostly luck. Judge the system after months, using
> `paper-report` and the drift alerts.

### 6.2 The fully automatic paper book

```bash
indquant paper-init --variant champion
```

Then **every trading day, in the evening** (after about 7 pm, once NSE has published the day's prices), run:

```bash
indquant paper-update
```

It reads the day's prices, "fills" yesterday's orders at today's open, applies stop-losses,
and writes the next day's plan to `output/signals/<date>/champion/summary.md`. Read that
file: it is exactly what you would be doing with real money.

**Let your computer do it automatically** (optional):

* **Windows:** open *Task Scheduler* → *Create Basic Task* → name "indquant" → *Weekly*, tick
  Mon–Fri, time 19:30 → *Start a program*:
  * Program: `C:\path\to\Trading\.venv\Scripts\indquant.exe`
  * Arguments: `paper-update`
  * Start in: `C:\path\to\Trading`
* **macOS / Linux:** run `crontab -e` and add (change the path):
  `30 19 * * 1-5 cd /path/to/Trading && .venv/bin/indquant paper-update >> logs/cron.log 2>&1`

**Every weekend:** run `indquant paper-report`. It shows returns, trades, and a **"live vs
backtest"** line. If it ever says **DRIFT ALERT**, live results are worse than 95% of what the
backtest predicted: stop and re-check before going further.

---

## 7. Prepare for real money

### 7.1 A broker account

You need a **demat + trading account** with any SEBI-registered broker, for example
Zerodha, Upstox, Angel One or Groww. Opening one is done online with PAN and Aadhaar. Make
sure your broker supports **GTT** orders (Good Till Triggered), which most do. GTT is how your
stop-loss stays active for weeks without you watching the screen.

### 7.2 Decide the real amount

Start **smaller than your paper capital**. You can give the live desk its own amount:

```bash
indquant desk init --capital 200000
```

The desk is the part of indquant that tracks your *real* account. It is separate from the
paper book, which keeps running as your reference.

### 7.3 Optional: switch on the AI red-flag reviewer

Before you buy, the AI reviewer reads recent news about the stock: SEBI action, fraud
allegations, auditor resignations, results due in a few days. It can **warn** or **block**
a trade, but it can never add one.

1. Create an account at <https://console.anthropic.com>, add a little credit, and create an
   API key.
2. Save the key on your computer:
   * Windows (PowerShell): `setx ANTHROPIC_API_KEY "sk-ant-..."`, then close and reopen the
     terminal.
   * macOS/Linux: add `export ANTHROPIC_API_KEY="sk-ant-..."` to the end of `~/.zshrc` (or
     `~/.bashrc`), then open a new terminal.
3. Typical cost: a few US cents per stock reviewed with the default model. Set
   `reviewer.model: claude-haiku-5-5` in `config/my.yaml` for roughly 40× less. Without a key,
   trades simply show "REVIEW" and you check the news yourself.

---

## 8. Your first real trade, step by step

Do this on a normal trading day, **in the evening after about 7 pm** (NSE publishes the day's prices
in the evening; with `data.source: yahoo` you can start soon after 4 pm).

### Day 1, evening: get the plan

1. Open a terminal, go to the `Trading` folder, activate the environment (step 1.5).
2. Run:
   ```bash
   indquant console
   ```
   Your browser opens the **Trade Desk**. The terminal prints the link; it contains a secret
   token, so don't share it. Keep the terminal open while you use the page.
3. Click **Refresh plan**. The first time takes a few minutes while the models are trained.
4. The **Next-session trade plan** table appears. Each row is one suggested order:

   | Column | Meaning |
   |---|---|
   | Order | BUY or SELL and the stock symbol |
   | Qty | number of shares |
   | Limit | the highest price you will pay (buys) or lowest you accept (sells) |
   | Stop-loss | where the trade is exited if it goes wrong |
   | Target | where a swing trade takes its profit |
   | Risk | rupees lost if the stop-loss is hit, and as % of your capital |
   | Why | the reason, e.g. "Core: new holding" or "Swing: model edge after costs" |
   | Check | the AI verdict; click it to read the flags and headlines |

5. **For your very first trade, pick just one**, ideally a swing trade with a small "Risk".
   Click **Approve** on it and **Skip** on the rest. (When you are comfortable you will
   approve the whole plan each day.)
6. After approving, an instruction appears, for example:
   > ➜ In your broker app: Buy TCS (NSE) · Product CNC (delivery) · Order type LIMIT ·
   > Price ₹4,012.50 · Qty 3. Place it as an AMO …

### Day 1, evening: place the order in your broker app

Example for **Zerodha Kite** (other apps look similar):

1. Search the stock (e.g. `TCS`) and choose the **NSE** listing, not BSE.
2. Click **Buy**.
3. Choose **AMO** (after-market order); outside market hours Kite usually offers it
   automatically.
4. Product: **CNC / Longterm** (delivery), **not** MIS/Intraday.
5. Order type: **Limit**. Enter the **price** and **quantity** exactly as shown in the console.
6. Place the order. It waits and goes to the exchange at the next morning's open.

### Day 2, morning: confirm the fill

1. After 9:30 am, open **Orders** in your broker app. The order should say *Executed /
   Complete*. Note the **average price**.
   * If it did not execute (the price opened above your limit), go back to the console and
     click **Cancel** on that row. Nothing else to do.
2. In the console (run `indquant console` again if you closed it), type the actual average
   price into the **fill price** box and click **Mark filled**.
3. The position now appears under **Open positions** with an instruction like:
   > Create a GTT → OCO for TCS (CNC, qty 3): stop-loss trigger ₹3,850.00 with limit ₹3,811.50;
   > target trigger ₹4,240.00 with limit ₹4,240.00.

### Day 2: protect the position with a GTT stop-loss

In your broker app, open **GTT** and create a new one. In Kite: *Orders → GTT → New GTT*, or
the *GTT* option in the stock's menu. Then:

1. Choose **OCO** (one-cancels-other) for swing trades. For core holdings the console asks for
   a **Single** GTT, which is a protective stop only.
2. Transaction type **Sell**, quantity as shown.
3. Enter the **stop-loss trigger and limit**, and the **target trigger and limit**, exactly as
   shown.
4. Create it.

Now the stock is protected. If the price falls to the stop or rises to the target, the broker
sells automatically, even when you are not watching. The exact menu names can differ slightly
between apps and versions; ask your broker's support for "GTT OCO" if you can't find it.

### The following days

* **Each trading evening,** open the console and click **Refresh plan**. It updates prices and
  shows alerts:
  * *"traded through stop … your GTT has probably triggered"*: check your broker. If the
    stock was sold, click **Stopped out** on that position and enter the sale price.
  * *"reached target"*: the same, with the target sale price.
  * A **SELL** row (e.g. *Swing: time stop* after 15 days, or *Core: dropped from top list*):
    approve it, place the sell order in your app (CNC, Limit, AMO), then **Mark filled** the
    next day **and delete that stock's GTT** in your broker app.
* Moved a stop by hand? Use **Move stop** in the console so its records match your broker.
* **Emergency:** **STOP ALL** cancels pending buys and blocks new ones; your GTT stop-losses
  stay active. **Exit ALL positions…** creates sell orders for everything (you must type
  `SELL ALL`).

Prefer the terminal to the browser? Every button has a command: `indquant desk status`,
`indquant desk approve 12`, `indquant desk fill 12 --qty 3 --price 4011.8`, and so on.

---

## 9. Daily and weekly routine

| When | What | Command / place |
|---|---|---|
| Trading days, evening (after ~7 pm) | Refresh, read alerts, approve or skip, place AMOs | `indquant console` |
| Next morning after 9:30 | Check executions → **Mark filled** → create GTT | broker app + console |
| When a GTT fires | **Stopped out** with the sale price | console |
| Every day (automatic) | Paper book keeps running as the reference | `indquant paper-update` |
| Weekends | Compare live vs paper vs backtest; read drift alerts | `indquant paper-report` |
| Monthly | Are your real fills costing more than assumed? Are the filters helping? | `indquant desk tca`, `indquant desk journal` |
| Rarely (months, not days) | Change strategy settings, then re-run the backtest *before* trading them | `config/my.yaml` |

**Ground rules that protect beginners**

1. Never move a stop-loss further away to "give it room".
2. Don't increase position sizes after a winning streak.
3. Don't override the AI veto or the kill switch out of impatience.
4. If the console or `paper-report` shows a **DRIFT ALERT**, pause and re-check.
5. Profits are taxable: short-term capital gains (held under a year) and long-term gains. Your
   broker's tax P&L report helps at filing time; consult a tax professional.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised (Windows) | Re-install Python and tick "Add python.exe to PATH", or use `py` instead of `python`. |
| `Activate.ps1 cannot be loaded … running scripts is disabled` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |
| `indquant` is not recognised | The environment is not active: run the activate line from step 1.5. |
| LightGBM error about `libomp` (macOS) | `brew install libomp`, then reinstall: `pip install -e ".[dev,ai]"`. |
| `No bhavcopy files … Run indquant fetch` | Run `indquant fetch`; if downloads fail, check your internet, then re-run (finished days are kept). |
| `today's prices are not published yet` | NSE posts the day's file in the evening. Try again later. If today was a market holiday, add the date under `execution.holidays` in `config/my.yaml`. |
| `Data is stale` / `Refusing to generate signals` | Data hasn't updated for several days (internet, NSE site down, or a long holiday). Fix the cause and retry. Only use `--force` if you are sure the data is right. |
| "Dropped today's incomplete bar" | You ran it during market hours; indquant uses yesterday's close. Run it in the evening. |
| AI column says **REVIEW** | No API key or the AI was unavailable. Read the stock's latest news yourself before approving. |
| Broker rejects: insufficient funds | Add funds, or reduce the quantity when approving (`indquant desk approve ID --qty N`). |
| Order only partly executed | Mark filled with the quantity that actually executed. |
| Paper order "not filled: limit not reached" | The stock opened above your limit (a gap-up). That is realistic; nothing to do. |
| Fake-money desk shows old trades / want a fresh start | Close the console and delete `state/desk_paper.db` (only the fake-money book). |
| A stock had a split or bonus | Nothing to do in indquant (it adjusts automatically and tells you), but **update the GTT** at your broker as the console says. |
| I want to start the desk over | Close the console and delete `state/desk.db` (this erases the desk's records). |

---

## Glossary

* **AMO:** after-market order. Placed in the evening, sent to the exchange at the next open.
* **ATR:** average true range, the typical daily price movement; stops and targets are set
  in ATR multiples.
* **Backtest:** replaying the strategy on past data to see how it would have done.
* **Bhavcopy:** NSE's official daily file of every traded stock's prices.
* **CNC:** delivery product. You own the shares and can hold them for weeks (MIS is intraday).
* **Circuit / price band:** the daily limit on how far a stock may move; at the limit, trading
  can freeze on one side.
* **Drawdown:** fall from the highest value so far.
* **GTT / OCO:** a standing order that triggers when a price is hit. OCO pairs a stop-loss
  with a target; when one fires, the other is cancelled.
* **Limit order:** buy no higher (or sell no lower) than a set price.
* **Paper trading:** following the system with fake money on real prices.
* **Sharpe ratio:** return per unit of risk.
* **Stop-loss:** an exit price that limits the loss on a trade.
* **Alpha:** return beyond what the market (and known factors) explain, i.e. genuine skill.
