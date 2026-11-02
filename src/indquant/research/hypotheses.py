"""Every factor must state *why* it should earn a return before it is allowed to trade.

Data-mined patterns without a causal story tend to vanish out of sample. Each hypothesis
records the economic mechanism, the expected sign of its information coefficient and the
key reference, so research results can be checked against a prior rather than accepted
because they look good.
"""
from __future__ import annotations

import pandas as pd

HYPOTHESES = {
    "mom_12_1": dict(sign=+1, mechanism="Investors under-react to news and herd; winners keep winning for 3-12 "
                     "months (behavioural). The most recent month is skipped to avoid short-term reversal.",
                     reference="Jegadeesh & Titman (1993); Asness, Moskowitz & Pedersen (2013)"),
    "mom_6_1": dict(sign=+1, mechanism="Same under-reaction mechanism at a shorter horizon.",
                    reference="Jegadeesh & Titman (1993)"),
    "idio_mom": dict(sign=+1, mechanism="Momentum in stock-specific (residual) returns is driven by firm news "
                     "diffusion rather than market beta, so it crashes less after market reversals.",
                     reference="Blitz, Huij & Martens (2011)"),
    "low_vol": dict(sign=+1, mechanism="Leverage-constrained and lottery-seeking investors overpay for volatile "
                    "stocks, so low-volatility stocks earn better risk-adjusted returns.",
                    reference="Ang et al. (2006); Baker, Bradley & Wurgler (2011)"),
    "low_beta": dict(sign=+1, mechanism="Betting against beta: constrained investors bid up high-beta stocks.",
                     reference="Frazzini & Pedersen (2014)"),
    "reversal_1m": dict(sign=+1, mechanism="Short-term over-reaction and liquidity-provision premium: last "
                        "month's losers bounce.", reference="Jegadeesh (1990); Lehmann (1990)"),
    "trend": dict(sign=+1, mechanism="Slow information diffusion; prices above their long-run average tend to "
                  "keep trending.", reference="Moskowitz, Ooi & Pedersen (2012); Faber (2007)"),
    "high_52w": dict(sign=+1, mechanism="Anchoring: investors under-react to good news near the 52-week high.",
                     reference="George & Hwang (2004)"),
    "quality": dict(sign=+1, mechanism="Profitable, low-leverage firms are under-priced relative to their "
                    "persistence of earnings.", reference="Asness, Frazzini & Pedersen (2019) 'Quality minus junk'"),
    "smooth_momentum": dict(sign=+1, mechanism="Information arriving in many small pieces is under-reacted to "
                            "(limited attention), so steady-drift momentum persists longer than jump momentum.",
                            reference="Da, Gurun & Warachka (2014) 'Frog in the pan'"),
    "lottery_aversion": dict(sign=+1, mechanism="Investors overpay for lottery-like stocks with extreme daily "
                             "spikes; avoiding them avoids subsequent underperformance.",
                             reference="Bali, Cakici & Whitelaw (2011) 'Maxing out'"),
    "circuit_risk": dict(sign=+1, mechanism="Stocks repeatedly pinned at NSE price bands are thinly traded / "
                         "operator-driven; their momentum is fragile and exits gap through stops.",
                         reference="indquant hypothesis (India-specific; must be validated by `indquant research`)"),
    "value": dict(sign=+1, mechanism="Risk premium for distressed firms and extrapolation errors by investors.",
                  reference="Fama & French (1992); Lakonishok, Shleifer & Vishny (1994)"),
}


def check_against_hypotheses(report: pd.DataFrame, t_threshold: float = 2.0) -> pd.DataFrame:
    """Annotate a factor report with the prior and whether the data supports it."""
    out = report.copy()
    signs, verdicts, refs = [], [], []
    for name, row in out.iterrows():
        h = HYPOTHESES.get(name)
        if h is None:
            signs.append(0)
            verdicts.append("NO HYPOTHESIS - data mining, do not trade")
            refs.append("")
            continue
        t = row.get("t_stat")
        signs.append(h["sign"])
        refs.append(h["reference"])
        if pd.isna(t):
            verdicts.append("insufficient data")
        elif t * h["sign"] >= t_threshold:
            verdicts.append("supported")
        elif t * h["sign"] <= -t_threshold:
            verdicts.append("CONTRADICTED - investigate before use")
        else:
            verdicts.append("not significant")
    out["expected_sign"] = signs
    out["verdict"] = verdicts
    out["reference"] = refs
    return out


def unsupported_factors(names) -> list[str]:
    return [n for n in names if n not in HYPOTHESES]
