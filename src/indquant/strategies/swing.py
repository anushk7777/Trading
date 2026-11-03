"""Swing sleeve: daily entries on liquid stocks with ATR stop / target brackets.

Ranking uses a walk-forward LightGBM regressor trained on triple-barrier outcomes that
match the trading rules exactly (same ATR stop / target and holding limit). The label is
the realised R-multiple (trade return divided by the risk taken to the stop), so the model
estimates the *expected value* of taking the trade. A trade is only considered when that
expected value exceeds round-trip costs plus a minimum edge - a decision rule grounded in
the economics of the trade rather than an arbitrary probability cut-off. Before enough
history exists for the model, a transparent trend/relative-strength rule is used instead.

Entry logic is deliberately asymmetric: a few conditions make a stock a *candidate*,
then a long list of veto filters look for reasons NOT to take the trade (stop outside a
sensible range, chasing, over-extension, erratic range, illiquidity, recent losing exit,
sector / correlation crowding, total open risk, negative news, upcoming events) and the
account-level risk governor can halt or shrink everything. Every rejection is counted
and the latest decisions are kept so the daily signal report can explain *why not*.
"""
from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np
import pandas as pd

from ..backtest.account import Order
from ..backtest.costs import CostModel
from ..data.events import blackout_mask
from ..data.sentiment import sentiment_frame
from ..data.universe import eligibility_mask, top_liquid
from ..features.preprocessing import cs_rank
from ..features.technical import atr, sma, technical_features
from ..models.labeling import triple_barrier
from ..models.ml_ranker import WalkForwardModel
from ..portfolio import governor

log = logging.getLogger(__name__)

RANK_FEATURES = ["ret_5", "ret_21", "ret_63", "rs_63", "vol_21", "vol_z"]


class SwingStrategy:
    name = "swing"

    def __init__(self, cfg):
        self.cfg = cfg
        self.s = cfg.swing
        self.f = cfg.swing.filters
        self.ml_model: WalkForwardModel | None = None
        self.rejections: Counter = Counter()
        self.last_decisions: list[dict] = []
        self.last_governor: governor.GovernorDecision | None = None

    def prepare(self, data) -> None:
        cfg, s = self.cfg, self.s
        self.data = data
        base = eligibility_mask(data, cfg)
        self.mask = top_liquid(base, data, int(s.universe_top_liquid))
        tech = technical_features(data)
        self.atr = atr(data.high, data.low, data.close, int(s.atr_window))
        c = data.close
        ma50, ma200 = sma(c, 50), sma(c, 200)
        self.ma20 = sma(c, 20)
        trend_ok = (c > ma50) & (ma50 > ma200)
        rsi14 = tech["rsi_14"]
        setup = trend_ok & (rsi14 >= 50) & (rsi14 <= 75) & (tech["breakout_20"] >= -0.03)
        self.rule_score = (cs_rank(tech["rs_63"], self.mask) + cs_rank(tech["breakout_20"], self.mask)).where(setup & self.mask)
        self.ret1 = c.pct_change(fill_method=None)
        self.day_range = data.high - data.low
        self.adv = (c * data.volume).rolling(63, min_periods=20).mean()
        from ..features.own_signals import circuit_days

        self.circuit = circuit_days(data)

        ma = int(cfg.risk.regime_ma)
        idx_ma = data.index.rolling(ma, min_periods=ma).mean()
        self.risk_on = (data.index >= idx_ma) | idx_ma.isna()

        self.sentiment = sentiment_frame(cfg.sentiment.file or cfg.sentiment.history_file, data.dates,
                                         data.symbols, float(cfg.sentiment.halflife_days))
        self.blackout = blackout_mask(cfg.events.file, data.dates, data.symbols, int(self.f.event_blackout_days))

        cm = CostModel.from_config(cfg.costs)
        self.round_trip_cost = cm.proportional_rate("buy") + cm.proportional_rate("sell") + 2 * cm.slippage_bps / 1e4

        self.expected_r = None
        if cfg.ml.enabled:
            feats = dict(tech)
            for k in RANK_FEATURES:
                feats[f"r_{k}"] = cs_rank(tech[k], self.mask)
            if getattr(data, "alt_data", None):
                from ..features.smart_money import smart_money_factors

                for k, v in smart_money_factors(data).items():
                    feats[f"r_{k}"] = cs_rank(v.where(data.close.notna()), self.mask)
            tb = triple_barrier(data, self.atr, float(s.stop_atr), float(s.target_atr), int(s.max_hold_days))
            entry = data.open.shift(-1)  # label only: the price the trade would have been entered at
            risk_pct = float(s.stop_atr) * self.atr / entry
            r_mult = (tb.ret / risk_pct).clip(-1.5, 1.5 * float(s.target_atr) / float(s.stop_atr))
            self.ml_model = WalkForwardModel("regressor", cfg.ml, horizon=int(s.max_hold_days),
                                             sample_every=int(cfg.ml.swing_sample_every), seed=int(cfg.general.seed))
            self.expected_r = self.ml_model.fit_predict(feats, r_mult, self.mask)
            self.base_rate = float(np.nanmean(tb.label.where(self.mask).to_numpy()))
            self.base_r = float(np.nanmean(r_mult.where(self.mask).to_numpy()))
            self.calibration = None
            if s.kelly_fraction:
                from ..models.calibration import selected_edge

                cost_r = self.round_trip_cost / (float(s.stop_atr) * self.atr / c).replace(0, np.nan)
                net = (self.expected_r - cost_r).where(self.mask & ((self.expected_r - cost_r) >= float(s.min_expected_r)))
                # judge the trades the strategy would actually take: the day's top-ranked candidates
                chosen = net.rank(axis=1, ascending=False, method="first") <= 2 * int(s.max_new_per_day)
                self.calibration = selected_edge(chosen, r_mult - cost_r, int(s.max_hold_days),
                                                 int(s.kelly_window), int(s.kelly_min_obs))

    # ------------------------------------------------------------------ veto filters
    def _veto(self, i: int, sym: str, px: float, a: float, account, accepted: list[str], open_risk: float,
              new_risk: float, equity: float) -> str | None:
        f, date = self.f, self.data.dates[i]
        stop_pct = float(self.s.stop_atr) * a / px
        if stop_pct < float(f.min_stop_pct):
            return "stop_too_tight"
        if stop_pct > float(f.max_stop_pct):
            return "stop_too_wide"
        r1 = self.ret1.iat[i, self.data.close.columns.get_loc(sym)]
        if not math.isnan(r1) and abs(r1) > float(f.max_day_move):
            return "chasing_big_move"
        ma20 = self.ma20.at[date, sym]
        if not math.isnan(ma20) and (px - ma20) / a > float(f.max_extension_atr):
            return "overextended"
        rng = self.day_range.at[date, sym]
        if not math.isnan(rng) and rng / a > float(f.max_range_atr):
            return "erratic_range"
        if self.circuit.at[date, sym] > float(f.max_circuit_days):
            return "circuit_prone"
        last = account.meta.get("last_exit", {}).get(sym)
        if last and last.get("pnl", 0) < 0:
            days = np.busday_count(pd.Timestamp(last["date"]).date(), date.date())
            if days < int(f.reentry_cooldown_days):
                return "reentry_cooldown"
        if self.blackout is not None and bool(self.blackout.at[date, sym]):
            return "event_blackout"
        if self.sentiment is not None and float(self.sentiment.at[date, sym]) < float(f.news_veto_below):
            return "negative_news"
        sectors = self.data.sectors
        same = sum(1 for s_ in list(account.positions) + accepted if sectors.get(s_) == sectors.get(sym))
        if sectors.get(sym, "Unknown") != "Unknown" and same >= int(f.max_per_sector):
            return "sector_crowded"
        peers = [p for p in list(account.positions) + accepted if p != sym]
        if peers and f.max_correlation is not None:
            window = self.ret1.iloc[max(0, i - 62): i + 1]
            corr = window[peers].corrwith(window[sym]).dropna()
            if len(corr) and corr.max() >= float(f.max_correlation):
                return "correlated_with_open_position"
        if open_risk + new_risk > float(f.max_portfolio_heat) * equity:
            return "portfolio_heat"
        return None

    def _open_risk(self, account, close: pd.Series) -> float:
        risk = 0.0
        for sym, pos in account.positions.items():
            if pos.stop is not None:
                px = close.get(sym, pos.last_price)
                px = pos.last_price if px is None or px != px else px
                risk += max(0.0, (px - pos.stop)) * pos.shares
        return risk

    # ------------------------------------------------------------------ daily logic
    def on_bar(self, i: int, ctx) -> list[Order]:
        s, f, account = self.s, self.f, ctx.account
        date = str(ctx.date.date())
        dates = self.data.dates
        close = ctx.close
        orders: list[Order] = []
        exiting = set()
        decisions: list[dict] = []

        for sym, pos in account.positions.items():
            entry_idx = int(dates.searchsorted(pd.Timestamp(pos.entry_date)))
            held = i - entry_idx + 1
            px = float(close[sym])
            if held >= int(s.max_hold_days):
                orders.append(Order(sym, -pos.shares, "time_stop", date, ttl=3, ref_price=px))
                exiting.add(sym)
                continue
            be = s.breakeven_after_atr
            if be and pos.atr and pos.stop is not None and px >= pos.entry_price + float(be) * pos.atr:
                pos.stop = max(pos.stop, pos.entry_price)

        gov = governor.evaluate(account, ctx.equity, ctx.date, self.cfg.governor)
        self.last_governor = gov
        self.last_decisions = decisions
        if not gov.allow_new_entries:
            self.rejections["governor_halt"] += 1
            decisions.append({"symbol": "*", "decision": "no new entries", "reason": "; ".join(gov.reasons)})
            return orders
        if self.cfg.risk.regime_filter and not bool(self.risk_on.iloc[i]):
            self.rejections["regime_risk_off"] += 1
            decisions.append({"symbol": "*", "decision": "no new entries", "reason": "index below its moving average"})
            return orders
        slots = int(s.max_positions) - (len(account.positions) - len(exiting))
        slots = min(slots, int(s.max_new_per_day))
        if slots <= 0:
            return orders

        cand, source = None, "rule"
        cost_r = pd.Series(dtype=float)
        if self.expected_r is not None:
            row = self.expected_r.iloc[i]
            if row.notna().any():
                # expected R after costs: cost as a fraction of the risk taken to the stop
                risk_pct = (float(s.stop_atr) * self.atr.iloc[i] / close).replace(0, np.nan)
                cost_r = self.round_trip_cost / risk_pct
                net_r = row - cost_r
                cand = net_r[self.mask.iloc[i] & (net_r >= float(s.min_expected_r))].dropna()
                source = "ml"
        cal = None
        if source == "ml" and getattr(self, "calibration", None) is not None:
            c_ = self.calibration.iloc[i]
            cal = c_ if not np.isnan(c_["mean"]) else None
        if cand is None:
            cand = self.rule_score.iloc[i].dropna()
        cand = cand.drop(labels=[c for c in cand.index if c in account.positions], errors="ignore")
        cand = cand.sort_values(ascending=False)

        equity = ctx.equity
        budget = account.cash * (1 - float(s.cash_buffer))
        open_risk = self._open_risk(account, close)
        accepted: list[str] = []
        for sym in cand.index:
            if slots <= 0:
                break
            a = float(self.atr.iloc[i][sym])
            px = float(close[sym])
            if not a or math.isnan(a) or not px or math.isnan(px):
                continue
            stop_dist = float(s.stop_atr) * a
            base_risk = float(s.risk_per_trade)
            risk = base_risk
            if cal is not None:   # calibrated fractional Kelly: can only shrink the fixed-risk size
                from ..models.calibration import kelly_risk

                risk = kelly_risk(float(cal["mean"]), float(cal["var"]), float(s.kelly_fraction), base_risk)
                if risk < 0.1 * base_risk:
                    self.rejections["kelly_edge_too_small"] += 1
                    decisions.append({"symbol": sym, "score": float(cand[sym]), "decision": "rejected",
                                      "reason": "kelly_edge_too_small"})
                    continue
            qty = math.floor(min(risk * equity / stop_dist, float(s.max_position_pct) * equity / px)
                             * gov.size_mult)
            adv = self.adv.at[ctx.date, sym]
            if not math.isnan(adv) and adv > 0:
                qty = min(qty, math.floor(float(f.max_pct_of_adv) * adv / px))
            qty = min(qty, math.floor(budget / (px * 1.003)))
            reason = "insufficient_cash_or_liquidity" if qty <= 0 else self._veto(
                i, sym, px, a, account, accepted, open_risk, qty * stop_dist, equity)
            if reason:
                self.rejections[reason] += 1
                decisions.append({"symbol": sym, "score": float(cand[sym]), "decision": "rejected", "reason": reason})
                continue
            budget -= qty * px * 1.003
            open_risk += qty * stop_dist
            accepted.append(sym)
            decisions.append({"symbol": sym, "score": float(cand[sym]), "decision": "BUY",
                              "reason": source if risk >= base_risk else f"{source} (Kelly risk {risk:.2%})"})
            orders.append(Order(sym, qty, f"entry_{source}", date, ttl=1, stop_dist=stop_dist,
                                target_dist=float(s.target_atr) * a, atr=a, ref_price=px))
            slots -= 1
        return orders
