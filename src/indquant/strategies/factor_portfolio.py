"""Core sleeve: long-only multi-factor portfolio with monthly rebalancing.

Pipeline: eligible universe -> cross-sectional factor ranks -> composite score (optionally
blended with a walk-forward LightGBM ranker) -> top-N with rank buffer -> inverse-vol
weights with name / sector caps -> volatility target, regime filter, drawdown brake.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..backtest.account import Order
from ..backtest.costs import CostModel
from ..data.universe import eligibility_mask
from ..features.factors import compute_factors
from ..features.preprocessing import combine, cs_rank
from ..features.technical import technical_features
from ..models.factor_model import composite_score, rank_factors
from ..models.labeling import rank_labels
from ..models.ml_ranker import WalkForwardModel
from ..portfolio.construction import apply_caps, raw_weights, select_with_buffer
from ..portfolio import governor
from ..portfolio.risk import portfolio_vol, regime_is_risk_on, vol_target_multiplier
from ..research.factor_analysis import forward_returns, rowwise_spearman, trailing_ic_weights
from ..research.hypotheses import unsupported_factors

log = logging.getLogger(__name__)

ML_TECH_FEATURES = ["ret_5", "ret_21", "vol_21", "rsi_14", "dist_ma50", "vol_z", "rs_63", "breakout_20", "atr_pct"]
CONTEXT_FEATURES = ["mkt_ret_21", "mkt_vol_21", "mkt_dist_ma200"]


def rebalance_flags(dates: pd.DatetimeIndex) -> np.ndarray:
    """True on the first trading day of each month (signal at close, trade next open)."""
    months = dates.year * 12 + dates.month
    flags = np.ones(len(dates), dtype=bool)
    flags[1:] = months[1:] != months[:-1]
    return flags


class FactorPortfolioStrategy:
    name = "core"

    def __init__(self, cfg):
        self.cfg = cfg
        self.c = cfg.core
        self.ml_model: WalkForwardModel | None = None
        self.diagnostics: list[dict] = []
        self.last_governor = None

    def prepare(self, data) -> None:
        cfg, c = self.cfg, self.c
        self.data = data
        self.mask = eligibility_mask(data, cfg)
        self.factors = compute_factors(data)
        names = [k for k in c.factor_weights if k in self.factors]
        unknown = [k for k in c.factor_weights if k not in self.factors]
        if unknown:
            log.warning("Ignoring factors without data: %s", unknown)
        no_rationale = unsupported_factors([k for k in names if float(c.factor_weights[k]) != 0])
        if no_rationale:
            log.warning("Factors without a registered economic hypothesis (data-mining risk): %s", no_rationale)
        self.ranked = rank_factors({k: self.factors[k] for k in self.factors}, self.mask, data.sectors,
                                   bool(c.sector_neutral))
        if c.ic_weighting:
            fwd = forward_returns(data, int(cfg.ml.core_horizon))
            weights = trailing_ic_weights({k: self.factors[k] for k in names}, fwd, self.mask, int(cfg.ml.core_horizon))
            fixed = {k: float(c.factor_weights[k]) for k in names}
            weights = weights.fillna(fixed)
            dead = weights.sum(axis=1) <= 0  # every factor's trailing IC <= 0 -> fall back to priors
            for k in names:
                weights.loc[dead, k] = fixed[k]
        else:
            weights = {k: float(c.factor_weights[k]) for k in names}
        self.factor_score = composite_score({k: self.ranked[k] for k in names}, weights, self.mask)

        self.flags = rebalance_flags(data.dates)
        self.ml_score = None
        if cfg.ml.enabled and float(c.ml_weight) > 0:
            self.ml_score = self._fit_ml(data)
            self.score = cs_rank(combine({"factor": self.factor_score, "ml": self.ml_score},
                                         {"factor": 1 - float(c.ml_weight), "ml": float(c.ml_weight)}), self.mask)
        else:
            self.score = self.factor_score

        rets = data.close.pct_change(fill_method=None)
        self.returns = rets
        self.vol_ann = rets.rolling(63, min_periods=40).std() * math.sqrt(252)
        self.score_ic = None
        if c.construction == "optimizer":
            # realised IC of the final score: the IC measured at s is only known at s+h+1
            h = int(cfg.ml.core_horizon)
            ic = rowwise_spearman(self.score, forward_returns(data, h), self.mask)
            self.score_ic = ic.shift(h + 1).rolling(504, min_periods=63).mean()

    def _fit_ml(self, data) -> pd.DataFrame:
        cfg = self.cfg
        h = int(cfg.ml.core_horizon)
        tech = technical_features(data)
        feats = {f"f_{k}": v for k, v in self.ranked.items()}
        for k in ML_TECH_FEATURES:
            feats[f"t_{k}"] = cs_rank(tech[k], self.mask)
        for k in CONTEXT_FEATURES:
            feats[k] = tech[k]
        label = rank_labels(data, h, self.mask)
        self.ml_model = WalkForwardModel("regressor", cfg.ml, horizon=h, sample_every=int(cfg.ml.core_sample_every),
                                         seed=int(cfg.general.seed))
        predict_dates = data.dates[self.flags]
        # always predict the last bar so live signals can rebalance on demand
        predict_dates = predict_dates.union(data.dates[-1:])
        pred = self.ml_model.fit_predict(feats, label, self.mask, predict_dates)
        return cs_rank(pred, self.mask)

    # ------------------------------------------------------------------------
    def rebalance_due(self, i: int, account) -> bool:
        return bool(self.flags[i]) or account.meta.get("last_rebalance") is None or account.meta.get("force_rebalance", False)

    def target_weights(self, i: int, account, equity: float, gov=None) -> tuple[pd.Series, dict]:
        c, risk = self.c, self.cfg.risk
        scores = self.score.iloc[i]
        scores = scores[self.mask.iloc[i]].dropna()
        held = list(account.positions)
        w, opt_info = None, {}
        if c.construction == "optimizer":
            w, opt_info = self._optimize(i, account, equity, scores)
        if w is None:
            selection = select_with_buffer(scores, held, int(c.top_n), float(c.buffer_mult))
            w = raw_weights(selection, self.vol_ann.iloc[i], c.weighting)
            w = apply_caps(w, float(c.max_weight), float(c.sector_cap) if c.sector_cap else None, self.data.sectors)
            if c.construction == "optimizer":
                opt_info["fallback"] = "heuristic"

        exposure = 1.0 - float(c.cash_buffer)
        lookback = int(risk.vol_lookback)
        pred_vol = float("nan")
        if risk.target_vol and len(w):
            window = self.returns.iloc[max(0, i - lookback + 1): i + 1][list(w.index)]
            pred_vol = portfolio_vol(w, window)
            exposure *= vol_target_multiplier(pred_vol, float(risk.target_vol))
        risk_on = True
        if risk.regime_filter:
            risk_on = regime_is_risk_on(self.data.index, i, int(risk.regime_ma))
            if not risk_on:
                exposure *= float(risk.regime_exposure)
        gov_mult = gov.exposure_mult if gov is not None else 1.0
        exposure *= gov_mult
        info = {"date": self.data.dates[i], "n_selected": len(w), "pred_vol": pred_vol, "risk_on": risk_on,
                "governor_mult": gov_mult,
                "governor": "; ".join(gov.reasons) if gov is not None else "",
                "exposure": exposure * float(w.sum()),
                "ml_active": bool(self.ml_score is not None and self.ml_score.iloc[i].notna().any()),
                **{f"opt_{k}": v for k, v in opt_info.items()}}
        return w * exposure, info

    def _optimize(self, i: int, account, equity: float, scores: pd.Series):
        """Alpha (Grinold) x risk (Ledoit-Wolf) x costs, solved with cvxpy."""
        from ..portfolio.optimizer import optimize_weights
        from ..portfolio.risk import ledoit_wolf_cov

        c, o = self.c, self.c.optimizer
        if len(scores) < 5:
            return None, {"status": "too few names"}
        h = int(self.cfg.ml.core_horizon)
        cand = list(scores.sort_values(ascending=False).index[: int(int(c.top_n) * float(o.candidate_mult))])
        cand += [s_ for s_ in account.positions if s_ in scores.index and s_ not in cand]
        pct = (scores.reindex(cand) + 0.5).clip(0.005, 0.995)
        z = pd.Series(norm.ppf(pct.to_numpy()), index=cand)
        ic = self.score_ic.iloc[i] if self.score_ic is not None else np.nan
        ic = float(o.ic_prior) if np.isnan(ic) else float(np.clip(ic, float(o.ic_floor), float(o.ic_cap)))
        sigma = self.vol_ann.iloc[i].reindex(cand)
        sigma = sigma.fillna(sigma.median()) * math.sqrt(h / 252)
        alpha = ic * sigma * z
        window = self.returns.iloc[max(0, i - int(o.cov_lookback) + 1): i + 1][cand]
        cov = ledoit_wolf_cov(window) * h
        if cov.empty:
            return None, {"status": "no covariance"}
        alpha = alpha.reindex(cov.index)
        prices = self.data.close.ffill().iloc[i]
        w0 = pd.Series({s_: p.shares * prices.get(s_, np.nan) / equity for s_, p in account.positions.items()})
        cm = CostModel.from_config(self.cfg.costs)
        tcost = (cm.proportional_rate("buy") + cm.slippage_bps / 1e4) * float(o.tcost_mult)
        budget = 1.0 - float(c.cash_buffer)
        w, info = optimize_weights(alpha, cov, w0.dropna(), self.data.sectors, max_weight=float(c.max_weight),
                                   sector_cap=float(c.sector_cap) if c.sector_cap else None, budget=budget,
                                   risk_aversion=float(o.risk_aversion), tcost=tcost,
                                   min_weight=float(o.min_weight), max_names=int(o.max_names))
        info["ic_used"] = ic
        if w is None or w.sum() <= 0:
            return None, info
        return w / budget, info  # exposure scaling (cash buffer, vol target, regime) is applied by the caller

    def on_bar(self, i: int, ctx) -> list[Order]:
        account = ctx.account
        equity = ctx.equity
        gov = governor.evaluate(account, equity, ctx.date, self.cfg.governor)
        self.last_governor = gov
        # a circuit breaker that tightens between rebalances de-risks immediately
        tightened = gov.exposure_mult < float(account.meta.get("governor_mult", 1.0)) - 1e-9
        if not self.rebalance_due(i, account) and not tightened:
            return []
        weights, info = self.target_weights(i, account, equity, gov)
        info["trigger"] = "governor" if tightened and not self.flags[i] else "schedule"
        self.diagnostics.append(info)
        account.meta["governor_mult"] = gov.exposure_mult
        account.meta["last_rebalance"] = str(ctx.date.date())
        account.meta["last_targets"] = {k: float(v) for k, v in weights.items()}
        account.meta.pop("force_rebalance", None)

        prices = ctx.close
        date = str(ctx.date.date())
        min_trade = float(self.c.min_trade_pct) * equity
        sells, buys = [], []
        for sym, pos in account.positions.items():
            if sym not in weights.index:
                sells.append(Order(sym, -pos.shares, "rebalance_exit", date, ttl=3, ref_price=float(prices[sym])))
        for sym, wt in weights.items():
            px = float(prices[sym])
            if not px or math.isnan(px):
                continue
            target = int(math.floor(wt * equity / px))
            current = account.positions[sym].shares if sym in account.positions else 0
            delta = target - current
            if delta == 0 or (abs(delta) * px < min_trade and target > 0):
                continue
            stop_dist = px * float(self.c.disaster_stop_pct) if self.c.disaster_stop_pct and delta > 0 else None
            order = Order(sym, delta, "rebalance" if current else "rebalance_entry", date, ttl=3, ref_price=px,
                          stop_dist=stop_dist)
            (buys if delta > 0 else sells).append(order)
        buys.sort(key=lambda o: -o.shares * (o.ref_price or 0))
        return sells + buys
