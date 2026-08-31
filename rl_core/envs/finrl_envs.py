"""FinRL-style multi-asset portfolio environments — `StockTradingEnv` and
`StockPortfolioEnv`, the two flagship environments of AI4Finance's FinRL
(Liu et al., 2020, "FinRL: A Deep Reinforcement Learning Library for
Automated Stock Trading in Quantitative Finance"; the framework most
academic deep-RL-for-trading papers of the last few years benchmark
against), reimplemented from scratch against our own synthetic multi-asset
market simulator instead of depending on the `finrl` PyPI package itself.

Why not just `pip install finrl`: its top-level `finrl/__init__.py`
unconditionally imports `finrl.trade`, which unconditionally imports the
Alpaca *live paper-trading* module, which needs `alpaca_trade_api` — so
even `from finrl.meta.env_stock_trading.env_stocktrading import
StockTradingEnv` (the one thing anyone training locally actually wants)
fails with a bare `ModuleNotFoundError` unless that unrelated live-broker
SDK is installed too. Installing it (verified directly) pulls in
`exchange_calendars`/`alpaca-trade-api`, which downgrade `websockets` and
`urllib3` hard enough to break other already-installed packages — the
exact same "this dependency's install silently wrecks the rest of the
environment" failure mode `rl_core/envs/robotics_envs.py` (Fetch/Hand) and
this project's earlier, reverted CityLearn attempt hit, and for the same
reason skipped here: not worth shipping a "looks available, degrades
whatever else the user has installed the moment you try" dependency for
two environments whose actual simulation logic is a few hundred lines of
portfolio bookkeeping over a price array. FinRL's own `StockTradingEnv`/
`StockPortfolioEnv` are themselves designed to run on *any* OHLCV+
indicator DataFrame the caller builds (typically via `yfinance`, itself a
network fetch this sandbox restricts by default and the same "goes stale,
no reproducible seed" issue `rl_core/envs/trading_envs.py` already
explains for the single-asset trading env) — so reproducing their
environment *mechanics* (portfolio accounting, per-share buy/sell with
transaction costs, softmax portfolio-weight allocation) against a fresh
synthetic multi-asset price path every episode is a strictly better fit
for this app than either depending on the fragile package or wiring it to
a live data feed.

- `FinRLStockTradingEnv` (`FinRL-StockTrading-v0`) — mirrors
  `StockTradingEnv`: a portfolio of `num_stocks` synthetic stocks plus a
  cash balance; the action is one continuous "shares to trade" value per
  stock (positive buys, negative sells, scaled by `hmax`), each trade pays
  a proportional cost, and the reward is the step-over-step change in
  total portfolio value (cash + mark-to-market holdings). The point of
  this env vs. `Trading-Discrete-v0`/`Trading-Continuous-v0` is genuinely
  multi-asset bookkeeping — cash constraints, per-share transaction costs,
  and diversification — not just "one position, one instrument".
- `FinRLPortfolioAllocationEnv` (`FinRL-PortfolioAllocation-v0`) — mirrors
  `StockPortfolioEnv`: always fully invested (no cash, no leverage/short),
  the action is a raw score per stock that gets softmax-normalized into
  portfolio weights, and the observation includes the assets' recent
  realized covariance matrix — this is the "how do I *balance* a fixed
  budget across correlated assets" problem (Markowitz-style, but learned)
  as opposed to `FinRLStockTradingEnv`'s "when do I buy/sell how much"
  problem.

Both share `_MultiAssetMarket`: the same regime-switching GARCH(1,1) +
Student-t ingredients as `rl_core/envs/trading_envs.py`'s single-asset
model (see that module's docstring for the stylized-facts rationale),
extended to `num_stocks` assets via a one-factor model — a shared "market"
GARCH process plus each asset's own idiosyncratic GARCH process, combined
through a random per-asset beta (market sensitivity) and drift multiplier
drawn once per episode — which is what actually gives the assets realistic
*correlation* (the reason a portfolio env's covariance-matrix observation
is worth anything at all) instead of independent noise.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

_REGISTERED = False

_REGIMES: dict[str, tuple[float, float]] = {
    "bull": (0.0005, 0.009),
    "bear": (-0.0005, 0.015),
    "sideways": (0.0, 0.005),
}
_REGIME_NAMES = list(_REGIMES.keys())
_SWITCH_PROB = 0.01
_STUDENT_T_DOF = 5.0
_GARCH_ALPHA = 0.10
_GARCH_BETA = 0.85
_REWARD_SCALE = 100.0


def _garch_t_shock(np_random, sigma2: float, last_eps: float, long_run_vol: float) -> tuple[float, float]:
    """One GARCH(1,1) + Student-t step — shared helper so the market
    factor and every per-asset idiosyncratic process use identical
    dynamics (see module docstring). Returns `(shock, new_sigma2)`."""
    omega = (1.0 - _GARCH_ALPHA - _GARCH_BETA) * long_run_vol ** 2
    sigma2 = omega + _GARCH_ALPHA * last_eps ** 2 + _GARCH_BETA * sigma2
    sigma = float(np.sqrt(max(sigma2, 1e-12)))
    raw_t = float(np_random.standard_t(_STUDENT_T_DOF))
    unit_t = raw_t / np.sqrt(_STUDENT_T_DOF / (_STUDENT_T_DOF - 2.0))
    return sigma * unit_t, sigma2


class _MultiAssetMarket:
    """`num_assets` correlated synthetic price series — a one-factor
    regime-switching GARCH-t model (see module docstring). Reseeded fresh
    (via the owning env's `np_random`) every `reset()`, exactly like
    `rl_core/envs/trading_envs.py`'s single-asset simulator."""

    def __init__(self, num_assets: int) -> None:
        self.num_assets = num_assets
        self.prices = np.ones(num_assets, dtype=np.float64)
        self.regime = "sideways"
        self._mkt_sigma2 = 0.0
        self._mkt_last_eps = 0.0
        self._idio_sigma2 = np.zeros(num_assets, dtype=np.float64)
        self._idio_last_eps = np.zeros(num_assets, dtype=np.float64)
        # Drawn once per episode (in `reset`) — asset `i` reacts to the
        # shared market shock `beta[i]`-strongly and drifts at
        # `drift_mult[i]` times the regime's baseline drift, so assets are
        # visibly distinguishable (some defensive/low-beta, some
        # aggressive/high-beta) rather than N copies of the same series.
        self.beta = np.ones(num_assets, dtype=np.float64)
        self.drift_mult = np.ones(num_assets, dtype=np.float64)
        self.idio_vol_scale = np.full(num_assets, 0.6, dtype=np.float64)
        self.price_history: list[np.ndarray] = []

    def reset(self, np_random) -> None:
        self.regime = _REGIME_NAMES[int(np_random.integers(0, len(_REGIME_NAMES)))]
        self.prices = np.full(self.num_assets, 100.0, dtype=np.float64)
        self._mkt_sigma2 = _REGIMES[self.regime][1] ** 2
        self._mkt_last_eps = 0.0
        self.beta = np_random.uniform(0.5, 1.5, size=self.num_assets)
        self.drift_mult = np_random.uniform(0.6, 1.4, size=self.num_assets)
        self.idio_vol_scale = np_random.uniform(0.3, 0.9, size=self.num_assets)
        self._idio_sigma2 = (self.idio_vol_scale * _REGIMES[self.regime][1]) ** 2
        self._idio_last_eps = np.zeros(self.num_assets, dtype=np.float64)
        self.price_history = [self.prices.copy()]

    def step(self, np_random) -> np.ndarray:
        """Advances one bar; returns the per-asset log-return vector."""
        drift, long_run_vol = _REGIMES[self.regime]
        mkt_eps, self._mkt_sigma2 = _garch_t_shock(np_random, self._mkt_sigma2, self._mkt_last_eps, long_run_vol)
        self._mkt_last_eps = mkt_eps
        log_returns = np.zeros(self.num_assets, dtype=np.float64)
        for i in range(self.num_assets):
            idio_long_run_vol = self.idio_vol_scale[i] * long_run_vol
            idio_eps, self._idio_sigma2[i] = _garch_t_shock(
                np_random, self._idio_sigma2[i], self._idio_last_eps[i], idio_long_run_vol,
            )
            self._idio_last_eps[i] = idio_eps
            log_returns[i] = drift * self.drift_mult[i] + self.beta[i] * mkt_eps + idio_eps
        self.prices = self.prices * np.exp(log_returns)
        self.price_history.append(self.prices.copy())
        if np_random.random() < _SWITCH_PROB:
            others = [r for r in _REGIME_NAMES if r != self.regime]
            self.regime = others[int(np_random.integers(0, len(others)))]
        return log_returns


class FinRLStockTradingEnv(gym.Env):
    """See module docstring — mirrors FinRL's `StockTradingEnv`."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 6}

    def __init__(
        self, num_stocks: int = 8, initial_amount: float = 1_000_000.0,
        hmax: int = 100, buy_cost_pct: float = 0.001, sell_cost_pct: float = 0.001,
        window: int = 20, max_steps: int = 252, render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.num_stocks = int(num_stocks)
        self.initial_amount = float(initial_amount)
        self.hmax = int(hmax)
        self.buy_cost_pct = float(buy_cost_pct)
        self.sell_cost_pct = float(sell_cost_pct)
        self.window = int(window)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_stocks,), dtype=np.float32)
        # [cash fraction, per-stock: price/100, holdings value fraction,
        # short-momentum, realized vol] — a flat Box mirroring FinRL's own
        # `state_space = 1 + 2*stock_dim + len(indicators)*stock_dim`
        # layout (cash, prices, holdings, then indicators).
        obs_dim = 1 + self.num_stocks * 4
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self._market = _MultiAssetMarket(self.num_stocks)
        self._cash = self.initial_amount
        self._shares = np.zeros(self.num_stocks, dtype=np.float64)
        self._returns_hist = np.zeros((self.window, self.num_stocks), dtype=np.float64)
        self._total_value_history: list[float] = [self.initial_amount]
        self._t = 0

    def _total_value(self) -> float:
        return float(self._cash + np.sum(self._shares * self._market.prices))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._market.reset(self.np_random)
        self._cash = self.initial_amount
        self._shares = np.zeros(self.num_stocks, dtype=np.float64)
        self._returns_hist = np.zeros((self.window, self.num_stocks), dtype=np.float64)
        for i in range(self.window):
            self._returns_hist[i] = self._market.step(self.np_random)
        self._total_value_history = [self._total_value()]
        self._t = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        cash_frac = self._cash / self.initial_amount
        price_norm = self._market.prices / 100.0
        holdings_value = self._shares * self._market.prices
        holdings_frac = holdings_value / max(self._total_value(), 1e-6)
        short_mom = self._returns_hist[-5:].sum(axis=0)
        vol = self._returns_hist.std(axis=0)
        return np.concatenate([[cash_frac], price_norm, holdings_frac, short_mom, vol]).astype(np.float32)

    def step(self, action: Any):
        prev_value = self._total_value()
        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        share_deltas = np.round(np.clip(raw, -1.0, 1.0) * self.hmax)

        # Sells first (frees cash for buys later in the same step, exactly
        # like FinRL's own `_sell_stock`/`_buy_stock` ordering) — can never
        # sell more shares than currently held (no shorting in this env).
        order = np.argsort(share_deltas)  # sells (negative) first, buys (positive) last
        for i in order:
            delta = share_deltas[i]
            price = float(self._market.prices[i])
            if delta < 0:
                sell_shares = min(-delta, self._shares[i])
                proceeds = sell_shares * price * (1.0 - self.sell_cost_pct)
                self._cash += proceeds
                self._shares[i] -= sell_shares
            elif delta > 0:
                affordable_shares = min(delta, self._cash / (price * (1.0 + self.buy_cost_pct) + 1e-9))
                cost = affordable_shares * price * (1.0 + self.buy_cost_pct)
                self._cash -= cost
                self._shares[i] += affordable_shares

        log_returns = self._market.step(self.np_random)
        self._returns_hist = np.roll(self._returns_hist, -1, axis=0)
        self._returns_hist[-1] = log_returns
        self._t += 1

        new_value = self._total_value()
        self._total_value_history.append(new_value)
        reward = (new_value - prev_value) / self.initial_amount * _REWARD_SCALE

        terminated = new_value <= 0.05 * self.initial_amount  # portfolio effectively wiped out
        truncated = (not terminated) and self._t >= self.max_steps
        info = {"total_value": new_value, "cash": self._cash, "regime": self._market.regime}
        return self._obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        holdings_value = self._shares * self._market.prices
        weights = np.concatenate([[self._cash], holdings_value]) / max(self._total_value(), 1e-6)
        return _render_portfolio_dashboard(
            price_history=self._market.price_history, equity_history=self._total_value_history,
            weights=weights, has_cash_slot=True, title=f"Value {self._total_value():,.0f}  Cash {self._cash / max(self._total_value(), 1e-6):.0%}",
        )


class FinRLPortfolioAllocationEnv(gym.Env):
    """See module docstring — mirrors FinRL's `StockPortfolioEnv`. Always
    fully invested: the action is a raw per-asset score, softmax-normalized
    into non-negative weights summing to 1 (no cash, no shorting) — the
    "how do I balance a fixed budget" problem, as opposed to
    `FinRLStockTradingEnv`'s discrete buy/sell-share bookkeeping."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 6}

    def __init__(
        self, num_stocks: int = 8, window: int = 20, max_steps: int = 252,
        turnover_cost_rate: float = 0.001, render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.num_stocks = int(num_stocks)
        self.window = int(window)
        self.max_steps = int(max_steps)
        self.turnover_cost_rate = float(turnover_cost_rate)
        self.render_mode = render_mode

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_stocks,), dtype=np.float32)
        # [flattened realized covariance matrix, per-asset mean recent
        # return, current weights] — mirrors `StockPortfolioEnv`'s own
        # `state = [covariance matrix, price array]` layout (mean returns
        # standing in for the raw price array, since a policy conditioning
        # on non-stationary absolute price levels generalizes poorly).
        obs_dim = self.num_stocks * self.num_stocks + self.num_stocks * 2
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self._market = _MultiAssetMarket(self.num_stocks)
        self._returns_hist = np.zeros((self.window, self.num_stocks), dtype=np.float64)
        self._weights = np.full(self.num_stocks, 1.0 / self.num_stocks, dtype=np.float64)
        self._equity = 1.0
        self._equity_history: list[float] = [1.0]
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._market.reset(self.np_random)
        self._returns_hist = np.zeros((self.window, self.num_stocks), dtype=np.float64)
        for i in range(self.window):
            self._returns_hist[i] = self._market.step(self.np_random)
        self._weights = np.full(self.num_stocks, 1.0 / self.num_stocks, dtype=np.float64)
        self._equity = 1.0
        self._equity_history = [1.0]
        self._t = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        cov = np.cov(self._returns_hist, rowvar=False) * 1e4  # scaled up: raw daily-return covariances are ~1e-4/1e-5
        mean_ret = self._returns_hist.mean(axis=0) * 100.0
        return np.concatenate([cov.reshape(-1), mean_ret, self._weights]).astype(np.float32)

    def _softmax_weights(self, action: Any) -> np.ndarray:
        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        z = raw - raw.max()
        exp = np.exp(z)
        return exp / exp.sum()

    def step(self, action: Any):
        new_weights = self._softmax_weights(action)
        turnover = float(np.abs(new_weights - self._weights).sum())
        log_returns = self._market.step(self.np_random)
        self._returns_hist = np.roll(self._returns_hist, -1, axis=0)
        self._returns_hist[-1] = log_returns
        # Weights are set *before* observing this bar's return (like every
        # other wrapped-forward formulation in this app's trading envs) —
        # portfolio log-return is the previous weights dotted with the
        # realized per-asset log-returns.
        portfolio_log_return = float(np.dot(self._weights, log_returns))
        cost = self.turnover_cost_rate * turnover
        step_return = portfolio_log_return - cost
        self._equity *= float(np.exp(step_return))
        self._equity_history.append(self._equity)
        self._weights = new_weights
        self._t += 1

        reward = step_return * _REWARD_SCALE
        terminated = self._equity <= 1e-3
        truncated = (not terminated) and self._t >= self.max_steps
        info = {"equity": self._equity, "regime": self._market.regime, "turnover": turnover}
        return self._obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return _render_portfolio_dashboard(
            price_history=self._market.price_history, equity_history=self._equity_history,
            weights=self._weights, has_cash_slot=False,
            title=f"Equity {self._equity:.3f}  {self._market.regime}",
        )


def _asset_color(i: int, n: int) -> tuple[int, int, int]:
    import colorsys

    hue = (i / max(n, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _render_portfolio_dashboard(
    *, price_history: list[np.ndarray], equity_history: list[float],
    weights: np.ndarray, has_cash_slot: bool, title: str,
) -> np.ndarray:
    """A compact multi-asset "portfolio dashboard" via Pillow, echoing the
    normalized-cumulative-return line charts FinRL's own backtest reports
    plot (each asset indexed to a common baseline so relative performance
    is visually comparable regardless of each stock's absolute price
    level) plus a bottom weight bar this app's Gantt/candlestick renders
    already established as the compact "what is the policy actually doing
    right now" readout (see `rl_core/envs/industrial_envs.py`'s status
    strip / `rl_core/envs/trading_envs.py`'s position strip)."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 340, 160
    margin_top, margin_bottom = 24, 22
    img = Image.new("RGB", (width, height), (18, 18, 22))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=11)
    draw.text((6, 4), title, font=font, fill=(210, 210, 220))

    prices = np.asarray(price_history[-120:])  # (T, num_assets)
    if prices.shape[0] >= 2:
        norm = prices / prices[0:1, :] * 100.0
        lo, hi = float(norm.min()), float(norm.max())
        span = max(hi - lo, 1e-6)
        plot_h = height - margin_top - margin_bottom
        n_pts = norm.shape[0]

        def xy(t: int, value: float) -> tuple[int, int]:
            x = 6 + int((width - 12) * (t / max(n_pts - 1, 1)))
            y = margin_top + int(plot_h * (1.0 - (value - lo) / span))
            return x, y

        for a in range(norm.shape[1]):
            pts = [xy(t, norm[t, a]) for t in range(n_pts)]
            draw.line(pts, fill=_asset_color(a, norm.shape[1]), width=1)

        eq = np.asarray(equity_history[-120:], dtype=np.float64)
        eq_norm = eq / eq[0] * 100.0
        eq_lo, eq_hi = float(eq_norm.min()), float(eq_norm.max())
        combined_lo, combined_hi = min(lo, eq_lo), max(hi, eq_hi)
        combined_span = max(combined_hi - combined_lo, 1e-6)

        def eq_xy(t: int, value: float) -> tuple[int, int]:
            x = 6 + int((width - 12) * (t / max(len(eq_norm) - 1, 1)))
            y = margin_top + int(plot_h * (1.0 - (value - combined_lo) / combined_span))
            return x, y

        draw.line([eq_xy(t, v) for t, v in enumerate(eq_norm)], fill=(235, 220, 90), width=2)

    # Weight strip: one horizontal segment per asset (plus a leading gray
    # cash segment when the env has one), width proportional to that
    # asset's current portfolio share — a stacked-bar readout of "what is
    # the policy actually holding right now".
    strip_y = height - margin_bottom + 6
    x = 6
    total_w = width - 12
    num_assets = len(weights) - (1 if has_cash_slot else 0)
    if has_cash_slot:
        seg_w = int(total_w * max(weights[0], 0.0))
        draw.rectangle([x, strip_y, x + seg_w, strip_y + 10], fill=(90, 90, 100))
        x += seg_w
        asset_weights = weights[1:]
    else:
        asset_weights = weights
    for a in range(num_assets):
        seg_w = int(total_w * max(asset_weights[a], 0.0))
        draw.rectangle([x, strip_y, x + seg_w, strip_y + 10], fill=_asset_color(a, num_assets))
        x += seg_w

    return np.asarray(img, dtype=np.uint8)


def register_finrl_envs() -> None:
    """Idempotent — safe to import/call from multiple modules. No optional
    third-party dependency to guard against (see module docstring) — always
    registered."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    gym.register(id="FinRL-StockTrading-v0", entry_point=FinRLStockTradingEnv, kwargs={"num_stocks": 8})
    gym.register(id="FinRL-PortfolioAllocation-v0", entry_point=FinRLPortfolioAllocationEnv, kwargs={"num_stocks": 8})


# Self-registers at import time, exactly like `rl_core/envs/trading_envs.py`.
register_finrl_envs()
