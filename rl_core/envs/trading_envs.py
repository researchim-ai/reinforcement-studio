"""From-scratch single-asset trading environment — the classic "decide a
position, pay for changing your mind" RL task (the turnover-cost
formulation used throughout the deep-RL-for-trading literature, e.g.
Zhang, Zohren & Roberts, 2020, "Deep Reinforcement Learning for Trading",
and Deng et al., 2016, "Deep Direct Reinforcement Learning for Financial
Signal Representation and Trading"), with the price simulator and cost
model brought in line with what the more realism-focused trading-gym
projects (TensorTrade's pluggable `Exchange`/slippage model; Gym-Trading-Env
and its candlestick renderer; ABIDES-Gym's tick-level market-impact
framing) actually model instead of a bare random walk:

Why a synthetic price simulator instead of bundling/fetching real market
data: real OHLCV data is a moving target (goes stale, licensing, needs
network access this sandbox restricts by default), and training/evaluating
against a *single* fixed historical series risks a policy that just
memorizes one particular path rather than learning something that
generalizes. Every ingredient below is instead a well-known *statistical
stylized fact* of real markets (Cont, 2001, "Empirical properties of asset
returns: stylized facts and statistical issues") reproduced synthetically,
reseeded fresh every episode via `self.np_random`:

- **Regime-switching drift** (bull/bear/sideways, randomly transitioning)
  — unconditional trend/direction changes over time.
- **GARCH(1, 1) conditional volatility** — the single most well-established
  stylized fact in financial econometrics: *volatility clustering* (large
  moves cluster together, calm periods cluster together), not i.i.d. noise
  like a bare GBM assumes.
- **Student-t innovations** — real returns have fatter tails than a normal
  distribution; a bare Gaussian systematically underestimates crash/spike
  risk.
- **Intra-bar OHLC simulation** — each step is a full open/high/low/close
  bar (built from several intra-bar sub-steps of the same process, open of
  bar *t* = close of bar *t-1*), not just a single close price, so the
  render actually is a real candlestick chart and a policy can in
  principle react to intrabar range, not only the close-to-close return.
- **Bid-ask spread + market-impact slippage** — TensorTrade/ABIDES-Gym's
  point that "cost = one flat commission rate" is unrealistic: real
  execution cost is a *spread* (which widens in high-volatility regimes,
  exactly like real order books) plus *slippage* that grows with how much
  you're trading and how volatile the market currently is, on top of a
  flat commission.

Two variants, mirroring the LunarLander/LunarLanderContinuous split already
in the gallery:
- `Trading-Discrete-v0` — `Discrete(3)`: go short / flat / long (target
  position in `{-1, 0, +1}`), for DQN/Rainbow/PPO/A2C.
- `Trading-Continuous-v0` — `Box(-1, 1, (1,))`: exact target position
  (fractional leverage, e.g. 0.3 = 30% of capital long), for
  PPO/A2C/SAC/DDPG/TD3 — lets the policy size positions instead of only
  picking a direction.

Observation: engineered features in the style of TensorTrade/FinRL's
"Observer" (not just a raw window of returns) — see `_obs()`.

Reward: the position's log-return over the bar (the position is decided
*before* the bar it's scored against, like every wrapped-forward
formulation in this literature) minus commission + spread + slippage costs
on any change in position, scaled by `_REWARD_SCALE` so rewards sit in a
PPO/A2C-friendly ~[-1, 1] range instead of raw ~1e-3-scale log-returns.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

_REGISTERED = False

# (drift, long-run volatility) per bar — not calibrated to any real market,
# only tuned so bull/bear/sideways regimes are visually and behaviorally
# distinguishable within one ~250-bar episode.
_REGIMES: dict[str, tuple[float, float]] = {
    "bull": (0.0006, 0.010),
    "bear": (-0.0006, 0.016),
    "sideways": (0.0, 0.006),
}
_REGIME_NAMES = list(_REGIMES.keys())
# Expected regime "dwell time" ~= 1 / switch_prob bars (~80 bars) before
# transitioning to a different regime (uniformly, among the other two) —
# long enough for a policy to detect a regime shift is worth reacting to,
# short enough that a single episode usually visits more than one regime.
_SWITCH_PROB = 0.0125
_REWARD_SCALE = 100.0  # log-returns are ~1e-3 scale; report them as roughly "%".
_SUBSTEPS_PER_BAR = 8  # intra-bar resolution used only to build a realistic OHLC range
_STUDENT_T_DOF = 5.0  # fat tails; variance of a df=5 Student-t is finite (df/(df-2)) and gets rescaled to 1 below
# GARCH(1,1) persistence/reactivity — standard-textbook values (see e.g.
# Engle, 2001), tuned only so clustering is visible within one episode
# rather than fit to any real instrument.
_GARCH_ALPHA = 0.10
_GARCH_BETA = 0.85


class TradingEnv(gym.Env):
    """See module docstring. `continuous=False` gives `Discrete(3)`
    (short/flat/long); `continuous=True` gives `Box(-1, 1, (1,))` (exact
    target position)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self, continuous: bool = False, window: int = 30, max_steps: int = 250,
        commission_rate: float = 0.0002, base_spread_bps: float = 2.0,
        impact_coef: float = 0.05, render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.continuous = bool(continuous)
        self.window = int(window)
        self.max_steps = int(max_steps)
        self.commission_rate = float(commission_rate)
        self.base_spread_bps = float(base_spread_bps)
        self.impact_coef = float(impact_coef)
        self.render_mode = render_mode

        # obs = window of bar log-returns, short/long momentum, realized
        # vol, current conditional vol, position, fraction of episode
        # remaining — see module docstring / `_obs()`.
        obs_dim = self.window + 6
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        if self.continuous:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(3)  # 0=short, 1=flat, 2=long

        self._returns = np.zeros(self.window, dtype=np.float32)
        self._position = 0.0
        self._price = 1.0
        self._equity = 1.0
        self._equity_history: list[float] = []
        self._regime = "sideways"
        self._sigma2 = 0.0  # current GARCH conditional variance (per-bar)
        self._last_eps = 0.0
        self._t = 0
        self._bars: list[tuple[float, float, float, float]] = []  # (open, high, low, close)
        self._position_history: list[float] = []

    def _target_position(self, action: Any) -> float:
        if self.continuous:
            return float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
        return mapping[int(action)]

    def _garch_step(self, long_run_vol: float) -> float:
        """One GARCH(1,1) innovation: updates the conditional variance from
        the *previous* step's realized shock, then draws a fresh
        fat-tailed shock scaled by the *new* conditional std. `omega` is
        chosen so the process's long-run variance target equals the
        current regime's `long_run_vol ** 2` — regime switches still move
        the general vol level, GARCH only adds clustering/memory around
        whatever that level currently is."""
        omega = (1.0 - _GARCH_ALPHA - _GARCH_BETA) * long_run_vol ** 2
        self._sigma2 = omega + _GARCH_ALPHA * self._last_eps ** 2 + _GARCH_BETA * self._sigma2
        sigma = float(np.sqrt(max(self._sigma2, 1e-12)))
        # Student-t rescaled to unit variance so `sigma` above is the
        # actual realized std, not a `df`-dependent multiple of it.
        raw_t = float(self.np_random.standard_t(_STUDENT_T_DOF))
        unit_t = raw_t / np.sqrt(_STUDENT_T_DOF / (_STUDENT_T_DOF - 2.0))
        eps = sigma * unit_t
        self._last_eps = eps
        return eps

    def _maybe_switch_regime(self) -> None:
        if self.np_random.random() < _SWITCH_PROB:
            others = [r for r in _REGIME_NAMES if r != self._regime]
            self._regime = others[int(self.np_random.integers(0, len(others)))]

    def _simulate_bar(self) -> tuple[float, float, float, float]:
        """Builds one OHLC bar from `_SUBSTEPS_PER_BAR` intra-bar GARCH/
        Student-t sub-steps (see module docstring) — `open` is the
        previous bar's `close` (bars are contiguous, no overnight gaps
        modeled), `high`/`low` track the running intra-bar extremes."""
        drift, long_run_vol = _REGIMES[self._regime]
        sub_drift = drift / _SUBSTEPS_PER_BAR
        sub_vol = long_run_vol / np.sqrt(_SUBSTEPS_PER_BAR)
        open_price = self._price
        high = low = open_price
        price = open_price
        for _ in range(_SUBSTEPS_PER_BAR):
            eps = self._garch_step(sub_vol)
            price *= float(np.exp(sub_drift + eps))
            high = max(high, price)
            low = min(low, price)
        self._price = price
        self._maybe_switch_regime()
        return open_price, high, low, price

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._regime = _REGIME_NAMES[int(self.np_random.integers(0, len(_REGIME_NAMES)))]
        self._price = 1.0
        self._equity = 1.0
        self._equity_history = [1.0]
        self._position = 0.0
        self._sigma2 = _REGIMES[self._regime][1] ** 2
        self._last_eps = 0.0
        self._t = 0
        self._bars = []
        # Matches `_bars`' eventual length 1:1 the whole episode (see
        # `render()`, which zips the two) — the warmup loop below only
        # ever grows bars, so backfill the same number of (flat, no
        # position held yet) entries here up front.
        self._position_history = []
        # Warm up `window` bars of return history before the episode
        # "starts" — otherwise the very first observation would be all
        # zeros, an unrealistically easy tell that no information is
        # available yet.
        self._returns = np.zeros(self.window, dtype=np.float32)
        for i in range(self.window):
            bar = self._simulate_bar()
            self._bars.append(bar)
            self._position_history.append(0.0)
            self._returns[i] = float(np.log(bar[3] / bar[0]))
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        frac_remaining = 1.0 - self._t / self.max_steps
        short_momentum = float(self._returns[-5:].sum())
        long_momentum = float(self._returns.sum())
        realized_vol = float(self._returns.std())
        current_vol = float(np.sqrt(max(self._sigma2, 1e-12)))
        extra = np.asarray(
            [short_momentum, long_momentum, realized_vol, current_vol, self._position, frac_remaining],
            dtype=np.float32,
        )
        return np.concatenate([self._returns, extra]).astype(np.float32)

    def step(self, action: Any):
        old_position = self._position
        new_position = self._target_position(action)
        bar = self._simulate_bar()
        self._bars.append(bar)
        self._position_history.append(new_position)
        log_return = float(np.log(bar[3] / bar[0]))
        self._returns = np.roll(self._returns, -1)
        self._returns[-1] = log_return

        turnover = abs(new_position - old_position)
        current_vol = float(np.sqrt(max(self._sigma2, 1e-12)))
        # Spread widens with current volatility (illiquid/choppy markets
        # quote wider) — `0.008` is the "sideways" regime's baseline vol,
        # used purely as the reference point "normal" spread was tuned
        # against.
        effective_spread = (self.base_spread_bps / 1e4) * max(1.0, current_vol / 0.008)
        spread_cost = effective_spread * turnover
        slippage_cost = self.impact_coef * turnover * current_vol
        commission_cost = self.commission_rate * turnover
        total_cost = spread_cost + slippage_cost + commission_cost

        pnl = old_position * log_return
        step_return = pnl - total_cost
        self._equity *= float(np.exp(step_return))
        self._equity_history.append(self._equity)
        self._position = new_position
        self._t += 1

        reward = step_return * _REWARD_SCALE
        # "Blown up" — lost effectively all equity (only reachable at all
        # with leveraged/short continuous positions; the discrete +-1
        # position alone can't actually reach zero equity in finite steps
        # at these vol levels) — end the episode early rather than let a
        # dead account keep "trading".
        terminated = self._equity <= 1e-3
        truncated = (not terminated) and self._t >= self.max_steps
        info = {
            "equity": self._equity, "price": self._price, "regime": self._regime,
            "spread_cost": spread_cost, "slippage_cost": slippage_cost, "commission_cost": commission_cost,
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return _render_candlestick_chart(
            bars=self._bars, positions=self._position_history,
            equity_history=self._equity_history, equity=self._equity,
            position=self._position, regime=self._regime,
        )


def _render_candlestick_chart(
    *, bars: list[tuple[float, float, float, float]], positions: list[float],
    equity_history: list[float], equity: float, position: float, regime: str,
) -> np.ndarray:
    """A real OHLC candlestick chart via Pillow (already an rl_core
    dependency — see `rl_core/algorithms/metrics_callback.py`'s GIF
    encoding), in the spirit of Gym-Trading-Env's renderer: green/red
    candle bodies + high-low wicks, a thin equity-curve sparkline above,
    and a position strip (green=long/red=short, width proportional to
    |position|) below — instead of a single hand-drawn price line with no
    OHLC or equity information at all."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 320, 150
    margin_top, margin_bottom = 26, 22
    img = Image.new("RGB", (width, height), (18, 18, 22))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=11)

    max_bars = 60
    recent_bars = bars[-max_bars:]
    recent_pos = positions[-max_bars:]
    n = len(recent_bars)
    draw.text((6, 4), f"Equity {equity:.3f}  Pos {position:+.2f}  {regime}", font=font, fill=(210, 210, 220))
    if n < 2:
        return np.asarray(img, dtype=np.uint8)

    lo = min(b[2] for b in recent_bars)
    hi = max(b[1] for b in recent_bars)
    span = max(hi - lo, 1e-9)
    plot_h = height - margin_top - margin_bottom
    slot_w = (width - 10) / n

    def y_of(price: float) -> int:
        return margin_top + int(plot_h * (1.0 - (price - lo) / span))

    for i, (o, h, low, c) in enumerate(recent_bars):
        x_center = 6 + int(slot_w * (i + 0.5))
        body_w = max(1, int(slot_w * 0.6))
        up = c >= o
        color = (80, 200, 120) if up else (220, 90, 90)
        draw.line([(x_center, y_of(h)), (x_center, y_of(low))], fill=(150, 150, 160))
        y_o, y_c = y_of(o), y_of(c)
        y_top, y_bot = min(y_o, y_c), max(y_o, y_c)
        draw.rectangle([x_center - body_w // 2, y_top, x_center + body_w // 2, max(y_bot, y_top + 1)], fill=color)

    # Equity sparkline (own y-scale, drawn as a thin line across the same
    # plot area) — lets a glance distinguish "price chopped sideways but
    # I bled to costs" from "price chopped sideways and I sat it out".
    eq_recent = equity_history[-max_bars:]
    if len(eq_recent) >= 2:
        eq_lo, eq_hi = min(eq_recent), max(eq_recent)
        eq_span = max(eq_hi - eq_lo, 1e-9)
        pts = [
            (6 + int(slot_w * (i + 0.5)), margin_top + int(plot_h * 0.15 * (1.0 - (e - eq_lo) / eq_span)))
            for i, e in enumerate(eq_recent)
        ]
        draw.line(pts, fill=(230, 210, 90), width=1)

    # Position strip along the bottom — width (not just color) encodes
    # |position|, so a half-size long and a full-size long are visibly
    # different, not just "both green".
    strip_y = height - margin_bottom + 6
    for i, p in enumerate(recent_pos):
        if abs(p) < 1e-6:
            continue
        x_center = 6 + int(slot_w * (i + 0.5))
        body_w = max(1, int(slot_w * 0.6))
        bar_h = max(1, int(10 * min(abs(p), 1.0)))
        color = (80, 200, 120) if p > 0 else (220, 90, 90)
        draw.rectangle([x_center - body_w // 2, strip_y + (10 - bar_h), x_center + body_w // 2, strip_y + 10], fill=color)

    return np.asarray(img, dtype=np.uint8)


def _discrete_entry(**kwargs) -> gym.Env:
    return TradingEnv(continuous=False, **kwargs)


def _continuous_entry(**kwargs) -> gym.Env:
    return TradingEnv(continuous=True, **kwargs)


def register_trading_envs() -> None:
    """Idempotent — safe to import/call from multiple modules. No
    third-party dependency to guard against (see module docstring), unlike
    every sibling `register_*_envs()` in this package."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    gym.register(id="Trading-Discrete-v0", entry_point=_discrete_entry)
    gym.register(id="Trading-Continuous-v0", entry_point=_continuous_entry)


# Self-registers at import time, exactly like `rl_core/envs/pomdp.py`.
register_trading_envs()
