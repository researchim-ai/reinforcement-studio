"""A second family of real, published MARL benchmarks — RWARE and
LBForaging — alongside `rl_core/envs/pettingzoo_envs.py`'s PettingZoo four.
`marlgym:{slug}` env ids, same `team_ids`-bearing `VectorEnv` contract as
`SceneMultiAgentEnv`/`PettingZooVectorEnv` so `ippo`/`qmix`/every native
algorithm work against these unmodified — the only reason this isn't just
more entries in `pettingzoo_envs.py` is that both ship as a single
old-style `gym.Env` with `Tuple(...)` observation/action spaces (one env
object for the whole team, not one PettingZoo `ParallelEnv` "agent" per
lane), so the adapter (`TupleMarlVectorEnv` below) is a different shape
than `PettingZooVectorEnv`.

Why these two specifically: `rware` (Christianos et al., "Shared
Experience Actor-Critic for Multi-Agent Reinforcement Learning", NeurIPS
2020) and `lbforaging` (Papoudakis et al., "Benchmarking Multi-Agent Deep
Reinforcement Learning Algorithms in Cooperative Tasks", NeurIPS 2021
Datasets & Benchmarks — the EPyMARL paper) are the two *non*-MPE, non-SMAC
cooperative benchmarks that paper's own QMIX/MADDPG/MAPPO/VDN comparison
runs on, precisely because — like this app's own scope — they're pure
Python, install without a game client or physics binary, and finish an
episode on CPU in a fraction of a second:

- `rware` — Multi-Robot Warehouse: 2/4 robots navigate a grid warehouse,
  each carrying at most one shelf at a time, racing to deliver requested
  shelves to a drop-off, then return them — a direct, simplified stand-in
  for real warehouse-robot fleet coordination (the kind Amazon/Ocado
  actually run), not just a toy grid. Sparse reward (only on successful
  delivery) makes it a genuinely harder credit-assignment problem than
  MPE's dense per-step shaping.
- `lbforaging` — Level-Based Foraging: agents (each with a random
  "level") must coordinate to collect food items, but each item only
  yields its reward once the *sum* of levels of agents simultaneously
  next to it reaches the item's own level — no single agent can collect a
  high-level item alone, forcing genuine coordination (not just
  parallel, independent foraging). The `-coop-` variants used here
  guarantee every item requires 2+ agents.

Both are single-team (fully cooperative, shared objective) — the direct
"1 team, 2+ agents" case `qmix` was relaxed for (see
`rl_core/algorithms/native/qmix.py`'s module docstring); `ippo` still
doesn't apply (needs 2+ teams) here, same as `simple_spread`/`pursuit` in
`pettingzoo_envs.py`.

A third family, `smac_*`, lives in this same module/prefix for the same
structural reason (`Tuple(...)` obs/action spaces, one shared-team `gym.
Env`): **SMAClite** (Michalski et al., 2023, `arXiv:2305.05566`,
`uoe-agents/smaclite`) — a (nearly) pure-Python/Numpy reimplementation of
**SMAC**, the StarCraft Multi-Agent Challenge (Samvelyan et al., 2019,
`arXiv:1902.04043` — QMIX's *own* paper's flagship benchmark, and to date
the single most-cited MARL benchmark for value-decomposition methods).
Real SMAC needs an actual, ~30GB StarCraft II game client (Blizzard's
free Linux headless build) plus map files — the same class of problem as
Dota (see the Academy's MARL overview lesson) and a non-starter for a
pip-installed desktop app; SMAClite is API- and reward-identical (verified
by its own paper via transfer-learning experiments between the two) but
needs nothing beyond `numpy`/`pygame`/`rtree`, so it's the one genuinely
"big" SMAC-shaped benchmark this app can actually ship.

One real structural difference from RWARE/LBForaging that the adapter
below has to account for: SMAC(lite)'s action space is *situational* —
most of `n_actions` (move/no-op/stop plus one "attack unit k" action per
possible enemy) is invalid at any given moment (e.g. you can't attack an
enemy that's dead or out of range), and `SMACliteEnv.step` raises
`ValueError` outright on a genuinely invalid action instead of silently
tolerating it the way RWARE/LBForaging's always-fully-valid action spaces
never require the caller to think about. `TupleMarlVectorEnv` exposes the
current mask via `get_avail_actions()` (only when the wrapped env itself
has one — `None` for RWARE/LBForaging) and clamps any invalid action to a
safe fallback right before stepping, so *no* algorithm can ever crash the
run — but only `qmix` (`rl_core/algorithms/native/qmix.py`) actually reads
that mask to pick/bootstrap from *legal* actions during training, which is
why `smac_*`'s `compatible_algorithms` is `["qmix"]` alone, not the usual
discrete lineup: running `dqn`/`ppo`/etc. here unmasked would still train
(the clamp prevents crashes) but would spend most of its exploration
budget on forced fallback actions instead of ever seeing the real
consequence of what it "chose" — misleading, not merely suboptimal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

_TUPLE_MARL_PREFIX = "marlgym:"


def is_tuple_marl_env_id(env_id: str) -> bool:
    return env_id.startswith(_TUPLE_MARL_PREFIX)


def tuple_marl_slug_from_env_id(env_id: str) -> str:
    if not is_tuple_marl_env_id(env_id):
        raise ValueError(f"Not a tuple-space MARL env id: {env_id}")
    return env_id[len(_TUPLE_MARL_PREFIX):]


def tuple_marl_env_id(slug: str) -> str:
    return f"{_TUPLE_MARL_PREFIX}{slug}"


@dataclass
class _TupleMarlSpec:
    slug: str
    name: str
    description: str
    # `render_mode=None` -> fastest headless training path; `"rgb_array"`
    # -> only ever requested by `TupleMarlRenderEnv`/the availability probe,
    # for GIF preview.
    factory: Callable[[str | None], gym.Env]
    recommended_total_timesteps: int = 150_000
    # pip extra shown in the Environments gallery ("Не установлен extra
    # ...") — see `EXTRA_HINT` in `src/pages/Environments.tsx`. Distinct
    # key for `smac_*` because it needs a different (2-step) install
    # command than plain `pip install rware lbforaging` — see that file.
    extra_requirement: str = "rware/lbforaging"
    # True only for `smac_*` — see module docstring's "situational action
    # space" section. Drives both `compatible_algorithms` gating
    # (registry.py) and whether `TupleMarlVectorEnv` even bothers probing
    # the wrapped env for `get_avail_actions`.
    mask_aware: bool = False


def _make_rware_tiny_2ag(render_mode: str | None = None):
    import rware  # noqa: F401 — registers gym ids as a side effect

    return gym.make("rware-tiny-2ag-v2", render_mode=render_mode)


def _make_rware_small_4ag(render_mode: str | None = None):
    import rware  # noqa: F401

    return gym.make("rware-small-4ag-v2", render_mode=render_mode)


def _make_lbforaging_8x8_2p(render_mode: str | None = None):
    import lbforaging  # noqa: F401

    return gym.make("Foraging-8x8-2p-2f-coop-v3", render_mode=render_mode)


def _make_lbforaging_10x10_4p(render_mode: str | None = None):
    import lbforaging  # noqa: F401

    return gym.make("Foraging-10x10-4p-3f-coop-v3", render_mode=render_mode)


def _make_smac(scenario: str, render_mode: str | None = None):
    import smaclite  # noqa: F401 — registers `smaclite/{scenario}-v0` gym ids

    # `use_cpp_rvo2=False` (the default) — the optional C++ collision-
    # avoidance backend needs its own separate native build step; the
    # bundled Numpy port is plenty fast for this app's scenario sizes (see
    # the throughput numbers in the module docstring) and, critically,
    # needs nothing beyond what `pip install smaclite` already pulls in.
    return gym.make(f"smaclite/{scenario}-v0", use_cpp_rvo2=False, render_mode=render_mode)


def _make_smac_2s3z(render_mode: str | None = None):
    return _make_smac("2s3z", render_mode)


def _make_smac_3s_vs_5z(render_mode: str | None = None):
    return _make_smac("3s_vs_5z", render_mode)


def _make_smac_mmm2(render_mode: str | None = None):
    return _make_smac("MMM2", render_mode)


# Timestep budgets below are deliberately much larger than every other env in
# this app and are taken directly from the RWARE/LBF benchmarking literature,
# not chosen for a snappy in-app demo: EPyMARL (Papoudakis et al., 2021) and
# the extended follow-up benchmark (Christianos et al./AAMAS 2025,
# arXiv:2502.04773) train QMIX/VDN for 10M steps on LBF and a full **40M**
# steps on RWARE, and independently reproduced QMIX-on-RWARE write-ups
# (e.g. arXiv:2512.04463) report *zero* reward with default hyperparams even
# after 2M-10M steps, needing a 5M+-step epsilon anneal, a much bigger replay
# buffer (~200k) and batch size (~256) before the first delivery is ever
# discovered. Seeing `episode_reward_mean == 0.0` for a long time on these
# two envs is the expected shape of the learning curve, not a sign QMIX is
# broken — do not "fix" it by shrinking these numbers back down.
_SPECS: dict[str, _TupleMarlSpec] = {
    "rware_tiny_2ag": _TupleMarlSpec(
        slug="rware_tiny_2ag",
        name="RWARE: Tiny Warehouse (2 robots)",
        description="Multi-Robot Warehouse (Christianos et al., 2020 / SEAC) — 2 робота на маленьком складе "
                     "забирают запрошенные стеллажи и везут их к точке выдачи. Награда ОЧЕНЬ разреженная "
                     "(только за успешную доставку): в бенчмарках QMIX/EPyMARL первая ненулевая награда "
                     "появляется лишь после ~5 млн шагов, а сходится алгоритм к ~20-30 млн — на дефолтных "
                     "гиперпараметрах reward_mean=0 в первые несколько миллионов шагов — это нормально, не "
                     "баг. Для реального обучения увеличьте buffer_size (~100-200k), batch_size (~128-256) и "
                     "exploration_fraction (~0.6-0.8) у qmix. Команда одна — qmix/dqn/ppo, не ippo (нужно 2+ "
                     "команды).",
        factory=_make_rware_tiny_2ag,
        recommended_total_timesteps=5_000_000,
    ),
    "rware_small_4ag": _TupleMarlSpec(
        slug="rware_small_4ag",
        name="RWARE: Small Warehouse (4 robots)",
        description="Та же задача склада, но 4 робота на складе побольше — на порядок сложнее с точки зрения "
                     "координации и разреженной награды, стандартный «большой» сценарий из статьи про RWARE. "
                     "В литературе (EPyMARL/AAMAS'25) QMIX тренируют здесь 30-40 млн шагов; при коротких "
                     "прогонах и дефолтных buffer_size/batch_size reward_mean останется 0.0 — это ожидаемая "
                     "форма кривой обучения для этой среды, а не расходимость алгоритма. Для более быстрой "
                     "обратной связи начните с rware_tiny_2ag.",
        factory=_make_rware_small_4ag,
        recommended_total_timesteps=10_000_000,
    ),
    "lbforaging_8x8_2p": _TupleMarlSpec(
        slug="lbforaging_8x8_2p",
        name="Level-Based Foraging: 8×8 (2 агента)",
        description="Level-Based Foraging (Papoudakis et al., 2021 / EPyMARL) — 2 агента с уровнями собирают "
                     "еду; забрать еду можно только суммой уровней соседних агентов >= уровня еды, поэтому "
                     "-coop- вариант требует настоящей координации, а не параллельного независимого сбора. "
                     "Награда плотнее, чем в RWARE, но EPyMARL всё равно тренирует off-policy алгоритмы "
                     "(qmix/dqn) здесь ~2 млн шагов для устойчивого результата.",
        factory=_make_lbforaging_8x8_2p,
        recommended_total_timesteps=2_000_000,
    ),
    "lbforaging_10x10_4p": _TupleMarlSpec(
        slug="lbforaging_10x10_4p",
        name="Level-Based Foraging: 10×10 (4 агента)",
        description="Та же задача, но 4 агента на поле 10×10 с 3 источниками еды — больше агентов и больше "
                     "целей одновременно, более требовательный к координации «большой» вариант. Из-за роста "
                     "числа агентов и целей закладывайте больше шагов, чем для 8×8/2p — в бенчмарках "
                     "используют до 10 млн.",
        factory=_make_lbforaging_10x10_4p,
        recommended_total_timesteps=3_000_000,
    ),
    # SMAClite (Michalski et al., 2023) scenarios — one per official SMAC
    # difficulty tier (Samvelyan et al., 2019's own classification), picked
    # for throughput as well as fame: `corridor`/`bane_vs_bane`/`2c_vs_64zg`
    # are all real "super hard"/"hard" SMAC scenarios too, but 25-45
    # steps/sec single-process (vs. 130-320 for the three below) makes them
    # a poor fit for a first pass — see the module docstring.
    "smac_2s3z": _TupleMarlSpec(
        slug="smac_2s3z",
        name="SMAC: 2s3z (Easy)",
        description="StarCraft Multi-Agent Challenge (Samvelyan et al., 2019 — QMIX's own paper's benchmark), через "
                     "SMAClite (Michalski et al., 2023) — pip-пакет, не требующий самой игры. 2 Stalker'а + 3 "
                     "Zealot'а против зеркальной команды противника, официальный SMAC-тир «Easy». В отличие от "
                     "RWARE/LBForaging, действие «атаковать юнита k» валидно только для части юнитов в любой момент "
                     "(вне радиуса/уже мёртв) — доступные действия читает только qmix (маскирует выбор и цель "
                     "Double-DQN), поэтому здесь только он в списке совместимых алгоритмов. В SMAC/PyMARL-литературе "
                     "«Easy»-сценарии обычно тренируют ~2 млн шагов.",
        factory=_make_smac_2s3z,
        recommended_total_timesteps=2_000_000,
        extra_requirement="smaclite",
        mask_aware=True,
    ),
    "smac_3s_vs_5z": _TupleMarlSpec(
        slug="smac_3s_vs_5z",
        name="SMAC: 3s_vs_5z (Hard)",
        description="Тот же бенчмарк, официальный тир «Hard»: 3 своих Stalker'а против 5 Zealot'ов противника — "
                     "асимметрия числа юнитов означает, что простое «стоять и стрелять» проигрывает, а выигрывает "
                     "только keyting (отход при перезарядке, разрозненный микро-контроль) — классический пример из "
                     "статей про QMIX/QTRAN/QPLEX, где именно эта асимметрия отделяет работающую value-decomposition "
                     "от неработающей. Требует заметно больше шагов, чем 2s3z.",
        factory=_make_smac_3s_vs_5z,
        recommended_total_timesteps=5_000_000,
        extra_requirement="smaclite",
        mask_aware=True,
    ),
    "smac_mmm2": _TupleMarlSpec(
        slug="smac_mmm2",
        name="SMAC: MMM2 (Super Hard)",
        description="Официальный тир «Super Hard» и, вероятно, самый цитируемый сценарий во всей SMAC-литературе: "
                     "1 Medivac (лечит и не атакует) + 2 Marauder'а + 7 Marine'ов против более сильной вражеской "
                     "группы — неоднородные роли (лекарь отдельно от урона) и большое число юнитов (10 агентов) "
                     "делают его серьёзным тестом на масштабируемость credit assignment. Медленнее по шагам/сек, чем "
                     "2s3z/3s_vs_5z (больше юнитов -> дороже расчёт коллизий) — закладывайте больше времени на прогон, "
                     "не только больше шагов.",
        factory=_make_smac_mmm2,
        recommended_total_timesteps=10_000_000,
        extra_requirement="smaclite",
        mask_aware=True,
    ),
}


def list_slugs() -> list[str]:
    return list(_SPECS.keys())


class TupleMarlVectorEnv(VectorEnv):
    """Adapts one `gym.Env` with `Tuple(...)` observation/action spaces
    (RWARE/LBForaging's own native API — one env object for the whole
    team, not one object per agent like PettingZoo) into the same
    `team_ids`-bearing `gymnasium.vector.VectorEnv` shape as
    `SceneMultiAgentEnv`/`PettingZooVectorEnv` — every per-agent Tuple slot
    is one fixed lane (`num_envs = len(obs_space.spaces)`), all in a
    single team (both benchmarks here are fully cooperative by design —
    see module docstring), so `qmix`/every plain native algorithm's
    `vec_env.py` helpers work against this unmodified. `ippo` doesn't
    apply here (needs 2+ teams).

    Both `terminated`/`truncated` come back from the underlying env as a
    single shared bool (the whole team's episode ends together, there's
    no per-agent elimination) — broadcast to every lane, same as
    `PettingZooVectorEnv`'s synchronized-episode-end case."""

    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(self, spec: _TupleMarlSpec, render_mode: str | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.render_mode = render_mode
        self._env = spec.factory(render_mode)
        # Only `smac_*` (`spec.mask_aware`) actually has a situational
        # action space — `getattr(..., "get_avail_actions", None)` is
        # still checked defensively (not just `if spec.mask_aware`) so a
        # future non-SMAC mask-aware env would pick this up automatically.
        self._mask_fn = getattr(self._env.unwrapped, "get_avail_actions", None) if spec.mask_aware else None
        obs_space, act_space = self._env.observation_space, self._env.action_space
        if not isinstance(obs_space, gym.spaces.Tuple) or not isinstance(act_space, gym.spaces.Tuple):
            raise ValueError(f"Tuple-space MARL env '{spec.slug}': expected Tuple observation/action spaces")
        self.num_envs = len(obs_space.spaces)
        self.team_ids: list[str] = ["agents"] * self.num_envs
        self.teams: list[str] = ["agents"]

        if not all(isinstance(sp, Discrete) for sp in act_space.spaces):
            raise ValueError(f"Tuple-space MARL env '{spec.slug}': only Discrete per-agent action spaces are supported")
        n_actions = {int(sp.n) for sp in act_space.spaces}
        if len(n_actions) != 1 and not spec.mask_aware:
            raise ValueError(f"Tuple-space MARL env '{spec.slug}': every agent must share one action space (got {n_actions})")
        # `mask_aware` envs (SMAClite's heterogeneous unit roles — e.g.
        # MMM2's Medivac has 15 actions (no attack, only heal-target) vs.
        # every other unit's 18) may have per-agent `Discrete` sizes that
        # genuinely differ. Use the widest one uniformly (every native
        # algorithm's Q-network already needs one fixed `n_actions` per
        # team) — `get_avail_actions()` already returns every lane's mask
        # pre-padded to this same width with the extra slots `False` (see
        # the throughput probe in the module docstring), so a smaller-
        # action-space lane's padding indices are simply never legal and
        # `_sanitize_actions` never lets one through to `self._env.step`.
        self.single_action_space = Discrete(max(n_actions))

        obs_shapes = [tuple(sp.shape) for sp in obs_space.spaces]
        # RWARE/LBForaging always give every agent an identical-shaped
        # observation (same sensors, same grid) — no heterogeneous-role
        # padding needed here, unlike `simple_adversary`/`simple_tag` in
        # `pettingzoo_envs.py`. Kept as a runtime check (not an
        # assumption) so a future, asymmetric variant would still degrade
        # gracefully via the same flatten+zero-pad-to-widest fallback.
        self._homogeneous_shape = obs_shapes[0] if len(set(obs_shapes)) == 1 else None
        if self._homogeneous_shape is not None:
            self.single_observation_space = Box(low=-np.inf, high=np.inf, shape=self._homogeneous_shape, dtype=np.float32)
        else:
            self._max_obs_dim = max(int(np.prod(shape)) for shape in obs_shapes)
            self.single_observation_space = Box(low=-np.inf, high=np.inf, shape=(self._max_obs_dim,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self._pending_reset = True
        self._obs_buffer = np.zeros((self.num_envs, *self.single_observation_space.shape), dtype=np.float32)

    def get_avail_actions(self) -> np.ndarray | None:
        """Boolean `(num_envs, n_actions)` mask of which actions are legal
        for each lane *right now* (i.e. for whichever obs was last
        returned by `reset()`/`step()`) — `None` for envs without a
        situational action space (RWARE/LBForaging). `qmix`
        (`rl_core/algorithms/native/qmix.py`) calls this directly (this is
        a plain in-process object, never wrapped in an `AsyncVectorEnv` —
        see `is_shared_world_env_id` in `rl_core/envs/factory.py` — so a
        live method call across algorithm/env is always safe here)."""
        if self._mask_fn is None:
            return None
        return np.asarray(self._mask_fn(), dtype=bool)

    def _sanitize_actions(self, raw: np.ndarray) -> np.ndarray:
        """Clamps any action a lane's *current* mask marks illegal to that
        lane's first legal action — the safety net every caller gets for
        free, regardless of whether it bothered to read `get_avail_
        actions()` itself (see module docstring: only `qmix` does).
        Without this, `SMACliteEnv.step` raises `ValueError` outright on an
        illegal action instead of tolerating it."""
        mask = self.get_avail_actions()
        if mask is None:
            return raw
        out = raw.copy()
        for i in range(self.num_envs):
            if not mask[i, out[i]]:
                legal = np.flatnonzero(mask[i])
                out[i] = int(legal[0]) if legal.size else 0
        return out

    def _fill_obs(self, obs_tuple: tuple) -> np.ndarray:
        if self._homogeneous_shape is not None:
            for i, o in enumerate(obs_tuple):
                self._obs_buffer[i] = np.asarray(o, dtype=np.float32)
            return self._obs_buffer.copy()
        self._obs_buffer[:] = 0.0
        for i, o in enumerate(obs_tuple):
            flat = np.asarray(o, dtype=np.float32).reshape(-1)
            self._obs_buffer[i, : flat.shape[0]] = flat
        return self._obs_buffer.copy()

    def reset(self, *, seed: int | list[int] | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        seed0 = seed[0] if isinstance(seed, list) and seed else (seed if isinstance(seed, int) else None)
        obs_tuple, _info = self._env.reset(seed=seed0)
        self._pending_reset = False
        return self._fill_obs(obs_tuple), {}

    def step(self, actions: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        n = self.num_envs
        if self._pending_reset:
            obs_tuple, _info = self._env.reset()
            self._pending_reset = False
            return self._fill_obs(obs_tuple), np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), {}

        raw = np.asarray(actions).reshape(n).astype(np.int64)
        raw = self._sanitize_actions(raw)
        actions_tuple = tuple(int(x) for x in raw)
        obs_tuple, rew, term, trunc, _info = self._env.step(actions_tuple)
        # RWARE/LBForaging return one reward *per agent* (a list/tuple);
        # SMAClite returns a single shared team scalar instead (there's no
        # per-unit reward in SMAC's own API either) — broadcast it to
        # every lane so `MultiAgentQMIX.learn`'s `np.mean(rewards[lanes])`
        # (already a no-op on an already-identical broadcast value) and
        # every other caller never need their own special case for this.
        rewards = (
            np.asarray(list(rew), dtype=np.float32)
            if isinstance(rew, (list, tuple, np.ndarray))
            else np.full(n, float(rew), dtype=np.float32)
        )
        done = bool(term) or bool(trunc)
        terminated = np.full(n, bool(term), dtype=bool)
        truncated = np.full(n, bool(trunc), dtype=bool)
        if done:
            self._pending_reset = True
        return self._fill_obs(obs_tuple), rewards, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        try:
            frame = self._env.render()
            return np.asarray(frame) if frame is not None else None
        except Exception:  # noqa: BLE001 — preview is best-effort, never fatal (some backends need a real display)
            return None

    def close(self) -> None:
        self._env.close()


class TupleMarlRenderEnv(gym.Env):
    """Single-agent view for GIF preview, controlling lane 0 and sampling
    a random action for every other lane — exactly `PettingZooRenderEnv`'s
    role in `rl_core/envs/pettingzoo_envs.py` (see its docstring), for the
    same reason: `render_episode` only knows how to drive a plain
    single-agent `gym.Env`."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, spec: _TupleMarlSpec, render_mode: str = "rgb_array") -> None:
        super().__init__()
        self._vec = TupleMarlVectorEnv(spec, render_mode=render_mode)
        self.render_mode = render_mode
        self.observation_space = self._vec.single_observation_space
        self.action_space = self._vec.single_action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self._vec.reset(seed=seed)
        return obs[0], info

    def step(self, action):
        actions = [action] + [self._vec.single_action_space.sample() for _ in range(self._vec.num_envs - 1)]
        obs, rewards, term, trunc, info = self._vec.step(actions)
        return obs[0], float(rewards[0]), bool(term[0]), bool(trunc[0]), info

    def get_avail_actions(self) -> np.ndarray | None:
        """Passthrough so `qmix`'s `predict()` (called with *this* wrapper
        as `env` during GIF preview — see `render_episode`) can still mask
        lane 0's action the same way `learn()`'s live rollout does. Not
        load-bearing for correctness (`TupleMarlVectorEnv.step` already
        clamps any illegal action before it ever reaches `smaclite` — see
        its own docstring), just for a preview that matches training."""
        return self._vec.get_avail_actions()

    def render(self):
        return self._vec.render()

    def close(self):
        self._vec.close()


def make_tuple_marl_env(slug: str, *, render_mode: str | None = None, for_gif: bool = False) -> VectorEnv | gym.Env:
    spec = _SPECS.get(slug)
    if spec is None:
        raise KeyError(f"Unknown tuple-space MARL env slug: {slug}")
    if for_gif or render_mode == "rgb_array":
        return TupleMarlRenderEnv(spec, render_mode=render_mode or "rgb_array")
    return TupleMarlVectorEnv(spec, render_mode=render_mode)


def team_count(_slug: str) -> int:
    return 1  # both benchmarks are single-team by design — see module docstring


def agent_count(slug: str) -> int:
    return len(_probe_obs_space(slug).spaces)


_obs_space_cache: dict[str, gym.spaces.Tuple] = {}


def _probe_obs_space(slug: str) -> gym.spaces.Tuple:
    if slug not in _obs_space_cache:
        env = _SPECS[slug].factory(None)
        try:
            _obs_space_cache[slug] = env.observation_space
        finally:
            env.close()
    return _obs_space_cache[slug]


def available(slug: str) -> bool:
    try:
        _probe_obs_space(slug)
        return True
    except Exception:  # noqa: BLE001 — availability probe, never fatal
        return False


def list_meta() -> list[dict[str, Any]]:
    """One entry per registered tuple-space MARL env, in the same shape
    `pettingzoo_envs.list_meta()` uses (see its docstring) — code-defined,
    not user-editable."""
    out: list[dict[str, Any]] = []
    for slug, spec in _SPECS.items():
        is_available = available(slug)
        out.append({
            "id": tuple_marl_env_id(slug),
            "slug": slug,
            "name": spec.name,
            "description": spec.description,
            "agent_count": agent_count(slug) if is_available else 0,
            "team_count": team_count(slug),
            "action_kind": "discrete",
            "available": is_available,
            "recommended_total_timesteps": spec.recommended_total_timesteps,
            "extra_requirement": spec.extra_requirement,
            "mask_aware": spec.mask_aware,
        })
    return out
