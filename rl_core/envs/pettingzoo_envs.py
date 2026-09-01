"""Real, published multi-agent RL benchmarks (PettingZoo) alongside Scene
Builder's user-authored scenes — `petting:{slug}` env ids, mirroring the
`scene:{slug}` convention in `rl_core/scene_store.py`/`rl_core/envs/
scene_env.py` closely enough that `ippo`/`qmix` (both written against
`SceneMultiAgentEnv.team_ids`) work against these completely unmodified.

Why these four specifically, out of everything PettingZoo ships: they're
the classic, most-cited MARL benchmarks (the ones QMIX/MADDPG/VDN's own
papers evaluate on) that (a) install with pure-Python wheels — no MuJoCo/
Box2D/SC2 binary — and (b) actually run fast on CPU, matching this app's
"train in the app, in minutes, on a laptop" scope (a real StarCraft II
match or a Dota 2 game needs a licensed game client plus, for anything
resembling OpenAI Five's training budget, thousands of GPU-years — flatly
outside what any desktop app could run; see the Academy MARL lesson):

- `simple_spread` — cooperative navigation (Lowe et al., 2017 / MADDPG's
  own benchmark): N agents must spread out to cover N landmarks, penalized
  for colliding. One team, everyone shares the same reward — the
  textbook case for `qmix`'s mixing network as well as `ippo`.
- `simple_adversary` — 2 teams, 1 adversary vs 2 cooperating "good" agents
  racing it to an unmarked goal landmark (the adversary doesn't know which
  landmark is the real target; the good agents do, but their shared
  reward only pays if they're not too close to the adversary either) —
  small, fast, asymmetric MARL.
- `simple_tag` — the MPE predator-prey benchmark: 3 cooperative adversaries
  chase 2 faster good agents. Same shape as Scene Builder's own
  `pack_hunt` template, just the field-standard version everyone else's
  published numbers are on.
- `pursuit` (SISL) — 8 pursuers cooperatively herd randomly-moving evaders
  into a corner on a discrete grid, each with a small egocentric image
  observation (not a flat vector) — one team, and the one env here that
  exercises the CNN path (`build_feature_extractor`/`is_image_space` in
  `rl_core/algorithms/native/networks.py`) instead of the MLP path.
- `knights_archers_zombies` (Butterfly) — 2 archers + 2 knights defend a
  castle from an endless wave of zombies; melee knights and ranged
  archers have genuinely different action semantics (`obs_method="vector"`
  gives every agent the same flat egocentric ray-cast, not pixels, so
  `qmix`'s shared mixer/`ippo`'s shared-per-role policy both still see a
  uniform input). 2 roles = 2 "teams" by this module's naming convention
  even though the objective is fully cooperative (both want the castle to
  survive) — real per-agent kill rewards, so `ippo` genuinely differs from
  plain parameter-shared `ppo`, unlike a single flat team.

Every one of these is genuinely "big" relative to Scene Builder's own
templates (up to 8 agents vs. Scene Builder's max of 6) while still
finishing an episode in well under a second on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

_PETTINGZOO_PREFIX = "petting:"


def is_pettingzoo_env_id(env_id: str) -> bool:
    return env_id.startswith(_PETTINGZOO_PREFIX)


def pettingzoo_slug_from_env_id(env_id: str) -> str:
    if not is_pettingzoo_env_id(env_id):
        raise ValueError(f"Not a PettingZoo env id: {env_id}")
    return env_id[len(_PETTINGZOO_PREFIX):]


def pettingzoo_env_id(slug: str) -> str:
    return f"{_PETTINGZOO_PREFIX}{slug}"


def _team_of_agent_default(agent_name: str) -> str:
    """`"pursuer_2"` -> `"pursuer"`, `"adversary_0"` -> `"adversary"` — every
    built-in PettingZoo env used here names agents `<role>_<index>`, so
    stripping the trailing `_<index>` recovers the team/role split for
    free, same idea as Scene Builder's own `team` field."""
    return agent_name.rsplit("_", 1)[0]


@dataclass
class _PettingZooSpec:
    slug: str
    name: str
    description: str
    # `render_mode=None` -> fastest headless training path (no pygame
    # surface allocated at all); `"rgb_array"` -> only ever requested by
    # `PettingZooRenderEnv`/the availability probe below, for GIF preview.
    factory: Callable[[str | None], Any]
    team_of_agent: Callable[[str], str] = field(default=_team_of_agent_default)
    recommended_total_timesteps: int = 100_000


def _make_simple_spread(render_mode: str | None = None):
    from mpe2 import simple_spread_v3

    return simple_spread_v3.parallel_env(N=3, local_ratio=0.5, continuous_actions=False, max_cycles=50, render_mode=render_mode)


def _make_simple_adversary(render_mode: str | None = None):
    from mpe2 import simple_adversary_v3

    return simple_adversary_v3.parallel_env(N=2, continuous_actions=False, max_cycles=50, render_mode=render_mode)


def _make_simple_tag(render_mode: str | None = None):
    from mpe2 import simple_tag_v3

    return simple_tag_v3.parallel_env(
        num_good=2, num_adversaries=3, num_obstacles=2, continuous_actions=False, max_cycles=60, render_mode=render_mode,
    )


def _make_pursuit(render_mode: str | None = None):
    from pettingzoo.sisl import pursuit_v5

    return pursuit_v5.parallel_env(n_pursuers=8, n_evaders=15, max_cycles=250, render_mode=render_mode)


def _make_kaz(render_mode: str | None = None):
    from pettingzoo.butterfly import knights_archers_zombies_v11

    return knights_archers_zombies_v11.parallel_env(
        num_archers=2, num_knights=2, max_zombies=10, max_cycles=200, render_mode=render_mode,
    )


_SPECS: dict[str, _PettingZooSpec] = {
    "simple_spread": _PettingZooSpec(
        slug="simple_spread",
        name="MPE: Simple Spread",
        description="Кооперативная навигация (Lowe et al., 2017 / MADDPG) — 3 агента одной команды должны "
                     "разъехаться по 3 ориентирам, не столкнувшись; награда общая на всех. Одна команда — "
                     "подходит для ppo/dqn/a2c, не для ippo/qmix (нужно 2+ команды; см. Simple Adversary/Tag).",
        factory=_make_simple_spread,
        team_of_agent=lambda _a: "agents",  # single cooperative team by design
    ),
    "simple_adversary": _PettingZooSpec(
        slug="simple_adversary",
        name="MPE: Simple Adversary",
        description="2 команды: 1 противник против 2 кооперирующихся «хороших» агентов, гонка к скрытой "
                     "цели-ориентиру — классический асимметричный MPE-бенчмарк (MADDPG/QMIX-статьи).",
        factory=_make_simple_adversary,
    ),
    "simple_tag": _PettingZooSpec(
        slug="simple_tag",
        name="MPE: Simple Tag",
        description="MPE-версия хищник/жертва: 3 кооперативных хищника (общая награда) гонятся за 2 быстрыми "
                     "жертвами — тот же сюжет, что у Scene Builder-шаблона «Стая против жертв», но "
                     "стандартный бенчмарк, на котором сравниваются MADDPG/QMIX в статьях.",
        factory=_make_simple_tag,
    ),
    "pursuit": _PettingZooSpec(
        slug="pursuit",
        name="SISL: Pursuit",
        description="8 преследователей кооперативно окружают 15 сбегающих на дискретной сетке — каждый "
                     "видит только маленькое (7×7) окно вокруг себя (картиночное наблюдение, не вектор). "
                     "Одна команда — подходит для ppo/dqn/a2c с CNN, не для ippo/qmix (нужно 2+ команды).",
        factory=_make_pursuit,
        team_of_agent=lambda _a: "pursuers",
        recommended_total_timesteps=200_000,
    ),
    "knights_archers_zombies": _PettingZooSpec(
        slug="knights_archers_zombies",
        name="Butterfly: Knights Archers Zombies",
        description="2 лучника + 2 рыцаря защищают замок от бесконечной волны зомби — вектор наблюдения "
                     "(egocentric ray-cast, не пиксели), 6 действий. Полностью кооперативная цель "
                     "(выжить), но награда — за личные убийства, так что роли 'archer'/'knight' считаются "
                     "2 командами: ippo/qmix здесь реально отличаются от общего ppo/dqn.",
        factory=_make_kaz,
        recommended_total_timesteps=150_000,
    ),
}


def list_slugs() -> list[str]:
    return list(_SPECS.keys())


class PettingZooVectorEnv(VectorEnv):
    """Adapts one PettingZoo `ParallelEnv` into the same `team_ids`-bearing
    `gymnasium.vector.VectorEnv` shape as `SceneMultiAgentEnv` (see its
    docstring in `rl_core/envs/scene_env.py`) — every possible agent is one
    fixed lane (`num_envs = len(possible_agents)`, order preserved across
    the whole run), so `ippo`/`qmix`/every plain native algorithm's
    `vec_env.py` helpers work against this completely unmodified.

    Two adapter details PettingZoo itself doesn't need to care about but a
    *uniform-space* `VectorEnv` does:

    - **Padding.** Different roles in the same env can have different
      observation sizes (e.g. `simple_adversary`'s lone adversary sees 8
      floats, its two "good" targets see 10) — real for these 4 envs, but
      `VectorEnv.single_observation_space` must be identical for every
      lane. Every per-agent observation is flattened and zero-padded up to
      the widest one; the extra zeros are indistinguishable from
      "everything else in range but far away" to the network and every
      built-in env already asks it to cope with variable-distance
      entities the same way (see `SceneMultiAgentEnv`'s own sensor
      padding), so this never needed special-casing in any algorithm.
    - **Synchronized episode end.** None of these 4 envs ever eliminates
      an agent mid-episode (unlike Scene Builder's own optional
      `rules.tag`) — every lane's episode always starts and ends on the
      same step, so autoreset (`NEXT_STEP`, like `SceneMultiAgentEnv`) only
      ever needs one shared "pending reset" flag, not lane-by-lane
      tracking.
    """

    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(self, spec: _PettingZooSpec, render_mode: str | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.render_mode = render_mode
        self._env = spec.factory(render_mode)
        self.possible_agents: list[str] = list(self._env.possible_agents)
        self.num_envs = len(self.possible_agents)
        self.team_ids: list[str] = [spec.team_of_agent(a) for a in self.possible_agents]
        self.teams: list[str] = sorted(set(self.team_ids))

        act_spaces = [self._env.action_space(a) for a in self.possible_agents]
        if not all(isinstance(sp, Discrete) for sp in act_spaces):
            raise ValueError(f"PettingZoo env '{spec.slug}': only Discrete per-agent action spaces are supported")
        n_actions = {int(sp.n) for sp in act_spaces}
        if len(n_actions) != 1:
            raise ValueError(f"PettingZoo env '{spec.slug}': every agent must share one action space (got {n_actions})")
        self.single_action_space = Discrete(n_actions.pop())

        obs_spaces = [self._env.observation_space(a) for a in self.possible_agents]
        obs_shapes = [tuple(sp.shape) for sp in obs_spaces]
        # If every agent already shares one observation shape (true for
        # every single-team env here — `pursuit`'s (7,7,3) egocentric crop
        # included), keep it exactly as-is: no flatten/pad needed, and
        # crucially preserves `pursuit`'s *image* shape so it still routes
        # through `build_feature_extractor`'s CNN path (`is_image_space`
        # checks `len(shape) == 3`) instead of silently becoming an
        # oversized flat MLP input. Only the heterogeneous-role MPE envs
        # (`simple_adversary`/`simple_tag`, different obs size per role)
        # need the flatten+zero-pad-to-widest fallback below.
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

    def _fill_obs(self, obs_dict: dict[str, Any]) -> np.ndarray:
        if self._homogeneous_shape is not None:
            for i, agent in enumerate(self.possible_agents):
                self._obs_buffer[i] = np.asarray(obs_dict[agent], dtype=np.float32)
            return self._obs_buffer.copy()
        self._obs_buffer[:] = 0.0
        for i, agent in enumerate(self.possible_agents):
            flat = np.asarray(obs_dict[agent], dtype=np.float32).reshape(-1)
            self._obs_buffer[i, : flat.shape[0]] = flat
        return self._obs_buffer.copy()

    def reset(self, *, seed: int | list[int] | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        seed0 = seed[0] if isinstance(seed, list) and seed else (seed if isinstance(seed, int) else None)
        obs_dict, _info = self._env.reset(seed=seed0)
        self._pending_reset = False
        return self._fill_obs(obs_dict), {}

    def step(self, actions: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        n = self.num_envs
        if self._pending_reset:
            obs_dict, _info = self._env.reset()
            self._pending_reset = False
            return self._fill_obs(obs_dict), np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), {}

        raw = np.asarray(actions).reshape(n)
        actions_dict = {agent: int(raw[i]) for i, agent in enumerate(self.possible_agents)}
        obs_dict, rew_dict, term_dict, trunc_dict, _info = self._env.step(actions_dict)
        rewards = np.array([float(rew_dict.get(a, 0.0)) for a in self.possible_agents], dtype=np.float32)
        terminated = np.array([bool(term_dict.get(a, False)) for a in self.possible_agents], dtype=bool)
        truncated = np.array([bool(trunc_dict.get(a, False)) for a in self.possible_agents], dtype=bool)
        if terminated.any() or truncated.any() or not self._env.agents:
            self._pending_reset = True
        return self._fill_obs(obs_dict), rewards, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        try:
            return self._env.render()
        except Exception:  # noqa: BLE001 — preview is best-effort, never fatal
            return None

    def close(self) -> None:
        self._env.close()


class PettingZooRenderEnv(gym.Env):
    """Single-agent view for GIF preview, controlling lane 0 (the first
    `possible_agents` entry) and sampling a random action for every other
    lane — exactly `SceneRenderEnv`'s role for Scene Builder scenes (see
    `rl_core/envs/scene_env.py`), and for the same reason: `render_episode`
    (`rl_core/algorithms/metrics_callback.py`) only knows how to drive a
    plain single-agent `gym.Env`. Wraps a `PettingZooVectorEnv` (rather
    than a second raw PettingZoo env) so its observation is padded/shaped
    *identically* to what the algorithm's network was trained on — a raw
    (unpadded) single-agent observation here would silently feed the net
    the wrong number of features."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, spec: _PettingZooSpec, render_mode: str = "rgb_array") -> None:
        super().__init__()
        self._vec = PettingZooVectorEnv(spec, render_mode=render_mode)
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

    def render(self):
        return self._vec.render()

    def close(self):
        self._vec.close()


def make_pettingzoo_env(slug: str, *, render_mode: str | None = None, for_gif: bool = False) -> VectorEnv | gym.Env:
    spec = _SPECS.get(slug)
    if spec is None:
        raise KeyError(f"Unknown PettingZoo env slug: {slug}")
    if for_gif or render_mode == "rgb_array":
        return PettingZooRenderEnv(spec, render_mode=render_mode or "rgb_array")
    return PettingZooVectorEnv(spec, render_mode=render_mode)


def team_count(slug: str) -> int:
    return len({_SPECS[slug].team_of_agent(a) for a in _probe_possible_agents(slug)})


def agent_count(slug: str) -> int:
    return len(_probe_possible_agents(slug))


_agent_cache: dict[str, list[str]] = {}


def _probe_possible_agents(slug: str) -> list[str]:
    if slug not in _agent_cache:
        env = _SPECS[slug].factory(None)
        try:
            _agent_cache[slug] = list(env.possible_agents)
        finally:
            env.close()
    return _agent_cache[slug]


def available(slug: str) -> bool:
    try:
        _probe_possible_agents(slug)
        return True
    except Exception:  # noqa: BLE001 — availability probe, never fatal
        return False


def list_meta() -> list[dict[str, Any]]:
    """One entry per registered PettingZoo env, in the shape `rl_core.envs.
    registry.list_environments()` expects to append (mirrors `scene_store.
    list_meta()`'s contract closely, minus the on-disk CRUD parts — these
    are code-defined, not user-editable)."""
    out: list[dict[str, Any]] = []
    for slug, spec in _SPECS.items():
        is_available = available(slug)
        out.append({
            "id": pettingzoo_env_id(slug),
            "slug": slug,
            "name": spec.name,
            "description": spec.description,
            "agent_count": agent_count(slug) if is_available else 0,
            "team_count": team_count(slug) if is_available else 1,
            "action_kind": "discrete",
            "available": is_available,
            "recommended_total_timesteps": spec.recommended_total_timesteps,
        })
    return out
