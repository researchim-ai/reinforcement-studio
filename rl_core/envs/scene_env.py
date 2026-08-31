"""Gymnasium environments built from Scene Builder JSON specs.

`SceneMultiAgentEnv` is a custom `VectorEnv` where every "lane" is one agent
in a *shared* 3D world (not N independent copies). All agents step together
each call to `step()`; kinematic movement + AABB/circle collisions; items with
configurable rewards; NEXT_STEP autoreset per lane (compatible with
`rl_core/algorithms/vec_env.py` and every native algorithm).

Agents come from one or more *groups* (`spec["agents"]`, a list) — each
group has its own `count`, `team` (a free-form label; agents that share a
team are "teammates", everyone else is an "opponent" for sensor/reward
purposes) and `role` (free-form too, e.g. `"predator"`/`"prey"`; only
consulted by the optional `rules.tag` mechanic below). Every group still
shares one `movement`/`sensors` schema scene-wide (taken from the *first*
group) since a `VectorEnv`'s `single_action_space`/`single_observation_space`
must be identical for every lane — that's the one thing that can't vary
per group; body radius/spawn/shape/color all can, per-lane.

`rules` (optional, scene-level) adds two general-purpose multi-agent reward
mechanics on top of the plain per-agent item rewards below:
- `rules.tag`: a predator-role agent touching a prey-role agent on a
  *different* team exchanges a reward (predator gains, prey loses),
  optionally respawning/terminating the prey — the classic predator-prey
  MARL benchmark.
- `rules.team_shared_reward`: every agent's reward for the step is replaced
  by the sum of its whole team's raw rewards that step — turns any of the
  above into a fully cooperative (shared-credit) task instead of an
  individual one.

`SceneRenderEnv` wraps the vector env as a plain `gym.Env` for GIF preview
(`render_episode`) — controls agent 0, others take no-op actions.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import VectorEnv
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space

from rl_core import scene_store

_DEFAULT_COLORS = [(80, 160, 255), (255, 200, 80), (200, 120, 255), (120, 255, 200)]


def make_scene_env(
    slug: str,
    *,
    render_mode: str | None = None,
    for_gif: bool = False,
) -> VectorEnv | gym.Env:
    """Build a scene env from a saved slug. `for_gif=True` returns a plain
    `gym.Env` (agent 0 only) for `render_episode`; otherwise the full
    multi-agent `VectorEnv` for training."""
    spec = scene_store.load(slug)
    if for_gif or render_mode == "rgb_array":
        return SceneRenderEnv(spec, render_mode=render_mode or "rgb_array")
    return SceneMultiAgentEnv(spec, render_mode=render_mode)


def _agent_groups(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Every agent group, `team`/`role` defaulted — never empty (a spec
    with no `agents` key at all still gets one implicit single-agent
    group, exactly as before groups/teams existed)."""
    groups = spec.get("agents") or [{}]
    return [{"team": "default", "role": "agent", **g} for g in groups]


def _movement_cfg(spec: dict[str, Any]) -> dict[str, Any]:
    groups = spec.get("agents") or [{}]
    return groups[0].get("movement") or {"type": "discrete4", "speed": 0.5}


def _sensor_cfg(spec: dict[str, Any]) -> dict[str, Any]:
    groups = spec.get("agents") or [{}]
    return groups[0].get("sensors") or {"type": "nearest_k", "k": 4, "range": 10.0}


def _tag_rule(spec: dict[str, Any]) -> dict[str, Any] | None:
    rule = (spec.get("rules") or {}).get("tag")
    if not rule or not rule.get("enabled", True):
        return None
    return {
        "predator_role": str(rule.get("predator_role", "predator")),
        "prey_role": str(rule.get("prey_role", "prey")),
        "predator_reward": float(rule.get("predator_reward", 1.0)),
        "prey_reward": float(rule.get("prey_reward", -1.0)),
        "prey_terminates": bool(rule.get("prey_terminates", False)),
        "prey_respawns": bool(rule.get("prey_respawns", True)),
        "catch_radius_bonus": float(rule.get("catch_radius_bonus", 0.0)),
    }


def _team_shared_reward_enabled(spec: dict[str, Any]) -> bool:
    return bool((spec.get("rules") or {}).get("team_shared_reward", False))


def build_action_space(spec: dict[str, Any]) -> gym.Space:
    movement = _movement_cfg(spec).get("type", "discrete4")
    if movement == "discrete4":
        return Discrete(5)  # noop, +z, -z, -x, +x
    if movement == "discrete8":
        return Discrete(9)  # noop + 8 directions
    return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def build_observation_space(spec: dict[str, Any]) -> gym.Space:
    sensors = _sensor_cfg(spec)
    k = max(1, int(sensors.get("k", 4)))
    # self (x, z, vx, vz) + k * (rel_x, rel_z, kind, dist)
    dim = 4 + k * 4
    return Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)


def _parse_walls(objects: list[dict[str, Any]]) -> list[tuple[float, float, float, float, float, float]]:
    """Return list of AABB (min_x, max_x, min_z, max_z, y, height)."""
    walls: list[tuple[float, float, float, float, float, float]] = []
    for obj in objects or []:
        if obj.get("type") != "wall":
            continue
        pos = obj.get("position") or [0.0, 0.0, 0.0]
        size = obj.get("size") or [1.0, 1.0, 1.0]
        cx, _cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
        sx, _sy, sz = float(size[0]), float(size[1]), float(size[2])
        walls.append((cx - sx / 2, cx + sx / 2, cz - sz / 2, cz + sz / 2, float(pos[1]), float(size[1])))
    return walls


def _circle_aabb_overlap(x: float, z: float, r: float, wall: tuple[float, float, float, float, float, float]) -> bool:
    min_x, max_x, min_z, max_z, _y, _h = wall
    closest_x = min(max(x, min_x), max_x)
    closest_z = min(max(z, min_z), max_z)
    dx, dz = x - closest_x, z - closest_z
    return dx * dx + dz * dz < r * r


def _resolve_circle_aabb(x: float, z: float, r: float, wall: tuple[float, float, float, float, float, float]) -> tuple[float, float]:
    min_x, max_x, min_z, max_z, _y, _h = wall
    closest_x = min(max(x, min_x), max_x)
    closest_z = min(max(z, min_z), max_z)
    dx, dz = x - closest_x, z - closest_z
    dist_sq = dx * dx + dz * dz
    if dist_sq < 1e-12 or dist_sq >= r * r:
        return x, z
    dist = math.sqrt(dist_sq)
    push = (r - dist) + 1e-4
    nx, nz = dx / dist, dz / dist
    return x + nx * push, z + nz * push


def _action_delta(action: int | np.ndarray, movement_type: str, speed: float) -> tuple[float, float]:
    if movement_type == "continuous":
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        dx = float(np.clip(arr[0], -1.0, 1.0)) * speed
        dz = float(np.clip(arr[1], -1.0, 1.0)) * speed
        return dx, dz
    a = int(action)
    if movement_type == "discrete8":
        dirs = {
            1: (0.0, speed),
            2: (0.0, -speed),
            3: (-speed, 0.0),
            4: (speed, 0.0),
            5: (speed * 0.707, speed * 0.707),
            6: (-speed * 0.707, speed * 0.707),
            7: (-speed * 0.707, -speed * 0.707),
            8: (speed * 0.707, -speed * 0.707),
        }
        return dirs.get(a, (0.0, 0.0))
    # discrete4
    dirs = {1: (0.0, speed), 2: (0.0, -speed), 3: (-speed, 0.0), 4: (speed, 0.0)}
    return dirs.get(a, (0.0, 0.0))


def _hex_to_rgb(color: str | None) -> tuple[int, int, int] | None:
    if not color or not color.startswith("#") or len(color) != 7:
        return None
    try:
        return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
    except ValueError:
        return None


class SceneMultiAgentEnv(VectorEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15, "autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(self, spec: dict[str, Any], render_mode: str | None = None):
        super().__init__()
        self.spec = spec
        self.render_mode = render_mode
        self.world = spec.get("world") or {"width": 20.0, "depth": 20.0}
        self.half_w = float(self.world.get("width", 20.0)) / 2.0
        self.half_d = float(self.world.get("depth", 20.0)) / 2.0
        self.walls = _parse_walls(spec.get("objects") or [])
        self.items = list(spec.get("items") or [])
        # Movement type/sensor config are scene-wide, not per-group: every
        # lane shares one `single_action_space`/`single_observation_space`
        # (a hard `gymnasium.vector.VectorEnv` requirement — lanes can't
        # have heterogeneous spaces), so MARL scenes get heterogeneous
        # *behavior* (team/role, rewards, who's "us" vs "them" in the
        # sensor readout) without needing per-agent Dict/Tuple spaces.
        self.movement_type = _movement_cfg(spec).get("type", "discrete4")
        self.speed = float(_movement_cfg(spec).get("speed", 0.5))
        self.sensor_cfg = _sensor_cfg(spec)
        self.sensor_k = max(1, int(self.sensor_cfg.get("k", 4)))
        self.sensor_range = float(self.sensor_cfg.get("range", 10.0))
        self.max_steps = int((spec.get("episode") or {}).get("max_steps", 500))
        self.tag_rule = _tag_rule(spec)
        self.team_shared_reward = _team_shared_reward_enabled(spec)

        # Expand every group into per-lane arrays, in group order (group
        # 0's `count` lanes, then group 1's, ...) — this is the *only*
        # place lane index <-> group membership is decided, so
        # `MultiAgentPPO` (native/marl_ppo.py) reading `self.team_ids`
        # sees exactly the lane order everything below does.
        groups = _agent_groups(spec)
        self.num_envs = sum(max(1, int(g.get("count", 1))) for g in groups) or 1
        self.team_ids: list[str] = []
        self.roles: list[str] = []
        self.body_radii = np.zeros(self.num_envs, dtype=np.float32)
        self.shapes: list[str] = []
        self.colors: list[tuple[int, int, int]] = []
        self._spawn_centers = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._spawn_radii = np.zeros(self.num_envs, dtype=np.float32)
        lane = 0
        for gi, g in enumerate(groups):
            count = max(1, int(g.get("count", 1)))
            spawn = g.get("spawn") or {}
            center = spawn.get("center") or [0.0, 0.0, 0.0]
            radius = float(spawn.get("radius", 3.0))
            body_radius = float(g.get("body_radius", 0.4))
            shape = str(g.get("shape") or "capsule")
            color = _hex_to_rgb((g.get("material") or {}).get("color")) or _DEFAULT_COLORS[gi % len(_DEFAULT_COLORS)]
            for _ in range(count):
                if lane >= self.num_envs:
                    break
                self.team_ids.append(str(g.get("team", "default")))
                self.roles.append(str(g.get("role", "agent")))
                self.body_radii[lane] = body_radius
                self.shapes.append(shape)
                self.colors.append(color)
                self._spawn_centers[lane] = (float(center[0]), float(center[2]))
                self._spawn_radii[lane] = radius
                lane += 1
        self.teams: list[str] = sorted(set(self.team_ids))

        self.single_action_space = build_action_space(spec)
        self.single_observation_space = build_observation_space(spec)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self._rng = np.random.default_rng()
        self._positions = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._velocities = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._item_active = np.ones(len(self.items), dtype=bool)
        self._item_cooldown = np.zeros(len(self.items), dtype=np.int32)
        self._autoreset_envs = np.zeros(self.num_envs, dtype=bool)
        self._obs_buffer = np.zeros((self.num_envs, self.single_observation_space.shape[0]), dtype=np.float32)
        self._rewards = np.zeros(self.num_envs, dtype=np.float64)
        self._terminations = np.zeros(self.num_envs, dtype=bool)
        self._truncations = np.zeros(self.num_envs, dtype=bool)

    def _sample_spawn(self, lane: int, avoid: np.ndarray | None = None) -> tuple[float, float]:
        center = self._spawn_centers[lane]
        radius = float(self._spawn_radii[lane])
        body_radius = float(self.body_radii[lane])
        for _ in range(32):
            ang = self._rng.uniform(0, 2 * math.pi)
            rad = self._rng.uniform(0, radius)
            x = float(center[0] + math.cos(ang) * rad)
            z = float(center[1] + math.sin(ang) * rad)
            x = float(np.clip(x, -self.half_w + body_radius, self.half_w - body_radius))
            z = float(np.clip(z, -self.half_d + body_radius, self.half_d - body_radius))
            if avoid is not None and len(avoid) > 0:
                d = np.sqrt(((avoid[:, 0] - x) ** 2 + (avoid[:, 1] - z) ** 2).min(initial=1e9))
                if d < body_radius * 2.5:
                    continue
            blocked = any(_circle_aabb_overlap(x, z, body_radius, w) for w in self.walls)
            if not blocked:
                return x, z
        return float(center[0]), float(center[1])

    def _reset_agent(self, i: int) -> None:
        others = self._positions[np.arange(self.num_envs) != i] if self.num_envs > 1 else None
        x, z = self._sample_spawn(i, others)
        self._positions[i] = (x, z)
        self._velocities[i] = 0.0
        self._episode_steps[i] = 0

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            if isinstance(seed, int):
                self._rng = np.random.default_rng(seed)
            else:
                self._rng = np.random.default_rng(seed[0] if seed else None)
        self._item_active[:] = True
        self._item_cooldown[:] = 0
        self._autoreset_envs[:] = False
        for i in range(self.num_envs):
            self._reset_agent(i)
        self._fill_observations()
        return self._obs_buffer.copy(), {}

    def _entities_for_sensors(self, agent_idx: int) -> list[tuple[float, float, float]]:
        """(x, z, kind) where kind: 1=reward, 2=hazard, 3=teammate,
        4=opponent (another agent on a *different* team than `agent_idx`,
        e.g. predator-vs-prey or red-vs-blue — lets a policy actually tell
        rivals apart from allies instead of every other agent looking
        identical, which single-team scenes never needed but any
        competitive/mixed scene does)."""
        out: list[tuple[float, float, float]] = []
        ax, az = self._positions[agent_idx]
        for j, item in enumerate(self.items):
            if not self._item_active[j]:
                continue
            pos = item.get("position") or [0.0, 0.0, 0.0]
            kind = 1.0 if item.get("type") == "reward" else 2.0
            out.append((float(pos[0]), float(pos[2]), kind))
        own_team = self.team_ids[agent_idx]
        for j in range(self.num_envs):
            if j == agent_idx:
                continue
            kind = 3.0 if self.team_ids[j] == own_team else 4.0
            out.append((float(self._positions[j, 0]), float(self._positions[j, 1]), kind))
        # sort by distance
        out.sort(key=lambda e: (e[0] - ax) ** 2 + (e[1] - az) ** 2)
        return out

    def _obs_for_agent(self, i: int) -> np.ndarray:
        obs = np.zeros(self.single_observation_space.shape[0], dtype=np.float32)
        x, z = self._positions[i]
        vx, vz = self._velocities[i]
        obs[0] = np.clip(x / max(self.half_w, 1e-6), -1.0, 1.0)
        obs[1] = np.clip(z / max(self.half_d, 1e-6), -1.0, 1.0)
        obs[2] = np.clip(vx / max(self.speed, 1e-6), -1.0, 1.0)
        obs[3] = np.clip(vz / max(self.speed, 1e-6), -1.0, 1.0)
        entities = self._entities_for_sensors(i)
        base = 4
        for k in range(self.sensor_k):
            off = base + k * 4
            if k < len(entities):
                ex, ez, kind = entities[k]
                dx, dz = ex - x, ez - z
                dist = math.sqrt(dx * dx + dz * dz)
                if dist > self.sensor_range:
                    obs[off:off + 4] = 0.0
                else:
                    obs[off] = np.clip(dx / self.sensor_range, -1.0, 1.0)
                    obs[off + 1] = np.clip(dz / self.sensor_range, -1.0, 1.0)
                    obs[off + 2] = kind / 4.0
                    obs[off + 3] = dist / self.sensor_range
        return obs

    def _fill_observations(self) -> None:
        for i in range(self.num_envs):
            self._obs_buffer[i] = self._obs_for_agent(i)

    def _apply_collisions(self) -> None:
        for i in range(self.num_envs):
            x, z = float(self._positions[i, 0]), float(self._positions[i, 1])
            body_radius = float(self.body_radii[i])
            for wall in self.walls:
                if _circle_aabb_overlap(x, z, body_radius, wall):
                    x, z = _resolve_circle_aabb(x, z, body_radius, wall)
            x = float(np.clip(x, -self.half_w + body_radius, self.half_w - body_radius))
            z = float(np.clip(z, -self.half_d + body_radius, self.half_d - body_radius))
            self._positions[i] = (x, z)
        # agent-agent separation
        for i in range(self.num_envs):
            for j in range(i + 1, self.num_envs):
                dx = self._positions[j, 0] - self._positions[i, 0]
                dz = self._positions[j, 1] - self._positions[i, 1]
                dist_sq = dx * dx + dz * dz
                min_dist = float(self.body_radii[i] + self.body_radii[j])
                if dist_sq < min_dist * min_dist and dist_sq > 1e-8:
                    dist = math.sqrt(dist_sq)
                    overlap = (min_dist - dist) / 2
                    nx, nz = dx / dist, dz / dist
                    self._positions[i, 0] -= nx * overlap
                    self._positions[i, 1] -= nz * overlap
                    self._positions[j, 0] += nx * overlap
                    self._positions[j, 1] += nz * overlap

    def _tick_items(self) -> None:
        for j in range(len(self.items)):
            if not self._item_active[j] and self._item_cooldown[j] > 0:
                self._item_cooldown[j] -= 1
                if self._item_cooldown[j] <= 0:
                    self._item_active[j] = True

    def _apply_tag_rule(self) -> None:
        """Predator-vs-prey "tag" mechanic (see `_tag_rule()` above) —
        checked once per env `step()` across every predator/prey pair,
        *after* movement/collisions/items have already set this step's
        base rewards, so a tag adds on top of (never replaces) whatever
        else happened this step."""
        rule = self.tag_rule
        if rule is None:
            return
        predators = [i for i in range(self.num_envs) if self.roles[i] == rule["predator_role"]]
        preys = [i for i in range(self.num_envs) if self.roles[i] == rule["prey_role"]]
        for pi in predators:
            for qi in preys:
                if self.team_ids[pi] == self.team_ids[qi]:
                    continue  # same-team "predator"/"prey" never tags — needs a real rivalry
                catch_r = float(self.body_radii[pi] + self.body_radii[qi]) + rule["catch_radius_bonus"]
                dist = math.hypot(self._positions[pi, 0] - self._positions[qi, 0], self._positions[pi, 1] - self._positions[qi, 1])
                if dist >= catch_r:
                    continue
                self._rewards[pi] += rule["predator_reward"]
                self._rewards[qi] += rule["prey_reward"]
                if rule["prey_terminates"]:
                    self._terminations[qi] = True
                elif rule["prey_respawns"]:
                    self._reset_agent(qi)

    def _apply_team_shared_reward(self) -> None:
        if not self.team_shared_reward:
            return
        totals: dict[str, float] = {t: 0.0 for t in self.teams}
        for i in range(self.num_envs):
            totals[self.team_ids[i]] += float(self._rewards[i])
        for i in range(self.num_envs):
            self._rewards[i] = totals[self.team_ids[i]]

    def _step_agent(self, i: int, action: Any) -> None:
        dx, dz = _action_delta(action, self.movement_type, self.speed)
        self._velocities[i] = (dx, dz)
        self._positions[i, 0] += dx
        self._positions[i, 1] += dz
        self._apply_collisions()

        reward = 0.0
        terminated = False
        for j, item in enumerate(self.items):
            if not self._item_active[j]:
                continue
            restrict_team = item.get("restrict_team")
            if restrict_team and restrict_team != self.team_ids[i]:
                continue  # this item only rewards a specific team (resource-race scenes)
            pos = item.get("position") or [0.0, 0.0, 0.0]
            ix, iz = float(pos[0]), float(pos[2])
            ir = float(item.get("radius", 0.4))
            dist = math.hypot(self._positions[i, 0] - ix, self._positions[i, 1] - iz)
            if dist < float(self.body_radii[i]) + ir:
                reward += float(item.get("reward", 0.0))
                if item.get("terminate"):
                    terminated = True
                if item.get("respawn", True):
                    self._item_active[j] = False
                    self._item_cooldown[j] = int(item.get("cooldown_steps", 30))

        self._episode_steps[i] += 1
        truncated = self._episode_steps[i] >= self.max_steps
        self._rewards[i] = reward
        self._terminations[i] = terminated
        self._truncations[i] = truncated and not terminated

    def step(
        self, actions: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        raw = np.asarray(actions)
        if isinstance(self.single_action_space, Discrete):
            actions_list = [int(a) for a in raw.reshape(self.num_envs)]
        else:
            actions_list = [raw.reshape(self.num_envs, 2)[i] for i in range(self.num_envs)]

        for i in range(self.num_envs):
            if self._autoreset_envs[i]:
                self._reset_agent(i)
                self._rewards[i] = 0.0
                self._terminations[i] = False
                self._truncations[i] = False
            else:
                self._step_agent(i, actions_list[i])

        self._apply_tag_rule()
        self._apply_team_shared_reward()
        self._tick_items()
        self._fill_observations()
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            self._obs_buffer.copy(),
            self._rewards.copy(),
            self._terminations.copy(),
            self._truncations.copy(),
            {},
        )

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return _render_topdown(
            self.half_w, self.half_d, self.walls, self.items,
            self._item_active, self._positions, self.body_radii, self.shapes, self.colors,
        )

    def close(self) -> None:
        pass


class SceneRenderEnv(gym.Env):
    """Plain Env wrapper for agent-0 GIF preview."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, spec: dict[str, Any], render_mode: str = "rgb_array"):
        super().__init__()
        self._vec = SceneMultiAgentEnv(spec, render_mode=render_mode)
        self.render_mode = render_mode
        self.observation_space = self._vec.single_observation_space
        self.action_space = self._vec.single_action_space
        self.metadata = self._vec.metadata

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self._vec.reset(seed=seed, options=options)
        return obs[0], info

    def step(self, action):
        noop = 0 if isinstance(self.action_space, Discrete) else np.zeros(2, dtype=np.float32)
        actions = [action] + [noop] * (self._vec.num_envs - 1)
        obs, rewards, term, trunc, info = self._vec.step(actions)
        return obs[0], float(rewards[0]), bool(term[0]), bool(trunc[0]), info

    def render(self):
        return self._vec.render()

    def close(self):
        self._vec.close()


def _blit_shape(
    img: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    color: tuple[int, int, int],
    shape: str,
) -> None:
    """Stamp a simple top-down silhouette for GIF preview (visual only)."""
    h, w = img.shape[:2]
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    if shape in ("box", "pyramid"):
        mask = (np.abs(xx) <= r) & (np.abs(yy) <= r)
        if shape == "pyramid":
            mask = np.abs(xx) + np.abs(yy) <= r
    elif shape in ("crystal",):
        mask = np.abs(xx) + np.abs(yy) <= r
    elif shape in ("cylinder", "capsule", "cone"):
        # capsule/cylinder look circular from above; cone slightly smaller tip
        rad = r if shape != "cone" else max(1, int(r * 0.7))
        mask = xx * xx + yy * yy <= rad * rad
    else:
        mask = xx * xx + yy * yy <= r * r
    y_slice = slice(max(0, cy - r), min(h, cy + r + 1))
    x_slice = slice(max(0, cx - r), min(w, cx + r + 1))
    sub = img[y_slice, x_slice]
    m = mask[: sub.shape[0], : sub.shape[1]]
    sub[m] = color


def _render_topdown(
    half_w: float,
    half_d: float,
    walls: list[tuple[float, float, float, float, float, float]],
    items: list[dict],
    item_active: np.ndarray,
    positions: np.ndarray,
    body_radii: np.ndarray,
    shapes: list[str],
    colors: list[tuple[int, int, int]],
    size: int = 256,
) -> np.ndarray:
    img = np.ones((size, size, 3), dtype=np.uint8) * 40

    def to_px(x: float, z: float) -> tuple[int, int]:
        px = int((x + half_w) / (2 * half_w) * (size - 1))
        py = int((z + half_d) / (2 * half_d) * (size - 1))
        return int(np.clip(px, 0, size - 1)), int(np.clip(py, 0, size - 1))

    def fill_rect(min_x: float, max_x: float, min_z: float, max_z: float, color: tuple[int, int, int]) -> None:
        x0 = int((min_x + half_w) / (2 * half_w) * size)
        x1 = int((max_x + half_w) / (2 * half_w) * size)
        y0 = int((min_z + half_d) / (2 * half_d) * size)
        y1 = int((max_z + half_d) / (2 * half_d) * size)
        img[max(0, y0):min(size, y1 + 1), max(0, x0):min(size, x1 + 1)] = color

    fill_rect(-half_w, half_w, -half_d, half_d, (55, 55, 60))
    for min_x, max_x, min_z, max_z, _y, _h in walls:
        fill_rect(min_x, max_x, min_z, max_z, (90, 90, 100))

    for j, item in enumerate(items):
        if not item_active[j]:
            continue
        pos = item.get("position") or [0, 0, 0]
        px, py = to_px(float(pos[0]), float(pos[2]))
        r_px = max(2, int(float(item.get("radius", 0.4)) / (2 * half_w) * size))
        color = (60, 200, 90) if item.get("type") == "reward" else (200, 60, 60)
        shape = str(item.get("shape") or ("crystal" if item.get("type") == "reward" else "sphere"))
        _blit_shape(img, px, py, r_px, color, shape)

    for i in range(len(positions)):
        px, py = to_px(float(positions[i, 0]), float(positions[i, 1]))
        r_px = max(3, int(float(body_radii[i]) / (2 * half_w) * size))
        _blit_shape(img, px, py, r_px, colors[i], shapes[i])

    return img
