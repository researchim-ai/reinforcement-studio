"""Gymnasium-Robotics (Farama) maze navigation tasks — `AntMaze`/`PointMaze`,
the goal-conditioned continuous-control benchmark family used throughout
offline/goal-conditioned RL research (the D4RL suite's maze tasks; also a
staple HER — Hindsight Experience Replay — benchmark). `PointMaze` is a
simple 2D ball-in-a-maze (2D force control); `AntMaze` is the same maze
layouts but navigated by a full MuJoCo quadruped, i.e. locomotion +
navigation combined — a genuinely harder, more "real robot" continuous-
control task than the fixed-goal MuJoCo locomotion tasks (`Ant-v5` etc.)
already in the gallery.

Deliberately *not* including `gymnasium_robotics`'s Fetch (`FetchReach`/
`FetchPush`/`FetchPickAndPlace`/`FetchSlide`) or Shadow Hand
(`HandManipulate*`) manipulation tasks — arguably the more "industrial"
half of this package (robot-arm/dexterous-hand manipulation) — because the
current PyPI release (gymnasium-robotics 1.4.2) ships a real bug in their
`_env_setup()` (an `assert joint_type in (mjJNT_HINGE, mjJNT_SLIDE)` in
`mujoco_utils.set_joint_qpos` fails on the gripper joint at construction
time, a known upstream issue — see Gymnasium-Robotics#251/#256 and
https://stackoverflow.com/q/79358200 — the published wheel's own `v4` ids
still hit it), so `gym.make(...)` for any of them raises before a single
episode can even run. Not worth wiring up an env this app would then show
in the gallery as "available" but that immediately crashes on first use;
revisit if a fixed release ships.

`AntMaze`/`PointMaze`'s raw observation is `Dict(observation=Box(...),
achieved_goal=Box(2,), desired_goal=Box(2,))` — three plain `Box`es, same
shape of problem as highway-env's `parking-v0` (see
`rl_core/envs/highway_envs.py`) — flattened the same way with
`gymnasium.wrappers.FlattenObservation` rather than needing a bespoke
wrapper. Training still optimizes the env's own sparse reach-the-goal
reward as normal; flattening only reshapes the *observation*, and (since
there's no HER replay buffer here) the policy has to infer "what the goal
is" from the flattened desired_goal slice like any other observation
feature instead of getting it re-injected into every replay sample."""
from __future__ import annotations

import gymnasium as gym

_REGISTERED = False

# (our wrapped id, gymnasium-robotics' own base id)
_MAZE_BASE_IDS: list[tuple[str, str]] = [
    ("PointMaze-UMaze-Flat-v0", "PointMaze_UMaze-v3"),
    ("PointMaze-Medium-Flat-v0", "PointMaze_Medium-v3"),
    ("AntMaze-UMaze-Flat-v0", "AntMaze_UMaze-v5"),
    ("AntMaze-Medium-Flat-v0", "AntMaze_Medium-v5"),
]


def _make_flat_maze(base_id: str, **kwargs) -> gym.Env:
    import gymnasium_robotics  # noqa: F401  (side-effect: registers gymnasium-robotics' own ids)

    return gym.wrappers.FlattenObservation(gym.make(base_id, **kwargs))


def _entry_point_for(base_id: str):
    def _entry(**kwargs) -> gym.Env:
        return _make_flat_maze(base_id, **kwargs)
    return _entry


def register_robotics_envs() -> None:
    """Idempotent — safe to import/call from multiple modules, even when
    `gymnasium_robotics` isn't installed (a plain no-op then, exactly like
    every other optional-extra env in `rl_core/envs/registry.py` —
    `_gym_available()` is what actually decides whether these show as
    greyed-out in the gallery)."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    try:
        import gymnasium_robotics  # noqa: F401
    except ImportError:
        return
    for wrapped_id, base_id in _MAZE_BASE_IDS:
        gym.register(id=wrapped_id, entry_point=_entry_point_for(base_id))


# Self-registers at import time, exactly like `rl_core/envs/highway_envs.py`
# — any module that merely imports this one gets the ids registered for
# free.
register_robotics_envs()
