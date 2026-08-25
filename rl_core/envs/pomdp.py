"""Partially-observable ("POMDP") variants of a few classic Gymnasium tasks,
plus one dedicated delayed-cue-recall task — added specifically to give the
memory-enabled (LSTM/GRU) algorithms (see
`rl_core/algorithms/native/networks.py`) something that actually *needs*
them: a purely feed-forward policy is mathematically capped at a fixed
ceiling on every environment below (it cannot see enough of the state to do
better), while a recurrent one is not.

Two standard recipes from the literature, applied to a couple of the
classic-control envs already in the gallery:

- **Velocity masking** (`_MaskObservation`) — drop the derivative terms
  (cart/angular velocity) from the observation so only "which way things are
  pointed right now" is visible, not "which way they're moving". This is the
  textbook way papers turn an MDP benchmark into a POMDP one — e.g. RLlib's
  `StatelessCartPole` example, or the velocity-masked control tasks in Heess
  et al., 2015, "Memory-based control with recurrent neural networks".
- **Flickering observations** (`_FlickeringObservation`) — with probability
  `p` on each step, the entire observation is blanked to zero instead of the
  real one. This is exactly the "Flickering Atari" recipe from Hausknecht &
  Stone, 2015, "Deep Recurrent Q-Learning for Partially Observable MDPs" (the
  paper that introduced DRQN) — `FlickeringPong-v0` below reproduces their
  experiment directly; `FlickeringCartPole-v0` is the same idea on a much
  cheaper env for quick experimentation.

Plus two from-scratch tasks (not wrappers around an existing env):

- `MemoryCorridorEnv` — a delayed-cue-recall ("T-maze style") task in the
  tradition of Bakker, 2001, "Reinforcement Learning with Long Short-Term
  Memory": a single cue bit is shown once, then hidden for several neutral
  steps, and only rewarded if recalled correctly at the very end.
  `MemoryCorridorLong-v0` is the same task with a much longer delay (20 vs
  8 neutral steps), to show memory *capacity* (bsuite's `memory_len` sweep
  does the same thing).
- `RepeatPreviousEnv` (`RepeatPrevious-v0`) — the classic "N-back" /
  delayed-match-to-sample task: every step shows a fresh random symbol, and
  the correct action is always the symbol shown exactly `delay` steps
  earlier. Unlike Memory Corridor (one cue, read once), this continuously
  exercises a sliding window of the last few observations on *every* step
  of the episode.
- `RockSampleEnv` (`RockSample-v0`) — the classic scalable POMDP *planning*
  benchmark (Smith & Simmons, 2004; the default 7x7-grid/8-rock config here
  is their `RockSample(7,8)`), reused as an RL task: rock positions are
  known, but each rock's quality (good/bad) is hidden and can only be
  probed with a noisy long-range sensor whose accuracy decays with
  distance. Good policies have to *integrate many noisy readings over
  time* into a belief about each rock before committing to `sample` —
  a fundamentally different memory demand than "recall one cue", closer to
  the belief-state-tracking POMDPs studied in the planning literature.
- `VisualMemoryMazeEnv` (`VisualMemoryMaze-v0`) — an image-observation
  maze in the spirit of MiniGrid's `MemoryEnv` (the flagship benchmark used
  to demo recurrent policies in, e.g., `sb3-contrib`'s `RecurrentPPO`
  docs): the agent starts in a room with a colored cue, walks down a
  corridor it can only see a few cells of at a time (a small egocentric
  crop, not the masking trick used above), and must pick whichever of two
  end rooms has an object matching the *long-gone* cue color. Unlike every
  other task in this module the partial observability comes from a
  genuinely limited field of view rather than a hand-picked masked
  channel — the more realistic mechanism behind most real POMDPs.

A further batch, each chasing a *different* flavour of "why memory matters"
than the delayed-single-cue-recall pattern above — straight out of the
classic POMDP-planning literature (Kaelbling, Littman & Cassandra, 1998;
Littman, Cassandra & Kaelbling, 1995) and the more recent POPGym /
memory-RL benchmark suites:

- `TigerEnv` (`Tiger-v0`) — *the* canonical toy POMDP used to introduce the
  whole field: integrate several independently-noisy `listen` actions into
  a confident belief before committing to a door.
- `HeavenHellEnv` (`HeavenHell-v0`) — *active* information gathering: the
  sign revealing which fork leads to Heaven sits down a side corridor, not
  on the main path, so a policy that never bothers to detour and check it
  is stuck guessing.
- `HallwayEnv` (`Hallway-v0`) — Littman et al.'s original state-*aliasing*
  benchmark, simplified to a small hand-built maze: two rooms give the
  identical local reading yet need opposite actions to reach the goal, so
  only remembering the path taken since a landmark disambiguates them.
- `BattleshipEnv` (`Battleship-v0`) and `MinesweeperPOMDPEnv`
  (`MinesweeperPOMDP-v0`) — classic grid search/logic games reformulated so
  the observation is only ever the single most-recent cell's result, never
  a rendering of the whole board — avoiding a wasted repeat shot/reveal (or
  reasoning globally about many past numbers) needs real memory, not
  pattern-matching a fully-observed grid.
- `ConcentrationEnv` (`Concentration-v0`) — the card-matching memory game,
  one of the POPGym suite's diagnostic memory tasks: a card's identity is
  only ever observed on the exact step it's flipped.
- `LaserTagEnv` (`LaserTag-v0`) — pursue a *moving* opponent visible only
  within a short, wall-blocked line of sight; unlike every static hidden
  fact above, what's being tracked changes on its own, calling for a
  running belief rather than a one-shot memory.
- `ActiveTMazeEnv` (`ActiveTMaze-v0`, `ActiveTMazeLong-v0`) — Ni et al.,
  2023's active T-maze: the cue is gated behind a `look` action the agent
  must choose to take at the very start of a one-way corridor, testing
  whether it learns *to query* for information, not just to retain it once
  handed over for free.

Both wrapper classes are plain `gymnasium.ObservationWrapper`s, so rendering
(`env.render()`) is inherited unchanged from the wrapped env — the physics
and the on-screen animation are identical to the base task, only what the
*agent* observes differs.

Registered as ordinary `gymnasium` env ids via `gym.register(...)` at import
time (the same mechanism used for ALE/Atari ids, see
`rl_core/algorithms/sb3_runner.py`), so they work anywhere a Gym env id
already does: `gym.make()` in the native/SB3 runners, live rendering in the
Training Monitor, and the Designer's `/environments/inspect` endpoint.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

_REGISTERED = False


class _MaskObservation(gym.ObservationWrapper):
    """Keeps only `keep_indices` of a flat Box observation, dropping the rest
    (typically the velocity terms — the part that makes the raw state
    Markovian)."""

    def __init__(self, env: gym.Env, keep_indices: tuple[int, ...]) -> None:
        super().__init__(env)
        self.keep_indices = keep_indices
        base = env.observation_space
        assert isinstance(base, gym.spaces.Box)
        idx = list(keep_indices)
        self.observation_space = gym.spaces.Box(low=base.low[idx], high=base.high[idx], dtype=base.dtype)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation)[list(self.keep_indices)]


class _FlickeringObservation(gym.ObservationWrapper):
    """Replaces the whole observation with zeros with probability
    `flicker_prob` on every step, independent of history."""

    def __init__(self, env: gym.Env, flicker_prob: float = 0.5) -> None:
        super().__init__(env)
        self.flicker_prob = flicker_prob

    def observation(self, observation: np.ndarray) -> np.ndarray:
        arr = np.asarray(observation)
        if self.np_random.random() < self.flicker_prob:
            return np.zeros_like(arr)
        return arr


def _po_cartpole_entry(**kwargs: object) -> gym.Env:
    env = gym.make("CartPole-v1", **kwargs)
    return _MaskObservation(env, keep_indices=(0, 2))  # cart position, pole angle


def _po_pendulum_entry(**kwargs: object) -> gym.Env:
    env = gym.make("Pendulum-v1", **kwargs)
    return _MaskObservation(env, keep_indices=(0, 1))  # cos(theta), sin(theta)


def _po_mountaincar_entry(**kwargs: object) -> gym.Env:
    env = gym.make("MountainCar-v0", **kwargs)
    return _MaskObservation(env, keep_indices=(0,))  # position only


def _po_acrobot_entry(**kwargs: object) -> gym.Env:
    env = gym.make("Acrobot-v1", **kwargs)
    # obs = [cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2]
    # keep both joint angles, drop both angular velocities.
    return _MaskObservation(env, keep_indices=(0, 1, 2, 3))


def _po_lunarlander_entry(**kwargs: object) -> gym.Env:
    env = gym.make("LunarLander-v3", **kwargs)
    # obs = [x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact]
    # keep position/angle/contacts, drop all three velocity terms — a much
    # harder mask than PO-CartPole since the agent must infer descent speed
    # and rotation rate purely from consecutive positions.
    return _MaskObservation(env, keep_indices=(0, 1, 4, 6, 7))


def _flickering_cartpole_entry(**kwargs: object) -> gym.Env:
    env = gym.make("CartPole-v1", **kwargs)
    return _FlickeringObservation(env, flicker_prob=0.5)


def _flickering_pong_entry(**kwargs: object) -> gym.Env:
    import ale_py

    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", **kwargs)
    return _FlickeringObservation(env, flicker_prob=0.5)


class MemoryCorridorEnv(gym.Env):
    """Delayed-cue-recall task: on the very first observation the agent sees
    a one-hot cue bit (0 or 1); for the next `CORRIDOR_LENGTH - 1` steps the
    observation is all zeros (the cue is gone, indistinguishable from every
    other neutral step); the episode ends after `CORRIDOR_LENGTH` steps, and
    the agent is rewarded +1 only if its *very last* action matches the
    original cue (any actions taken on the neutral steps are unscored).

    A feed-forward policy cannot do better than chance (0.5 expected reward)
    here by construction — nothing in later observations carries any
    information about the cue. A recurrent policy can solve it perfectly by
    latching the cue into its hidden state and holding it there. See the
    "POMDP-среды" section of the README for a concrete DQN-vs-LSTM-DQN A/B
    run on this exact env.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
    CORRIDOR_LENGTH = 8  # default used by MemoryCorridor-v0

    def __init__(self, render_mode: str | None = None, corridor_length: int | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.corridor_length = int(corridor_length or self.CORRIDOR_LENGTH)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._cue = 0
        self._t = 0
        self._last_action: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._cue = int(self.np_random.integers(0, 2))
        self._t = 0
        self._last_action = None
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        obs = np.zeros(2, dtype=np.float32)
        if self._t == 0:
            obs[self._cue] = 1.0
        return obs

    def step(self, action: int):
        self._last_action = int(action)
        self._t += 1
        terminated = self._t >= self.corridor_length
        reward = 1.0 if (terminated and self._last_action == self._cue) else 0.0
        return self._obs(), reward, terminated, False, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        cell = 28
        n = self.corridor_length
        width, height = cell * n, cell * 3
        img = np.full((height, width, 3), 21, dtype=np.uint8)
        img[cell:cell * 2, :, :] = (48, 48, 64)
        cue_color = (220, 70, 70) if self._cue == 0 else (70, 120, 220)
        img[cell:cell * 2, 0:cell, :] = cue_color
        pos = min(self._t, n - 1)
        x0 = pos * cell
        img[cell // 2:cell // 2 + cell, x0:x0 + cell, :] = (90, 220, 130)
        return img


def _memory_corridor_long_entry(**kwargs: object) -> gym.Env:
    # Same task, a much longer delay (20 vs 8 neutral steps) — the recurrent
    # hidden state has to survive a lot more irrelevant transitions before
    # being read out, which is exactly how bsuite's `memory_len` sweep
    # demonstrates memory *capacity* rather than just memory *presence*.
    return MemoryCorridorEnv(corridor_length=20, **kwargs)


class RepeatPreviousEnv(gym.Env):
    """N-back / delayed-match-to-sample memory task: every step shows one of
    `num_symbols` random one-hot symbols, and the correct action is *always*
    the symbol that was shown exactly `delay` steps ago (the very first
    `delay` steps have no defined correct action and are excluded from
    scoring). Unlike `MemoryCorridorEnv` (one cue, read once at the very
    end), this demands *continuously* maintaining a sliding window of the
    last `delay` observations and keeps testing it on every single step of
    the episode — the classic "copy"/N-back paradigm used across the
    memory-augmented RL and neuroscience literature.

    A feed-forward policy is capped at chance (`1/num_symbols` accuracy) by
    construction, since a lone observation carries zero information about
    what was shown `delay` steps earlier. A recurrent policy can solve it
    perfectly (in this noise-free version) by latching a `delay`-slot FIFO
    queue into its hidden state.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    _COLORS = (
        (220, 70, 70), (70, 150, 220), (90, 220, 130), (230, 190, 60),
        (190, 90, 220), (240, 140, 60),
    )

    def __init__(
        self,
        render_mode: str | None = None,
        num_symbols: int = 4,
        delay: int = 3,
        episode_len: int = 30,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.num_symbols = int(num_symbols)
        self.delay = int(delay)
        self.episode_len = int(episode_len)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_symbols,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.num_symbols)
        self._sequence: list[int] = []
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._sequence = [int(self.np_random.integers(0, self.num_symbols)) for _ in range(self.episode_len)]
        self._t = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.num_symbols, dtype=np.float32)
        obs[self._sequence[self._t]] = 1.0
        return obs

    def step(self, action: int):
        target = self._sequence[self._t - self.delay] if self._t >= self.delay else None
        reward = 1.0 if (target is not None and int(action) == target) else 0.0
        self._t += 1
        terminated = self._t >= self.episode_len
        obs = self._obs() if not terminated else np.zeros(self.num_symbols, dtype=np.float32)
        return obs, reward, terminated, False, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        cell = 24
        window = min(self.episode_len, 16)
        width, height = cell * window, cell * 3
        img = np.full((height, width, 3), 21, dtype=np.uint8)
        cur_step = min(self._t, self.episode_len - 1)
        start = max(0, cur_step - window + 1)
        # Middle row: a scrolling timeline of the last `window` symbols shown so
        # far, brightest/outlined cell marks "right now". Top/bottom rows mark
        # the current symbol and the one from `delay` steps ago (the actual
        # answer the agent must have output this step) so a human watching the
        # replay can see at a glance whether the two match.
        for i, step in enumerate(range(start, cur_step + 1)):
            color = self._COLORS[self._sequence[step] % len(self._COLORS)]
            x0 = i * cell
            img[cell:cell * 2 - 2, x0 + 2:x0 + cell - 2, :] = color
            if step == cur_step:
                img[cell - 3:cell, x0:x0 + cell, :] = (245, 245, 245)
        cur_color = self._COLORS[self._sequence[cur_step] % len(self._COLORS)]
        img[2:cell - 4, 2:cell - 4, :] = cur_color
        if cur_step >= self.delay:
            target_color = self._COLORS[self._sequence[cur_step - self.delay] % len(self._COLORS)]
            img[height - cell + 4:height - 2, 2:cell - 4, :] = target_color
        return img


class RockSampleEnv(gym.Env):
    """RockSample(n, k) — Smith & Simmons, 2004, "Heuristic Search Value
    Iteration for POMDPs": one of the standard scalable benchmarks in the
    POMDP *planning* literature (alongside Hallway/Hallway2), used ever
    since to compare belief-tracking algorithms. Default config
    (`n=7, k=8`) is their `RockSample(7,8)`.

    An agent on an `n`x`n` grid must collect "good" rocks (worth +10 to
    sample) while avoiding "bad" ones (-10) among `k` rocks at fixed (but
    randomized-per-episode) positions. Rock *positions* are always known —
    only each rock's *quality* is hidden, and can only be probed with a
    long-range sensor (one `check_i` action per rock) whose reading is
    correct with probability `0.5 + 0.5 * 2**(-distance / sensor_efficiency)`
    — i.e. near-certain up close, coin-flip from far away. Exiting east off
    the grid ends the episode.

    Optimal behaviour requires *integrating several noisy checks of the
    same rock over time* into a confident belief before deciding whether to
    detour to sample it — a feed-forward policy only ever sees the single
    most recent noisy bit and is stuck acting on that alone, while a
    recurrent policy can accumulate evidence. This is a different flavour
    of "memory" than every cue-recall task above: aggregating repeated
    noisy evidence rather than recalling one clean signal untouched by
    noise.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        n: int = 7,
        k: int = 8,
        sensor_efficiency: float = 3.0,
        max_steps: int = 60,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.n = int(n)
        self.k = int(k)
        self.sensor_efficiency = float(sensor_efficiency)
        self.max_steps = int(max_steps)
        # obs = [agent_x, agent_y] + [rock_dx, rock_dy]*k + [sampled]*k + [last_check_signal]*k
        obs_dim = 2 + 2 * self.k + self.k + self.k
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        # 0=North 1=South 2=East 3=West 4=Sample 5..5+k-1=Check_i
        self.action_space = gym.spaces.Discrete(5 + self.k)
        self._agent = (0, 0)
        self._rocks: list[tuple[int, int]] = []
        self._good = np.zeros(self.k, dtype=bool)
        self._sampled = np.zeros(self.k, dtype=bool)
        self._last_check = np.zeros(self.k, dtype=np.float32)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._agent = (0, self.n // 2)
        # Random, non-overlapping rock cells (rock positions are common
        # knowledge in RockSample — only quality is hidden).
        cells = [(x, y) for x in range(self.n) for y in range(self.n) if (x, y) != self._agent]
        idx = self.np_random.choice(len(cells), size=self.k, replace=False)
        self._rocks = [cells[i] for i in idx]
        self._good = self.np_random.integers(0, 2, size=self.k).astype(bool)
        self._sampled = np.zeros(self.k, dtype=bool)
        self._last_check = np.zeros(self.k, dtype=np.float32)
        self._t = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        ax, ay = self._agent
        denom = max(self.n - 1, 1)
        parts = [np.array([ax / denom, ay / denom], dtype=np.float32)]
        for rx, ry in self._rocks:
            parts.append(np.array([(rx - ax) / denom, (ry - ay) / denom], dtype=np.float32))
        parts.append(self._sampled.astype(np.float32))
        parts.append(self._last_check.copy())
        return np.concatenate(parts).astype(np.float32)

    def step(self, action: int):
        action = int(action)
        self._last_check[:] = 0.0
        reward = 0.0
        terminated = False
        ax, ay = self._agent
        if action in (0, 1, 2, 3):
            dx, dy = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}[action]
            nx, ny = ax + dx, ay + dy
            if action == 2 and nx >= self.n:
                terminated = True  # exit east off the grid
            elif 0 <= nx < self.n and 0 <= ny < self.n:
                self._agent = (nx, ny)
        elif action == 4:  # Sample
            hit = next((i for i, r in enumerate(self._rocks) if r == self._agent and not self._sampled[i]), None)
            if hit is None:
                reward = -1.0  # nothing (useful) to sample here
            else:
                reward = 10.0 if self._good[hit] else -10.0
                self._sampled[hit] = True
        else:  # Check_i
            i = action - 5
            if 0 <= i < self.k:
                rx, ry = self._rocks[i]
                dist = float(np.hypot(rx - ax, ry - ay))
                p_correct = 0.5 + 0.5 * (2.0 ** (-dist / max(self.sensor_efficiency, 1e-6)))
                true_bit = bool(self._good[i])
                observed = true_bit if self.np_random.random() < p_correct else (not true_bit)
                self._last_check[i] = 1.0 if observed else -1.0
        self._t += 1
        truncated = (not terminated) and self._t >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        cell = 24
        img = np.full((cell * self.n, cell * self.n, 3), 18, dtype=np.uint8)
        for x in range(self.n):
            img[:, x * cell, :] = (40, 40, 50)
        for y in range(self.n):
            img[y * cell, :, :] = (40, 40, 50)
        for i, (rx, ry) in enumerate(self._rocks):
            row = self.n - 1 - ry
            if self._sampled[i]:
                color = (70, 70, 78)
            else:
                color = (90, 220, 130) if self._good[i] else (220, 90, 90)
            pad = cell // 5
            img[row * cell + pad:(row + 1) * cell - pad, rx * cell + pad:(rx + 1) * cell - pad, :] = color
        ax, ay = self._agent
        row = self.n - 1 - ay
        cx, cy = ax * cell + cell // 2, row * cell + cell // 2
        yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (cell // 3) ** 2
        img[mask] = (240, 240, 245)
        return img


def _rocksample_entry(**kwargs: object) -> gym.Env:
    return RockSampleEnv(**kwargs)


class VisualMemoryMazeEnv(gym.Env):
    """Image-observation maze in the spirit of MiniGrid's `MemoryEnv` — the
    flagship benchmark used across the recurrent-RL literature to
    demonstrate that memory matters (see, e.g., `sb3-contrib`'s
    `RecurrentPPO` documentation, which uses exactly this task).

    Layout: a one-cell-wide corridor connects a starting room (containing a
    single colored cue object) to a junction, which forks into an upper and
    a lower end-room, each containing a colored object. The agent must
    reach whichever end-room's object color matches the cue seen many
    steps earlier at the very start — one color is assigned to the cue and
    (independently, per episode) one of the two end-rooms.

    Unlike `_MaskObservation`'s velocity-masking trick, the partial
    observability here is the "real" mechanism behind most POMDPs: the
    agent only ever sees a small egocentric crop of the map (default 5x5
    cells) around itself, so the cue object simply scrolls out of view once
    the agent leaves the starting room, and the end-room objects are
    invisible until the agent gets close to the junction. No observation is
    ever artificially hidden — everything the agent can't see is exactly
    what a limited camera/field-of-view genuinely wouldn't show.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    WIDTH = 11
    HEIGHT = 7
    VIEW_RADIUS = 2
    CELL_PX = 6
    FULL_CELL_PX = 16
    MAX_STEPS = 60
    _COLOR_A = (70, 150, 220)
    _COLOR_B = (230, 140, 40)
    _WALL = (26, 26, 34)
    _FLOOR = (46, 46, 58)
    _CUE_BG = (66, 66, 84)
    _AGENT = (240, 240, 245)

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        win = 2 * self.VIEW_RADIUS + 1
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(win * self.CELL_PX, win * self.CELL_PX, 3), dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(4)  # 0=up 1=down 2=left 3=right
        self._open: set[tuple[int, int]] = set()
        for x in range(self.WIDTH - 1):
            self._open.add((x, 3))
        for y in range(1, 6):
            self._open.add((self.WIDTH - 2, y))
        self._goal_top = (self.WIDTH - 1, 1)
        self._goal_bottom = (self.WIDTH - 1, 5)
        self._open.add(self._goal_top)
        self._open.add(self._goal_bottom)
        self._cue_cell = (0, 3)
        self._agent = self._cue_cell
        self._cue_color = self._COLOR_A
        self._top_color = self._COLOR_A
        self._bottom_color = self._COLOR_B
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._agent = self._cue_cell
        cue_is_a = bool(self.np_random.integers(0, 2))
        self._cue_color = self._COLOR_A if cue_is_a else self._COLOR_B
        other = self._COLOR_B if cue_is_a else self._COLOR_A
        if bool(self.np_random.integers(0, 2)):
            self._top_color, self._bottom_color = self._cue_color, other
        else:
            self._top_color, self._bottom_color = other, self._cue_color
        self._t = 0
        return self._egocentric_obs(), {}

    def _cell_visual(self, cell: tuple[int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int] | None]:
        if cell == self._cue_cell:
            return self._CUE_BG, self._cue_color
        if cell == self._goal_top:
            return self._FLOOR, self._top_color
        if cell == self._goal_bottom:
            return self._FLOOR, self._bottom_color
        if cell in self._open:
            return self._FLOOR, None
        return self._WALL, None

    def _draw_cell(self, img: np.ndarray, row0: int, col0: int, size: int, cell: tuple[int, int]) -> None:
        base, marker = self._cell_visual(cell)
        img[row0:row0 + size, col0:col0 + size, :] = base
        if marker is not None:
            pad = max(1, size // 5)
            img[row0 + pad:row0 + size - pad, col0 + pad:col0 + size - pad, :] = marker

    def _egocentric_obs(self) -> np.ndarray:
        r, size = self.VIEW_RADIUS, self.CELL_PX
        win = 2 * r + 1
        img = np.zeros((win * size, win * size, 3), dtype=np.uint8)
        ax, ay = self._agent
        for j, dy in enumerate(range(-r, r + 1)):
            for i, dx in enumerate(range(-r, r + 1)):
                cell = (ax + dx, ay + dy)
                if 0 <= cell[0] < self.WIDTH and 0 <= cell[1] < self.HEIGHT:
                    self._draw_cell(img, j * size, i * size, size, cell)
                else:
                    img[j * size:(j + 1) * size, i * size:(i + 1) * size, :] = self._WALL
        return img

    def step(self, action: int):
        dx, dy = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[int(action)]
        nx, ny = self._agent[0] + dx, self._agent[1] + dy
        if (nx, ny) in self._open:
            self._agent = (nx, ny)
        self._t += 1
        terminated = False
        reward = -0.01
        if self._agent == self._goal_top:
            reward = 1.0 if self._top_color == self._cue_color else -1.0
            terminated = True
        elif self._agent == self._goal_bottom:
            reward = 1.0 if self._bottom_color == self._cue_color else -1.0
            terminated = True
        truncated = (not terminated) and self._t >= self.MAX_STEPS
        return self._egocentric_obs(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        size = self.FULL_CELL_PX
        img = np.zeros((self.HEIGHT * size, self.WIDTH * size, 3), dtype=np.uint8)
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                self._draw_cell(img, y * size, x * size, size, (x, y))
        ax, ay = self._agent
        cx, cy = ax * size + size // 2, ay * size + size // 2
        yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (size // 3) ** 2
        img[mask] = self._AGENT
        return img


class TigerEnv(gym.Env):
    """The Tiger problem (Kaelbling, Littman & Cassandra, 1998) — *the*
    textbook toy POMDP, used to introduce the whole field. A tiger is behind
    one of two doors (fixed for the episode, unknown to the agent). Each
    step the agent can `listen` (a noisy, costly hint about the tiger's
    side: 85% accurate) or commit by opening a door (+10 for the empty one,
    -100 for the tiger). Optimal behaviour listens two or three times to
    build up enough confidence before committing — a single listen's 85%
    accuracy alone isn't good enough to beat blind guessing's expected
    value, but *several* independent noisy listens integrated together
    comfortably are. Tiny by design (2 hidden states, 3 actions) — meant as
    a fast, canonical first example of "why does POMDP memory matter" more
    than a serious training challenge.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}
    LISTEN, OPEN_LEFT, OPEN_RIGHT = 0, 1, 2
    _LISTEN_ACCURACY = 0.85

    def __init__(self, render_mode: str | None = None, max_steps: int = 30) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        # obs = [heard_left, heard_right] — both 0 before the first listen.
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._tiger_left = True
        self._last_obs = np.zeros(2, dtype=np.float32)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._tiger_left = bool(self.np_random.integers(0, 2))
        self._last_obs = np.zeros(2, dtype=np.float32)
        self._t = 0
        return self._last_obs.copy(), {}

    def step(self, action: int):
        action = int(action)
        self._t += 1
        if action == self.LISTEN:
            heard_left = self._tiger_left if self.np_random.random() < self._LISTEN_ACCURACY else not self._tiger_left
            self._last_obs = np.array([1.0, 0.0] if heard_left else [0.0, 1.0], dtype=np.float32)
            reward = -1.0
            terminated = False
        else:
            opened_left = action == self.OPEN_LEFT
            reward = -100.0 if opened_left == self._tiger_left else 10.0
            self._last_obs = np.zeros(2, dtype=np.float32)
            terminated = True
        truncated = (not terminated) and self._t >= self.max_steps
        return self._last_obs.copy(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        img = np.full((140, 220, 3), 18, dtype=np.uint8)
        for side, x0 in ((self._tiger_left, 20), (not self._tiger_left, 120)):
            color = (200, 90, 70) if side else (60, 60, 72)
            img[30:110, x0:x0 + 80, :] = color
        return img


class HeavenHellEnv(gym.Env):
    """Heaven/Hell — a T-maze that additionally requires *active*
    information gathering (a distinct POMDP flavour from every "cue you
    passively see" task above): the sign that reveals which end of the fork
    is Heaven (+1) vs Hell (-1) sits down a side corridor, not on the main
    path, so a reactive policy that never detours to read it is stuck
    guessing at the fork — while a policy with memory can visit the sign
    once, remember the answer, and then always walk straight to Heaven no
    matter how many neutral steps pass in between.

    Layout (a "plus" of corridors): a vertical spine from the start up to a
    fork, a horizontal side corridor partway up leading to the sign room,
    and the fork opening onto the Heaven/Hell cells left and right. The
    sign can be revisited at any time (there's no time pressure to read it
    immediately), so this is purely a "did you bother to go look, and did
    you remember what you saw" test, not a race against a timer.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
    NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
    SPINE_LEN = 4  # start (y=0) .. fork (y=SPINE_LEN)
    MAX_STEPS = 40

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        # obs = [wall_N, wall_S, wall_E, wall_W, at_sign, cue] (cue is 0
        # unless standing in the sign room, then +1/-1 for heaven left/right).
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self._open: set[tuple[int, int]] = {(0, y) for y in range(self.SPINE_LEN + 1)}
        self._sign_room = (-1, 2)
        self._open.add(self._sign_room)
        self._fork = (0, self.SPINE_LEN)
        self._left_goal = (-1, self.SPINE_LEN)
        self._right_goal = (1, self.SPINE_LEN)
        self._open.add(self._left_goal)
        self._open.add(self._right_goal)
        self._agent = (0, 0)
        self._heaven_left = True
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._agent = (0, 0)
        self._heaven_left = bool(self.np_random.integers(0, 2))
        self._t = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        x, y = self._agent
        walls = np.array([
            1.0 if (x, y + 1) not in self._open else 0.0,
            1.0 if (x, y - 1) not in self._open else 0.0,
            1.0 if (x + 1, y) not in self._open else 0.0,
            1.0 if (x - 1, y) not in self._open else 0.0,
        ], dtype=np.float32)
        at_sign = 1.0 if self._agent == self._sign_room else 0.0
        cue = (1.0 if self._heaven_left else -1.0) if at_sign else 0.0
        return np.concatenate([walls, [at_sign, cue]]).astype(np.float32)

    def step(self, action: int):
        dx, dy = {self.NORTH: (0, 1), self.SOUTH: (0, -1), self.EAST: (1, 0), self.WEST: (-1, 0)}[int(action)]
        nxt = (self._agent[0] + dx, self._agent[1] + dy)
        if nxt in self._open:
            self._agent = nxt
        self._t += 1
        terminated = False
        reward = -0.02
        if self._agent == self._left_goal:
            reward = 1.0 if self._heaven_left else -1.0
            terminated = True
        elif self._agent == self._right_goal:
            reward = -1.0 if self._heaven_left else 1.0
            terminated = True
        truncated = (not terminated) and self._t >= self.MAX_STEPS
        return self._obs(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        cell = 28
        ox, oy = 2, 1  # grid-to-pixel origin offset (in cells) so x=-1 fits
        img = np.full(((self.SPINE_LEN + 2) * cell, 5 * cell, 3), 16, dtype=np.uint8)
        for (x, y) in self._open:
            r0, c0 = (self.SPINE_LEN - y + oy - 1) * cell, (x + ox - 1) * cell
            color = (46, 46, 58)
            if (x, y) == self._sign_room:
                color = (66, 66, 84)
            elif (x, y) == self._left_goal:
                color = (90, 220, 130) if self._heaven_left else (220, 90, 90)
            elif (x, y) == self._right_goal:
                color = (220, 90, 90) if self._heaven_left else (90, 220, 130)
            img[r0:r0 + cell, c0:c0 + cell, :] = color
        ax, ay = self._agent
        r0, c0 = (self.SPINE_LEN - ay + oy - 1) * cell, (ax + ox - 1) * cell
        cx, cy = c0 + cell // 2, r0 + cell // 2
        yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (cell // 3) ** 2
        img[mask] = (240, 240, 245)
        return img


class HallwayEnv(gym.Env):
    """A compact rendition of Littman/Cassandra/Kaelbling's `Hallway`
    benchmark (1995 QMDP paper) — the defining property of that family is
    *state aliasing*: several different rooms produce the exact same local
    sensor reading, so the reading alone can never tell you which one
    you're in; only remembering the path you took since a known landmark
    can. Simplified here to a small hand-built maze graph (named
    straight/left/right exits per room rather than full heading + N/E/S/W
    wall bits) to keep the essential lesson while staying easy to reason
    about: `FORK0` and both `TWIN1`/`TWIN2` rooms emit the *identical*
    observation (`open: straight=no, left=yes, right=yes`), yet reaching
    the goal from `TWIN1` requires the opposite turn to the one that leads
    to `TWIN1` from `FORK0` in the first place (and symmetrically for
    `TWIN2`) — so *any* fixed reactive rule for that observation is
    guaranteed to fail at one of the two twins, while a policy that
    remembers which way it turned at `FORK0` can always pick correctly.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}
    STRAIGHT, LEFT, RIGHT = 0, 1, 2
    MAX_STEPS = 20
    # room -> {action: next_room}; a missing action is a wall (no-op).
    _GRAPH = {
        "S": {STRAIGHT: "FORK0"},
        "FORK0": {LEFT: "A1", RIGHT: "B1"},
        "A1": {STRAIGHT: "A2"},
        "A2": {STRAIGHT: "TWIN1"},
        "TWIN1": {LEFT: "DEAD1", RIGHT: "GOAL"},
        "B1": {STRAIGHT: "B2"},
        "B2": {STRAIGHT: "TWIN2"},
        "TWIN2": {LEFT: "GOAL", RIGHT: "DEAD2"},
    }
    _TERMINALS = {"GOAL": 1.0, "DEAD1": -1.0, "DEAD2": -1.0}
    _ORDER = ["S", "FORK0", "A1", "A2", "TWIN1", "B1", "B2", "TWIN2", "GOAL", "DEAD1", "DEAD2"]

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        # obs = one-hot [straight_open, left_open, right_open].
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._room = "S"
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._room = "S"
        self._t = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        exits = self._GRAPH.get(self._room, {})
        return np.array([
            1.0 if self.STRAIGHT in exits else 0.0,
            1.0 if self.LEFT in exits else 0.0,
            1.0 if self.RIGHT in exits else 0.0,
        ], dtype=np.float32)

    def step(self, action: int):
        action = int(action)
        self._t += 1
        exits = self._GRAPH.get(self._room, {})
        if action in exits:
            self._room = exits[action]
            reward = -0.02
        else:
            reward = -0.05  # invalid exit for this room — stay put
        terminated = self._room in self._TERMINALS
        if terminated:
            reward = self._TERMINALS[self._room]
        truncated = (not terminated) and self._t >= self.MAX_STEPS
        return self._obs(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        cell = 26
        img = np.full((cell * 3, cell * len(self._ORDER), 3), 16, dtype=np.uint8)
        colors = {"GOAL": (90, 220, 130), "DEAD1": (220, 90, 90), "DEAD2": (220, 90, 90)}
        for i, name in enumerate(self._ORDER):
            color = colors.get(name, (60, 60, 78) if name in ("FORK0", "TWIN1", "TWIN2") else (44, 44, 56))
            img[cell:cell * 2, i * cell:(i + 1) * cell, :] = color
        idx = self._ORDER.index(self._room)
        img[0:cell, idx * cell:(idx + 1) * cell, :] = (240, 240, 245)
        return img


class BattleshipEnv(gym.Env):
    """Battleship as a POMDP (a well-known scale-up benchmark in the
    planning literature, e.g. Silver & Veness, 2010's POMCP paper) —
    reformulated here specifically to *need* memory rather than let a
    feed-forward net solve it by pattern-matching a fully-rendered board:
    the observation on every step is only the *single most recent* shot's
    result (row, column, hit/miss/sunk), never the whole grid of past
    shots. Re-firing an already-tried cell is explicitly penalized, which a
    feed-forward policy — blind to everything except the last shot — has no
    way to reliably avoid, while a recurrent one can track the full history
    of cells already tried."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}
    SIZE = 6
    SHIP_LENGTHS = (3, 2)
    MAX_STEPS = 40

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        # obs = [row_norm, col_norm, result, remaining_frac]; result in
        # {-1: repeat shot, 0: miss, 0.5: hit, 1: hit+sunk}, before the
        # first shot it's all zeros.
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.SIZE * self.SIZE)
        self._ship_cells: set[tuple[int, int]] = set()
        self._ship_id: dict[tuple[int, int], int] = {}
        self._sunk: list[bool] = []
        self._fired: set[tuple[int, int]] = set()
        self._last_obs = np.zeros(4, dtype=np.float32)
        self._t = 0

    def _place_ships(self) -> None:
        self._ship_cells = set()
        self._ship_id = {}
        occupied: set[tuple[int, int]] = set()
        for ship_idx, length in enumerate(self.SHIP_LENGTHS):
            while True:
                horizontal = bool(self.np_random.integers(0, 2))
                if horizontal:
                    r = int(self.np_random.integers(0, self.SIZE))
                    c = int(self.np_random.integers(0, self.SIZE - length + 1))
                    cells = [(r, c + i) for i in range(length)]
                else:
                    r = int(self.np_random.integers(0, self.SIZE - length + 1))
                    c = int(self.np_random.integers(0, self.SIZE))
                    cells = [(r + i, c) for i in range(length)]
                if not occupied.intersection(cells):
                    occupied.update(cells)
                    for cell in cells:
                        self._ship_id[cell] = ship_idx
                    break
        self._ship_cells = occupied

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._place_ships()
        self._sunk = [False] * len(self.SHIP_LENGTHS)
        self._fired = set()
        self._last_obs = np.zeros(4, dtype=np.float32)
        self._t = 0
        return self._last_obs.copy(), {}

    def step(self, action: int):
        action = int(action)
        r, c = divmod(action, self.SIZE)
        self._t += 1
        remaining_frac = 1.0 - (sum(self._sunk) / max(1, len(self._sunk)))
        if (r, c) in self._fired:
            reward = -1.0
            result = -1.0
        else:
            self._fired.add((r, c))
            if (r, c) in self._ship_cells:
                ship_idx = self._ship_id[(r, c)]
                ship_cells = [cell for cell, sid in self._ship_id.items() if sid == ship_idx]
                sunk_now = all(cell in self._fired for cell in ship_cells)
                if sunk_now:
                    self._sunk[ship_idx] = True
                    reward, result = 2.0, 1.0
                else:
                    reward, result = 1.0, 0.5
            else:
                reward, result = -0.1, 0.0
        won = all(self._sunk)
        if won:
            reward += 5.0
        remaining_frac = 1.0 - (sum(self._sunk) / max(1, len(self._sunk)))
        self._last_obs = np.array([r / (self.SIZE - 1), c / (self.SIZE - 1), result, remaining_frac], dtype=np.float32)
        terminated = won
        truncated = (not terminated) and self._t >= self.MAX_STEPS
        return self._last_obs.copy(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        cell = 24
        img = np.full((cell * self.SIZE, cell * self.SIZE, 3), 24, dtype=np.uint8)
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if (r, c) not in self._fired:
                    continue
                color = (90, 220, 130) if (r, c) in self._ship_cells else (70, 70, 84)
                pad = cell // 6
                img[r * cell + pad:(r + 1) * cell - pad, c * cell + pad:(c + 1) * cell - pad, :] = color
        return img


class MinesweeperPOMDPEnv(gym.Env):
    """Minesweeper as a memory-requiring POMDP (tagged specifically as a
    strong-memory task in the POPGym benchmark suite). Like
    `BattleshipEnv` above, the observation deliberately exposes only the
    single most-recently-revealed cell's result — not a rendering of the
    whole board — so avoiding a wasted re-reveal, and eventually inferring
    which cells are safe from *many* past numbers, both require actually
    remembering the board rather than reading it off the current
    observation."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}
    SIZE = 5
    NUM_MINES = 4
    MAX_STEPS = 30

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        # obs = [row_norm, col_norm, result, is_repeat]; result is
        # adjacent-mine-count/8 for a fresh safe reveal, -1 for a fresh
        # mine hit, 0 for a repeat (whose `is_repeat` flag is set instead).
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.SIZE * self.SIZE)
        self._mines: set[tuple[int, int]] = set()
        self._revealed: set[tuple[int, int]] = set()
        self._last_obs = np.zeros(4, dtype=np.float32)
        self._t = 0

    def _adjacent_mines(self, r: int, c: int) -> int:
        return sum(
            (r + dr, c + dc) in self._mines
            for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cells = [(r, c) for r in range(self.SIZE) for c in range(self.SIZE)]
        idx = self.np_random.choice(len(cells), size=self.NUM_MINES, replace=False)
        self._mines = {cells[i] for i in idx}
        self._revealed = set()
        self._last_obs = np.zeros(4, dtype=np.float32)
        self._t = 0
        return self._last_obs.copy(), {}

    def step(self, action: int):
        action = int(action)
        r, c = divmod(action, self.SIZE)
        self._t += 1
        terminated = False
        if (r, c) in self._revealed:
            reward, result, is_repeat = -0.5, 0.0, 1.0
        elif (r, c) in self._mines:
            reward, result, is_repeat = -1.0, -1.0, 0.0
            terminated = True
        else:
            self._revealed.add((r, c))
            reward, result, is_repeat = 0.2, self._adjacent_mines(r, c) / 8.0, 0.0
            if len(self._revealed) == self.SIZE * self.SIZE - self.NUM_MINES:
                reward += 5.0
                terminated = True
        self._last_obs = np.array([r / (self.SIZE - 1), c / (self.SIZE - 1), result, is_repeat], dtype=np.float32)
        truncated = (not terminated) and self._t >= self.MAX_STEPS
        return self._last_obs.copy(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        cell = 28
        img = np.full((cell * self.SIZE, cell * self.SIZE, 3), 24, dtype=np.uint8)
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if (r, c) in self._revealed:
                    img[r * cell + 2:(r + 1) * cell - 2, c * cell + 2:(c + 1) * cell - 2, :] = (46, 46, 60)
        return img


class ConcentrationEnv(gym.Env):
    """The card-matching memory game ("Concentration"/"Memory") — flip two
    face-down cards per turn; a match keeps both face-up for good, a
    mismatch flips both back down. The identity of any card is only ever
    part of the observation on the exact step it (or its turn-partner) is
    flipped — the rest of the time it's just "face-down, unknown" again —
    so scoring well above chance means actually remembering what turned up
    under cards seen many turns ago, the textbook definition of the game
    and one of the POPGym suite's diagnostic memory tasks."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}
    NUM_PAIRS = 4
    MAX_STEPS = 40

    def __init__(self, render_mode: str | None = None, num_pairs: int | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.num_pairs = int(num_pairs or self.NUM_PAIRS)
        self.num_cards = self.num_pairs * 2
        # obs, per card: [matched, currently_visible, value_norm_if_visible].
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_cards * 3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.num_cards)
        self._values = np.zeros(self.num_cards, dtype=np.int64)
        self._matched = np.zeros(self.num_cards, dtype=bool)
        self._pending_first: int | None = None
        self._just_revealed: list[int] = []
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        pairs = np.repeat(np.arange(self.num_pairs), 2)
        self._values = self.np_random.permutation(pairs)
        self._matched = np.zeros(self.num_cards, dtype=bool)
        self._pending_first = None
        self._just_revealed = []
        self._t = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        out = np.zeros((self.num_cards, 3), dtype=np.float32)
        out[:, 0] = self._matched.astype(np.float32)
        for i in self._just_revealed:
            out[i, 1] = 1.0
            out[i, 2] = self._values[i] / max(1, self.num_pairs - 1)
        return out.reshape(-1)

    def step(self, action: int):
        action = int(action)
        self._t += 1
        self._just_revealed = []
        if action < 0 or action >= self.num_cards or self._matched[action] or action == self._pending_first:
            reward = -0.5  # invalid pick: already matched, or re-picking the pending card
        elif self._pending_first is None:
            self._pending_first = action
            self._just_revealed = [action]
            reward = 0.0
        else:
            first = self._pending_first
            self._just_revealed = [first, action]
            if self._values[first] == self._values[action]:
                self._matched[first] = True
                self._matched[action] = True
                reward = 1.0
            else:
                reward = -0.1
            self._pending_first = None
        terminated = bool(self._matched.all())
        truncated = (not terminated) and self._t >= self.MAX_STEPS
        return self._obs(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        cell = 30
        cols = self.num_cards
        img = np.full((cell, cell * cols, 3), 24, dtype=np.uint8)
        for i in range(cols):
            if self._matched[i]:
                color = (90, 220, 130)
            elif i in self._just_revealed:
                color = (230, 190, 60)
            else:
                color = (54, 54, 68)
            img[2:cell - 2, i * cell + 2:(i + 1) * cell - 2, :] = color
        return img


class LaserTagEnv(gym.Env):
    """A single-agent rendition of `LaserTag`/`Tag` (Pineau et al., 2003's
    PBVI paper) — pursue a fleeing opponent whose position is only visible
    within a short line-of-sight (blocked by a wall block in the middle of
    the arena, and always out of range beyond a few cells). Unlike every
    static-hidden-fact task above, the thing being tracked here *moves on
    its own* — a feed-forward policy only reacts to the opponent's current
    (frequently invisible) position, while a recurrent one can keep a
    running belief ("last seen here, heading that way") through the gaps.
    The opponent runs a small scripted evasion policy (flee when seen, else
    random-walk) rather than being co-trained, keeping this a genuine
    single-agent POMDP.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
    SIZE = 8
    VISION_RANGE = 4
    MAX_STEPS = 60
    _WALLS = {(3, 3), (3, 4), (4, 3), (4, 4)}
    _MOVES = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        # obs = [ax_norm, ay_norm, dx_norm, dy_norm, visible] — dx/dy (the
        # opponent's position *relative* to the agent) are 0 when not seen.
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self._agent = (0, 0)
        self._opponent = (self.SIZE - 1, self.SIZE - 1)
        self._t = 0

    def _open_cells(self) -> list[tuple[int, int]]:
        return [(x, y) for x in range(self.SIZE) for y in range(self.SIZE) if (x, y) not in self._WALLS]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cells = self._open_cells()
        idx = self.np_random.choice(len(cells), size=2, replace=False)
        self._agent, self._opponent = cells[idx[0]], cells[idx[1]]
        self._t = 0
        return self._obs(), {}

    def _visible(self) -> bool:
        ax, ay = self._agent
        ox, oy = self._opponent
        dist = abs(ax - ox) + abs(ay - oy)
        if dist <= 1:
            return True
        if dist > self.VISION_RANGE:
            return False
        if ax == ox:
            lo, hi = sorted((ay, oy))
            return not any((ax, y) in self._WALLS for y in range(lo + 1, hi))
        if ay == oy:
            lo, hi = sorted((ax, ox))
            return not any((x, ay) in self._WALLS for x in range(lo + 1, hi))
        return False

    def _obs(self) -> np.ndarray:
        ax, ay = self._agent
        denom = max(self.SIZE - 1, 1)
        if self._visible():
            ox, oy = self._opponent
            return np.array([ax / denom, ay / denom, (ox - ax) / denom, (oy - ay) / denom, 1.0], dtype=np.float32)
        return np.array([ax / denom, ay / denom, 0.0, 0.0, 0.0], dtype=np.float32)

    def _move_opponent(self) -> None:
        ox, oy = self._opponent
        candidates = [
            (ox + dx, oy + dy) for dx, dy in self._MOVES.values()
            if 0 <= ox + dx < self.SIZE and 0 <= oy + dy < self.SIZE and (ox + dx, oy + dy) not in self._WALLS
        ] or [self._opponent]
        if self._visible():
            ax, ay = self._agent
            candidates.sort(key=lambda c: -(abs(c[0] - ax) + abs(c[1] - ay)))
            self._opponent = candidates[0]
        else:
            self._opponent = candidates[int(self.np_random.integers(0, len(candidates)))]

    def step(self, action: int):
        dx, dy = self._MOVES[int(action)]
        ax, ay = self._agent
        nx, ny = ax + dx, ay + dy
        if 0 <= nx < self.SIZE and 0 <= ny < self.SIZE and (nx, ny) not in self._WALLS:
            self._agent = (nx, ny)
        self._t += 1
        if self._agent == self._opponent:
            return self._obs(), 1.0, True, False, {}
        self._move_opponent()
        terminated = self._agent == self._opponent
        reward = 1.0 if terminated else -0.05
        truncated = (not terminated) and self._t >= self.MAX_STEPS
        return self._obs(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        cell = 20
        img = np.full((cell * self.SIZE, cell * self.SIZE, 3), 18, dtype=np.uint8)
        for (x, y) in self._WALLS:
            row = self.SIZE - 1 - y
            img[row * cell:(row + 1) * cell, x * cell:(x + 1) * cell, :] = (60, 60, 72)
        for pos, color in ((self._opponent, (220, 90, 90)), (self._agent, (90, 180, 220))):
            x, y = pos
            row = self.SIZE - 1 - y
            pad = cell // 5
            img[row * cell + pad:(row + 1) * cell - pad, x * cell + pad:(x + 1) * cell - pad, :] = color
        return img


class ActiveTMazeEnv(gym.Env):
    """The *active* variant of the T-maze memory task (Ni et al., 2023,
    "When Do Transformers Shine in RL? Decoupling Memory from Credit
    Assignment"): unlike `MemoryCorridorEnv` above, where the cue is simply
    shown once whether the agent likes it or not, here the agent must
    *choose* to `look` while still at the start of a one-way corridor —
    walking past without looking loses the cue for good (there's no
    turning back). This tests a different, meta-behavioural piece of
    memory-driven competence: not just holding onto information, but
    learning that a particular moment is worth actively querying for it in
    the first place. `ActiveTMazeLong-v0` repeats the test with a longer
    corridor to probe how that capability degrades with the memory gap.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
    LOOK, FORWARD, CHOOSE_LEFT, CHOOSE_RIGHT = 0, 1, 2, 3
    CORRIDOR_LENGTH = 10

    def __init__(self, render_mode: str | None = None, corridor_length: int | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.corridor_length = int(corridor_length or self.CORRIDOR_LENGTH)
        # obs = [pos_norm, cue] — cue is nonzero only on the very step a
        # successful `look` happens (always at position 0).
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self._pos = 0
        self._cue_left = True
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._pos = 0
        self._cue_left = bool(self.np_random.integers(0, 2))
        self._t = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def step(self, action: int):
        action = int(action)
        self._t += 1
        cue_signal = 0.0
        reward = -0.02
        terminated = False
        if action == self.LOOK:
            if self._pos == 0:
                cue_signal = 1.0 if self._cue_left else -1.0
                reward = -0.01
            else:
                reward = -0.1  # look only does anything at the very start
        elif action == self.FORWARD:
            if self._pos < self.corridor_length:
                self._pos += 1
            else:
                reward = -0.1
        else:  # CHOOSE_LEFT / CHOOSE_RIGHT
            if self._pos == self.corridor_length:
                chose_left = action == self.CHOOSE_LEFT
                reward = 1.0 if chose_left == self._cue_left else -1.0
                terminated = True
            else:
                reward = -0.1
        obs = np.array([self._pos / self.corridor_length, cue_signal], dtype=np.float32)
        truncated = (not terminated) and self._t >= self.corridor_length + 15
        return obs, reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        cell = 22
        img = np.full((cell * 3, cell * (self.corridor_length + 1), 3), 18, dtype=np.uint8)
        img[cell:cell * 2, :, :] = (46, 46, 58)
        img[0:cell, self.corridor_length * cell:(self.corridor_length + 1) * cell, :] = (
            (90, 220, 130) if self._cue_left else (220, 90, 90)
        )
        img[cell * 2:cell * 3, self.corridor_length * cell:(self.corridor_length + 1) * cell, :] = (
            (220, 90, 90) if self._cue_left else (90, 220, 130)
        )
        pad = cell // 5
        img[cell + pad:cell * 2 - pad, self._pos * cell + pad:(self._pos + 1) * cell - pad, :] = (240, 240, 245)
        return img


def _active_tmaze_long_entry(**kwargs: object) -> gym.Env:
    return ActiveTMazeEnv(corridor_length=25, **kwargs)


def register_pomdp_envs() -> None:
    """Idempotent — safe to import/call from multiple modules."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    gym.register(id="POCartPole-v0", entry_point=_po_cartpole_entry)
    gym.register(id="POPendulum-v0", entry_point=_po_pendulum_entry)
    gym.register(id="POMountainCar-v0", entry_point=_po_mountaincar_entry)
    gym.register(id="POAcrobot-v0", entry_point=_po_acrobot_entry)
    gym.register(id="POLunarLander-v0", entry_point=_po_lunarlander_entry)
    gym.register(id="FlickeringCartPole-v0", entry_point=_flickering_cartpole_entry)
    gym.register(id="FlickeringPong-v0", entry_point=_flickering_pong_entry)
    # No max_episode_steps override on either of these two — both envs
    # already terminate themselves exactly at their own step count.
    gym.register(id="MemoryCorridor-v0", entry_point=MemoryCorridorEnv)
    # Deliberately *not* named "MemoryCorridor-v1" — gymnasium's registry
    # treats a higher trailing version number on the same base name as a
    # newer release of the *same* task and prints a "consider upgrading"
    # deprecation warning, which would be misleading here since this is a
    # harder difficulty variant, not a replacement for v0.
    gym.register(id="MemoryCorridorLong-v0", entry_point=_memory_corridor_long_entry)
    gym.register(id="RepeatPrevious-v0", entry_point=RepeatPreviousEnv)
    gym.register(id="RockSample-v0", entry_point=_rocksample_entry)
    gym.register(id="VisualMemoryMaze-v0", entry_point=VisualMemoryMazeEnv)
    gym.register(id="Tiger-v0", entry_point=TigerEnv)
    gym.register(id="HeavenHell-v0", entry_point=HeavenHellEnv)
    gym.register(id="Hallway-v0", entry_point=HallwayEnv)
    gym.register(id="Battleship-v0", entry_point=BattleshipEnv)
    gym.register(id="MinesweeperPOMDP-v0", entry_point=MinesweeperPOMDPEnv)
    gym.register(id="Concentration-v0", entry_point=ConcentrationEnv)
    gym.register(id="LaserTag-v0", entry_point=LaserTagEnv)
    gym.register(id="ActiveTMaze-v0", entry_point=ActiveTMazeEnv)
    gym.register(id="ActiveTMazeLong-v0", entry_point=_active_tmaze_long_entry)


register_pomdp_envs()
