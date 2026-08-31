"""From-scratch industrial operations-research environments — the kind of
combinatorial scheduling/allocation problem real factories, warehouses and
cloud schedulers actually run, as opposed to a locomotion/Atari benchmark.
No third-party dependency (unlike `robotics_envs.py`/`trading_envs.py`'s
neighbours): both tasks below are just discrete-event bookkeeping over a
handful of arrays, so writing them directly against `gym.Env` (same
approach as `rl_core/envs/pomdp.py`'s from-scratch tasks) is both simpler
and far more robust than depending on one of the handful of unmaintained
job-shop-scheduling gym packages on PyPI.

- `JobShopEnv` (`JobShop-6x6-v0` / `JobShop-10x10-v0`) — the classical Job
  Shop Scheduling Problem (JSSP; Taillard, 1993's benchmark instances are
  the same 6x6/10x10/etc. sizes used here), extended with the three
  "real-world constraint" ingredients that recent shopfloor-focused RL
  frameworks (JobShopLab/Lang et al., 2025, "A Production Scheduling
  Framework for Reinforcement Learning Under Real-World Constraints";
  job_shop_lib) add on top of the bare textbook version because pure
  makespan-on-deterministic-times is a poor proxy for an actual factory:
  - **Due dates + weighted tardiness**: each job gets a due date (Total
    Work Content method — due date = total processing time × a random
    tightness factor, the standard due-date-assignment recipe in the JSSP
    literature) and a random priority weight (some jobs are "rush
    orders"); being late on a *heavy* job hurts more than being late on a
    light one. Minimizing pure makespan can quietly let an urgent job run
    dead last — weighted tardiness is what real production scheduling is
    actually judged on.
  - **Sequence-dependent setup times**: jobs belong to a "family" (e.g.
    a product line/tooling requirement), and a machine pays a fixed setup
    penalty whenever the family it's about to run differs from the one it
    just finished — the reason real dispatching rules batch same-family
    jobs together instead of always taking "shortest processing time
    next".
  - **Stochastic processing times**: actual duration is the nominal one
    times log-normal noise (mean 1, small sigma) drawn *at dispatch time*
    — a schedule can't be planned once and blindly executed; a good
    dispatcher has to stay robust to processing times not landing exactly
    where the instance said they would (JobShopLab's "stochastic
    processing conditions").

  Formulated as a priority-dispatching RL task the way Zhang, Song, Cao et
  al., 2020 ("Learning to Dispatch for Job Shop Scheduling via Deep
  Reinforcement Learning", L2D/NeurIPS 2020) do: at each step the agent
  names *one* not-yet-finished job, its next operation gets appended to
  that job's machine's queue (waiting for "this job's previous operation
  finished", "this machine is free", and any needed setup), and the agent
  is scored by how much that choice grew the running makespan plus any
  weighted tardiness realized — a *dense*, step-by-step version of the
  sparse "-makespan - weighted tardiness" episode return every dispatching
  policy is ultimately judged on (the makespan term telescopes exactly to
  that total; see `step()`).
- `BinPackingEnv` (`BinPacking-v0`) — online 1D bin packing (items arrive
  one at a time, must be placed in *some* open bin — of fixed capacity —
  immediately and irrevocably, or a new bin opened): the classic model for
  warehouse pallet/container loading and cloud-VM bin-packing (e.g. "Ali
  Baba cluster scheduling"-style resource-allocation research), minimizing
  the number of bins used. Simple to state, genuinely NP-hard, and (unlike
  JobShop's fixed `num_jobs`/`num_machines` action space) exercises
  variable, capacity-driven action masking — the agent picks *which open
  bin* (or "open a new one") to drop the current item into.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

_REGISTERED = False


class JobShopEnv(gym.Env):
    """See module docstring. A fresh random instance (processing times +
    per-job machine routing) is drawn every episode (via `self.np_random`,
    so it's fully reproducible from a seed) rather than training against
    one fixed instance — the point of a *learned dispatching policy* is
    that it generalizes across instances, not that it memorizes one
    schedule; a single fixed instance would let the agent just memorize
    the optimal dispatch order instead of actually learning "what makes a
    good dispatching decision" from the observation."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self, num_jobs: int = 6, num_machines: int = 6,
        min_duration: int = 1, max_duration: int = 20,
        num_families: int | None = None, setup_time: float = 5.0,
        due_date_tightness: tuple[float, float] = (1.3, 2.2),
        weight_choices: tuple[int, ...] = (1, 2, 4),
        proc_time_noise_std: float = 0.10, tardiness_coef: float = 1.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.num_jobs = int(num_jobs)
        self.num_machines = int(num_machines)
        self.min_duration = int(min_duration)
        self.max_duration = int(max_duration)
        # A "family" (product line / shared tooling) per job — machines pay
        # `setup_time` when switching between families (see module
        # docstring). Defaults to a handful of families regardless of
        # `num_jobs` so even the 10x10 instance still has real batching
        # incentive (one family per job would make every switch a setup,
        # collapsing this into "setup after every single operation").
        self.num_families = int(num_families) if num_families else max(2, min(4, self.num_jobs))
        self.setup_time = float(setup_time)
        self.due_date_tightness = due_date_tightness
        self.weight_choices = tuple(int(w) for w in weight_choices)
        self.proc_time_noise_std = float(proc_time_noise_std)
        self.tardiness_coef = float(tardiness_coef)
        self.render_mode = render_mode
        # A generous but finite time bound most random instances' optimal
        # (let alone learned) makespan sits well under — used only to
        # normalize observation features into a roughly-[0, 1] range, not
        # to cap the actual schedule. Padded for setup time on top of raw
        # processing time so it stays a genuine upper bound.
        self._time_norm = float(self.num_jobs * self.num_machines * (self.max_duration + self.setup_time))
        self._tardiness_norm = self._time_norm * max(self.weight_choices)

        self.action_space = gym.spaces.Discrete(self.num_jobs)
        # Per job: [progress fraction, next-op duration, job-ready time,
        # is-done flag, due-date slack, priority weight] (6 features) +
        # per machine: [machine-free time] — everything normalized by
        # `_time_norm`/weight range so the observation stays roughly
        # bounded even though schedule length isn't capped up front (due
        # slack in particular can go negative once a job is already late).
        obs_dim = self.num_jobs * 6 + self.num_machines
        self.observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)

        self._proc_times = np.zeros((self.num_jobs, self.num_machines), dtype=np.float32)
        self._machine_order = np.zeros((self.num_jobs, self.num_machines), dtype=np.int64)
        self._families = np.zeros(self.num_jobs, dtype=np.int64)
        self._weights = np.ones(self.num_jobs, dtype=np.float32)
        self._due_dates = np.zeros(self.num_jobs, dtype=np.float32)
        self._op_idx = np.zeros(self.num_jobs, dtype=np.int64)
        self._job_free_time = np.zeros(self.num_jobs, dtype=np.float32)
        self._machine_free_time = np.zeros(self.num_machines, dtype=np.float32)
        self._machine_last_family = np.full(self.num_machines, -1, dtype=np.int64)
        self._makespan = 0.0
        self._total_weighted_tardiness = 0.0
        self._tardy_jobs = 0
        self._t = 0
        self._max_steps = self.num_jobs * self.num_machines * 3  # generous slack for wasted (invalid-action) steps
        # (job, machine, op_idx, family, setup_start, proc_start, end) — for render().
        self._schedule: list[tuple[int, int, int, int, float, float, float]] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._proc_times = self.np_random.integers(
            self.min_duration, self.max_duration + 1, size=(self.num_jobs, self.num_machines),
        ).astype(np.float32)
        self._machine_order = np.stack([
            self.np_random.permutation(self.num_machines) for _ in range(self.num_jobs)
        ])
        self._families = self.np_random.integers(0, self.num_families, size=self.num_jobs)
        self._weights = self.np_random.choice(self.weight_choices, size=self.num_jobs).astype(np.float32)
        # Total Work Content (TWK) due-date assignment — the standard
        # recipe in the JSSP literature: due date = total processing time
        # (across all of a job's operations) times a random tightness
        # factor, so tighter-drawn jobs are (by construction) harder to
        # deliver on time than looser ones, without needing any notion of
        # "shop load" to define tightness against.
        total_proc = self._proc_times.sum(axis=1)
        tightness = self.np_random.uniform(*self.due_date_tightness, size=self.num_jobs)
        self._due_dates = (total_proc * tightness).astype(np.float32)
        self._op_idx = np.zeros(self.num_jobs, dtype=np.int64)
        self._job_free_time = np.zeros(self.num_jobs, dtype=np.float32)
        self._machine_free_time = np.zeros(self.num_machines, dtype=np.float32)
        self._machine_last_family = np.full(self.num_machines, -1, dtype=np.int64)
        self._makespan = 0.0
        self._total_weighted_tardiness = 0.0
        self._tardy_jobs = 0
        self._t = 0
        self._schedule = []
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        done = self._op_idx >= self.num_machines
        next_op_idx = np.minimum(self._op_idx, self.num_machines - 1)
        next_dur = np.where(done, 0.0, self._proc_times[np.arange(self.num_jobs), next_op_idx])
        due_slack = (self._due_dates - self._job_free_time) / self._time_norm
        job_feats = np.stack([
            self._op_idx.astype(np.float32) / self.num_machines,
            next_dur / self.max_duration,
            self._job_free_time / self._time_norm,
            done.astype(np.float32),
            due_slack,
            self._weights / max(self.weight_choices),
        ], axis=1).reshape(-1)
        machine_feats = self._machine_free_time / self._time_norm
        return np.concatenate([job_feats, machine_feats]).astype(np.float32)

    def step(self, action: Any):
        job = int(action)
        self._t += 1
        prev_makespan = self._makespan

        if job < 0 or job >= self.num_jobs or self._op_idx[job] >= self.num_machines:
            # Invalid dispatch (bad index, or that job already has every
            # operation scheduled) — small penalty, no state change, so a
            # policy that keeps picking finished jobs just wastes steps
            # (bounded by `_max_steps`) rather than crashing or stalling
            # forever.
            reward = -1.0
        else:
            op = int(self._op_idx[job])
            machine = int(self._machine_order[job, op])
            family = int(self._families[job])
            # Sequence-dependent setup: only paid when the machine's last
            # family differs from this job's (see module docstring) — a
            # dispatcher that keeps same-family jobs together on a machine
            # avoids it entirely, one that ping-pongs between families
            # pays it every single operation.
            needs_setup = self._machine_last_family[machine] not in (-1, family)
            setup = self.setup_time if needs_setup else 0.0
            setup_start = max(float(self._job_free_time[job]), float(self._machine_free_time[machine]))
            proc_start = setup_start + setup
            nominal_dur = float(self._proc_times[job, op])
            actual_dur = nominal_dur
            if self.proc_time_noise_std > 0:
                # Stochastic processing time, realized only at dispatch —
                # a schedule built assuming nominal durations can't be
                # blindly trusted (see module docstring).
                actual_dur *= float(self.np_random.lognormal(mean=0.0, sigma=self.proc_time_noise_std))
            end = proc_start + actual_dur

            self._machine_last_family[machine] = family
            self._job_free_time[job] = end
            self._machine_free_time[machine] = end
            self._op_idx[job] += 1
            self._makespan = max(self._makespan, end)
            self._schedule.append((job, machine, op, family, setup_start, proc_start, end))

            # Dense reward, two telescoping components:
            # 1) -(growth in makespan this step) — sums exactly to
            #    `-final_makespan` over the episode.
            # 2) -(growth in total weighted tardiness this step), only
            #    nonzero on the single step that finishes a job's *last*
            #    operation — sums exactly to `-total_weighted_tardiness`.
            # Both far easier to credit-assign than either sparse total
            # would be alone at episode end.
            makespan_term = -(self._makespan - prev_makespan) / self.max_duration
            tardiness_term = 0.0
            if self._op_idx[job] >= self.num_machines:
                tardiness = max(0.0, end - float(self._due_dates[job]))
                if tardiness > 0:
                    self._tardy_jobs += 1
                weighted_tardiness = tardiness * float(self._weights[job])
                self._total_weighted_tardiness += weighted_tardiness
                tardiness_term = -weighted_tardiness / self._tardiness_norm
            reward = makespan_term + self.tardiness_coef * tardiness_term

        all_done = bool(np.all(self._op_idx >= self.num_machines))
        terminated = all_done
        truncated = (not terminated) and self._t >= self._max_steps
        info = (
            {
                "makespan": self._makespan,
                "total_weighted_tardiness": self._total_weighted_tardiness,
                "tardy_jobs": self._tardy_jobs,
            }
            if terminated
            else {}
        )
        return self._obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return _render_gantt_chart(
            num_machines=self.num_machines, num_jobs=self.num_jobs,
            schedule=self._schedule, makespan=self._makespan,
            due_dates=self._due_dates, job_free_time=self._job_free_time,
            op_idx=self._op_idx, num_ops=self.num_machines,
            total_weighted_tardiness=self._total_weighted_tardiness,
        )


def _job_color(job: int, num_jobs: int) -> tuple[int, int, int]:
    """Evenly spaced, maximally distinguishable hues around the color
    wheel — evenly-spaced RGB tuples (the previous approach) start
    repeating visually similar colors well before `num_jobs` gets past
    single digits."""
    import colorsys

    hue = (job / max(num_jobs, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 0.92)
    return int(r * 255), int(g * 255), int(b * 255)


def _render_gantt_chart(
    *, num_machines: int, num_jobs: int,
    schedule: list[tuple[int, int, int, int, float, float, float]],
    makespan: float, due_dates: np.ndarray, job_free_time: np.ndarray,
    op_idx: np.ndarray, num_ops: int, total_weighted_tardiness: float,
) -> np.ndarray:
    """A real, labeled Gantt chart (machine rows × time) via Pillow —
    already an rl_core dependency (see `rl_core/algorithms/metrics_callback.py`'s
    GIF encoding) — rather than plain filled `numpy` rectangles: per-job
    color legend, a job number baked into each bar, a lighter hatch-free
    block for setup time distinct from the colored processing block, a
    time-axis grid, and a compact "on-time / late / pending" status strip
    per job (the thing weighted tardiness is actually scoring) along the
    bottom."""
    from PIL import Image, ImageDraw, ImageFont

    left_margin, top_margin = 34, 22
    row_h = 24
    width = 460
    strip_h = 20
    height = top_margin + num_machines * row_h + strip_h + 10
    img = Image.new("RGB", (width, height), (20, 20, 25))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=11)

    span = max(makespan, 1.0)
    plot_w = width - left_margin - 10

    def x_of(t: float) -> int:
        return left_margin + int(plot_w * (t / span))

    draw.text((6, 4), f"Job Shop  t={makespan:.0f}  tardiness={total_weighted_tardiness:.0f}", font=font, fill=(210, 210, 220))

    # Time-axis grid (5 ticks) + machine-row labels/gridlines.
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x_of(frac * span)
        draw.line([(x, top_margin), (x, top_margin + num_machines * row_h)], fill=(40, 40, 48))
        draw.text((min(x, width - 24), top_margin + num_machines * row_h + 1), f"{frac * span:.0f}", font=font, fill=(120, 120, 130))
    for m in range(num_machines):
        y = top_margin + m * row_h
        draw.line([(left_margin, y), (width - 6, y)], fill=(34, 34, 40))
        draw.text((4, y + row_h // 2 - 5), f"M{m}", font=font, fill=(150, 150, 160))

    for job, machine, _op, _family, setup_start, proc_start, end in schedule:
        color = _job_color(job, num_jobs)
        y0 = top_margin + machine * row_h + 3
        y1 = y0 + row_h - 7
        if proc_start > setup_start:
            # Setup block: a dim, unfilled outline rather than the job's
            # own color — visually reads as "machine busy but not actually
            # making this part yet", the standard Gantt-chart convention.
            draw.rectangle([x_of(setup_start), y0, max(x_of(proc_start), x_of(setup_start) + 1), y1], outline=(90, 90, 60), width=1)
        x0, x1 = x_of(proc_start), max(x_of(end), x_of(proc_start) + 1)
        draw.rectangle([x0, y0, x1, y1], fill=color)
        if x1 - x0 >= 10:
            draw.text((x0 + 2, y0 + 1), str(job), font=font, fill=(15, 15, 15))

    # Per-job status strip: green = finished on time, red = finished late,
    # amber = still in progress but already past its due date, gray =
    # pending/not-yet-due — a compact, always-visible readout of the exact
    # objective (weighted tardiness) the Gantt bars above don't make
    # obvious at a glance.
    strip_y = top_margin + num_machines * row_h + 12
    cell_w = max(4, min(18, plot_w // max(num_jobs, 1)))
    for job in range(num_jobs):
        finished = op_idx[job] >= num_ops
        late = float(job_free_time[job]) > float(due_dates[job]) if finished else False
        overdue_in_progress = (not finished) and makespan > float(due_dates[job])
        if finished and not late:
            color = (90, 190, 110)
        elif finished and late:
            color = (210, 90, 90)
        elif overdue_in_progress:
            color = (215, 165, 70)
        else:
            color = (80, 80, 90)
        x0 = left_margin + job * cell_w
        draw.rectangle([x0, strip_y, x0 + cell_w - 2, strip_y + 8], fill=color)

    return np.asarray(img, dtype=np.uint8)


class BinPackingEnv(gym.Env):
    """Online 1D bin packing — see module docstring. Every step draws one
    new item (size in `(0, bin_capacity]`) that must be placed *now*: the
    action picks which currently-open bin to drop it into, or "open a new
    bin". An action naming a bin that's already too full for the current
    item is treated the same as picking "open a new bin" instead — this
    keeps the action space a fixed `Discrete(max_bins + 1)` (no dynamic
    masking machinery needed) while still guaranteeing every placement is
    always physically valid; a policy that never bothers to check
    remaining capacity before choosing just ends up opening more bins than
    it needed to, which the reward already penalizes."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self, bin_capacity: int = 100, num_items: int = 50,
        min_item_size: int = 10, max_item_size: int = 70,
        max_bins: int = 20, render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.bin_capacity = int(bin_capacity)
        self.num_items = int(num_items)
        self.min_item_size = int(min_item_size)
        self.max_item_size = int(max_item_size)
        self.max_bins = int(max_bins)
        self.render_mode = render_mode

        # "Open a new bin" is action `max_bins` (one past the last real
        # bin slot) — naming an already-open bin that can't fit the
        # current item, or a bin slot not opened yet, both fall back to
        # opening a new bin (see class docstring).
        self.action_space = gym.spaces.Discrete(self.max_bins + 1)
        # [current item size] + per bin-slot: [remaining capacity] (0 for
        # not-yet-opened slots) — fixed-size flat Box, `max_bins` caps how
        # many *simultaneously open* bins the observation can describe,
        # not how many total items/bins an episode can use.
        obs_dim = 1 + self.max_bins
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._remaining = np.zeros(self.max_bins, dtype=np.float32)
        self._num_open = 0
        self._current_item = 0.0
        self._items_placed = 0
        self._items: np.ndarray = np.zeros(0, dtype=np.float32)
        self._placements: list[int] = []  # bin index chosen for each placed item so far (for render())

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._items = self.np_random.integers(
            self.min_item_size, self.max_item_size + 1, size=self.num_items,
        ).astype(np.float32)
        self._remaining = np.zeros(self.max_bins, dtype=np.float32)
        self._num_open = 0
        self._items_placed = 0
        self._placements = []
        self._current_item = float(self._items[0])
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        return np.concatenate([
            [self._current_item / self.bin_capacity],
            self._remaining / self.bin_capacity,
        ]).astype(np.float32)

    def step(self, action: Any):
        bin_idx = int(action)
        item = self._current_item
        fits_existing = 0 <= bin_idx < self._num_open and self._remaining[bin_idx] >= item
        if fits_existing:
            self._remaining[bin_idx] -= item
            reward = 0.0  # packed into an already-open bin — free (no new-bin cost)
        elif self._num_open < self.max_bins:
            bin_idx = self._num_open
            self._remaining[bin_idx] = self.bin_capacity - item
            self._num_open += 1
            reward = -1.0  # opening a new bin is the thing we're trying to minimize
        else:
            # Ran out of bin slots entirely (a genuinely bad, avoidable
            # outcome — a decent policy keeps enough slack open) — reject
            # the item; count it as a heavily-penalized "wasted" bin of
            # its own rather than silently dropping it.
            bin_idx = -1
            reward = -3.0

        self._placements.append(bin_idx)
        self._items_placed += 1
        terminated = self._items_placed >= self.num_items
        if not terminated:
            self._current_item = float(self._items[self._items_placed])
        truncated = False
        info = {"num_bins_used": self._num_open} if terminated else {}
        return self._obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        cols = max(1, self._num_open)
        width, height = min(320, 20 + cols * 28), 90
        img = np.full((height, width, 3), 22, dtype=np.uint8)
        for b in range(min(cols, self.max_bins)):
            fill_frac = 1.0 - float(self._remaining[b]) / self.bin_capacity
            x0 = 10 + b * 28
            bar_h = int(60 * max(0.0, min(1.0, fill_frac)))
            img[80 - bar_h:80, x0:x0 + 20, :] = (90, 170, 110)
            img[20:80, x0:x0 + 20, :] = np.maximum(img[20:80, x0:x0 + 20, :], (40, 40, 46))
        return img


def register_industrial_envs() -> None:
    """Idempotent — safe to import/call from multiple modules. No optional
    third-party dependency to guard against (see module docstring), unlike
    every sibling `register_*_envs()` in this package."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    gym.register(id="JobShop-6x6-v0", entry_point=JobShopEnv, kwargs={"num_jobs": 6, "num_machines": 6})
    gym.register(id="JobShop-10x10-v0", entry_point=JobShopEnv, kwargs={"num_jobs": 10, "num_machines": 10})
    gym.register(id="BinPacking-v0", entry_point=BinPackingEnv)


# Self-registers at import time, exactly like `rl_core/envs/pomdp.py`.
register_industrial_envs()
