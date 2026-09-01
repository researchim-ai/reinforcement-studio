"""QMIX — Monotonic Value Function Factorisation (Rashid et al., 2018),
the `qmix` algorithm in the Experiment Designer, selectable for any env
exposing 2+ agents via `team_ids` (single *or* multi-team) with *discrete*
movement — see `compatible_algorithms` in `rl_core/envs/registry.py`
(gated on `agent_count`, not `team_count`, precisely because a single
cooperative team is QMIX's own classic setting — see below).

Where `ippo` (`rl_core/algorithms/native/marl_ppo.py`) gives every team its
own independent, on-policy PPO learner, `qmix` gives every team its own
independent, *cooperative, off-policy, value-based* learner instead — the
two are direct alternatives, not building blocks of one another. Unlike
`ippo` — which, run on a single team, degenerates into an unnecessarily
split-up copy of plain `ppo` — a *single* team of 2+ agents is exactly
VDN/QMIX's own textbook setting (their original papers' running example is
one cooperative team, not two competing ones): the mixer still buys
something there (credit assignment across teammates via the shared
`Q_tot`), so `ippo` stays gated on 2+ *teams* while `qmix` only needs 2+
*agents*, in one team or several (`MultiAgentQMIX` below just runs one
independent `_TeamQMIX` per distinct value in `team_ids`, so a
single-team env simply ends up with exactly one `_TeamQMIX`).

Within one team:

- Every teammate shares one Q-network `Qi(oi, a)` (parameter sharing, same
  idea as `_TeamPolicy` sharing one `ActorCriticNet` across a team's lanes
  in `marl_ppo.py`) — homogeneous teammates are the common case for this
  app's built-in scenes, so no per-agent id is fed in.
- A `QMixer` hypernetwork combines the whole team's chosen-action Q-values
  into one team-level `Q_tot`, conditioned on a "global state" — since
  `SceneMultiAgentEnv` doesn't expose one explicitly, the concatenation of
  every teammate's own local observation is used instead (fully
  determined by what the team already collectively observes, which is
  all a within-team mixer needs).
- The mixer's hypernetwork weights are constrained non-negative (`abs`),
  which keeps `∂Q_tot/∂Qi ≥ 0` for every teammate `i` — the *monotonicity*
  that makes "each teammate argmax's their own Qi" exactly equivalent to
  the joint `argmax_a Q_tot` (Individual-Global-Max), so execution stays
  fully decentralized even though `Q_tot` is trained centrally.

A team's reward signal is the *mean* of its teammates' raw per-lane
rewards that step (see `learn()`) — when the scene's `team_shared_reward`
rule is on, every teammate's raw reward is already the identical
team-pooled total (`_apply_team_shared_reward` in `rl_core/envs/
scene_env.py`), so the mean just recovers that same value unchanged; when
it's off, the mean gives QMIX a genuinely cooperative training signal
without inflating its scale by team size. A team's transition is treated
as *terminal* (for TD bootstrapping) whenever *any* of its lanes
terminates/truncates that step — a simplifying approximation for scenes
where teammates can end their own episode independently (e.g. a tagged
prey respawning) while the rest of the team keeps going; built-in scenes
without that mechanic (Team Battle) always have every teammate's episode
end in lockstep anyway, so the approximation is exact there.

Like `ippo`, deliberately not built on any shared single-team base class —
`learn()` below is a plain step loop (much closer to `NativeDQN`'s than
`OnPolicyAlgorithm`'s, since QMIX is off-policy/value-based) generalized
over `self.teams` instead of one team.

Stability note (why `_train_step` looks slightly more defensive than
`NativeDQN._train_step`): the mixer's hypernetwork gives Q_tot an *extra,
learned multiplicative degree of freedom* on top of the per-agent Q-values
(the hypernetwork's own weights, not just activations, scale every
teammate's Qi) — empirically (see the QMIX section of the training-center
docs) this makes vanilla single-Q, MSE-loss, frequent-hard-target-update
DQN-style training diverge (Q_tot runs away to the thousands within a few
thousand gradient steps, even at tiny learning rates, on e.g. `simple_
spread`) far more easily than plain per-agent DQN/VDN does on the exact
same rollout data. Three standard, well-established fixes close that gap
without touching the mixer's architecture: Double-DQN's action selection
(pick `argmax` via the *online* net, evaluate it via the *target* net —
decouples selection from evaluation, cutting the overestimation bias that
feeds the runaway loop), Huber (`smooth_l1`) instead of MSE loss (bounds
the gradient magnitude once an estimate is already off, instead of
squaring it), and a tighter default `max_grad_norm` + a much less
frequent `target_update_interval` (a fast-moving target is exactly what
lets Q_tot and its own bootstrap target grow together unchecked). None of
these change QMIX's algorithm on paper — they're exactly the standard
"deadly triad" mitigations for semi-gradient TD with function
approximation (Double DQN: Van Hasselt et al., 2016) — just tuned as this
app's *defaults* because the mixer needs them more than a plain Q-network
does.

Action-masking note (SMAClite via `rl_core/envs/tuple_marl_envs.py`):
whenever `env` (or its `.unwrapped`) exposes `get_avail_actions() ->
(n_agents, n_actions) bool`, every ε-greedy/greedy action choice below is
restricted to that lane's currently-legal actions, and `_train_step`'s
Double-DQN action *selection* (not evaluation — see above) is masked the
same way using each transition's stored *next-state* mask — both halves
matter: unmasked selection could still argmax onto an illegal action that
happened to look good, and unmasked ε-exploration would spend most of its
budget on actions the env would've silently overridden anyway (see
`TupleMarlVectorEnv._sanitize_actions`), teaching the Q-network from a
signal that doesn't reflect what it actually asked for. This is exactly
why `smac_*`'s `compatible_algorithms` in `registry.py` is `["qmix"]`
alone — no other native algorithm reads this mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.buffers import JointReplayBuffer
from rl_core.algorithms.native.exploration.schedules import linear_schedule
from rl_core.algorithms.native.networks import QNetwork
from rl_core.algorithms.native.preprocessing import action_to_env, obs_batch_to_array, obs_to_array
from rl_core.algorithms.vec_env import action_space, num_envs_of, obs_space, vec_reset, vec_step

DEFAULT_HYPERPARAMS = {
    # Lower than `dqn`'s 1e-3/`5e-4` — the mixer's hypernetwork makes
    # Q_tot noticeably more prone to runaway growth than a plain
    # per-agent Q-network at the same LR (see the module docstring's
    # "Stability note"); 1e-4 was the smallest change that reliably keeps
    # `_train_step`'s loss bounded across a full training run in testing.
    "learning_rate": 1e-4,
    "buffer_size": 20_000,
    "batch_size": 32,
    "gamma": 0.99,
    "exploration_fraction": 0.3,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "learning_starts": 1_000,
    "train_freq": 4,
    # Deliberately much less frequent than `dqn`'s 1,000 (in real env
    # steps, not gradient steps) — a target net that tracks the online
    # net too closely gives the mixer's runaway-growth direction almost no
    # friction to fight against (see "Stability note" above); PyMARL's
    # own QMIX defaults to syncing every 200 *episodes* (thousands of
    # steps), which this is closer to in spirit than `dqn`'s per-step
    # value would be.
    "target_update_interval": 2_000,
    # Hidden size of the mixing network's hypernetwork output — see
    # `QMixer`. Bigger lets the mixer represent a richer (still monotonic)
    # combination of the team's individual Q-values at the cost of more
    # hypernetwork parameters; 32 (the QMIX paper's default) is plenty for
    # this app's small teams (rarely more than 4-5 agents).
    "mixing_embed_dim": 32,
    # Tighter than `dqn`'s 10.0 — same reasoning as `learning_rate` above.
    "max_grad_norm": 5.0,
}


class QMixer(nn.Module):
    """Rashid et al. (2018)'s mixing network: combines `n_agents`
    per-teammate Q-values into one scalar `Q_tot`, via a 2-layer MLP whose
    *weights* (not activations) are themselves produced by a hypernetwork
    fed the global state — `abs()` on both weight matrices is what keeps
    the whole thing monotonically increasing in every input Qi (see module
    docstring)."""

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Linear(state_dim, embed_dim * n_agents)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """`agent_qs`: `(B, n_agents)` chosen-action Q-values, `states`:
        `(B, state_dim)`. Returns `(B,)` `Q_tot`."""
        bs = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(states)).view(bs, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(bs, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.view(bs, 1, self.n_agents), w1) + b1)  # (B, 1, embed_dim)
        w2 = torch.abs(self.hyper_w2(states)).view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2  # (B, 1, 1)
        return q_tot.view(bs)


class _TeamQMIX:
    """Everything one team owns: shared per-teammate Q-network, mixer,
    their target copies, one joint optimizer over both, and (once
    `learn()` starts collecting) a `JointReplayBuffer` — the QMIX
    equivalent of `_TeamPolicy` in `marl_ppo.py`."""

    def __init__(self, obs_sp: gym.Space, n_actions: int, n_agents: int, hyperparams: dict[str, Any], device: str) -> None:
        self.n_agents = n_agents
        self.state_dim = n_agents * int(np.prod(obs_sp.shape))
        embed_dim = int(hyperparams.get("mixing_embed_dim", 32))
        self.agent_net = QNetwork(obs_sp, n_actions).to(device)
        self.target_agent_net = QNetwork(obs_sp, n_actions).to(device)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.mixer = QMixer(n_agents, self.state_dim, embed_dim).to(device)
        self.target_mixer = QMixer(n_agents, self.state_dim, embed_dim).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimizer = torch.optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            lr=float(hyperparams.get("learning_rate", 5e-4)),
        )
        self.buffer: JointReplayBuffer | None = None
        self.last_metrics: dict[str, float] = {}

    def sync_target(self) -> None:
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())


class MultiAgentQMIX(CustomAlgorithm):
    """Independent-per-team QMIX across every distinct team of a MARL
    Scene Builder scene / PettingZoo benchmark — one *or* more teams (see
    module docstring for why, unlike `ippo`, a single team is a fully
    valid, genuinely useful case here). Requires `env` (or its
    `.unwrapped`) to expose a `team_ids` list with 2+ agents total and a
    `Discrete` single action space — QMIX's value-decomposition only makes
    sense for discrete joint action spaces."""

    def __init__(self, env: Any, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        base_env = getattr(env, "unwrapped", env)
        team_ids = getattr(base_env, "team_ids", None)
        if not team_ids or len(team_ids) < 2:
            raise ValueError(
                "MultiAgentQMIX ('qmix') требует среду с 2+ агентами, объединёнными "
                "в team_ids (одна общая команда или несколько) — обычная одноагентная "
                "gym-среда сюда не подходит, используйте 'dqn'."
            )
        obs_sp, act_sp = obs_space(env), action_space(env)
        if not isinstance(act_sp, gym.spaces.Discrete):
            raise ValueError(
                "MultiAgentQMIX ('qmix') поддерживает только дискретные действия "
                "(movement: discrete4/discrete8) — для continuous-сцен используйте 'ippo'."
            )
        self.team_ids: list[str] = list(team_ids)
        self.teams: list[str] = sorted(set(self.team_ids))
        self.lanes: dict[str, np.ndarray] = {
            team: np.asarray([i for i, t in enumerate(self.team_ids) if t == team], dtype=np.int64)
            for team in self.teams
        }
        self.n_actions = int(act_sp.n)
        self._obs_space = obs_sp
        self._action_space = act_sp
        # SMAClite (`marlgym:smac_*`) is the only built-in env with a
        # situational action space — see module docstring's "Action-
        # masking note". `None` for every other env (RWARE/LBForaging,
        # every Scene/PettingZoo benchmark), which just means "every
        # action is always legal", exactly today's behavior.
        self._mask_fn = getattr(base_env, "get_avail_actions", None)
        self.teams_state: dict[str, _TeamQMIX] = {
            team: _TeamQMIX(obs_sp, self.n_actions, len(lanes), hyperparams, device)
            for team, lanes in self.lanes.items()
        }

        self.buffer_size = int(hyperparams.get("buffer_size", 20_000))
        self.batch_size = int(hyperparams.get("batch_size", 32))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.exploration_fraction = float(hyperparams.get("exploration_fraction", 0.3))
        self.exploration_initial_eps = float(hyperparams.get("exploration_initial_eps", 1.0))
        self.exploration_final_eps = float(hyperparams.get("exploration_final_eps", 0.05))
        self.learning_starts = int(hyperparams.get("learning_starts", 500))
        self.train_freq = max(1, int(hyperparams.get("train_freq", 4)))
        self.target_update_interval = max(1, int(hyperparams.get("target_update_interval", 200)))
        self.max_grad_norm = float(hyperparams.get("max_grad_norm", 10.0))
        # `predict()` (live GIF preview via `SceneRenderEnv`, always lane
        # 0 — see its docstring) always renders whichever team owns lane 0,
        # same convention as `MultiAgentPPO._render_team`.
        self._render_team = self.team_ids[0]

    def _epsilon(self, num_timesteps: int, total_timesteps: int) -> float:
        return linear_schedule(
            num_timesteps, total_timesteps, self.exploration_fraction,
            self.exploration_initial_eps, self.exploration_final_eps,
        )

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        n_envs = num_envs_of(self.env)
        act_sp = action_space(self.env)

        obs_arr = obs_batch_to_array(vec_reset(self.env, seed=self.seed), self._obs_space)
        # `self._mask_fn` may exist-but-return-`None` (e.g. RWARE/
        # LBForaging's `TupleMarlVectorEnv.get_avail_actions()` — see its
        # docstring) — probe the actual return value, not just presence of
        # the attribute, so the buffer only allocates a mask array for
        # genuinely mask-aware envs (SMAClite).
        mask_aware = self._mask_fn is not None and self._mask_fn() is not None
        for team, lanes in self.lanes.items():
            state = self.teams_state[team]
            if state.buffer is None:
                state.buffer = JointReplayBuffer(
                    self.buffer_size, len(lanes), obs_arr.shape[1:],
                    n_actions=self.n_actions if mask_aware else None,
                )

        ep_reward = np.zeros(n_envs, dtype=np.float64)
        ep_length = np.zeros(n_envs, dtype=np.int64)
        num_timesteps = 0

        while num_timesteps < total_timesteps:
            eps = self._epsilon(num_timesteps, total_timesteps)
            # Reflects whatever obs `obs_arr` already holds (see
            # `TupleMarlVectorEnv.get_avail_actions`'s docstring) — fetched
            # once per step, before any lane's action is chosen.
            mask_arr = self._mask_fn() if mask_aware else None
            actions = np.zeros(n_envs, dtype=np.int64)
            for team, lanes in self.lanes.items():
                state = self.teams_state[team]
                team_obs = obs_arr[lanes]
                obs_t = torch.as_tensor(team_obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    q = state.agent_net(obs_t)
                    if mask_arr is not None:
                        team_mask_t = torch.as_tensor(mask_arr[lanes], dtype=torch.bool, device=self.device)
                        q = q.masked_fill(~team_mask_t, float("-inf"))
                    greedy = torch.argmax(q, dim=-1).cpu().numpy()
                random_mask = np.random.rand(len(lanes)) < eps
                if mask_arr is not None:
                    # Masked ε-exploration (PyMARL's convention): sample
                    # uniformly among *legal* actions only, never a
                    # uniform `act_sp.sample()` over the full action space
                    # — see module docstring's "Action-masking note".
                    random_actions = np.array(
                        [np.random.choice(np.flatnonzero(mask_arr[lane])) for lane in lanes], dtype=np.int64,
                    )
                else:
                    random_actions = np.array([act_sp.sample() for _ in lanes], dtype=np.int64)
                actions[lanes] = np.where(random_mask, random_actions, greedy)

            next_obs_list, rewards, terminated, truncated, _infos = vec_step(self.env, [int(a) for a in actions])
            dones = terminated | truncated
            next_obs_arr = obs_batch_to_array(next_obs_list, self._obs_space)
            # Reflects the just-produced `next_obs_arr` (see
            # `TupleMarlVectorEnv.step`: the pending-reset flag delays the
            # actual `env.reset()` call to the *following* step, so the
            # wrapped env's internal state still matches this step's
            # terminal obs right now) — stored per-transition so
            # `_train_step` can mask the *next* state's Q-values, not
            # (wrongly) this step's.
            next_mask_arr = self._mask_fn() if mask_aware else None

            for team, lanes in self.lanes.items():
                state = self.teams_state[team]
                team_reward = float(np.mean(rewards[lanes]))
                team_done = bool(np.any(dones[lanes]))
                state.buffer.add(
                    obs_arr[lanes], actions[lanes], team_reward, next_obs_arr[lanes], team_done,
                    next_action_mask=next_mask_arr[lanes] if next_mask_arr is not None else None,
                )

            obs_arr = next_obs_arr
            ep_reward += rewards
            ep_length += 1
            prev_num_timesteps = num_timesteps
            num_timesteps += n_envs

            if num_timesteps // self.train_freq != prev_num_timesteps // self.train_freq:
                for team, state in self.teams_state.items():
                    if len(state.buffer) >= max(self.learning_starts, self.batch_size):
                        self._train_step(team)
            if num_timesteps // self.target_update_interval != prev_num_timesteps // self.target_update_interval:
                for state in self.teams_state.values():
                    state.sync_target()

            any_finished = False
            for i in range(n_envs):
                if dones[i]:
                    any_finished = True
                    keep_going = callback.on_step(
                        num_timesteps, float(ep_reward[i]), int(ep_length[i]), self._combined_metrics(eps),
                    )
                    ep_reward[i], ep_length[i] = 0.0, 0
                    if not keep_going:
                        return
            if not any_finished:
                keep_going = callback.on_step(num_timesteps, metrics=self._combined_metrics(eps))
                if not keep_going:
                    return

    def _train_step(self, team: str) -> None:
        state = self.teams_state[team]
        batch = state.buffer.sample(self.batch_size)
        n_agents = state.n_agents
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)  # (B, n_agents, *obs_shape)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)  # (B, n_agents)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)  # (B,)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)  # (B,)
        batch_size = obs_t.shape[0]

        flat_obs = obs_t.reshape(batch_size * n_agents, *obs_t.shape[2:])
        q_all = state.agent_net(flat_obs).view(batch_size, n_agents, self.n_actions)
        chosen_q = q_all.gather(-1, actions_t.unsqueeze(-1)).squeeze(-1)  # (B, n_agents)
        state_vec = obs_t.reshape(batch_size, -1)
        q_tot = state.mixer(chosen_q, state_vec)

        with torch.no_grad():
            flat_next_obs = next_obs_t.reshape(batch_size * n_agents, *next_obs_t.shape[2:])
            # Double-DQN action selection (Van Hasselt et al., 2016): pick each
            # teammate's next greedy action via the *online* net, but evaluate
            # that action's Q-value via the *target* net — decouples "which
            # action looks best" from "how good is it", which is what keeps a
            # positive Q-estimation bias from compounding through the mixer
            # every training step (see the module docstring's "Stability
            # note"; plain `next_q_all.max(...)`, the original QMIX paper's
            # own simpler choice, empirically diverges here far more easily
            # than it does for a plain per-agent `NativeDQN`, precisely
            # because the mixer gives that bias an extra multiplicative
            # degree of freedom to run away through).
            next_q_online = state.agent_net(flat_next_obs).view(batch_size, n_agents, self.n_actions)
            if "next_action_mask" in batch:
                # Mask *before* argmax — an unmasked online net could still
                # pick an illegal action to evaluate via the target net
                # even though `learn()`'s rollout never lets it actually
                # execute one (see module docstring's "Action-masking
                # note"); masking only the target's `.gather` afterwards
                # wouldn't fix that, since the wrong *index* would already
                # be locked in.
                next_mask_t = torch.as_tensor(batch["next_action_mask"], dtype=torch.bool, device=self.device)
                next_q_online = next_q_online.masked_fill(~next_mask_t, float("-inf"))
            next_actions = next_q_online.argmax(dim=-1)  # (B, n_agents)
            next_q_all = state.target_agent_net(flat_next_obs).view(batch_size, n_agents, self.n_actions)
            next_greedy_q = next_q_all.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)  # (B, n_agents)
            next_state_vec = next_obs_t.reshape(batch_size, -1)
            next_q_tot = state.target_mixer(next_greedy_q, next_state_vec)
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q_tot

        # Huber (`smooth_l1`), not MSE — same reasoning as `NativeDQN`'s own
        # `_train_step`, but more load-bearing here: squaring an already-off
        # Q_tot estimate (MSE) hands the mixer's runaway direction an
        # ever-growing gradient exactly when it should be getting damped.
        loss = F.smooth_l1_loss(q_tot, target)
        state.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(state.agent_net.parameters()) + list(state.mixer.parameters()), self.max_grad_norm,
        )
        state.optimizer.step()
        state.last_metrics = {"loss": float(loss.detach().item()), "q_tot_mean": float(q_tot.detach().mean().item())}

    def _combined_metrics(self, epsilon: float) -> dict[str, float]:
        """Every team's loss metrics, flattened into one dict keyed
        `<team>__<metric>` — same convention as `MultiAgentPPO.
        _combined_metrics`, so the Training Monitor renders each team's
        loss as its own line for free."""
        out: dict[str, float] = {"exploration_epsilon": float(epsilon)}
        for team, state in self.teams_state.items():
            for key, value in state.last_metrics.items():
                out[f"{team}__{key}"] = value
        return out

    def predict(self, obs: Any, deterministic: bool = True, episode_start: bool = False) -> tuple[Any, Any]:
        del episode_start  # no memory support (see module docstring)
        state = self.teams_state[self._render_team]
        obs_arr = obs_to_array(obs, self._obs_space)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = state.agent_net(obs_t)
        # Lane-0 mask only — `predict()` always renders `self._render_team`
        # (see `__init__`), and `env` here is the single-lane render
        # wrapper (`TupleMarlRenderEnv.get_avail_actions`), not the full
        # vector env. Best-effort/optional (see that method's docstring) —
        # `TupleMarlVectorEnv.step` clamps regardless if this is skipped.
        lane_mask = None
        if self._mask_fn is not None:
            full_mask = self._mask_fn()
            if full_mask is not None and len(full_mask) > 0:
                lane_mask = np.asarray(full_mask[0], dtype=bool)
        if lane_mask is not None:
            q = q.masked_fill(~torch.as_tensor(lane_mask, dtype=torch.bool, device=self.device), float("-inf"))
            if not deterministic and np.random.rand() < self.exploration_final_eps:
                return int(np.random.choice(np.flatnonzero(lane_mask))), None
        elif not deterministic and np.random.rand() < self.exploration_final_eps:
            return int(self._action_space.sample()), None
        action = int(torch.argmax(q, dim=-1).item())
        return action_to_env(np.asarray(action), action_space(self.env)), None

    def save(self, path: Path) -> None:
        torch.save({
            "teams": self.teams,
            "team_ids": self.team_ids,
            "agent_state_dicts": {team: s.agent_net.state_dict() for team, s in self.teams_state.items()},
            "mixer_state_dicts": {team: s.mixer.state_dict() for team, s in self.teams_state.items()},
            "optimizer_state_dicts": {team: s.optimizer.state_dict() for team, s in self.teams_state.items()},
            "hyperparams": self.hyperparams,
        }, path)

    @classmethod
    def load(cls, path: Path, env: Any, device: str = "cpu") -> "MultiAgentQMIX":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        for team, state_dict in payload.get("agent_state_dicts", {}).items():
            if team in algo.teams_state:
                algo.teams_state[team].agent_net.load_state_dict(state_dict)
                algo.teams_state[team].target_agent_net.load_state_dict(state_dict)
        for team, state_dict in payload.get("mixer_state_dicts", {}).items():
            if team in algo.teams_state:
                algo.teams_state[team].mixer.load_state_dict(state_dict)
                algo.teams_state[team].target_mixer.load_state_dict(state_dict)
        for team, opt_state in payload.get("optimizer_state_dicts", {}).items():
            if team in algo.teams_state:
                algo.teams_state[team].optimizer.load_state_dict(opt_state)
        return algo
