"""Independent PPO (IPPO) — the `ippo` algorithm in the Experiment Designer,
selectable only for multi-team Scene Builder scenes (see `team_count()` in
`rl_core/scene_store.py` and the `compatible_algorithms` gating in
`rl_core/envs/registry.py`).

One full, independent PPO policy per distinct `team` in the scene (see
`SceneMultiAgentEnv.team_ids` in `rl_core/envs/scene_env.py`) — every
team's `ActorCriticNet`/optimizer/`RolloutBuffer` is completely separate
(no shared critic, no cross-team gradients, no communication channel
between policies); teams only ever interact through the environment's
shared dynamics and reward rules (`rules.tag`/`rules.team_shared_reward`).
This is the standard, simplest MARL baseline — "Independent Learning"
(e.g. de Witt et al. 2020, "Is Independent Learning All You Need in the
StarCraft Multi-Agent Challenge?") — and a natural first native MARL
algorithm to build: `NativePPO`'s single-team update loop already does
everything one team's policy needs, so this file mostly just runs that
same loop N times per rollout against N disjoint lane-slices of one shared
environment step, instead of against N independent environments.

Deliberately *not* built on `OnPolicyAlgorithm` (rl_core/algorithms/native/
on_policy.py) — that class assumes one net/buffer for the whole batch of
lanes, whereas here every team needs its own; `learn()` below re-implements
just enough of that loop, generalized over `self.policies` instead of a
single `self.net`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.buffers import RolloutBuffer
from rl_core.algorithms.native.networks import ActorCriticNet
from rl_core.algorithms.native.on_policy import _LossAccumulator
from rl_core.algorithms.native.preprocessing import action_to_env, obs_batch_to_array, obs_to_array
from rl_core.algorithms.vec_env import action_space, num_envs_of, obs_space, vec_reset, vec_step

DEFAULT_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_epochs": 10,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    # 0 = constant `learning_rate`, 1 = linearly decay to 0 over
    # `total_timesteps` — same idea as NativePPO's, applied independently
    # per team (each team anneals its own optimizer's LR on the same
    # shared `num_timesteps` clock).
    "lr_schedule": 0,
    # Beta-distribution policy head (see `_BetaPolicyDistribution` in
    # networks.py) instead of a clipped Gaussian, applied per-team — see
    # NativePPO's `use_beta` docstring for why this matters for bounded
    # continuous action spaces. Ignored for Discrete movement (the common
    # case for Scene Builder scenes).
    "use_beta": 0,
    # Extra per-head hidden layer between features and each of pi/value —
    # see `ActorCriticNet.head_hidden_size`. Same default (0) as NativePPO.
    "head_hidden_size": 0,
    # gSDE for continuous per-team movement — same idea as NativePPO's
    # `use_sde`, resampled independently per team (each team's noise
    # matrix only ever needs to match its own lane count, not the whole
    # env's) on the same `sde_sample_freq` schedule, driven by `learn()`'s
    # own step loop below (this class isn't built on `OnPolicyAlgorithm`,
    # so there's no shared collection loop to piggyback on via `getattr`
    # like NativePPO/NativeA2C do). Beta wins if both are set (see
    # `ActorCriticNet`).
    "use_sde": 0,
    "sde_sample_freq": 4,
    "sde_log_std_init": -2.0,
}


class _TeamPolicy:
    """Everything one team owns: its own network, optimizer, and (once
    `learn()` starts a rollout) `RolloutBuffer` slice — the per-team
    equivalent of what `NativePPO` alone would hold for a whole (single-
    team) run."""

    def __init__(self, obs_sp: gym.Space, act_sp: gym.Space, hyperparams: dict[str, Any], device: str) -> None:
        use_beta = bool(int(hyperparams.get("use_beta", 0)))
        head_hidden_size = int(hyperparams.get("head_hidden_size", 0))
        use_sde = bool(int(hyperparams.get("use_sde", 0)))
        sde_log_std_init = float(hyperparams.get("sde_log_std_init", -2.0))
        self.net = ActorCriticNet(
            obs_sp, act_sp, use_sde=use_sde, sde_log_std_init=sde_log_std_init,
            head_hidden_size=head_hidden_size, use_beta=use_beta,
        ).to(device)
        # Reflects whether this team's net really ended up SDE-capable and
        # switched on (e.g. never true for Discrete movement — see
        # `ActorCriticNet.use_sde`), same distinction NativePPO draws.
        self.use_sde = self.net.use_sde
        self.initial_lr = float(hyperparams.get("learning_rate", 3e-4))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.initial_lr)
        self.last_metrics: dict[str, float] = {}


class MultiAgentPPO(CustomAlgorithm):
    """Independent PPO across every distinct team of a MARL Scene Builder
    scene. Requires `env` (or its `.unwrapped`) to expose a `team_ids`
    list — currently only `SceneMultiAgentEnv` does; the Designer only
    ever offers `ippo` for scenes with `team_count() >= 2` (see
    `rl_core/envs/registry.py`), so this should never be reached with an
    incompatible env through normal use of the app."""

    def __init__(self, env: Any, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        base_env = getattr(env, "unwrapped", env)
        team_ids = getattr(base_env, "team_ids", None)
        if not team_ids or len(set(team_ids)) < 2:
            raise ValueError(
                "MultiAgentPPO ('ippo') требует сцену с 2+ командами агентов "
                "(разные значения 'team' в группах агентов) — обычная gym-среда "
                "или однокомандная сцена сюда не подходят, используйте 'ppo'."
            )
        self.team_ids: list[str] = list(team_ids)
        self.teams: list[str] = sorted(set(self.team_ids))
        self.lanes: dict[str, np.ndarray] = {
            team: np.asarray([i for i, t in enumerate(self.team_ids) if t == team], dtype=np.int64)
            for team in self.teams
        }

        obs_sp, act_sp = obs_space(env), action_space(env)
        self.discrete = isinstance(act_sp, gym.spaces.Discrete)
        self.action_dim = 0 if self.discrete else int(np.prod(act_sp.shape))
        self.policies: dict[str, _TeamPolicy] = {
            team: _TeamPolicy(obs_sp, act_sp, hyperparams, device) for team in self.teams
        }

        self.n_steps = int(hyperparams.get("n_steps", 1024))
        self.batch_size = int(hyperparams.get("batch_size", 64))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.gae_lambda = float(hyperparams.get("gae_lambda", 0.95))
        self.clip_range = float(hyperparams.get("clip_range", 0.2))
        self.n_epochs = int(hyperparams.get("n_epochs", 10))
        self.ent_coef = float(hyperparams.get("ent_coef", 0.0))
        self.vf_coef = float(hyperparams.get("vf_coef", 0.5))
        self.max_grad_norm = float(hyperparams.get("max_grad_norm", 0.5))
        self.lr_schedule = "linear" if int(hyperparams.get("lr_schedule", 0)) == 1 else "constant"
        self.sde_sample_freq = max(1, int(hyperparams.get("sde_sample_freq", 4)))
        for team, lanes in self.lanes.items():
            policy = self.policies[team]
            if policy.use_sde:
                policy.net.reset_noise(len(lanes))
        # `predict()` (live GIF preview via `SceneRenderEnv`, which always
        # controls lane 0 — see its docstring) gets one unbatched
        # observation with no lane/team tag attached, so we fix "which
        # team's policy renders the preview" once, up front, to whichever
        # team actually owns lane 0.
        self._render_team = self.team_ids[0]

    def _obs_space(self) -> gym.Space:
        return obs_space(self.env)

    def _sample_team_actions(self, team: str, obs_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        net = self.policies[team].net
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist, value = net.distribution(obs_t)
            action_t = dist.sample()
            log_prob_t = net.log_prob(dist, action_t)
        return action_t.cpu().numpy(), log_prob_t.cpu().numpy(), value.cpu().numpy()

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        n_envs = num_envs_of(self.env)
        obs_sp = self._obs_space()
        act_sp = action_space(self.env)

        obs_arr = obs_batch_to_array(vec_reset(self.env, seed=self.seed), obs_sp)
        ep_reward = np.zeros(n_envs, dtype=np.float64)
        ep_length = np.zeros(n_envs, dtype=np.int64)
        num_timesteps = 0
        actions_dtype = np.int64 if self.discrete else np.float32
        actions_shape = (n_envs,) if self.discrete else (n_envs, self.action_dim)
        # See `OnPolicyAlgorithm.learn`'s identical `last_sde_reset`
        # tracking — resampled every `sde_sample_freq` *steps*, per team,
        # against only that team's own lane count.
        last_sde_reset = 0

        while num_timesteps < total_timesteps:
            remaining_lane_steps = -(-(total_timesteps - num_timesteps) // n_envs)  # ceil div
            capacity = max(1, min(self.n_steps, remaining_lane_steps))
            buffers = {
                team: RolloutBuffer(capacity, len(lanes), obs_arr.shape[1:], self.action_dim, self.discrete)
                for team, lanes in self.lanes.items()
            }
            dones = np.zeros(n_envs, dtype=bool)

            for _ in range(capacity):
                if num_timesteps - last_sde_reset >= self.sde_sample_freq:
                    last_sde_reset = num_timesteps
                    for team, lanes in self.lanes.items():
                        if self.policies[team].use_sde:
                            self.policies[team].net.reset_noise(len(lanes))
                actions = np.zeros(actions_shape, dtype=actions_dtype)
                per_team = {}
                for team, lanes in self.lanes.items():
                    acts, log_probs, values = self._sample_team_actions(team, obs_arr[lanes])
                    per_team[team] = (acts, log_probs, values)
                    actions[lanes] = acts

                env_actions = [action_to_env(actions[i], act_sp) for i in range(n_envs)]
                next_obs_list, rewards, terminated, truncated, _infos = vec_step(self.env, env_actions)
                dones = terminated | truncated
                next_obs_arr = obs_batch_to_array(next_obs_list, obs_sp)

                for team, lanes in self.lanes.items():
                    acts, log_probs, values = per_team[team]
                    buffers[team].add(
                        obs_arr[lanes], acts, log_probs, values,
                        rewards[lanes], dones[lanes].astype(np.float32),
                        np.zeros(len(lanes), dtype=np.float32),
                    )

                ep_reward += rewards
                ep_length += 1
                num_timesteps += n_envs

                any_finished = False
                for i in range(n_envs):
                    if dones[i]:
                        any_finished = True
                        keep_going = callback.on_step(
                            num_timesteps, float(ep_reward[i]), int(ep_length[i]), self._combined_metrics(),
                        )
                        ep_reward[i], ep_length[i] = 0.0, 0
                        if not keep_going:
                            return
                if not any_finished:
                    keep_going = callback.on_step(num_timesteps, metrics=self._combined_metrics())
                    if not keep_going:
                        return

                obs_arr = next_obs_arr

            with torch.no_grad():
                obs_t_all = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
                last_values = np.zeros(n_envs, dtype=np.float32)
                for team, lanes in self.lanes.items():
                    _, v = self.policies[team].net.forward(obs_t_all[lanes])
                    last_values[lanes] = v.cpu().numpy()

            progress = min(1.0, num_timesteps / total_timesteps)
            for team, lanes in self.lanes.items():
                buffers[team].compute_returns_and_advantage(
                    last_values[lanes], dones[lanes].astype(np.float32), self.gamma, self.gae_lambda,
                )
                if self.lr_schedule == "linear":
                    policy = self.policies[team]
                    new_lr = policy.initial_lr * (1.0 - progress)
                    for group in policy.optimizer.param_groups:
                        group["lr"] = new_lr
                self._update_team(team, buffers[team])

    def _update_team(self, team: str, buf: RolloutBuffer) -> None:
        policy = self.policies[team]
        acc = _LossAccumulator()
        for _ in range(self.n_epochs):
            for batch in buf.minibatches(self.batch_size):
                obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
                actions_t = torch.as_tensor(batch["actions"], device=self.device)
                old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
                advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)

                dist, values = policy.net.distribution(obs_t)
                log_probs = policy.net.log_prob(dist, actions_t)
                entropy = policy.net.entropy(dist).mean()

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.net.parameters(), self.max_grad_norm)
                policy.optimizer.step()
                acc.add(policy_loss, value_loss, entropy)
        policy.last_metrics = acc.means()

    def _combined_metrics(self) -> dict[str, float]:
        """Every team's loss metrics, flattened into one dict keyed
        `<team>__<metric>` (e.g. `predator__policy_loss`) — the Training
        Monitor already renders whatever keys `on_step(metrics=...)`
        reports without any MARL-specific chart code, so each team's
        losses show up as their own lines for free."""
        out: dict[str, float] = {}
        for team, policy in self.policies.items():
            for key, value in policy.last_metrics.items():
                out[f"{team}__{key}"] = value
        return out

    def predict(self, obs: Any, deterministic: bool = True, episode_start: bool = False) -> tuple[Any, Any]:
        del episode_start  # no memory support yet (see module docstring)
        net = self.policies[self._render_team].net
        obs_arr = obs_to_array(obs, self._obs_space())
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action_t = net.deterministic_action(obs_t)
            else:
                dist, _ = net.distribution(obs_t)
                action_t = dist.sample()
        action = action_t.squeeze(0).cpu().numpy()
        return action_to_env(action, action_space(self.env)), None

    def save(self, path: Path) -> None:
        torch.save({
            "teams": self.teams,
            "team_ids": self.team_ids,
            "state_dicts": {team: p.net.state_dict() for team, p in self.policies.items()},
            "optimizer_state_dicts": {team: p.optimizer.state_dict() for team, p in self.policies.items()},
            "hyperparams": self.hyperparams,
        }, path)

    @classmethod
    def load(cls, path: Path, env: Any, device: str = "cpu") -> "MultiAgentPPO":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        for team, state_dict in payload.get("state_dicts", {}).items():
            if team in algo.policies:
                algo.policies[team].net.load_state_dict(state_dict)
        for team, opt_state in payload.get("optimizer_state_dicts", {}).items():
            if team in algo.policies:
                algo.policies[team].optimizer.load_state_dict(opt_state)
        return algo
