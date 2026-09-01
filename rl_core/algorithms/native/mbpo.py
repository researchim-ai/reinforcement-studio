"""MBPO — Model-Based Policy Optimization (Janner et al., 2019 —
https://arxiv.org/abs/1906.08253). SAC (`rl_core/algorithms/native/sac.py`'s
exact policy/critic networks and update rule, reused as-is) plus a
probabilistic ensemble dynamics model (`rl_core/world_models/ensemble.py`)
that's continuously retrained on real transitions and used to generate
*short* imagined rollouts branched from real replay states — those get
added to a second buffer, and every SAC update draws a mix of real and
model-generated transitions (`real_ratio` controls the split) rather than
only ever seeing real data. The core insight (and MBPO's main theoretical
contribution): branching short rollouts from real states rather than
generating entire imagined episodes bounds how far compounding model
error can drift the data SAC actually trains on.

Deliberately keeps the imagined rollouts *short* (`rollout_length` defaults
to 1) and the model reasonably fresh (`model_train_freq` retrains it every
handful of real steps) — both match the paper's own recommendation that a
"slightly-branched, mostly-real" replay mix outperforms either extreme
(pure real-data SAC, or fully model-generated Dyna-style rollouts)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.buffers import ContinuousReplayBuffer
from rl_core.algorithms.native.networks import GaussianPolicy, QCritic
from rl_core.algorithms.native.preprocessing import action_to_env, obs_batch_to_array, obs_to_array
from rl_core.algorithms.vec_env import action_space, num_envs_of, obs_space, vec_reset, vec_step
from rl_core.world_models.losses import train_step_ensemble
from rl_core.world_models.spec import resolve_or_build_for_algo, save_checkpoint

DEFAULT_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "model_buffer_size": 100_000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": 0.2,
    "learning_starts": 1_000,
    "train_freq": 1,
    "model_learning_rate": 1e-3,
    "model_train_freq": 250,
    "rollout_batch_size": 400,
    "rollout_length": 1,
    "real_ratio": 0.1,
}


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


class NativeMBPO(CustomAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        obs_sp, act_sp = obs_space(env), action_space(env)
        if not isinstance(act_sp, gym.spaces.Box):
            raise ValueError("NativeMBPO поддерживает только непрерывные (Box) действия")

        self.action_dim = int(np.prod(act_sp.shape))
        low = np.asarray(act_sp.low, dtype=np.float32).reshape(-1)
        high = np.asarray(act_sp.high, dtype=np.float32).reshape(-1)

        self.policy = GaussianPolicy(obs_sp, low, high).to(device)
        self.q1 = QCritic(obs_sp, self.action_dim).to(device)
        self.q2 = QCritic(obs_sp, self.action_dim).to(device)
        self.q1_target = QCritic(obs_sp, self.action_dim).to(device)
        self.q2_target = QCritic(obs_sp, self.action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        lr = float(hyperparams.get("learning_rate", 3e-4))
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        self.world_model, self.world_model_config, self.world_model_slug = resolve_or_build_for_algo(
            hyperparams.get("world_model_spec"), "ensemble", obs_sp, act_sp, device,
        )
        self.world_model.to(device)
        self.model_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=float(hyperparams.get("model_learning_rate", 1e-3)),
        )

        self.buffer_size = int(hyperparams.get("buffer_size", 100_000))
        self.model_buffer_size = int(hyperparams.get("model_buffer_size", 100_000))
        self.batch_size = int(hyperparams.get("batch_size", 256))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.tau = float(hyperparams.get("tau", 0.005))
        self.ent_coef = float(hyperparams.get("ent_coef", 0.2))
        self.learning_starts = int(hyperparams.get("learning_starts", 1_000))
        self.train_freq = max(1, int(hyperparams.get("train_freq", 1)))
        self.model_train_freq = max(1, int(hyperparams.get("model_train_freq", 250)))
        self.rollout_batch_size = max(1, int(hyperparams.get("rollout_batch_size", 400)))
        self.rollout_length = max(1, int(hyperparams.get("rollout_length", 1)))
        self.real_ratio = float(np.clip(hyperparams.get("real_ratio", 0.1), 0.0, 1.0))

        self.real_buffer: ContinuousReplayBuffer | None = None
        self.model_buffer: ContinuousReplayBuffer | None = None
        self._last_metrics: dict[str, float] = {}
        self._obs_space = obs_sp
        self._action_space = act_sp

    def _obs_arr(self, obs: Any) -> np.ndarray:
        return obs_to_array(obs, self._obs_space)

    def _sample_actions(self, obs_arr: np.ndarray, deterministic: bool) -> np.ndarray:
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, deterministic_action = self.policy.sample(obs_t)
            chosen = deterministic_action if deterministic else action
        return chosen.cpu().numpy()

    @torch.no_grad()
    def _generate_model_rollouts(self) -> None:
        """Branches `rollout_batch_size` short (`rollout_length`-step)
        imagined rollouts off real states sampled from `real_buffer`,
        using the *current* policy to pick actions at every imagined step
        (exactly what makes this data useful for improving that same
        policy) — every resulting `(obs, action, reward, next_obs, done)`
        goes into `model_buffer`. `done` is always `False`: the ensemble
        has no notion of episode termination at all (see `ensemble.py`),
        so treating every imagined step as non-terminal is the standard
        MBPO simplification (the alternative — a hand-written per-env
        termination function — is explicitly out of scope here)."""
        batch = self.real_buffer.sample(min(self.rollout_batch_size, len(self.real_buffer)))
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        flat_obs = self.world_model.flat_target(obs)
        model_indices = self.world_model.sample_model_indices(flat_obs.shape[0], self.device)
        raw_obs = obs

        for _ in range(self.rollout_length):
            action_t, _, _ = self.policy.sample(raw_obs)
            next_flat, reward = self.world_model.predict_step(flat_obs, action_t, model_indices=model_indices, sample=True)
            next_raw = self.world_model.to_raw_obs(next_flat)
            obs_np = raw_obs.cpu().numpy()
            action_np = action_t.cpu().numpy()
            next_obs_np = next_raw.cpu().numpy()
            reward_np = reward.cpu().numpy()
            for i in range(obs_np.shape[0]):
                self.model_buffer.add(obs_np[i], action_np[i], float(reward_np[i]), next_obs_np[i], False)
            flat_obs, raw_obs = next_flat, next_raw

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        n_envs = num_envs_of(self.env)
        obs_arr = obs_batch_to_array(vec_reset(self.env, seed=self.seed), self._obs_space)
        if self.real_buffer is None:
            self.real_buffer = ContinuousReplayBuffer(self.buffer_size, obs_arr.shape[1:], self.action_dim)
            self.model_buffer = ContinuousReplayBuffer(self.model_buffer_size, obs_arr.shape[1:], self.action_dim)
        ep_reward = np.zeros(n_envs, dtype=np.float64)
        ep_length = np.zeros(n_envs, dtype=np.int64)
        num_timesteps = 0

        while num_timesteps < total_timesteps:
            if len(self.real_buffer) < self.learning_starts:
                actions = np.stack(
                    [np.asarray(self._action_space.sample(), dtype=np.float32).reshape(-1) for _ in range(n_envs)],
                )
            else:
                actions = self._sample_actions(obs_arr, deterministic=False)

            env_actions = [action_to_env(actions[i], self._action_space) for i in range(n_envs)]
            next_obs_list, rewards, terminated, truncated, _infos = vec_step(self.env, env_actions)
            dones = terminated | truncated
            next_obs_arr = obs_batch_to_array(next_obs_list, self._obs_space)
            for i in range(n_envs):
                self.real_buffer.add(obs_arr[i], actions[i], float(rewards[i]), next_obs_arr[i], bool(dones[i]))
            obs_arr = next_obs_arr
            ep_reward += rewards
            ep_length += 1
            prev_num_timesteps = num_timesteps
            num_timesteps += n_envs

            if len(self.real_buffer) >= max(self.learning_starts, self.batch_size):
                if num_timesteps // self.model_train_freq != prev_num_timesteps // self.model_train_freq:
                    model_losses = train_step_ensemble(
                        self.world_model, self.model_optimizer, self.real_buffer, self.batch_size, self.device,
                    )
                    self._last_metrics.update(model_losses)
                    self._generate_model_rollouts()
                if num_timesteps // self.train_freq != prev_num_timesteps // self.train_freq and len(self.model_buffer) > 0:
                    self._train_step()

            any_finished = False
            for i in range(n_envs):
                if dones[i]:
                    any_finished = True
                    keep_going = callback.on_step(num_timesteps, float(ep_reward[i]), int(ep_length[i]), self._last_metrics)
                    ep_reward[i], ep_length[i] = 0.0, 0
                    if not keep_going:
                        return
            if not any_finished:
                keep_going = callback.on_step(num_timesteps, metrics=self._last_metrics)
                if not keep_going:
                    return

    def _train_step(self) -> None:
        real_n = max(1, round(self.batch_size * self.real_ratio))
        model_n = max(0, self.batch_size - real_n)
        model_n = min(model_n, len(self.model_buffer)) if self.model_buffer is not None else 0
        real_n = self.batch_size - model_n

        real_batch = self.real_buffer.sample(real_n)
        if model_n > 0:
            model_batch = self.model_buffer.sample(model_n)
            batch = {key: np.concatenate([real_batch[key], model_batch[key]], axis=0) for key in real_batch}
        else:
            batch = real_batch

        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_obs_t)
            target_q1 = self.q1_target(next_obs_t, next_action)
            target_q2 = self.q2_target(next_obs_t, next_action)
            min_target_q = torch.min(target_q1, target_q2) - self.ent_coef * next_log_prob
            target = rewards_t + self.gamma * (1.0 - dones_t) * min_target_q

        current_q1 = self.q1(obs_t, actions_t)
        current_q2 = self.q2(obs_t, actions_t)
        q_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        critic_loss = float(q_loss.detach().item())

        action_pi, log_prob_pi, _ = self.policy.sample(obs_t)
        q1_pi = self.q1(obs_t, action_pi)
        q2_pi = self.q2(obs_t, action_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.ent_coef * log_prob_pi - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        _soft_update(self.q1_target, self.q1, self.tau)
        _soft_update(self.q2_target, self.q2, self.tau)

        self._last_metrics.update({
            "actor_loss": float(policy_loss.detach().item()),
            "critic_loss": critic_loss,
            "model_buffer_size": float(len(self.model_buffer)) if self.model_buffer is not None else 0.0,
        })

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]:
        obs_arr = self._obs_arr(obs)
        action = self._sample_actions(obs_arr[np.newaxis, ...], deterministic=deterministic)[0]
        return action.reshape(self._action_space.shape), None

    def save(self, path: Path) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "world_model_state_dict": self.world_model.state_dict(),
                "world_model_config": self.world_model_config,
                "hyperparams": self.hyperparams,
            },
            path,
        )

    def save_world_model_checkpoint(self, path: Path) -> None:
        save_checkpoint("ensemble", self.world_model, path, self.world_model_config)

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativeMBPO":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.policy.load_state_dict(payload["policy_state_dict"])
        algo.q1.load_state_dict(payload["q1_state_dict"])
        algo.q2.load_state_dict(payload["q2_state_dict"])
        algo.q1_target.load_state_dict(payload["q1_state_dict"])
        algo.q2_target.load_state_dict(payload["q2_state_dict"])
        algo.world_model.load_state_dict(payload["world_model_state_dict"])
        return algo
