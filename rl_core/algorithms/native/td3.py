"""From-scratch Twin Delayed DDPG (Fujimoto et al., 2018) — the standard
fix for DDPG's three main failure modes, all three implemented here:

1. **Twin critics**: two independent `QCritic`s, target value = min of both
   targets. Same "don't trust a single critic's optimism" idea as SAC's two
   critics / Double DQN, just applied to a deterministic policy.
2. **Delayed policy updates**: the actor (and both target networks) only
   update every `policy_delay` critic updates — the critic needs to settle
   somewhat before the actor's gradient through it means anything.
3. **Target policy smoothing**: small clipped Gaussian noise is added to the
   *target* actor's action before it's fed into the target critics, so the
   critic can't be exploited by narrow, easily-overestimated Q-value spikes
   at a single sharp action.

Deterministic policy + replay buffer like `NativeDDPG` (see that module for
the shared exploration/collection logic) — TD3 differs only in `_train_step`
and the extra `policy_delay`/`policy_noise`/`noise_clip` hyperparameters.
Generally the more reliable default for continuous control among this app's
off-policy options when the actual environment dynamics are (close to)
deterministic; SAC's entropy-driven stochastic policy tends to explore
better on its own but needs its `ent_coef` tuned for the reward scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.buffers import ContinuousReplayBuffer
from rl_core.algorithms.native.networks import DeterministicPolicy, QCritic
from rl_core.algorithms.native.preprocessing import action_to_env, obs_batch_to_array, obs_to_array
from rl_core.algorithms.vec_env import action_space, num_envs_of, obs_space, vec_reset, vec_step

DEFAULT_HYPERPARAMS = {
    "learning_rate": 1e-3,
    "buffer_size": 100_000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "exploration_noise": 0.1,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "policy_delay": 2,
    "learning_starts": 1_000,
    "train_freq": 1,
}


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


class NativeTD3(CustomAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        obs_sp, act_sp = obs_space(env), action_space(env)
        if not isinstance(act_sp, gym.spaces.Box):
            raise ValueError("NativeTD3 only supports continuous (Box) action spaces")

        self.action_dim = int(np.prod(act_sp.shape))
        low = np.asarray(act_sp.low, dtype=np.float32).reshape(-1)
        high = np.asarray(act_sp.high, dtype=np.float32).reshape(-1)
        self._action_low = low
        self._action_high = high
        self._action_range = np.where(np.isfinite(high - low), high - low, 2.0)

        self.actor = DeterministicPolicy(obs_sp, low, high).to(device)
        self.q1 = QCritic(obs_sp, self.action_dim).to(device)
        self.q2 = QCritic(obs_sp, self.action_dim).to(device)
        self.actor_target = DeterministicPolicy(obs_sp, low, high).to(device)
        self.q1_target = QCritic(obs_sp, self.action_dim).to(device)
        self.q2_target = QCritic(obs_sp, self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        lr = float(hyperparams.get("learning_rate", 1e-3))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        self.buffer_size = int(hyperparams.get("buffer_size", 100_000))
        self.batch_size = int(hyperparams.get("batch_size", 256))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.tau = float(hyperparams.get("tau", 0.005))
        self.exploration_noise = float(hyperparams.get("exploration_noise", 0.1))
        self.policy_noise = float(hyperparams.get("policy_noise", 0.2))
        self.noise_clip = float(hyperparams.get("noise_clip", 0.5))
        self.policy_delay = max(1, int(hyperparams.get("policy_delay", 2)))
        self.learning_starts = int(hyperparams.get("learning_starts", 1_000))
        self.train_freq = max(1, int(hyperparams.get("train_freq", 1)))
        self.replay_buffer: ContinuousReplayBuffer | None = None
        self._last_metrics: dict[str, float] = {}
        self._critic_updates = 0
        self._obs_space = obs_sp
        self._action_space = act_sp

    def _obs_arr(self, obs: Any) -> np.ndarray:
        return obs_to_array(obs, self._obs_space)

    def _sample_actions(self, obs_arr: np.ndarray, deterministic: bool) -> np.ndarray:
        """`obs_arr`: `(num_envs, ...)` -> `(num_envs, action_dim)`."""
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions = self.actor(obs_t).cpu().numpy()
        if not deterministic:
            actions = actions + np.random.normal(0.0, self.exploration_noise * self._action_range, size=actions.shape)
            actions = np.clip(actions, self._action_space.low, self._action_space.high)
        return actions.astype(np.float32)

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        n_envs = num_envs_of(self.env)
        obs_arr = obs_batch_to_array(vec_reset(self.env, seed=self.seed), self._obs_space)
        if self.replay_buffer is None:
            self.replay_buffer = ContinuousReplayBuffer(self.buffer_size, obs_arr.shape[1:], self.action_dim)
        ep_reward = np.zeros(n_envs, dtype=np.float64)
        ep_length = np.zeros(n_envs, dtype=np.int64)
        num_timesteps = 0

        while num_timesteps < total_timesteps:
            if len(self.replay_buffer) < self.learning_starts:
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
                self.replay_buffer.add(obs_arr[i], actions[i], float(rewards[i]), next_obs_arr[i], bool(dones[i]))
            obs_arr = next_obs_arr
            ep_reward += rewards
            ep_length += 1
            prev_num_timesteps = num_timesteps
            num_timesteps += n_envs

            if len(self.replay_buffer) >= max(self.learning_starts, self.batch_size) and (
                num_timesteps // self.train_freq != prev_num_timesteps // self.train_freq
            ):
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
        batch = self.replay_buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        low_t = torch.as_tensor(self._action_low, dtype=torch.float32, device=self.device)
        high_t = torch.as_tensor(self._action_high, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action = self.actor_target(next_obs_t)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(low_t, high_t)
            target_q1 = self.q1_target(next_obs_t, next_action)
            target_q2 = self.q2_target(next_obs_t, next_action)
            min_target_q = torch.min(target_q1, target_q2)
            target = rewards_t + self.gamma * (1.0 - dones_t) * min_target_q

        current_q1 = self.q1(obs_t, actions_t)
        current_q2 = self.q2(obs_t, actions_t)
        q_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self._critic_updates += 1
        self._last_metrics["critic_loss"] = float(q_loss.detach().item())

        if self._critic_updates % self.policy_delay == 0:
            actor_loss = -self.q1(obs_t, self.actor(obs_t)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            _soft_update(self.actor_target, self.actor, self.tau)
            _soft_update(self.q1_target, self.q1, self.tau)
            _soft_update(self.q2_target, self.q2, self.tau)

            self._last_metrics["actor_loss"] = float(actor_loss.detach().item())

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]:
        obs_arr = self._obs_arr(obs)
        action = self._sample_actions(obs_arr[np.newaxis, ...], deterministic=deterministic)[0]
        return action.reshape(self._action_space.shape), None

    def save(self, path: Path) -> None:
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "q_optimizer_state_dict": self.q_optimizer.state_dict(),
                "hyperparams": self.hyperparams,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativeTD3":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.actor.load_state_dict(payload["actor_state_dict"])
        algo.q1.load_state_dict(payload["q1_state_dict"])
        algo.q2.load_state_dict(payload["q2_state_dict"])
        algo.actor_target.load_state_dict(payload["actor_state_dict"])
        algo.q1_target.load_state_dict(payload["q1_state_dict"])
        algo.q2_target.load_state_dict(payload["q2_state_dict"])
        if payload.get("actor_optimizer_state_dict"):
            algo.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        if payload.get("q_optimizer_state_dict"):
            algo.q_optimizer.load_state_dict(payload["q_optimizer_state_dict"])
        return algo
