"""From-scratch Soft Actor-Critic — the first *off-policy, continuous-action*
option in the Experiment Designer (see `rl_core/algorithms/native_runner.py`).
PPO/A2C already handle continuous actions on-policy; DQN/Rainbow are
discrete-only. SAC fills that gap: it learns a stochastic policy that
maximizes reward *and* entropy (Haarnoja et al., 2018), using two Q-critics
with target networks (soft-updated via Polyak averaging) to keep training
stable, and reuses experience through a replay buffer like DQN does —
generally far more sample-efficient than PPO/A2C on continuous-control
tasks, at the cost of more hyperparameters to get right.

Uses a fixed (not automatically tuned) entropy coefficient `ent_coef` — the
original SAC paper's simpler variant — to keep every hyperparameter here a
plain number the Designer's numeric hyperparam UI already knows how to
render, rather than adding a special-cased "auto" string option."""
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
from rl_core.algorithms.native.preprocessing import obs_to_array

DEFAULT_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": 0.2,
    "learning_starts": 1_000,
    "train_freq": 1,
}


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


class NativeSAC(CustomAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("NativeSAC only supports continuous (Box) action spaces")

        self.action_dim = int(np.prod(env.action_space.shape))
        low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

        self.policy = GaussianPolicy(env.observation_space, low, high).to(device)
        self.q1 = QCritic(env.observation_space, self.action_dim).to(device)
        self.q2 = QCritic(env.observation_space, self.action_dim).to(device)
        self.q1_target = QCritic(env.observation_space, self.action_dim).to(device)
        self.q2_target = QCritic(env.observation_space, self.action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        lr = float(hyperparams.get("learning_rate", 3e-4))
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        self.buffer_size = int(hyperparams.get("buffer_size", 100_000))
        self.batch_size = int(hyperparams.get("batch_size", 256))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.tau = float(hyperparams.get("tau", 0.005))
        self.ent_coef = float(hyperparams.get("ent_coef", 0.2))
        self.learning_starts = int(hyperparams.get("learning_starts", 1_000))
        self.train_freq = max(1, int(hyperparams.get("train_freq", 1)))
        self.replay_buffer: ContinuousReplayBuffer | None = None

    def _obs_arr(self, obs: Any) -> np.ndarray:
        return obs_to_array(obs, self.env.observation_space)

    def _sample_action(self, obs_arr: np.ndarray, deterministic: bool) -> np.ndarray:
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, deterministic_action = self.policy.sample(obs_t)
            chosen = deterministic_action if deterministic else action
        return chosen.squeeze(0).cpu().numpy()

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        obs, _ = self.env.reset(seed=self.seed)
        obs_arr = self._obs_arr(obs)
        if self.replay_buffer is None:
            self.replay_buffer = ContinuousReplayBuffer(self.buffer_size, obs_arr.shape, self.action_dim)
        ep_reward, ep_length = 0.0, 0

        for num_timesteps in range(1, total_timesteps + 1):
            if len(self.replay_buffer) < self.learning_starts:
                action = np.asarray(self.env.action_space.sample(), dtype=np.float32).reshape(-1)
            else:
                action = self._sample_action(obs_arr, deterministic=False)

            next_obs, reward, terminated, truncated, _info = self.env.step(action.reshape(self.env.action_space.shape))
            done = terminated or truncated
            next_obs_arr = self._obs_arr(next_obs)
            self.replay_buffer.add(obs_arr, action, float(reward), next_obs_arr, done)
            obs_arr = next_obs_arr
            ep_reward += float(reward)
            ep_length += 1

            if len(self.replay_buffer) >= max(self.learning_starts, self.batch_size) and num_timesteps % self.train_freq == 0:
                self._train_step()

            if done:
                finished_reward, finished_length = ep_reward, ep_length
                ep_reward, ep_length = 0.0, 0
                next_obs2, _ = self.env.reset()
                obs_arr = self._obs_arr(next_obs2)
                keep_going = callback.on_step(num_timesteps, finished_reward, finished_length)
            else:
                keep_going = callback.on_step(num_timesteps)
            if not keep_going:
                return

    def _train_step(self) -> None:
        batch = self.replay_buffer.sample(self.batch_size)
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

        # Policy update: maximize (Q - ent_coef * log_prob), i.e. minimize
        # its negative — reward *and* action entropy both pull the policy,
        # so it doesn't collapse onto a single action too early.
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

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]:
        obs_arr = self._obs_arr(obs)
        action = self._sample_action(obs_arr, deterministic=deterministic)
        return action.reshape(self.env.action_space.shape), None

    def save(self, path: Path) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "hyperparams": self.hyperparams,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, env: gym.Env) -> "NativeSAC":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, "cpu")
        algo.policy.load_state_dict(payload["policy_state_dict"])
        algo.q1.load_state_dict(payload["q1_state_dict"])
        algo.q2.load_state_dict(payload["q2_state_dict"])
        algo.q1_target.load_state_dict(payload["q1_state_dict"])
        algo.q2_target.load_state_dict(payload["q2_state_dict"])
        return algo
