"""From-scratch Deep Q-Network — the default `dqn` algorithm in the
Experiment Designer (see `rl_core/algorithms/native_runner.py`). Discrete
actions only (enforced by the environment/algorithm compatibility table in
`rl_core/envs/registry.py`, same as before)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.buffers import ReplayBuffer
from rl_core.algorithms.native.networks import QNetwork
from rl_core.algorithms.native.preprocessing import obs_to_array

DEFAULT_HYPERPARAMS = {
    "learning_rate": 1e-3,
    "buffer_size": 50_000,
    "batch_size": 64,
    "gamma": 0.99,
    "exploration_fraction": 0.2,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "learning_starts": 1_000,
    "train_freq": 4,
    "target_update_interval": 1_000,
}


class NativeDQN(CustomAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("NativeDQN only supports Discrete action spaces")

        self.n_actions = int(env.action_space.n)
        self.q_net = QNetwork(env.observation_space, self.n_actions).to(device)
        self.target_net = QNetwork(env.observation_space, self.n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(hyperparams.get("learning_rate", 1e-3)))

        self.buffer_size = int(hyperparams.get("buffer_size", 50_000))
        self.batch_size = int(hyperparams.get("batch_size", 64))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.exploration_fraction = float(hyperparams.get("exploration_fraction", 0.2))
        self.exploration_initial_eps = float(hyperparams.get("exploration_initial_eps", 1.0))
        self.exploration_final_eps = float(hyperparams.get("exploration_final_eps", 0.05))
        self.learning_starts = int(hyperparams.get("learning_starts", 1_000))
        self.train_freq = max(1, int(hyperparams.get("train_freq", 4)))
        self.target_update_interval = max(1, int(hyperparams.get("target_update_interval", 1_000)))
        self.replay_buffer: ReplayBuffer | None = None

    def _obs_arr(self, obs: Any) -> np.ndarray:
        return obs_to_array(obs, self.env.observation_space)

    def _epsilon(self, num_timesteps: int, total_timesteps: int) -> float:
        exploration_steps = max(1, int(self.exploration_fraction * total_timesteps))
        progress = min(1.0, num_timesteps / exploration_steps)
        return self.exploration_initial_eps + progress * (self.exploration_final_eps - self.exploration_initial_eps)

    def _greedy_action(self, obs_arr: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.q_net(obs_t), dim=-1).item())

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        obs, _ = self.env.reset(seed=self.seed)
        obs_arr = self._obs_arr(obs)
        if self.replay_buffer is None:
            self.replay_buffer = ReplayBuffer(self.buffer_size, obs_arr.shape)
        ep_reward, ep_length = 0.0, 0

        for num_timesteps in range(1, total_timesteps + 1):
            eps = self._epsilon(num_timesteps, total_timesteps)
            if len(self.replay_buffer) < self.learning_starts or np.random.rand() < eps:
                action = int(self.env.action_space.sample())
            else:
                action = self._greedy_action(obs_arr)

            next_obs, reward, terminated, truncated, _info = self.env.step(action)
            done = terminated or truncated
            next_obs_arr = self._obs_arr(next_obs)
            self.replay_buffer.add(obs_arr, action, float(reward), next_obs_arr, done)
            obs_arr = next_obs_arr
            ep_reward += float(reward)
            ep_length += 1

            if len(self.replay_buffer) >= max(self.learning_starts, self.batch_size) and num_timesteps % self.train_freq == 0:
                self._train_step()
            if num_timesteps % self.target_update_interval == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

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
        actions_t = torch.as_tensor(batch["actions"], device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(dim=-1)[0]
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q
        current_q = self.q_net(obs_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]:
        obs_arr = self._obs_arr(obs)
        if not deterministic and np.random.rand() < self.exploration_final_eps:
            return int(self.env.action_space.sample()), None
        return self._greedy_action(obs_arr), None

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.q_net.state_dict(), "hyperparams": self.hyperparams}, path)

    @classmethod
    def load(cls, path: Path, env: gym.Env) -> "NativeDQN":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, "cpu")
        algo.q_net.load_state_dict(payload["state_dict"])
        algo.target_net.load_state_dict(payload["state_dict"])
        return algo
