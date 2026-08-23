"""Shared rollout-collection loop for the two on-policy algorithms
(NativePPO, NativeA2C) — they only differ in how `_update()` turns a filled
`RolloutBuffer` into a gradient step."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.buffers import RolloutBuffer
from rl_core.algorithms.native.networks import ActorCriticNet
from rl_core.algorithms.native.preprocessing import action_to_env, obs_to_array


class OnPolicyAlgorithm(CustomAlgorithm):
    net: ActorCriticNet
    n_steps: int
    gamma: float
    gae_lambda: float

    def _obs_arr(self, obs: Any) -> np.ndarray:
        return obs_to_array(obs, self.env.observation_space)

    def _sample_action(self, obs_arr: np.ndarray) -> tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.net.distribution(obs_t)
            action_t = dist.sample()
            log_prob_t = self.net.log_prob(dist, action_t)
        action = action_t.squeeze(0).cpu().numpy()
        return action, float(log_prob_t.item()), float(value.item())

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        obs, _ = self.env.reset(seed=self.seed)
        obs_arr = self._obs_arr(obs)
        action_dim = 0 if self.net.discrete else int(np.prod(self.env.action_space.shape))
        ep_reward, ep_length = 0.0, 0
        num_timesteps = 0

        while num_timesteps < total_timesteps:
            capacity = min(self.n_steps, total_timesteps - num_timesteps)
            buf = RolloutBuffer(capacity, obs_arr.shape, action_dim, self.net.discrete)
            done = False
            for _ in range(capacity):
                action, log_prob, value = self._sample_action(obs_arr)
                env_action = action_to_env(action, self.env.action_space)
                next_obs, reward, terminated, truncated, _info = self.env.step(env_action)
                done = terminated or truncated
                buf.add(obs_arr, action, log_prob, value, float(reward), done)
                ep_reward += float(reward)
                ep_length += 1
                num_timesteps += 1

                if done:
                    finished_reward, finished_length = ep_reward, ep_length
                    ep_reward, ep_length = 0.0, 0
                    next_obs, _ = self.env.reset()
                    keep_going = callback.on_step(num_timesteps, finished_reward, finished_length)
                else:
                    keep_going = callback.on_step(num_timesteps)
                obs_arr = self._obs_arr(next_obs)
                if not keep_going:
                    return

            with torch.no_grad():
                obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
                _, last_value = self.net.forward(obs_t)
            buf.compute_returns_and_advantage(float(last_value.item()), done, self.gamma, self.gae_lambda)
            self._update(buf)

    def _update(self, buf: RolloutBuffer) -> None:
        raise NotImplementedError

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]:
        obs_arr = self._obs_arr(obs)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action_t = self.net.deterministic_action(obs_t)
            else:
                dist, _ = self.net.distribution(obs_t)
                action_t = dist.sample()
        action = action_t.squeeze(0).cpu().numpy()
        return action_to_env(action, self.env.action_space), None

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.net.state_dict(), "hyperparams": self.hyperparams}, path)

    @classmethod
    def load(cls, path: Path, env: gym.Env) -> "OnPolicyAlgorithm":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, "cpu")
        algo.net.load_state_dict(payload["state_dict"])
        return algo
