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
from rl_core.algorithms.native.networks import ActorCriticNet, Hidden, RecurrentActorCriticNet
from rl_core.algorithms.native.preprocessing import action_to_env, obs_to_array
from rl_core.netbuilder import SpecActorCriticNet


class OnPolicyAlgorithm(CustomAlgorithm):
    net: ActorCriticNet | SpecActorCriticNet | RecurrentActorCriticNet
    n_steps: int
    gamma: float
    gae_lambda: float
    # Set by subclasses' __init__ once `self.net` is built (see
    # ppo.py/a2c.py: `isinstance(self.net, RecurrentActorCriticNet)`).
    # `memory_seq_len` only matters when `recurrent` is True.
    recurrent: bool = False
    memory_seq_len: int = 32
    # Hidden state threaded across rollout-collection steps and (separately)
    # across `predict()` calls — `None` until the first reset seeds it via
    # `net.initial_state(...)`.
    _hidden: Hidden | None = None
    _predict_hidden: Hidden | None = None

    def _obs_arr(self, obs: Any) -> np.ndarray:
        return obs_to_array(obs, self.env.observation_space)

    def _sample_action(self, obs_arr: np.ndarray) -> tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.recurrent:
                dist, value, self._hidden = self.net.distribution_step(obs_t, self._hidden)
            else:
                dist, value = self.net.distribution(obs_t)
            action_t = dist.sample()
            log_prob_t = self.net.log_prob(dist, action_t)
        action = action_t.squeeze(0).cpu().numpy()
        return action, float(log_prob_t.item()), float(value.item())

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        obs, _ = self.env.reset(seed=self.seed)
        obs_arr = self._obs_arr(obs)
        action_dim = 0 if self.net.discrete else int(np.prod(self.env.action_space.shape))
        if self.recurrent:
            self._hidden = self.net.initial_state(1, self.device)
        episode_start = True
        ep_reward, ep_length = 0.0, 0
        num_timesteps = 0

        while num_timesteps < total_timesteps:
            capacity = min(self.n_steps, total_timesteps - num_timesteps)
            buf = RolloutBuffer(capacity, obs_arr.shape, action_dim, self.net.discrete)
            # Hidden state as it was *entering* this rollout — the correct
            # seed for re-running the whole chunk through the RNN during
            # `_update()` (every epoch restarts from here, not from
            # wherever collection ended up, since collection already moved
            # on to the *next* rollout by the time training happens).
            rollout_hidden = self._hidden
            done = False
            for _ in range(capacity):
                action, log_prob, value = self._sample_action(obs_arr)
                env_action = action_to_env(action, self.env.action_space)
                next_obs, reward, terminated, truncated, _info = self.env.step(env_action)
                done = terminated or truncated
                buf.add(obs_arr, action, log_prob, value, float(reward), done, episode_start)
                episode_start = False
                ep_reward += float(reward)
                ep_length += 1
                num_timesteps += 1

                if done:
                    finished_reward, finished_length = ep_reward, ep_length
                    ep_reward, ep_length = 0.0, 0
                    next_obs, _ = self.env.reset()
                    if self.recurrent:
                        self._hidden = self.net.initial_state(1, self.device)
                    episode_start = True
                    keep_going = callback.on_step(num_timesteps, finished_reward, finished_length)
                else:
                    keep_going = callback.on_step(num_timesteps)
                obs_arr = self._obs_arr(next_obs)
                if not keep_going:
                    return

            with torch.no_grad():
                obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
                if self.recurrent:
                    _, last_value, _ = self.net.step(obs_t, self._hidden)
                else:
                    _, last_value = self.net.forward(obs_t)
            buf.compute_returns_and_advantage(float(last_value.item()), done, self.gamma, self.gae_lambda)
            self._update(buf, rollout_hidden)

    def _update(self, buf: RolloutBuffer, rollout_hidden: Hidden | None = None) -> None:
        raise NotImplementedError

    def predict(self, obs: Any, deterministic: bool = True, episode_start: bool = False) -> tuple[Any, Any]:
        obs_arr = self._obs_arr(obs)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.recurrent:
                if episode_start or self._predict_hidden is None:
                    self._predict_hidden = self.net.initial_state(1, self.device)
                if deterministic:
                    action_t, self._predict_hidden = self.net.deterministic_action_step(obs_t, self._predict_hidden)
                else:
                    dist, _, self._predict_hidden = self.net.distribution_step(obs_t, self._predict_hidden)
                    action_t = dist.sample()
            elif deterministic:
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
