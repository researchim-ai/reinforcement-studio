"""PETS — Probabilistic Ensembles with Trajectory Sampling (Chua et al.,
2018 — https://arxiv.org/abs/1805.12114). No learned policy or value
function at all, unlike every other algorithm in this app (including
`mbpo.py`, its closest relative here — same ensemble dynamics model, but
MBPO uses it to generate *training data* for a separately learned SAC
policy, while PETS uses it directly as the world model for online planning,
learning nothing but the dynamics model itself): every real action comes
from `rl_core.world_models.ensemble.cem_plan` re-solving a short-horizon
Cross-Entropy-Method search from scratch, using the current ensemble as a
stand-in for the real environment.

This makes PETS the most direct "model accuracy proxy" of any algorithm
here — its return curve is a fairly faithful proxy for how good the world
model has gotten so far (a fresh PPO/SAC still has *some* learned skill to
fall back on if its most recent gradient step happened to be a bad one; a
PETS agent's next action is entirely at the mercy of whatever imagined
future the ensemble models this step), which makes it a good pick for
literally watching a world model improve during training rather than
just its own loss curves.

Only supports `training.num_envs=1` — the whole point is genuinely
online planning against the real env's current state, one step (then
replan from scratch) at a time, which has no obvious "N independent lanes"
analogue the way collecting experience with a fixed policy does."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.buffers import ContinuousReplayBuffer
from rl_core.algorithms.native.preprocessing import obs_to_array
from rl_core.algorithms.vec_env import is_vector_env
from rl_core.world_models.ensemble import cem_plan
from rl_core.world_models.losses import train_step_ensemble
from rl_core.world_models.spec import resolve_or_build_for_algo, save_checkpoint

DEFAULT_HYPERPARAMS = {
    "model_learning_rate": 1e-3,
    "buffer_size": 100_000,
    "batch_size": 256,
    "learning_starts": 500,
    "model_train_freq": 250,
    "cem_horizon": 15,
    "cem_candidates": 400,
    "cem_elites": 40,
    "cem_iterations": 5,
}


class NativePETS(CustomAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if is_vector_env(env):
            raise ValueError("NativePETS поддерживает только training.num_envs=1")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("NativePETS поддерживает только непрерывные (Box) действия (нужны для CEM-планирования)")
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self._obs_space = env.observation_space
        self._action_space = env.action_space

        self.world_model, self.world_model_config, self.world_model_slug = resolve_or_build_for_algo(
            hyperparams.get("world_model_spec"), "ensemble", self._obs_space, self._action_space, device,
        )
        self.world_model.to(device)
        self.model_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=float(hyperparams.get("model_learning_rate", 1e-3)),
        )

        self.buffer_size = int(hyperparams.get("buffer_size", 100_000))
        self.batch_size = int(hyperparams.get("batch_size", 256))
        self.learning_starts = int(hyperparams.get("learning_starts", 500))
        self.model_train_freq = max(1, int(hyperparams.get("model_train_freq", 250)))
        self.cem_horizon = max(1, int(hyperparams.get("cem_horizon", 15)))
        self.cem_candidates = max(2, int(hyperparams.get("cem_candidates", 400)))
        self.cem_elites = max(1, int(hyperparams.get("cem_elites", 40)))
        self.cem_iterations = max(1, int(hyperparams.get("cem_iterations", 5)))

        self.buffer: ContinuousReplayBuffer | None = None
        self._last_metrics: dict[str, float] = {}

    def _obs_arr(self, obs: Any) -> np.ndarray:
        return obs_to_array(obs, self._obs_space)

    def _plan_action(self, obs_arr: np.ndarray) -> np.ndarray:
        flat_obs0 = self.world_model.flat_target(
            torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0),
        ).squeeze(0)
        action = cem_plan(
            self.world_model, flat_obs0, self._action_space,
            horizon=self.cem_horizon, num_candidates=self.cem_candidates,
            num_elites=self.cem_elites, num_iterations=self.cem_iterations,
        )
        return action.cpu().numpy()

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        obs, _info = self.env.reset(seed=self.seed)
        obs_arr = self._obs_arr(obs)
        if self.buffer is None:
            self.buffer = ContinuousReplayBuffer(self.buffer_size, obs_arr.shape, int(np.prod(self._action_space.shape)))
        ep_reward, ep_length = 0.0, 0
        num_timesteps = 0

        while num_timesteps < total_timesteps:
            if num_timesteps < self.learning_starts:
                action_arr = np.asarray(self._action_space.sample(), dtype=np.float32).reshape(-1)
            else:
                action_arr = self._plan_action(obs_arr)

            env_action = np.clip(action_arr, self._action_space.low, self._action_space.high).reshape(self._action_space.shape)
            next_obs, reward, terminated, truncated, _info = self.env.step(env_action)
            done = bool(terminated or truncated)
            next_obs_arr = self._obs_arr(next_obs)
            self.buffer.add(obs_arr, action_arr, float(reward), next_obs_arr, done)
            obs_arr = next_obs_arr
            ep_reward += float(reward)
            ep_length += 1
            num_timesteps += 1

            if len(self.buffer) >= max(self.learning_starts, self.batch_size) and num_timesteps % self.model_train_freq == 0:
                model_losses = train_step_ensemble(self.world_model, self.model_optimizer, self.buffer, self.batch_size, self.device)
                self._last_metrics.update(model_losses)

            if done:
                keep_going = callback.on_step(num_timesteps, ep_reward, ep_length, self._last_metrics)
                obs, _info = self.env.reset()
                obs_arr = self._obs_arr(obs)
                ep_reward, ep_length = 0.0, 0
            else:
                keep_going = callback.on_step(num_timesteps, metrics=self._last_metrics)
            if not keep_going:
                return

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]:
        del deterministic  # CEM planning has no separate "exploration" mode to turn off
        obs_arr = self._obs_arr(obs)
        action_arr = self._plan_action(obs_arr)
        return np.clip(action_arr, self._action_space.low, self._action_space.high).reshape(self._action_space.shape), None

    def save(self, path: Path) -> None:
        torch.save(
            {
                "world_model_state_dict": self.world_model.state_dict(),
                "world_model_config": self.world_model_config,
                "hyperparams": self.hyperparams,
            },
            path,
        )

    def save_world_model_checkpoint(self, path: Path) -> None:
        save_checkpoint("ensemble", self.world_model, path, self.world_model_config)

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativePETS":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.world_model.load_state_dict(payload["world_model_state_dict"])
        return algo
