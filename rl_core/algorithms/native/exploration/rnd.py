"""Random Network Distillation intrinsic motivation (Burda et al., 2019)."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_core.algorithms.native.preprocessing import obs_flat_dim


class RunningMeanStd:
    """Numerically stable running moments with checkpoint support."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == self.mean.ndim:
            arr = arr.reshape(1, *arr.shape)
        if arr.shape[0] == 0:
            return
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0)
        batch_count = arr.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = float(total)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


def _rnd_network(input_dim: int, feature_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, feature_dim),
    )


class RNDModule(nn.Module):
    """Frozen random target plus a trainable predictor.

    The prediction error on the next observation is a novelty bonus. Both
    observations and bonuses are normalized online so one coefficient works
    across environments with very different scales.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        device: torch.device | str,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        bonus_clip: float = 5.0,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.input_dim = obs_flat_dim(observation_space)
        self.target = _rnd_network(self.input_dim, feature_dim, hidden_dim)
        self.predictor = _rnd_network(self.input_dim, feature_dim, hidden_dim)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.target.eval()
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=float(learning_rate))
        self.obs_rms = RunningMeanStd((self.input_dim,))
        self.reward_rms = RunningMeanStd(())
        self.bonus_clip = float(bonus_clip)
        self.last_raw_bonus = 0.0
        self.last_normalized_bonus = 0.0
        self.bonus_mean = 0.0
        self._bonus_count = 0
        self.last_loss = 0.0

    def _flat_numpy(self, observations: np.ndarray) -> np.ndarray:
        arr = np.asarray(observations, dtype=np.float32)
        if arr.size == self.input_dim:
            return arr.reshape(1, self.input_dim)
        return arr.reshape(arr.shape[0], self.input_dim)

    def _normalize_numpy(self, observations: np.ndarray, update: bool) -> np.ndarray:
        flat = self._flat_numpy(observations)
        if update:
            self.obs_rms.update(flat)
        normalized = (flat - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(normalized, -5.0, 5.0).astype(np.float32)

    def _error(self, normalized: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_features = self.target(normalized)
        predicted_features = self.predictor(normalized)
        return F.mse_loss(predicted_features, target_features, reduction="none").mean(dim=-1)

    @torch.no_grad()
    def bonus(self, next_observation: np.ndarray, update_stats: bool = True) -> float:
        normalized = self._normalize_numpy(next_observation, update=update_stats)
        obs_t = torch.as_tensor(normalized, device=self.device)
        error = self._error(obs_t)
        raw = float(error.mean().item())
        if update_stats:
            self.reward_rms.update(np.asarray([raw], dtype=np.float64))
        scaled = raw / float(np.sqrt(self.reward_rms.var + 1e-8))
        scaled = float(np.clip(scaled, 0.0, self.bonus_clip))
        self.last_raw_bonus = raw
        self.last_normalized_bonus = scaled
        self._bonus_count += 1
        self.bonus_mean += (scaled - self.bonus_mean) / min(self._bonus_count, 1_000)
        return scaled

    def update_predictor(self, observations: np.ndarray, mask: np.ndarray | None = None) -> float:
        # Observation moments are updated once on collection in `bonus()`.
        # Updating them again from replay batches would overweight frequently
        # sampled/PER transitions and make normalization depend on replay
        # policy rather than the environment visitation distribution.
        normalized = self._normalize_numpy(observations, update=False)
        obs_t = torch.as_tensor(normalized, device=self.device)
        errors = self._error(obs_t)
        if mask is not None:
            mask_t = torch.as_tensor(np.asarray(mask).reshape(-1), dtype=torch.float32, device=self.device)
            errors = errors * mask_t
            loss = errors.sum() / mask_t.sum().clamp(min=1.0)
        else:
            loss = errors.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 10.0)
        self.optimizer.step()
        self.last_loss = float(loss.detach().item())
        return self.last_loss

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "module": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "reward_rms": self.reward_rms.state_dict(),
            "bonus_mean": self.bonus_mean,
            "bonus_count": self._bonus_count,
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        self.load_state_dict(state["module"])
        if state.get("optimizer"):
            self.optimizer.load_state_dict(state["optimizer"])
        self.obs_rms.load_state_dict(state["obs_rms"])
        self.reward_rms.load_state_dict(state["reward_rms"])
        self.bonus_mean = float(state.get("bonus_mean", 0.0))
        self._bonus_count = int(state.get("bonus_count", 0))
