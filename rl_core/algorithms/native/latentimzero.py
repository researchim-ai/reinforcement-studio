"""LatentImZero v7: ResearchImZero with an isolated uncertainty sidecar.

The ResearchImZero implementation is the complete primary algorithm.  The
optional probe is trained only from detached action-token hidden features and
can only add a bonus to rewards inside tree search.  It never changes replay
rewards, TD targets, the core loss, or the core optimizer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_core.algorithms.native.researchimzero import (
    DEFAULT_HYPERPARAMS as RESEARCH_DEFAULT_HYPERPARAMS,
    NativeResearchImZero,
    _signed_hyperbolic,
)


LATENTIMZERO_CHECKPOINT_VERSION = 7

DEFAULT_HYPERPARAMS = {
    **RESEARCH_DEFAULT_HYPERPARAMS,
    "force_research_mode": 0,
    "uncertainty_enabled": 1,
    "uncertainty_bonus_coef": 0.05,
    "uncertainty_scale": 1.0,
    "uncertainty_members": 2,
    "uncertainty_aux_learning_rate": 1e-4,
    "uncertainty_warmup_steps": 5_000,
    "uncertainty_ramp_steps": 20_000,
    "uncertainty_anneal_start": 100_000,
    "uncertainty_anneal_end": 250_000,
    "uncertainty_bonus_clip": 1.0,
    "uncertainty_ema_decay": 0.99,
    "uncertainty_sparse_threshold": 0.5,
}


class _SearchUncertaintyProbe(nn.Module):
    """Bootstrap reward heads over detached Research action-token features."""

    def __init__(self, feature_dim: int, members: int = 2) -> None:
        super().__init__()
        self.members = max(2, int(members))
        hidden_dim = max(16, feature_dim // 2)
        self.reward_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.members)
            ],
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(features.detach()) for head in self.reward_heads], dim=-1)

    def bootstrap_loss(
        self,
        features: torch.Tensor,
        reward: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        predictions = self(features.detach())
        terms: list[torch.Tensor] = []
        valid = valid.bool()
        for member in range(self.members):
            bootstrap = (torch.rand_like(reward) < 0.5) & valid
            if not bootstrap.any():
                bootstrap = valid
            if bootstrap.any():
                terms.append(F.mse_loss(predictions[..., member][bootstrap], reward[bootstrap]))
        return torch.stack(terms).mean() if terms else predictions.sum() * 0.0


class NativeLatentImZero(NativeResearchImZero):
    """ResearchImZero core plus a conservative search-only uncertainty probe."""

    COMPOSITE_FAMILY = "latentimzero"

    def __init__(
        self,
        env: gym.Env,
        hyperparams: dict[str, Any],
        seed: int | None,
        device: str,
    ) -> None:
        super().__init__(env, hyperparams, seed, device)
        self.force_research_mode = bool(int(hyperparams.get("force_research_mode", 0)))
        self.uncertainty_enabled = bool(int(hyperparams.get("uncertainty_enabled", 1)))
        self.uncertainty_bonus_coef = max(
            0.0, float(hyperparams.get("uncertainty_bonus_coef", 0.05)),
        )
        self.uncertainty_scale = max(0.0, float(hyperparams.get("uncertainty_scale", 1.0)))
        self.uncertainty_members = max(2, int(hyperparams.get("uncertainty_members", 2)))
        self.uncertainty_aux_learning_rate = float(
            hyperparams.get("uncertainty_aux_learning_rate", 1e-4),
        )
        self.uncertainty_warmup_steps = max(
            0, int(hyperparams.get("uncertainty_warmup_steps", 5_000)),
        )
        self.uncertainty_ramp_steps = max(
            1, int(hyperparams.get("uncertainty_ramp_steps", 20_000)),
        )
        self.uncertainty_anneal_start = max(
            0, int(hyperparams.get("uncertainty_anneal_start", 100_000)),
        )
        self.uncertainty_anneal_end = max(
            self.uncertainty_anneal_start + 1,
            int(hyperparams.get("uncertainty_anneal_end", 250_000)),
        )
        self.uncertainty_bonus_clip = min(
            1.0, max(0.0, float(hyperparams.get("uncertainty_bonus_clip", 1.0))),
        )
        self.uncertainty_ema_decay = min(
            0.9999, max(0.0, float(hyperparams.get("uncertainty_ema_decay", 0.99))),
        )
        self.uncertainty_sparse_threshold = min(
            0.9999, max(0.0, float(hyperparams.get("uncertainty_sparse_threshold", 0.5))),
        )
        self._uncertainty_ema_initialized = False
        self._uncertainty_probe_loss_ema = 0.0
        self._uncertainty_disagreement_ema = 0.0
        self._uncertainty_zero_reward_fraction_ema = 0.0
        self._last_uncertainty_normalized_mean = 0.0
        self._last_uncertainty_bonus_mean = 0.0
        self._last_uncertainty_bonus_max = 0.0

        # Sidecar construction must not advance RNG relative to ResearchImZero.
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all()
            if self._is_cuda and torch.cuda.is_available()
            else None
        )
        try:
            self.uncertainty_probe = _SearchUncertaintyProbe(
                self.embed_dim, self.uncertainty_members,
            ).to(device)
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        self.uncertainty_optimizer = torch.optim.Adam(
            self.uncertainty_probe.parameters(),
            lr=self.uncertainty_aux_learning_rate,
        )

    @property
    def uncertainty_runtime_scale(self) -> float:
        if self.force_research_mode or not self.uncertainty_enabled:
            return 0.0
        step = self._num_timesteps
        if step <= self.uncertainty_warmup_steps:
            return 0.0
        ramp = min(
            1.0,
            (step - self.uncertainty_warmup_steps) / self.uncertainty_ramp_steps,
        )
        if step <= self.uncertainty_anneal_start:
            anneal = 1.0
        elif step >= self.uncertainty_anneal_end:
            anneal = 0.0
        else:
            anneal = 1.0 - (
                (step - self.uncertainty_anneal_start)
                / (self.uncertainty_anneal_end - self.uncertainty_anneal_start)
            )
        return self.uncertainty_scale * ramp * anneal

    @property
    def uncertainty_calibration(self) -> float:
        if not self._uncertainty_ema_initialized:
            return 0.0
        return 1.0 / (1.0 + max(0.0, self._uncertainty_probe_loss_ema))

    @property
    def uncertainty_sparsity_gate(self) -> float:
        if not self._uncertainty_ema_initialized:
            return 0.0
        fraction = self._uncertainty_zero_reward_fraction_ema
        threshold = self.uncertainty_sparse_threshold
        return min(1.0, max(0.0, (fraction - threshold) / (1.0 - threshold)))

    def _adjust_search_reward(
        self,
        action_hidden: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.uncertainty_runtime_scale
        if scale == 0.0:
            self._last_uncertainty_normalized_mean = 0.0
            self._last_uncertainty_bonus_mean = 0.0
            self._last_uncertainty_bonus_max = 0.0
            return reward
        predictions = self.uncertainty_probe(action_hidden.detach().float())
        disagreement = predictions.std(dim=-1, unbiased=False)
        ratio = disagreement / max(self._uncertainty_disagreement_ema, 1e-6)
        normalized = (ratio / (1.0 + ratio)).clamp(0.0, self.uncertainty_bonus_clip)
        bonus = (
            self.uncertainty_bonus_coef
            * scale
            * self.uncertainty_calibration
            * self.uncertainty_sparsity_gate
            * normalized
        )
        self._last_uncertainty_normalized_mean = float(normalized.mean().item())
        self._last_uncertainty_bonus_mean = float(bonus.mean().item())
        self._last_uncertainty_bonus_max = float(bonus.max().item())
        return reward + bonus

    def _after_core_train_step(
        self,
        batch: dict[str, Any],
        hidden: torch.Tensor,
        obs0_pos: torch.Tensor,
    ) -> dict[str, float]:
        if self.force_research_mode or not self.uncertainty_enabled:
            return {}
        batch_size = hidden.shape[0]
        batch_idx = torch.arange(batch_size, device=self.device)
        features = torch.stack(
            [
                hidden[batch_idx, obs0_pos + 2 * step + 1]
                for step in range(self.unroll_steps)
            ],
            dim=1,
        ).detach().float()
        reward = _signed_hyperbolic(
            torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device),
        )
        valid = torch.as_tensor(batch["mask"], dtype=torch.bool, device=self.device)

        self.uncertainty_optimizer.zero_grad(set_to_none=True)
        loss = self.uncertainty_probe.bootstrap_loss(features, reward, valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.uncertainty_probe.parameters(), self.max_grad_norm)
        self.uncertainty_optimizer.step()
        with torch.no_grad():
            disagreement = self.uncertainty_probe(features).std(dim=-1, unbiased=False)
            mean_disagreement = (
                disagreement[valid].mean() if valid.any() else disagreement.new_zeros(())
            )
            raw_reward = torch.as_tensor(
                batch["reward"], dtype=torch.float32, device=self.device,
            )
            zero_fraction = (
                (raw_reward[valid].abs() <= 1e-8).float().mean()
                if valid.any()
                else raw_reward.new_zeros(())
            )
            observed_loss = float(loss.detach().item())
            observed_disagreement = float(mean_disagreement.item())
            observed_zero_fraction = float(zero_fraction.item())
            if not self._uncertainty_ema_initialized:
                self._uncertainty_probe_loss_ema = observed_loss
                self._uncertainty_disagreement_ema = observed_disagreement
                self._uncertainty_zero_reward_fraction_ema = observed_zero_fraction
                self._uncertainty_ema_initialized = True
            else:
                decay = self.uncertainty_ema_decay
                self._uncertainty_probe_loss_ema = (
                    decay * self._uncertainty_probe_loss_ema
                    + (1.0 - decay) * observed_loss
                )
                self._uncertainty_disagreement_ema = (
                    decay * self._uncertainty_disagreement_ema
                    + (1.0 - decay) * observed_disagreement
                )
                self._uncertainty_zero_reward_fraction_ema = (
                    decay * self._uncertainty_zero_reward_fraction_ema
                    + (1.0 - decay) * observed_zero_fraction
                )
        return {
            "uncertainty_aux_loss": float(loss.detach().item()),
            "uncertainty_disagreement": float(mean_disagreement.item()),
            "uncertainty_schedule_scale": float(self.uncertainty_runtime_scale),
            "uncertainty_calibration": float(self.uncertainty_calibration),
            "uncertainty_sparsity_gate": float(self.uncertainty_sparsity_gate),
            "uncertainty_normalized_disagreement": self._last_uncertainty_normalized_mean,
            "uncertainty_effective_bonus_mean": self._last_uncertainty_bonus_mean,
            "uncertainty_effective_bonus_max": self._last_uncertainty_bonus_max,
            "uncertainty_zero_reward_fraction": self._uncertainty_zero_reward_fraction_ema,
        }

    def save(self, path: Path) -> None:
        super().save(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update(
            {
                "checkpoint_version": LATENTIMZERO_CHECKPOINT_VERSION,
                "algorithm": "latentimzero",
                "architecture": "researchimzero_core+search_uncertainty_probe",
                "optimizer_state_dict": self.optimizer.state_dict(),
                "grad_scaler_state_dict": self._grad_scaler.state_dict(),
                "uncertainty_probe_state_dict": self.uncertainty_probe.state_dict(),
                "uncertainty_optimizer_state_dict": self.uncertainty_optimizer.state_dict(),
                "uncertainty_ema_initialized": self._uncertainty_ema_initialized,
                "uncertainty_probe_loss_ema": self._uncertainty_probe_loss_ema,
                "uncertainty_disagreement_ema": self._uncertainty_disagreement_ema,
                "uncertainty_zero_reward_fraction_ema": self._uncertainty_zero_reward_fraction_ema,
                "last_uncertainty_normalized_mean": self._last_uncertainty_normalized_mean,
                "last_uncertainty_bonus_mean": self._last_uncertainty_bonus_mean,
                "last_uncertainty_bonus_max": self._last_uncertainty_bonus_max,
                "num_timesteps": self._num_timesteps,
                "total_timesteps_estimate": self._total_timesteps_estimate,
                "grad_step_count": self._grad_step_count,
                "model_error_ema": self._model_error_ema,
                "last_search_num_simulations": self._last_search_num_simulations,
            },
        )
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: Path,
        env: gym.Env,
        device: str = "cpu",
    ) -> "NativeLatentImZero":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        version = payload.get("checkpoint_version")
        if version == 6 or "world_model" in payload or "actor" in payload:
            raise ValueError(
                "LatentImZero checkpoint v6 is incompatible with v7: v7 uses the "
                "ResearchImZero core and cannot safely migrate the deleted stochastic "
                "world-model/actor/critic architecture.",
            )
        if version != LATENTIMZERO_CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported LatentImZero checkpoint version {version!r}; expected v7.",
            )

        algo = super().load(path, env, device)
        assert isinstance(algo, cls)
        algo.uncertainty_probe.load_state_dict(payload["uncertainty_probe_state_dict"])
        if "optimizer_state_dict" in payload:
            algo.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "grad_scaler_state_dict" in payload:
            algo._grad_scaler.load_state_dict(payload["grad_scaler_state_dict"])
        if "uncertainty_optimizer_state_dict" in payload:
            algo.uncertainty_optimizer.load_state_dict(
                payload["uncertainty_optimizer_state_dict"],
            )
        algo._uncertainty_ema_initialized = bool(
            payload.get("uncertainty_ema_initialized", False),
        )
        algo._uncertainty_probe_loss_ema = float(
            payload.get("uncertainty_probe_loss_ema", 0.0),
        )
        algo._uncertainty_disagreement_ema = float(
            payload.get("uncertainty_disagreement_ema", 0.0),
        )
        algo._uncertainty_zero_reward_fraction_ema = float(
            payload.get("uncertainty_zero_reward_fraction_ema", 0.0),
        )
        algo._last_uncertainty_normalized_mean = float(
            payload.get("last_uncertainty_normalized_mean", 0.0),
        )
        algo._last_uncertainty_bonus_mean = float(
            payload.get("last_uncertainty_bonus_mean", 0.0),
        )
        algo._last_uncertainty_bonus_max = float(
            payload.get("last_uncertainty_bonus_max", 0.0),
        )
        algo._num_timesteps = int(payload.get("num_timesteps", 0))
        algo._total_timesteps_estimate = int(payload.get("total_timesteps_estimate", 1))
        algo._grad_step_count = int(payload.get("grad_step_count", 0))
        algo._model_error_ema = float(payload.get("model_error_ema", algo._model_error_ema))
        algo._last_search_num_simulations = int(
            payload.get("last_search_num_simulations", algo._last_search_num_simulations),
        )
        return algo
