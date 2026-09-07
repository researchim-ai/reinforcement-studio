"""LatentImZero v10.1: search as an optional learned computational action.

The ResearchImZero implementation remains the primary algorithm.  Its
uncertainty probe is trained only from detached hidden features and never
changes replay rewards or receives gradients from the core optimizer.  The
probe can conservatively alter search-only rewards and reanalysis mixing.
An isolated value-of-computation sidecar controls a nested max-schedule
planner only after stage-specific audit gates pass. Marginal labels compare
checkpoint representatives with one shared EMA one-step evaluator, never
visit-scaled completed-Q. Path consistency is the sole additional loss that
deliberately updates the core value model.
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_core.algorithms.native.researchimzero import (
    DEFAULT_HYPERPARAMS as RESEARCH_DEFAULT_HYPERPARAMS,
    NativeResearchImZero,
    _HalvingSchedule,
    _MinMaxStats,
    _SearchNode,
    _backpropagate,
    _cache_positions,
    _select_action,
    _sequential_halving,
    _signed_hyperbolic,
    _transformed_completed_qs,
)


LATENTIMZERO_CHECKPOINT_VERSION = 11
VOC_LABEL_SCHEMA_VERSION = 3

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
    "adaptive_train_steps": 1,
    "adaptive_train_steps_min": 2,
    "adaptive_train_steps_max": 4,
    "replay_success_fraction": 0.15,
    "replay_success_top_quantile": 0.25,
    "adaptive_closed_loop": 1,
    "adaptive_closed_loop_max_horizon": 5,
    "path_consistency_coef": 0.1,
    "uncertainty_sve_beta": 2.0,
    "uncertainty_sve_min_mix": 0.0,
    "uncertainty_extra_simulations_max": 4,
    "learning_progress_priority_weight": 0.1,
    "voc_enabled": 1,
    "voc_budgets": [0, 4, 16, 32],
    "voc_learning_rate": 1e-4,
    "voc_replay_size": 20_000,
    "voc_batch_size": 64,
    "voc_min_labels": 128,
    "voc_threshold": 0.5,
    "voc_compute_cost_per_expansion": 0.001,
    "voc_label_margin": 0.0,
    "voc_return_scale": 1.0,
    "voc_label_sample_fraction": 0.25,
    "voc_uncertainty_filter": 0.8,
    "voc_random_audit_fraction": 0.05,
    "voc_targeted_audit_fraction": 0.05,
    "voc_min_audits": 64,
    "voc_safety_patience": 20,
    "voc_max_actor_regret": 0.1,
    "voc_max_policy_divergence": 0.2,
    "voc_max_brier": 0.15,
    "voc_min_stop_precision": 0.8,
    "voc_min_stop_audits": 64,
    "voc_min_class_samples": 64,
    "voc_min_class_fraction": 0.05,
    "voc_premature_regret_epsilon": [0.01, 0.02, 0.05],
    "voc_max_premature_exceedance": [0.01, 0.025, 0.05],
    "voc_max_p95_premature_regret": [0.02, 0.05, 0.1],
    "voc_max_mean_premature_regret": [0.005, 0.015, 0.03],
    "voc_catastrophic_multiplier": 3.0,
    "voc_health_check_steps": 100_000,
    "voc_health_desync_window": 1_000,
    "voc_health_desync_ratio": 0.7,
    "voc_health_actual_max_ratio": 0.95,
    "voc_health_unsafe_actor_fraction": 0.9,
}


class _SearchUncertaintyProbe(nn.Module):
    """Matched reward/value bootstrap heads over detached Research features."""

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
        self.value_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.members)
            ],
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        detached = features.detach()
        return self.reward(detached), self.value(detached)

    def reward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(features.detach()) for head in self.reward_heads], dim=-1)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(features.detach()) for head in self.value_heads], dim=-1)

    def bootstrap_loss(
        self,
        reward_features: torch.Tensor,
        value_features: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        reward_predictions = self.reward(reward_features)
        value_predictions = self.value(value_features)
        terms: list[torch.Tensor] = []
        valid = valid.bool()
        for member in range(self.members):
            bootstrap = (torch.rand_like(reward) < 0.5) & valid
            if not bootstrap.any():
                bootstrap = valid
            if bootstrap.any():
                terms.append(
                    F.mse_loss(reward_predictions[..., member][bootstrap], reward[bootstrap])
                    + F.mse_loss(value_predictions[..., member][bootstrap], value[bootstrap])
                )
        return (
            torch.stack(terms).mean()
            if terms
            else (reward_predictions.sum() + value_predictions.sum()) * 0.0
        )


class _VoCClassifier(nn.Module):
    """Detached stage-conditioned value-of-computation classifier."""

    def __init__(self, embed_dim: int, scalar_dim: int = 7) -> None:
        super().__init__()
        hidden = max(32, embed_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + scalar_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, root_hidden: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([root_hidden.detach(), scalars.detach()], dim=-1)).squeeze(-1)


class NativeLatentImZero(NativeResearchImZero):
    """ResearchImZero core plus conservative confidence-aware extensions."""

    COMPOSITE_FAMILY = "latentimzero"

    def __init__(
        self,
        env: gym.Env,
        hyperparams: dict[str, Any],
        seed: int | None,
        device: str,
    ) -> None:
        force_research_mode = bool(int(hyperparams.get("force_research_mode", 0)))
        core_hyperparams = (
            {
                **hyperparams,
                "adaptive_train_steps": 0,
                "replay_success_fraction": 0.0,
                "adaptive_closed_loop": 0,
                "path_consistency_coef": 0.0,
                "learning_progress_priority_weight": 0.0,
            }
            if force_research_mode
            else hyperparams
        )
        super().__init__(env, core_hyperparams, seed, device)
        self.force_research_mode = force_research_mode
        self.uncertainty_enabled = bool(int(hyperparams.get("uncertainty_enabled", 1)))
        self.uncertainty_bonus_coef = max(
            0.0, float(hyperparams.get("uncertainty_bonus_coef", 0.05)),
        )
        self.uncertainty_scale = max(0.0, float(hyperparams.get("uncertainty_scale", 1.0)))
        self.uncertainty_members = max(2, int(hyperparams.get("uncertainty_members", 2)))
        self.uncertainty_aux_learning_rate = float(
            hyperparams.get("uncertainty_aux_learning_rate", 1e-4),
        )
        self.uncertainty_sve_beta = max(
            0.0, float(hyperparams.get("uncertainty_sve_beta", 2.0)),
        )
        self.uncertainty_sve_min_mix = min(
            1.0, max(0.0, float(hyperparams.get("uncertainty_sve_min_mix", 0.0))),
        )
        self.uncertainty_extra_simulations_max = max(
            0, int(hyperparams.get("uncertainty_extra_simulations_max", 4)),
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
        self._uncertainty_ema_initialized = False
        self._uncertainty_probe_loss_ema = 0.0
        self._uncertainty_disagreement_ema = 0.0
        self._uncertainty_surprise_ema = 0.0
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
        self.voc_enabled = bool(int(hyperparams.get("voc_enabled", 1)))
        self.voc_budgets = self._sanitize_voc_budgets(
            hyperparams.get("voc_budgets", [0, 4, 16, 32]),
            self.num_simulations,
        )
        self.voc_stage_ids = self._make_voc_stage_ids(self.voc_budgets)
        self.voc_edge_ids = [
            f"{self.voc_stage_ids[index]}_to_{self.voc_stage_ids[index + 1]}"
            for index in range(len(self.voc_stage_ids) - 1)
        ]
        self.voc_threshold = min(1.0, max(0.0, float(hyperparams.get("voc_threshold", 0.5))))
        self.voc_compute_cost_per_expansion = max(
            0.0, float(hyperparams.get("voc_compute_cost_per_expansion", 0.001)),
        )
        self.voc_label_margin = float(hyperparams.get("voc_label_margin", 0.0))
        self.voc_return_scale = max(
            1e-6, float(hyperparams.get("voc_return_scale", 1.0)),
        )
        self.voc_label_sample_fraction = min(
            1.0,
            max(0.0, float(hyperparams.get("voc_label_sample_fraction", 0.25))),
        )
        self.voc_uncertainty_filter = min(
            1.0, max(0.0, float(hyperparams.get("voc_uncertainty_filter", 0.8))),
        )
        self.voc_batch_size = max(2, int(hyperparams.get("voc_batch_size", 64)))
        self.voc_min_labels = max(1, int(hyperparams.get("voc_min_labels", 128)))
        self.voc_random_audit_fraction = min(
            1.0, max(0.0, float(hyperparams.get("voc_random_audit_fraction", 0.05))),
        )
        self.voc_targeted_audit_fraction = min(
            1.0 - self.voc_random_audit_fraction,
            max(0.0, float(hyperparams.get("voc_targeted_audit_fraction", 0.05))),
        )
        self.voc_min_audits = max(1, int(hyperparams.get("voc_min_audits", 64)))
        self.voc_safety_patience = max(1, int(hyperparams.get("voc_safety_patience", 20)))
        self.voc_max_actor_regret = max(
            0.0, float(hyperparams.get("voc_max_actor_regret", 0.1)),
        )
        self.voc_max_policy_divergence = max(
            0.0, float(hyperparams.get("voc_max_policy_divergence", 0.2)),
        )
        self.voc_max_brier = max(0.0, float(hyperparams.get("voc_max_brier", 0.15)))
        self.voc_min_stop_precision = min(
            1.0, max(0.0, float(hyperparams.get("voc_min_stop_precision", 0.8))),
        )
        num_edges = len(self.voc_budgets) - 1
        self.voc_min_stop_audits = max(
            1, int(hyperparams.get("voc_min_stop_audits", self.voc_min_audits)),
        )
        self.voc_min_class_fraction = min(
            0.5, max(0.0, float(hyperparams.get("voc_min_class_fraction", 0.05))),
        )
        self.voc_min_class_samples = max(
            1, int(hyperparams.get("voc_min_class_samples", 64)),
        )
        self.voc_premature_regret_epsilon = self._stage_values(
            hyperparams.get("voc_premature_regret_epsilon", [0.01, 0.02, 0.05]),
            num_edges,
            [0.01, 0.02, 0.05],
        )
        self.voc_max_premature_exceedance = self._stage_values(
            hyperparams.get("voc_max_premature_exceedance", [0.01, 0.025, 0.05]),
            num_edges,
            [0.01, 0.025, 0.05],
        )
        self.voc_max_p95_premature_regret = self._stage_values(
            hyperparams.get("voc_max_p95_premature_regret", [0.02, 0.05, 0.1]),
            num_edges,
            [0.02, 0.05, 0.1],
        )
        self.voc_max_mean_premature_regret = self._stage_values(
            hyperparams.get("voc_max_mean_premature_regret", [0.005, 0.015, 0.03]),
            num_edges,
            [0.005, 0.015, 0.03],
        )
        self.voc_catastrophic_multiplier = max(
            1.0, float(hyperparams.get("voc_catastrophic_multiplier", 3.0)),
        )
        self.voc_health_check_steps = max(
            1, int(hyperparams.get("voc_health_check_steps", 100_000)),
        )
        self.voc_health_desync_window = max(
            1, int(hyperparams.get("voc_health_desync_window", 1_000)),
        )
        self.voc_health_desync_ratio = min(
            1.0, max(0.0, float(hyperparams.get("voc_health_desync_ratio", 0.7))),
        )
        self.voc_health_actual_max_ratio = min(
            1.0,
            max(0.0, float(hyperparams.get("voc_health_actual_max_ratio", 0.95))),
        )
        self.voc_health_unsafe_actor_fraction = min(
            1.0,
            max(0.0, float(hyperparams.get("voc_health_unsafe_actor_fraction", 0.9))),
        )
        self._voc_rng = np.random.default_rng(
            (0 if seed is None else int(seed)) ^ 0x51DEC0DE,
        )

        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all()
            if self._is_cuda and torch.cuda.is_available()
            else None
        )
        try:
            self.voc_classifier = _VoCClassifier(self.embed_dim).to(device)
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        self.voc_optimizer = torch.optim.Adam(
            self.voc_classifier.parameters(),
            lr=float(hyperparams.get("voc_learning_rate", 1e-4)),
        )
        self._voc_replay: deque[dict[str, Any]] = deque(
            maxlen=max(1, int(hyperparams.get("voc_replay_size", 20_000))),
        )
        self._voc_audits: deque[dict[str, float]] = deque(maxlen=1_000)
        self._voc_shadow_mode = True
        self._voc_minimum_budget_index = len(self.voc_budgets) - 1
        self._voc_safety_streaks = [0] * max(0, len(self.voc_budgets) - 1)
        self._voc_update_count = 0
        self._last_voc_loss = 0.0
        self._last_voc_accuracy = 0.0
        self._last_voc_brier = 1.0
        self._last_voc_stop_precision = 0.0
        self._last_voc_ece = 1.0
        self._executed_budget_history: deque[int] = deque(maxlen=1_000)
        self._model_expansion_history: deque[int] = deque(maxlen=1_000)
        self._predicted_budget_history: deque[int] = deque(
            maxlen=self.voc_health_desync_window,
        )
        self._requested_budget_history: deque[int] = deque(
            maxlen=self.voc_health_desync_window,
        )
        self._permitted_budget_history: deque[int] = deque(
            maxlen=self.voc_health_desync_window,
        )
        self._executed_search_history: deque[int] = deque(
            maxlen=self.voc_health_desync_window,
        )
        self._last_voc_search_metrics: dict[str, float] = {}

    @staticmethod
    def _sanitize_voc_budgets(raw: Any, maximum: int) -> list[int]:
        values = [0]
        if isinstance(raw, (list, tuple)):
            for value in raw:
                try:
                    values.append(max(0, min(maximum, int(value))))
                except (TypeError, ValueError):
                    continue
        values.append(maximum)
        return sorted(set(values))

    @staticmethod
    def _make_voc_stage_ids(budgets: list[int]) -> list[str]:
        if budgets == [0, 4, 16, 32]:
            return ["actor", "small", "medium", "max"]
        return [f"stage_{index}" for index in range(len(budgets))]

    @staticmethod
    def _stage_values(raw: Any, count: int, defaults: list[float]) -> list[float]:
        values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        fallback = list(defaults)
        while len(fallback) < count:
            fallback.append(fallback[-1])
        output: list[float] = []
        for index in range(count):
            value = values[index] if index < len(values) else fallback[index]
            output.append(max(0.0, float(value)))
        return output

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
    def uncertainty_reliability_scale(self) -> float:
        """Persistent post-warmup scale for SVE trust and label filtering."""
        if (
            self.force_research_mode
            or not self.uncertainty_enabled
            or not self._uncertainty_ema_initialized
        ):
            return 0.0
        ramp = min(
            1.0,
            max(
                0.0,
                (self._num_timesteps - self.uncertainty_warmup_steps)
                / self.uncertainty_ramp_steps,
            ),
        )
        return self.uncertainty_scale * ramp

    @property
    def uncertainty_calibration(self) -> float:
        if not self._uncertainty_ema_initialized:
            return 0.0
        surprise = max(0.0, self._uncertainty_surprise_ema)
        loss_scale = math.sqrt(max(0.0, self._uncertainty_probe_loss_ema))
        signal = 0.5 * (surprise + loss_scale)
        return signal / (1.0 + signal)

    def _effective_train_steps_per_iter(self) -> int:
        if self.force_research_mode or not bool(int(self.hyperparams.get("adaptive_train_steps", 1))):
            return self.train_steps_per_iter
        minimum = max(1, int(self.hyperparams.get("adaptive_train_steps_min", 2)))
        maximum = max(minimum, int(self.hyperparams.get("adaptive_train_steps_max", 4)))
        normalized_error = (
            self._model_error_ema - self.search_model_error_low
        ) / (self.search_model_error_high - self.search_model_error_low)
        pressure = min(1.0, max(0.0, normalized_error))
        return int(round(minimum + pressure * (maximum - minimum)))

    def _effective_closed_loop_horizon(self) -> int:
        return (
            self.closed_loop_horizon
            if self.force_research_mode
            else super()._effective_closed_loop_horizon()
        )

    def _update_closed_loop_error_ema(self, errors: list[torch.Tensor]) -> None:
        if not self.force_research_mode:
            super()._update_closed_loop_error_ema(errors)

    def _return_predictions(
        self,
        action_hidden: torch.Tensor,
        next_hidden: torch.Tensor,
    ) -> torch.Tensor:
        reward_predictions = self.uncertainty_probe.reward(action_hidden.detach().float())
        value_predictions = self.uncertainty_probe.value(next_hidden.detach().float())
        return reward_predictions + self.gamma * value_predictions

    def _adjust_search_predictions(
        self,
        action_hidden: torch.Tensor,
        next_hidden: torch.Tensor,
        reward: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.uncertainty_runtime_scale
        if scale == 0.0:
            self._last_uncertainty_normalized_mean = 0.0
            self._last_uncertainty_bonus_mean = 0.0
            self._last_uncertainty_bonus_max = 0.0
            return reward, next_value
        predictions = self._return_predictions(action_hidden, next_hidden)
        disagreement = predictions.std(dim=-1, unbiased=False)
        ratio = disagreement / max(self._uncertainty_disagreement_ema, 1e-6)
        normalized = (ratio / (1.0 + ratio)).clamp(0.0, self.uncertainty_bonus_clip)
        bonus = (
            self.uncertainty_bonus_coef
            * scale
            * self.uncertainty_calibration
            * normalized
        )
        self._last_uncertainty_normalized_mean = float(normalized.mean().item())
        self._last_uncertainty_bonus_mean = float(bonus.mean().item())
        self._last_uncertainty_bonus_max = float(bonus.max().item())
        return reward + bonus, next_value

    def _effective_reanalyze_value_mix(
        self,
        action_hidden: torch.Tensor,
        next_hidden: torch.Tensor,
        base_mix: float,
    ) -> torch.Tensor:
        if (
            self.force_research_mode
            or not self.uncertainty_enabled
            or not self._uncertainty_ema_initialized
            or self.uncertainty_reliability_scale == 0.0
        ):
            return torch.as_tensor(base_mix, dtype=torch.float32, device=self.device)
        predictions = self._return_predictions(action_hidden, next_hidden)
        disagreement = predictions.std(dim=-1, unbiased=False)
        ratio = disagreement / max(self._uncertainty_disagreement_ema, 1e-6)
        uncertainty = (ratio / (1.0 + ratio)).clamp(0.0, 1.0)
        reliability_fraction = min(
            1.0,
            self.uncertainty_reliability_scale
            / max(self.uncertainty_scale, 1e-8),
        )
        mix = base_mix * torch.exp(
            -self.uncertainty_sve_beta * reliability_fraction * uncertainty,
        )
        return mix.clamp(min=self.uncertainty_sve_min_mix, max=base_mix)

    def _effective_search_simulations(
        self,
        base_simulations: int,
        policy_logits: torch.Tensor | None,
        policy_std: torch.Tensor | None,
    ) -> int:
        if (
            self.force_research_mode
            or self.voc_enabled
            or self.uncertainty_runtime_scale == 0.0
        ):
            return base_simulations
        if policy_logits is not None:
            probabilities = F.softmax(policy_logits.float(), dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
            ambiguity = float((entropy / math.log(policy_logits.shape[-1])).mean().item())
        elif policy_std is not None:
            ambiguity = float((policy_std.float() / (1.0 + policy_std.float())).mean().item())
        else:
            ambiguity = 0.0
        schedule = min(
            1.0,
            self.uncertainty_runtime_scale / max(self.uncertainty_scale, 1e-8),
        )
        extra = round(self.uncertainty_extra_simulations_max * schedule * ambiguity)
        return min(self.num_simulations, base_simulations + max(0, int(extra)))

    def _continuous_anytime_candidates(
        self, mean: np.ndarray, std: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        candidates, priors = self._sample_continuous_candidates(mean, std)
        candidates[0] = np.clip(mean, self._action_space.low, self._action_space.high).astype(
            np.float32,
        )
        variance = np.clip(std, 1e-6, None) ** 2
        priors[0] = -0.5 * np.log(2.0 * math.pi * variance).sum()
        return candidates, priors

    def _voc_stage_features(
        self,
        root_hidden: torch.Tensor,
        actor_entropy: float,
        policy: np.ndarray,
        budget: int,
        stage: int,
        maximum_budget: int,
        normalized_epistemic: float | None = None,
    ) -> tuple[torch.Tensor, float]:
        sorted_policy = np.sort(np.asarray(policy, dtype=np.float64))
        margin = float(sorted_policy[-1] - sorted_policy[-2]) if len(sorted_policy) > 1 else 1.0
        if normalized_epistemic is None:
            with torch.inference_mode():
                value_members = self.uncertainty_probe.value(root_hidden[None].float())[0]
                root_epistemic = float(value_members.std(unbiased=False).item())
            normalized_epistemic = root_epistemic / max(
                self._uncertainty_disagreement_ema, 1e-6,
            )
            normalized_epistemic = normalized_epistemic / (1.0 + normalized_epistemic)
        normalized_model_error = (
            self._model_error_ema - self.search_model_error_low
        ) / (self.search_model_error_high - self.search_model_error_low)
        scalars = torch.tensor(
            [
                actor_entropy,
                margin,
                normalized_epistemic,
                min(1.0, max(0.0, normalized_model_error)),
                budget / max(1, maximum_budget),
                stage / max(1, len(self.voc_budgets) - 2),
                self.uncertainty_calibration,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return scalars, float(normalized_epistemic)

    @staticmethod
    def _common_voc_labels(
        common_q: list[float],
        budgets: list[int],
        compute_cost: float,
        margin: float,
        return_scale: float,
    ) -> list[tuple[float, float]]:
        """Labels from one shared EMA one-step evaluator scale.

        ``compute_cost`` and ``margin`` are normalized-return quantities;
        multiplying them by ``return_scale`` makes them comparable to raw
        one-step Q gains.
        """
        labels: list[tuple[float, float]] = []
        for stage in range(len(common_q) - 1):
            gain = float(common_q[stage + 1] - common_q[stage])
            normalized_cost = (
                compute_cost * (budgets[stage + 1] - budgets[stage]) + margin
            )
            cost = return_scale * normalized_cost
            labels.append((gain, float(gain > cost)))
        return labels

    def _common_one_step_evaluator(
        self,
        roots: list[_SearchNode],
        checkpoint_actions: list[dict[int, Any]],
        full_lanes: np.ndarray,
    ) -> dict[int, dict[str, Any]]:
        """Evaluate unique checkpoint decisions in one detached EMA batch."""
        entries: list[tuple[int, Any]] = []
        lookup: dict[tuple[int, bytes], int] = {}
        budget_to_entry: dict[int, dict[int, int]] = {}
        for lane in np.flatnonzero(full_lanes):
            lane = int(lane)
            budget_to_entry[lane] = {}
            for budget, action in checkpoint_actions[lane].items():
                action_array = np.asarray(action, dtype=np.float32).reshape(-1)
                key = (lane, action_array.tobytes())
                if key not in lookup:
                    lookup[key] = len(entries)
                    entries.append((lane, action))
                budget_to_entry[lane][budget] = lookup[key]
        if not entries:
            return {}
        parent_caches = [roots[lane].cache for lane, _ in entries]
        if self.discrete:
            indices = torch.as_tensor(
                [int(action) for _, action in entries],
                dtype=torch.long,
                device=self.device,
            )
            action_tensor = F.one_hot(indices, num_classes=self.n_actions).float()
        else:
            action_tensor = torch.as_tensor(
                np.stack([np.asarray(action, dtype=np.float32) for _, action in entries]),
                dtype=torch.float32,
                device=self.device,
            )
        with torch.inference_mode():
            _, action_hidden, next_hidden = self._step_imagine(parent_caches, action_tensor)
            target_q = (
                self.target_heads.reward(action_hidden)
                + self.gamma * self.target_heads.value(next_hidden)
            )
            online_q = (
                self.heads.reward(action_hidden)
                + self.gamma * self.heads.value(next_hidden)
            )
            probe_q = (
                self.uncertainty_probe.reward(action_hidden)
                + self.gamma * self.uncertainty_probe.value(next_hidden)
            )
            probe_uncertainty = probe_q.std(dim=-1, unbiased=False)
            stability_error = (target_q - online_q).abs()
            confidence = 1.0 / (
                1.0
                + stability_error / self.voc_return_scale
                + probe_uncertainty / self.voc_return_scale
            )
        target_np = target_q.cpu().numpy()
        confidence_np = confidence.cpu().numpy()
        uncertainty_np = (probe_uncertainty / self.voc_return_scale).cpu().numpy()
        output: dict[int, dict[str, Any]] = {}
        for lane, mapping in budget_to_entry.items():
            ordered_budgets = sorted(mapping)
            output[lane] = {
                "q": {budget: float(target_np[mapping[budget]]) for budget in ordered_budgets},
                "confidence": {
                    budget: float(confidence_np[mapping[budget]])
                    for budget in ordered_budgets
                },
                "uncertainty": {
                    budget: float(uncertainty_np[mapping[budget]])
                    for budget in ordered_budgets
                },
                "unique_actions": len(set(mapping.values())),
            }
        return output

    def _sample_audit_types(
        self,
        root_epistemic: np.ndarray,
        *,
        random_draws: np.ndarray | None = None,
        targeted_count: int | None = None,
    ) -> list[str]:
        """Sample unconditional random/targeted rates without collision loss."""
        batch_size = len(root_epistemic)
        draws = (
            self._voc_rng.random(batch_size)
            if random_draws is None
            else np.asarray(random_draws, dtype=np.float64)
        )
        random_mask = draws < self.voc_random_audit_fraction
        audit_types = ["random" if selected else "none" for selected in random_mask]
        eligible = np.flatnonzero(~random_mask)
        if targeted_count is None:
            denominator = max(1e-12, 1.0 - self.voc_random_audit_fraction)
            conditional_probability = min(
                1.0, self.voc_targeted_audit_fraction / denominator,
            )
            targeted_count = int(
                self._voc_rng.binomial(len(eligible), conditional_probability),
            )
        targeted_count = min(max(0, int(targeted_count)), len(eligible))
        if targeted_count:
            ranked = eligible[np.argsort(root_epistemic[eligible])[::-1]]
            for lane in ranked[:targeted_count]:
                audit_types[int(lane)] = "targeted"
        return audit_types

    def _sample_label_lanes(
        self,
        full_lanes: np.ndarray,
        audit_types: list[str],
        *,
        draws: np.ndarray | None = None,
    ) -> np.ndarray:
        """Always label audits; subsample other full-budget lanes."""
        full = np.asarray(full_lanes, dtype=bool)
        audited = np.asarray([audit_type != "none" for audit_type in audit_types])
        random_draws = (
            self._voc_rng.random(len(full))
            if draws is None
            else np.asarray(draws, dtype=np.float64)
        )
        sampled_shadow = (
            full
            & ~audited
            & (random_draws < self.voc_label_sample_fraction)
        )
        return full & (audited | sampled_shadow)

    def _voc_predict_batch(
        self,
        root_hidden: torch.Tensor,
        scalars: torch.Tensor,
    ) -> np.ndarray:
        with torch.inference_mode():
            return torch.sigmoid(
                self.voc_classifier(root_hidden, scalars),
            ).cpu().numpy()

    def _train_voc(self) -> None:
        valid_examples = [example for example in self._voc_replay if example["valid"]]
        if len(valid_examples) < self.voc_min_labels:
            return
        count = min(self.voc_batch_size, len(valid_examples))
        indices = self._voc_rng.choice(len(valid_examples), size=count, replace=False)
        examples = [valid_examples[int(index)] for index in indices]
        roots = torch.stack([example["root_hidden"] for example in examples]).to(self.device)
        scalars = torch.stack([example["scalars"] for example in examples]).to(self.device)
        labels = torch.tensor(
            [example["label"] for example in examples],
            dtype=torch.float32,
            device=self.device,
        )
        logits = self.voc_classifier(roots, scalars)
        positives = labels.sum()
        negatives = labels.numel() - positives
        positive_weight = negatives / positives.clamp_min(1.0)
        weights = torch.where(labels > 0.5, positive_weight, torch.ones_like(labels))
        loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        self.voc_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.voc_classifier.parameters(), self.max_grad_norm)
        self.voc_optimizer.step()
        with torch.no_grad():
            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= self.voc_threshold
            stops = ~predictions
            correct_stops = stops & (labels < 0.5)
            stop_precision = (
                correct_stops.float().sum() / stops.float().sum().clamp_min(1.0)
            )
            ece = torch.zeros((), device=self.device)
            for lower in torch.linspace(0.0, 0.8, 5, device=self.device):
                selected = (probabilities >= lower) & (probabilities < lower + 0.2)
                if selected.any():
                    ece = ece + selected.float().mean() * (
                        probabilities[selected].mean() - labels[selected].mean()
                    ).abs()
        self._voc_update_count += 1
        self._last_voc_loss = float(loss.item())
        self._last_voc_accuracy = float((predictions == (labels > 0.5)).float().mean().item())
        self._last_voc_brier = float(((probabilities - labels) ** 2).mean().item())
        self._last_voc_stop_precision = float(stop_precision.item())
        self._last_voc_ece = float(ece.item())

    def _voc_stage_summary(self, stage: int) -> dict[str, float]:
        labels = [
            item
            for item in self._voc_replay
            if item.get("valid", False)
            and item.get("edge_index") == stage
        ]
        audits = [
            item
            for item in self._voc_audits
            if item.get("edge_index") == stage
        ]
        label_values = np.asarray(
            [item["label"] for item in labels],
            dtype=np.float64,
        )
        stop_fraction = float(np.mean(label_values < 0.5)) if label_values.size else 0.0
        continue_fraction = float(np.mean(label_values >= 0.5)) if label_values.size else 0.0
        stop_count = int(np.sum(label_values < 0.5))
        continue_count = int(np.sum(label_values >= 0.5))
        if not audits:
            return {
                "labels": float(len(labels)),
                "audits": 0.0,
                "stop_samples": 0.0,
                "label_stop_fraction": stop_fraction,
                "label_continue_fraction": continue_fraction,
                "label_stop_count": float(stop_count),
                "label_continue_count": float(continue_count),
                "regret": float("inf"),
                "divergence": float("inf"),
                "overall_regret": float("inf"),
                "overall_divergence": float("inf"),
                "premature_exceedance": 1.0,
                "premature_p95": float("inf"),
                "premature_mean_positive": float("inf"),
                "premature_positive_case_mean": float("inf"),
                "brier": float("inf"),
                "stop_precision": 0.0,
                "confident_mistake": 1.0,
                "activated": float(stage >= self._voc_minimum_budget_index),
            }
        probabilities = np.asarray([item["probability"] for item in audits], dtype=np.float64)
        targets = np.asarray([item["label"] for item in audits], dtype=np.float64)
        predicted_stops = probabilities < self.voc_threshold
        stop_audits = [
            item for item, is_stop in zip(audits, predicted_stops.tolist()) if is_stop
        ]
        stop_regrets = np.asarray(
            [max(0.0, item["regret"]) for item in stop_audits],
            dtype=np.float64,
        )
        positive_regrets = stop_regrets[stop_regrets > 0.0]
        epsilon = self.voc_premature_regret_epsilon[stage]
        return {
            "labels": float(len(labels)),
            "audits": float(len(audits)),
            "stop_samples": float(len(stop_audits)),
            "label_stop_fraction": stop_fraction,
            "label_continue_fraction": continue_fraction,
            "label_stop_count": float(stop_count),
            "label_continue_count": float(continue_count),
            "regret": (
                float(stop_regrets.mean())
                if stop_regrets.size
                else float("inf")
            ),
            "divergence": (
                float(np.mean([item["divergence"] for item in stop_audits]))
                if stop_audits
                else float("inf")
            ),
            "overall_regret": float(np.mean([item["regret"] for item in audits])),
            "overall_divergence": float(
                np.mean([item["divergence"] for item in audits]),
            ),
            "premature_exceedance": (
                float(np.mean(stop_regrets > epsilon))
                if stop_regrets.size
                else 1.0
            ),
            "premature_p95": (
                float(np.percentile(stop_regrets, 95))
                if stop_regrets.size
                else float("inf")
            ),
            # Primary expected harm: E[max(R_premature, 0)] over every
            # predicted STOP, including the safe zero-regret decisions.
            "premature_mean_positive": (
                float(stop_regrets.mean())
                if stop_regrets.size
                else float("inf")
            ),
            # Diagnostic severity conditional on a positive-regret case;
            # intentionally never used as a release/relock gate.
            "premature_positive_case_mean": (
                float(positive_regrets.mean()) if positive_regrets.size else 0.0
            ),
            "brier": float(np.mean((probabilities - targets) ** 2)),
            "stop_precision": (
                float(np.mean(targets[predicted_stops] < 0.5))
                if predicted_stops.any()
                else 0.0
            ),
            "confident_mistake": (
                float(np.mean([item["confident_mistake"] for item in stop_audits]))
                if stop_audits
                else 1.0
            ),
            "activated": float(stage >= self._voc_minimum_budget_index),
        }

    def _voc_stage_is_safe(self, stage: int, summary: dict[str, float]) -> bool:
        return (
            summary["labels"] >= self.voc_min_labels
            and summary["audits"] >= self.voc_min_audits
            and summary["stop_samples"] >= self.voc_min_stop_audits
            and summary["label_stop_count"] >= self.voc_min_class_samples
            and summary["label_continue_count"] >= self.voc_min_class_samples
            and summary["premature_exceedance"]
            <= self.voc_max_premature_exceedance[stage]
            and summary["premature_p95"]
            <= self.voc_max_p95_premature_regret[stage]
            and summary["premature_mean_positive"]
            <= self.voc_max_mean_premature_regret[stage]
            and summary["stop_precision"] >= self.voc_min_stop_precision
            and summary["confident_mistake"] <= 1.0 - self.voc_min_stop_precision
        )

    def _voc_stage_regressed(self, stage: int, summary: dict[str, float]) -> bool:
        enough_stops = summary["stop_samples"] >= self.voc_min_stop_audits
        tail_unsafe = enough_stops and (
            summary["premature_exceedance"]
            > self.voc_max_premature_exceedance[stage]
            or summary["premature_p95"]
            > self.voc_max_p95_premature_regret[stage]
            or summary["premature_mean_positive"]
            > self.voc_max_mean_premature_regret[stage]
            or summary["stop_precision"] < self.voc_min_stop_precision
        )
        catastrophic = (
            summary["audits"] >= self.voc_min_audits
            and (
                summary["brier"]
                > min(1.0, self.voc_catastrophic_multiplier * self.voc_max_brier)
                or summary["overall_divergence"]
                > max(
                    1.0,
                    self.voc_catastrophic_multiplier
                    * self.voc_max_policy_divergence,
                )
            )
        )
        return (
            summary["audits"] >= self.voc_min_audits
            and (tail_unsafe or catastrophic)
        )

    def _update_voc_safety(self) -> None:
        maximum_index = len(self.voc_budgets) - 1
        regressed_stages: list[int] = []
        for stage in range(self._voc_minimum_budget_index, maximum_index):
            if self._voc_stage_regressed(stage, self._voc_stage_summary(stage)):
                regressed_stages.append(stage)
        if regressed_stages:
            self._voc_minimum_budget_index = min(
                maximum_index,
                max(
                    self._voc_minimum_budget_index + 1,
                    max(stage + 1 for stage in regressed_stages),
                ),
            )
            self._voc_safety_streaks = [0] * len(self._voc_safety_streaks)
            return
        if self._voc_minimum_budget_index <= 0:
            return
        unlock_stage = self._voc_minimum_budget_index - 1
        safe = self._voc_stage_is_safe(
            unlock_stage,
            self._voc_stage_summary(unlock_stage),
        )
        self._voc_safety_streaks[unlock_stage] = (
            self._voc_safety_streaks[unlock_stage] + 1 if safe else 0
        )
        if self._voc_safety_streaks[unlock_stage] >= self.voc_safety_patience:
            self._voc_minimum_budget_index -= 1
            self._voc_shadow_mode = False
            self._voc_safety_streaks[unlock_stage] = 0

    def _voc_health_metrics(self) -> dict[str, float]:
        summaries = [
            self._voc_stage_summary(stage)
            for stage in range(len(self.voc_edge_ids))
        ]
        missing_edge = any(
            summary["labels"] < self.voc_min_labels
            or summary["audits"] < self.voc_min_audits
            or summary["stop_samples"] < self.voc_min_stop_audits
            for summary in summaries
        )
        class_collapse = any(
            summary["labels"] >= self.voc_min_labels
            and (
                summary["label_stop_count"] < self.voc_min_class_samples
                or summary["label_continue_count"] < self.voc_min_class_samples
            )
            for summary in summaries
        )
        history_ready = (
            len(self._predicted_budget_history) >= self.voc_health_desync_window
        )
        maximum = max(1, self.num_simulations)
        requested_history = np.asarray(self._requested_budget_history)
        executed_history = np.asarray(self._executed_search_history)
        permitted_history = np.asarray(self._permitted_budget_history)
        contract_history_ready = (
            len(permitted_history) == self.voc_health_desync_window
            and len(requested_history) == len(permitted_history)
            and len(executed_history) == len(permitted_history)
        )
        unexplained_gap = (
            (
                (requested_history > permitted_history + 0.05 * maximum)
                | (executed_history > permitted_history + 0.05 * maximum)
            )
            if contract_history_ready
            else np.zeros(0, dtype=bool)
        )
        budget_desync = bool(
            history_ready
            and not missing_edge
            and contract_history_ready
            and float(np.mean(permitted_history))
            < self.voc_health_desync_ratio * maximum
            and float(np.mean(requested_history))
            > self.voc_health_actual_max_ratio * maximum
            and float(np.mean(executed_history))
            > self.voc_health_actual_max_ratio * maximum
            and float(np.mean(unexplained_gap))
            > self.voc_health_actual_max_ratio
        )
        actor_summary = summaries[0] if summaries else None
        unsafe_compute_collapse = bool(
            history_ready
            and actor_summary is not None
            and np.mean(np.asarray(self._executed_search_history) == 0)
            > self.voc_health_unsafe_actor_fraction
            and actor_summary["stop_samples"] >= self.voc_min_stop_audits
            and actor_summary["premature_p95"]
            > self.voc_max_p95_premature_regret[0]
        )
        deadline = self._num_timesteps >= self.voc_health_check_steps
        if unsafe_compute_collapse:
            status = 5
        elif not deadline:
            status = 1
        elif missing_edge:
            status = 2
        elif class_collapse:
            status = 3
        elif self._voc_shadow_mode:
            status = 1
        elif budget_desync:
            status = 4
        else:
            status = 0
        names = [
            "healthy",
            "warming_up",
            "activation_failure_missing_edge",
            "activation_failure_class_collapse",
            "activation_failure_budget_desync",
            "unsafe_compute_collapse",
        ]
        metrics = {
            "voc_health_status": float(status),
            "voc_fail_fast_recommended": float(status >= 2),
            "voc_rollout_depth_control_deferred": 1.0,
            "voc_health_shadow_prediction_gap": (
                float(
                    np.mean(
                        np.maximum(
                            0.0,
                            requested_history
                            - np.asarray(self._predicted_budget_history),
                        )
                    )
                    / maximum
                )
                if self._voc_shadow_mode and requested_history.size
                else 0.0
            ),
        }
        metrics.update(
            {
                f"voc_health_{name}": float(index == status)
                for index, name in enumerate(names)
            }
        )
        return metrics

    def _voc_metrics(self, executed: list[int], expansions: list[int], predicted: list[int], audits: list[str]) -> dict[str, float]:
        budget_array = np.asarray(executed, dtype=np.float64)
        expansion_array = np.asarray(expansions, dtype=np.float64)
        labels = np.asarray(
            [item["label"] for item in self._voc_replay if item["valid"]],
            dtype=np.float64,
        )
        audit_count = max(1, len(audits))
        metrics = {
            "search_budget_mean": float(budget_array.mean()),
            "search_budget_p50": float(np.percentile(budget_array, 50)),
            "search_budget_p90": float(np.percentile(budget_array, 90)),
            "model_expansions_mean": float(expansion_array.mean()),
            "shadow_predicted_budget_mean": float(np.mean(predicted)),
            "audit_random_fraction": audits.count("random") / audit_count,
            "audit_targeted_fraction": audits.count("targeted") / audit_count,
            "voc_label_count": float(labels.size),
            "voc_continue_fraction": float(labels.mean()) if labels.size else 0.0,
            "voc_bce": self._last_voc_loss,
            "voc_accuracy": self._last_voc_accuracy,
            "voc_brier": self._last_voc_brier,
            "voc_ece": self._last_voc_ece,
            "voc_stop_precision": self._last_voc_stop_precision,
            "voc_minimum_budget": float(self.voc_budgets[self._voc_minimum_budget_index]),
            "voc_shadow_mode": float(self._voc_shadow_mode),
        }
        for budget in self.voc_budgets:
            metrics[f"search_budget_fraction_{budget}"] = float(np.mean(budget_array == budget))
        auditable = [item for item in self._voc_audits if "current_budget" in item]
        if auditable:
            probabilities = np.asarray(
                [item["probability"] for item in auditable],
                dtype=np.float64,
            )
            audit_labels = np.asarray(
                [item["label"] for item in auditable],
                dtype=np.float64,
            )
            predicted_stops = probabilities < self.voc_threshold
            metrics["voc_normalized_actor_regret"] = float(
                np.mean([item["regret"] for item in auditable]),
            )
            metrics["voc_policy_divergence"] = float(
                np.mean([item["divergence"] for item in auditable]),
            )
            metrics["voc_overall_normalized_regret"] = metrics[
                "voc_normalized_actor_regret"
            ]
            metrics["voc_overall_policy_divergence"] = metrics[
                "voc_policy_divergence"
            ]
            stop_auditable = [
                item
                for item in auditable
                if item["probability"] < self.voc_threshold
            ]
            metrics["voc_predicted_stop_normalized_regret"] = (
                float(np.mean([item["regret"] for item in stop_auditable]))
                if stop_auditable
                else float("inf")
            )
            metrics["voc_predicted_stop_policy_divergence"] = (
                float(np.mean([item["divergence"] for item in stop_auditable]))
                if stop_auditable
                else float("inf")
            )
            metrics["voc_confident_mistake_rate"] = float(
                np.mean([item["confident_mistake"] for item in auditable]),
            )
            metrics["voc_audit_brier"] = float(np.mean((probabilities - audit_labels) ** 2))
            metrics["voc_audit_stop_precision"] = (
                float(np.mean(audit_labels[predicted_stops] < 0.5))
                if predicted_stops.any()
                else 0.0
            )
        valid_confidences = [
            item["confidence"] for item in self._voc_replay if item["valid"]
        ]
        metrics["voc_common_evaluator_confidence"] = (
            float(np.mean(valid_confidences)) if valid_confidences else 0.0
        )
        for stage in range(len(self.voc_budgets) - 1):
            summary = self._voc_stage_summary(stage)
            prefix = f"voc_edge_{self.voc_edge_ids[stage]}"
            for key, value in summary.items():
                metrics[f"{prefix}_{key}"] = value
        metrics.update(self._voc_health_metrics())
        return metrics

    def search(
        self,
        obs_batch: np.ndarray,
        root_caches: list[Any],
        deterministic: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Run one max-schedule planner and expose true prefix checkpoints.

        The Gumbels, fixed root candidate set, and max-budget sequential-halving
        schedule are sampled once.  A B checkpoint is therefore the state of
        this exact planner after B model expansions, not an independently
        scheduled B-simulation search.  Explicit deterministic/evaluation and
        replay-reanalysis calls always execute the current maximum budget and
        never train VoC or mutate its safety state.
        """
        if self.force_research_mode or not self.voc_enabled:
            return super().search(obs_batch, root_caches, deterministic)

        batch_size = len(obs_batch)
        collection = deterministic is None
        uncertainty_diagnostics = (
            self._last_uncertainty_normalized_mean,
            self._last_uncertainty_bonus_mean,
            self._last_uncertainty_bonus_max,
        )
        if deterministic is None:
            deterministic_flags = np.zeros(batch_size, dtype=bool)
        else:
            deterministic_flags = np.asarray(deterministic, dtype=bool)
        # VoC is the sole breadth controller in v10.1.  The Research
        # model-error scheduler remains reachable only through the identity
        # path above (force mode or voc_enabled=0).
        maximum_budget = self.num_simulations
        budgets = self.voc_budgets
        self._last_search_base_simulations = maximum_budget
        self._last_search_num_simulations = maximum_budget
        minimum_budget = min(
            self.voc_budgets[self._voc_minimum_budget_index],
            maximum_budget,
        )

        with torch.inference_mode():
            observations = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
            obs_emb = self.tokenizer(observations)
            positions, embed_positions = _cache_positions(root_caches, self.device)
            root_hidden, root_states = self.transformer.forward_incremental_batch(
                obs_emb, positions, root_caches, embed_positions,
            )
            root_values = self.heads.value(root_hidden)
            root_values_np = root_values.cpu().numpy()
            if self.discrete:
                root_logits = self.heads.policy_logits(root_hidden)
                root_logits_np = root_logits.cpu().numpy().astype(np.float64)
                actor_probabilities = torch.softmax(root_logits, dim=-1).cpu().numpy()
                actor_means = None
                actor_stds = None
            else:
                actor_means_tensor, actor_stds_tensor = self.heads.policy_gaussian(root_hidden)
                actor_means = actor_means_tensor.cpu().numpy()
                actor_stds = actor_stds_tensor.cpu().numpy()
                actor_probabilities = None

        num_root_actions = self.n_actions if self.discrete else self.num_sampled_actions
        num_top_actions = max(
            2,
            min(
                self.num_top_actions,
                num_root_actions,
                max(2, maximum_budget // 2),
            ),
        )
        schedule = _HalvingSchedule(maximum_budget, num_top_actions)
        minmax = [_MinMaxStats(self.value_minmax_delta) for _ in range(batch_size)]
        roots: list[_SearchNode] = []
        root_candidates: list[list[Any]] = []
        actor_entropies: list[float] = []
        for lane in range(batch_size):
            if self.discrete:
                candidates = list(range(num_root_actions))
                priors = root_logits_np[lane, :num_root_actions]
                probabilities = actor_probabilities[lane, :num_root_actions]
                entropy = -float(
                    np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None))),
                ) / max(math.log(max(2, len(probabilities))), 1e-6)
            else:
                candidates, priors = self._continuous_anytime_candidates(
                    actor_means[lane], actor_stds[lane],
                )
                candidate_probabilities = np.exp(priors - np.max(priors))
                candidate_probabilities /= candidate_probabilities.sum()
                probabilities = candidate_probabilities
                gaussian_entropy = float(
                    np.log(np.clip(actor_stds[lane], 1e-6, None) * math.sqrt(2.0 * math.pi * math.e)).mean(),
                )
                entropy = float(1.0 / (1.0 + math.exp(-gaussian_entropy)))
            root = _SearchNode(prior=1.0)
            root.expand(
                priors,
                root_states[lane],
                0.0,
                candidate_actions=None if self.discrete else candidates,
            )
            root.visit_count = 1
            root.value_sum = float(root_values_np[lane])
            roots.append(root)
            root_candidates.append(candidates)
            actor_entropies.append(entropy)

        fixed_gumbels = (
            np.random.gumbel(size=(batch_size, num_root_actions))
            * self.policy_target_temperature
            * self._current_exploration_scale()
        )
        for lane, root in enumerate(roots):
            scores = fixed_gumbels[lane] + np.asarray([child.prior for child in root.children])
            root.selected_children_idx = list(np.argsort(-scores)[:num_top_actions])
        checkpoint_indices: list[dict[int, int]] = [dict() for _ in range(batch_size)]
        checkpoint_policies: list[dict[int, np.ndarray]] = [dict() for _ in range(batch_size)]
        checkpoint_actions: list[dict[int, Any]] = [dict() for _ in range(batch_size)]
        checkpoint_features: list[dict[int, dict[str, Any]]] = [dict() for _ in range(batch_size)]
        actor_behavior: list[np.ndarray] = []
        with torch.inference_mode():
            root_epistemic_tensor = self.uncertainty_probe.value(
                root_hidden.float(),
            ).std(dim=-1, unbiased=False)
            root_epistemic_tensor = root_epistemic_tensor / max(
                self._uncertainty_disagreement_ema, 1e-6,
            )
            root_epistemic_tensor = root_epistemic_tensor / (
                1.0 + root_epistemic_tensor
            )
            root_epistemic = root_epistemic_tensor.cpu().numpy()
        root_scalars: list[torch.Tensor] = []
        for lane in range(batch_size):
            if self.discrete:
                behavior = actor_probabilities[lane, :num_root_actions].astype(np.float32)
                actor_index = int(np.argmax(behavior))
                feature_policy = behavior
            else:
                behavior = actor_means[lane].astype(np.float32)
                actor_index = 0
                priors = np.asarray([child.prior for child in roots[lane].children])
                feature_policy = np.exp(priors - np.max(priors))
                feature_policy /= feature_policy.sum()
            actor_behavior.append(behavior)
            checkpoint_indices[lane][0] = actor_index
            checkpoint_policies[lane][0] = behavior
            checkpoint_actions[lane][0] = actor_index if self.discrete else behavior.copy()
            scalars, _ = self._voc_stage_features(
                root_hidden[lane],
                actor_entropies[lane],
                feature_policy,
                0,
                0,
                maximum_budget,
                float(root_epistemic[lane]),
            )
            root_scalars.append(scalars)
            checkpoint_features[lane][0] = {
                "root_hidden": root_hidden[lane].detach().cpu(),
                "scalars": scalars.detach().cpu(),
                "probability": 0.0,
            }
        root_probabilities = self._voc_predict_batch(
            root_hidden,
            torch.stack(root_scalars),
        )
        for lane, probability in enumerate(root_probabilities.tolist()):
            checkpoint_features[lane][0]["probability"] = float(probability)

        audit_types = (
            self._sample_audit_types(root_epistemic)
            if collection
            else ["none"] * batch_size
        )
        execute_full = np.asarray(
            [
                (not collection)
                or self._voc_shadow_mode
                or audit_types[lane] != "none"
                for lane in range(batch_size)
            ],
            dtype=bool,
        )
        predicted_budgets = np.zeros(batch_size, dtype=np.int64)
        controller_done = np.zeros(batch_size, dtype=bool)
        expansion_counts = np.zeros(batch_size, dtype=np.int64)
        first_next_budget = budgets[1] if len(budgets) > 1 else 0
        active_lanes: list[int] = []
        for lane in range(batch_size):
            probability = checkpoint_features[lane][0]["probability"]
            if probability >= self.voc_threshold:
                predicted_budgets[lane] = first_next_budget
            else:
                controller_done[lane] = True
            if execute_full[lane] or first_next_budget <= minimum_budget or not controller_done[lane]:
                active_lanes.append(lane)

        for simulation in range(maximum_budget):
            if not active_lanes:
                break
            leaf_nodes: list[_SearchNode] = []
            parent_caches: list[Any] = []
            actions: list[Any] = []
            search_paths: list[list[_SearchNode]] = []
            for lane in active_lanes:
                node = roots[lane]
                path = [node]
                while node.expanded():
                    child_index = _select_action(
                        node,
                        minmax[lane],
                        self.gamma,
                        self.c_visit,
                        self.c_scale,
                    )
                    node = node.children[child_index]
                    path.append(node)
                parent = path[-2]
                leaf_nodes.append(node)
                parent_caches.append(parent.cache)
                actions.append(
                    node.candidate_action
                    if not self.discrete
                    else parent.children.index(node)
                )
                search_paths.append(path)

            if self.discrete:
                action_indices = torch.as_tensor(actions, dtype=torch.long, device=self.device)
                action_tensor = F.one_hot(action_indices, num_classes=self.n_actions).float()
            else:
                action_tensor = torch.as_tensor(
                    np.stack(actions), dtype=torch.float32, device=self.device,
                )
            with torch.inference_mode():
                child_caches, action_hidden, next_hidden = self._step_imagine(
                    parent_caches, action_tensor,
                )
                rewards = self.heads.reward(action_hidden)
                next_values = self.heads.value(next_hidden)
                rewards, next_values = self._adjust_search_predictions(
                    action_hidden, next_hidden, rewards, next_values,
                )
                if self.discrete:
                    next_policy = self.heads.policy_logits(next_hidden)
                    next_policy_np = next_policy.cpu().numpy().astype(np.float64)
                else:
                    next_means, next_stds = self.heads.policy_gaussian(next_hidden)
                    next_means_np = next_means.cpu().numpy()
                    next_stds_np = next_stds.cpu().numpy()
                rewards_np = rewards.cpu().numpy()
                next_values_np = next_values.cpu().numpy()
            for local_index, lane in enumerate(active_lanes):
                leaf = leaf_nodes[local_index]
                if self.discrete:
                    leaf_candidates = None
                    leaf_priors = next_policy_np[local_index]
                else:
                    leaf_candidates, leaf_priors = self._continuous_anytime_candidates(
                        next_means_np[local_index],
                        next_stds_np[local_index],
                    )
                leaf.expand(
                    leaf_priors,
                    child_caches[local_index],
                    float(rewards_np[local_index]),
                    candidate_actions=leaf_candidates,
                )
                _backpropagate(
                    search_paths[local_index],
                    float(next_values_np[local_index]),
                    minmax[lane],
                    self.gamma,
                )
                expansion_counts[lane] += 1
            if schedule.maybe_advance(simulation):
                for lane in active_lanes:
                    _sequential_halving(
                        roots[lane],
                        fixed_gumbels[lane],
                        minmax[lane],
                        schedule.current_top,
                        self.gamma,
                        self.c_visit,
                        self.c_scale,
                    )

            completed_budget = simulation + 1
            if completed_budget not in budgets[1:]:
                continue
            stage = budgets.index(completed_budget)
            next_budget = budgets[stage + 1] if stage + 1 < len(budgets) else None
            stage_lanes: list[int] = []
            stage_scalars: list[torch.Tensor] = []
            for lane in active_lanes:
                completed_q = _transformed_completed_qs(
                    roots[lane], minmax[lane], self.gamma, self.c_visit, self.c_scale,
                )
                improved = roots[lane].improved_policy(completed_q).astype(np.float32)
                selected_index = int(np.argmax(improved))
                checkpoint_indices[lane][completed_budget] = selected_index
                if self.discrete:
                    checkpoint_policies[lane][completed_budget] = improved
                    checkpoint_actions[lane][completed_budget] = selected_index
                else:
                    policy_mean = np.sum(
                        improved[:, None] * np.stack(root_candidates[lane]),
                        axis=0,
                    ).astype(np.float32)
                    checkpoint_policies[lane][completed_budget] = policy_mean
                    checkpoint_actions[lane][completed_budget] = policy_mean.copy()
                if next_budget is None:
                    continue
                scalars, _ = self._voc_stage_features(
                    root_hidden[lane],
                    actor_entropies[lane],
                    improved,
                    completed_budget,
                    stage,
                    maximum_budget,
                    float(root_epistemic[lane]),
                )
                stage_lanes.append(lane)
                stage_scalars.append(scalars)
                checkpoint_features[lane][completed_budget] = {
                    "root_hidden": root_hidden[lane].detach().cpu(),
                    "scalars": scalars.detach().cpu(),
                    "probability": 0.0,
                }
            if next_budget is None:
                active_lanes = []
                continue
            stage_probabilities = self._voc_predict_batch(
                root_hidden[stage_lanes],
                torch.stack(stage_scalars),
            )
            next_active: list[int] = []
            for lane, probability in zip(stage_lanes, stage_probabilities.tolist()):
                probability = float(probability)
                checkpoint_features[lane][completed_budget]["probability"] = probability
                if not controller_done[lane]:
                    if probability >= self.voc_threshold:
                        predicted_budgets[lane] = next_budget
                    else:
                        controller_done[lane] = True
                must_continue = completed_budget < minimum_budget
                if execute_full[lane] or must_continue or not controller_done[lane]:
                    next_active.append(lane)
            active_lanes = next_active

        full_lanes = expansion_counts == maximum_budget
        label_lanes = (
            self._sample_label_lanes(full_lanes, audit_types)
            if collection
            else np.zeros(batch_size, dtype=bool)
        )
        common_evaluation = (
            self._common_one_step_evaluator(roots, checkpoint_actions, label_lanes)
            if label_lanes.any()
            else {}
        )
        evaluator_expansions = np.zeros(batch_size, dtype=np.int64)
        for lane, evaluation in common_evaluation.items():
            evaluator_expansions[lane] = int(evaluation["unique_actions"])

        results: list[dict[str, Any]] = []
        for lane in range(batch_size):
            executed_search = int(expansion_counts[lane])
            requested_budget = executed_search
            if requested_budget not in checkpoint_indices[lane]:
                requested_budget = max(
                    budget for budget in checkpoint_indices[lane] if budget <= requested_budget
                )
            requested_stage_index = budgets.index(requested_budget)
            predicted_budget = int(predicted_budgets[lane])
            predicted_stage_index = budgets.index(predicted_budget)
            permitted_budget = (
                maximum_budget
                if execute_full[lane]
                else max(predicted_budget, minimum_budget)
            )
            behavior_policy = checkpoint_policies[lane][requested_budget]
            if self.discrete:
                if deterministic_flags[lane]:
                    action_index = int(np.argmax(behavior_policy))
                else:
                    probabilities = behavior_policy / max(float(behavior_policy.sum()), 1e-8)
                    action_index = int(np.random.choice(len(probabilities), p=probabilities))
                action: Any = action_index
            else:
                action = behavior_policy
                action = np.clip(action, self._action_space.low, self._action_space.high).astype(np.float32)
            if executed_search >= maximum_budget:
                termination_reason = "budget_exhausted"
            elif predicted_budget < minimum_budget and executed_search >= minimum_budget:
                termination_reason = "safety_floor"
            else:
                termination_reason = "controller_stop"
            total_model_calls = int(
                executed_search + evaluator_expansions[lane]
            )
            results.append(
                {
                    "env_action": action,
                    "policy_target": behavior_policy,
                    "behavior_policy": behavior_policy,
                    "policy_target_valid": requested_budget > 0,
                    "value_target": float(roots[lane].value()),
                    # Backward-compatible aliases: search_budget is the
                    # requested planner budget; model_expansions is total
                    # model calls including detached common evaluation.
                    "search_budget": requested_budget,
                    "model_expansions": total_model_calls,
                    "stage_index": requested_stage_index,
                    "stage_id": self.voc_stage_ids[requested_stage_index],
                    "nominal_stage_budget": budgets[requested_stage_index],
                    "requested_budget": requested_budget,
                    "predicted_stage_index": predicted_stage_index,
                    "predicted_stage_id": self.voc_stage_ids[predicted_stage_index],
                    "predicted_requested_budget": predicted_budget,
                    "permitted_routing_budget": int(permitted_budget),
                    "executed_search_expansions": executed_search,
                    "common_evaluator_expansions": int(evaluator_expansions[lane]),
                    "total_model_calls": total_model_calls,
                    "termination_reason": termination_reason,
                    "audit_type": audit_types[lane],
                    "root_cache": roots[lane].cache,
                    "budget_checkpoints": {
                        budget: {
                            "policy": checkpoint_policies[lane][budget].copy(),
                            "representative_action": (
                                int(checkpoint_actions[lane][budget])
                                if self.discrete
                                else np.asarray(
                                    checkpoint_actions[lane][budget],
                                    dtype=np.float32,
                                ).copy()
                            ),
                        }
                        for budget in sorted(checkpoint_policies[lane])
                    },
                }
            )

        if collection:
            for lane in np.flatnonzero(label_lanes):
                lane = int(lane)
                evaluation = common_evaluation[lane]
                common_q = [evaluation["q"][budget] for budget in budgets]
                labels = self._common_voc_labels(
                    common_q,
                    budgets,
                    self.voc_compute_cost_per_expansion,
                    self.voc_label_margin,
                    self.voc_return_scale,
                )
                for stage, (gain, label) in enumerate(labels):
                    current_budget = budgets[stage]
                    next_budget = budgets[stage + 1]
                    feature = checkpoint_features[lane][current_budget]
                    probability = float(feature["probability"])
                    confidence = min(
                        evaluation["confidence"][current_budget],
                        evaluation["confidence"][next_budget],
                    )
                    uncertainty = max(
                        float(feature["scalars"][2].item()),
                        float(feature["scalars"][3].item()),
                        evaluation["uncertainty"][current_budget],
                        evaluation["uncertainty"][next_budget],
                    )
                    valid_label = bool(
                        np.isfinite(common_q[stage])
                        and np.isfinite(common_q[stage + 1])
                        and np.isfinite(probability)
                        and np.isfinite(confidence)
                        and uncertainty <= self.voc_uncertainty_filter
                    )
                    self._voc_replay.append(
                        {
                            "root_hidden": feature["root_hidden"],
                            "scalars": feature["scalars"],
                            "label": label,
                            "raw_gain": gain,
                            "normalized_gain": gain / self.voc_return_scale,
                            "valid": valid_label,
                            "confidence": confidence,
                            "stage": stage,
                            "edge_index": stage,
                            "edge_id": self.voc_edge_ids[stage],
                            "from_stage_id": self.voc_stage_ids[stage],
                            "to_stage_id": self.voc_stage_ids[stage + 1],
                            "current_budget": current_budget,
                            "next_budget": next_budget,
                            "from_nominal_budget": current_budget,
                            "to_nominal_budget": next_budget,
                            "delta_budget": next_budget - current_budget,
                        }
                    )
                    if audit_types[lane] == "none" or not valid_label:
                        continue
                    if self.discrete:
                        current_policy = np.asarray(
                            checkpoint_policies[lane][current_budget],
                            dtype=np.float64,
                        )
                        next_policy = np.asarray(
                            checkpoint_policies[lane][next_budget],
                            dtype=np.float64,
                        )
                        divergence = float(
                            np.sum(
                                next_policy
                                * (
                                    np.log(np.clip(next_policy, 1e-12, None))
                                    - np.log(np.clip(current_policy, 1e-12, None))
                                )
                            )
                        )
                    else:
                        action_range = np.maximum(
                            self._action_space.high - self._action_space.low,
                            1e-6,
                        )
                        divergence = float(
                            np.mean(
                                np.abs(
                                    np.asarray(checkpoint_actions[lane][next_budget])
                                    - np.asarray(checkpoint_actions[lane][current_budget])
                                )
                                / action_range
                            ),
                        )
                    self._voc_audits.append(
                        {
                            "stage": stage,
                            "edge_index": stage,
                            "edge_id": self.voc_edge_ids[stage],
                            "from_stage_id": self.voc_stage_ids[stage],
                            "to_stage_id": self.voc_stage_ids[stage + 1],
                            "current_budget": current_budget,
                            "next_budget": next_budget,
                            "from_nominal_budget": current_budget,
                            "to_nominal_budget": next_budget,
                            "delta_budget": next_budget - current_budget,
                            "regret": max(gain / self.voc_return_scale, 0.0),
                            "divergence": divergence,
                            "probability": probability,
                            "label": label,
                            "confident_mistake": float(
                                probability < self.voc_threshold and label > 0.5
                            ),
                        }
                    )
            self._train_voc()
            self._update_voc_safety()
            requested = [result["requested_budget"] for result in results]
            executed_search = [
                result["executed_search_expansions"] for result in results
            ]
            total_calls = [result["total_model_calls"] for result in results]
            predicted_requested = [
                result["predicted_requested_budget"] for result in results
            ]
            permitted_requested = [
                result["permitted_routing_budget"] for result in results
            ]
            self._executed_budget_history.extend(requested)
            self._model_expansion_history.extend(total_calls)
            self._predicted_budget_history.extend(predicted_requested)
            self._requested_budget_history.extend(requested)
            self._permitted_budget_history.extend(permitted_requested)
            self._executed_search_history.extend(executed_search)
            self._last_voc_search_metrics = self._voc_metrics(
                requested,
                total_calls,
                predicted_requested,
                audit_types,
            )
            full_count = int(full_lanes.sum())
            self._last_voc_search_metrics.update(
                {
                    "voc_label_sample_fraction_actual": (
                        float(label_lanes.sum() / full_count) if full_count else 0.0
                    ),
                    "voc_common_evaluator_expansions_mean": float(
                        evaluator_expansions.mean(),
                    ),
                    "requested_budget_mean": float(np.mean(requested)),
                    "requested_budget_p50": float(np.percentile(requested, 50)),
                    "requested_budget_p90": float(np.percentile(requested, 90)),
                    "executed_search_expansions_mean": float(
                        np.mean(executed_search),
                    ),
                    "executed_search_expansions_p50": float(
                        np.percentile(executed_search, 50),
                    ),
                    "executed_search_expansions_p90": float(
                        np.percentile(executed_search, 90),
                    ),
                    "total_model_calls_mean": float(np.mean(total_calls)),
                    "total_model_calls_p50": float(np.percentile(total_calls, 50)),
                    "total_model_calls_p90": float(np.percentile(total_calls, 90)),
                }
            )
            for reason in ("budget_exhausted", "controller_stop", "safety_floor"):
                self._last_voc_search_metrics[
                    f"termination_fraction_{reason}"
                ] = float(
                    np.mean(
                        [result["termination_reason"] == reason for result in results]
                    )
                )
            for stage_index, stage_id in enumerate(self.voc_stage_ids):
                self._last_voc_search_metrics[
                    f"requested_stage_fraction_{stage_id}"
                ] = float(
                    np.mean(
                        [result["stage_index"] == stage_index for result in results]
                    )
                )
                self._last_voc_search_metrics[
                    f"predicted_stage_fraction_{stage_id}"
                ] = float(
                    np.mean(
                        [
                            result["predicted_stage_index"] == stage_index
                            for result in results
                        ]
                    )
                )
            self._last_voc_search_metrics.update(
                {
                    "voc_root_epistemic_mean": float(root_epistemic.mean()),
                    "voc_action_ambiguity_mean": float(1.0 - np.mean([
                        max(np.asarray(policy, dtype=np.float64))
                        if self.discrete
                        else 1.0 - actor_entropies[index]
                        for index, policy in enumerate(actor_behavior)
                    ])),
                    "voc_model_error_normalized": float(
                        min(
                            1.0,
                            max(
                                0.0,
                                (self._model_error_ema - self.search_model_error_low)
                                / (self.search_model_error_high - self.search_model_error_low),
                            ),
                        )
                    ),
                    "uncertainty_exploration_bonus_scale": float(
                        self.uncertainty_runtime_scale,
                    ),
                    "uncertainty_reliability_scale": float(
                        self.uncertainty_reliability_scale,
                    ),
                    "uncertainty_return_disagreement_ema": float(
                        self._uncertainty_disagreement_ema,
                    ),
                }
            )
            self._last_metrics.update(self._last_voc_search_metrics)
        else:
            (
                self._last_uncertainty_normalized_mean,
                self._last_uncertainty_bonus_mean,
                self._last_uncertainty_bonus_max,
            ) = uncertainty_diagnostics
        return results

    def _after_core_train_step(
        self,
        batch: dict[str, Any],
        hidden: torch.Tensor,
        obs0_pos: torch.Tensor,
        value_targets: torch.Tensor,
    ) -> dict[str, float]:
        if self.force_research_mode or not self.uncertainty_enabled:
            return {}
        batch_size = hidden.shape[0]
        batch_idx = torch.arange(batch_size, device=self.device)
        reward_features = torch.stack(
            [
                hidden[batch_idx, obs0_pos + 2 * step + 1]
                for step in range(self.unroll_steps)
            ],
            dim=1,
        ).detach().float()
        value_features = torch.stack(
            [
                hidden[batch_idx, obs0_pos + 2 * step]
                for step in range(self.unroll_steps)
            ],
            dim=1,
        ).detach().float()
        reward = _signed_hyperbolic(
            torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device),
        )
        value = _signed_hyperbolic(value_targets.detach().float())
        valid = torch.as_tensor(batch["mask"], dtype=torch.bool, device=self.device)

        self.uncertainty_optimizer.zero_grad(set_to_none=True)
        loss = self.uncertainty_probe.bootstrap_loss(
            reward_features, value_features, reward, value, valid,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.uncertainty_probe.parameters(), self.max_grad_norm)
        self.uncertainty_optimizer.step()
        with torch.no_grad():
            reward_predictions = self.uncertainty_probe.reward(reward_features)
            value_predictions = self.uncertainty_probe.value(value_features)
            pair_valid = valid[:, :-1] & valid[:, 1:]
            has_return_pairs = bool(pair_valid.any())
            if has_return_pairs:
                return_predictions = (
                    reward_predictions[:, :-1]
                    + self.gamma * value_predictions[:, 1:]
                )
                return_target = reward[:, :-1] + self.gamma * value[:, 1:]
                disagreement = return_predictions.std(dim=-1, unbiased=False)
                mean_disagreement = disagreement[pair_valid].mean()
                mean_prediction = return_predictions.mean(dim=-1)
                surprise = (
                    mean_prediction[pair_valid] - return_target[pair_valid]
                ).abs().mean()
            else:
                mean_disagreement = reward.new_zeros(())
                surprise = reward.new_zeros(())
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
            observed_surprise = float(surprise.item())
            observed_zero_fraction = float(zero_fraction.item())
            if has_return_pairs and not self._uncertainty_ema_initialized:
                self._uncertainty_probe_loss_ema = observed_loss
                self._uncertainty_disagreement_ema = observed_disagreement
                self._uncertainty_surprise_ema = observed_surprise
                self._uncertainty_zero_reward_fraction_ema = observed_zero_fraction
                self._uncertainty_ema_initialized = True
            elif has_return_pairs:
                decay = self.uncertainty_ema_decay
                self._uncertainty_probe_loss_ema = (
                    decay * self._uncertainty_probe_loss_ema
                    + (1.0 - decay) * observed_loss
                )
                self._uncertainty_disagreement_ema = (
                    decay * self._uncertainty_disagreement_ema
                    + (1.0 - decay) * observed_disagreement
                )
                self._uncertainty_surprise_ema = (
                    decay * self._uncertainty_surprise_ema
                    + (1.0 - decay) * observed_surprise
                )
                self._uncertainty_zero_reward_fraction_ema = (
                    decay * self._uncertainty_zero_reward_fraction_ema
                    + (1.0 - decay) * observed_zero_fraction
                )
        return {
            "uncertainty_aux_loss": float(loss.detach().item()),
            "uncertainty_disagreement": float(mean_disagreement.item()),
            "uncertainty_schedule_scale": float(self.uncertainty_runtime_scale),
            "uncertainty_exploration_bonus_scale": float(
                self.uncertainty_runtime_scale,
            ),
            "uncertainty_reliability_scale": float(
                self.uncertainty_reliability_scale,
            ),
            "uncertainty_return_disagreement_ema": float(
                self._uncertainty_disagreement_ema,
            ),
            "uncertainty_calibration": float(self.uncertainty_calibration),
            "uncertainty_surprise_ema": float(self._uncertainty_surprise_ema),
            "uncertainty_normalized_disagreement": self._last_uncertainty_normalized_mean,
            "uncertainty_effective_bonus_mean": self._last_uncertainty_bonus_mean,
            "uncertainty_effective_bonus_max": self._last_uncertainty_bonus_max,
            "uncertainty_zero_reward_fraction": self._uncertainty_zero_reward_fraction_ema,
            **self._last_voc_search_metrics,
        }

    def save(self, path: Path) -> None:
        super().save(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update(
            {
                "checkpoint_version": LATENTIMZERO_CHECKPOINT_VERSION,
                "algorithm": "latentimzero",
                "architecture": "researchimzero_core+return_uncertainty_probe+voc_anytime_v10_1",
                "optimizer_state_dict": self.optimizer.state_dict(),
                "grad_scaler_state_dict": self._grad_scaler.state_dict(),
                "uncertainty_probe_state_dict": self.uncertainty_probe.state_dict(),
                "uncertainty_optimizer_state_dict": self.uncertainty_optimizer.state_dict(),
                "uncertainty_ema_initialized": self._uncertainty_ema_initialized,
                "uncertainty_probe_loss_ema": self._uncertainty_probe_loss_ema,
                "uncertainty_disagreement_ema": self._uncertainty_disagreement_ema,
                "uncertainty_surprise_ema": self._uncertainty_surprise_ema,
                "uncertainty_zero_reward_fraction_ema": self._uncertainty_zero_reward_fraction_ema,
                "last_uncertainty_normalized_mean": self._last_uncertainty_normalized_mean,
                "last_uncertainty_bonus_mean": self._last_uncertainty_bonus_mean,
                "last_uncertainty_bonus_max": self._last_uncertainty_bonus_max,
                "num_timesteps": self._num_timesteps,
                "total_timesteps_estimate": self._total_timesteps_estimate,
                "grad_step_count": self._grad_step_count,
                "model_error_ema": self._model_error_ema,
                "last_search_num_simulations": self._last_search_num_simulations,
                "adaptive_closed_loop_horizon": self._adaptive_closed_loop_horizon,
                "adaptive_closed_loop_stable_count": self._adaptive_closed_loop_stable_count,
                "closed_loop_error_ema": self._closed_loop_error_ema,
                "voc_classifier_state_dict": self.voc_classifier.state_dict(),
                "voc_optimizer_state_dict": self.voc_optimizer.state_dict(),
                "voc_label_schema_version": VOC_LABEL_SCHEMA_VERSION,
                "voc_rng_state": self._voc_rng.bit_generator.state,
                "voc_replay": list(self._voc_replay),
                "voc_audits": list(self._voc_audits),
                "voc_shadow_mode": self._voc_shadow_mode,
                "voc_minimum_budget_index": self._voc_minimum_budget_index,
                "voc_safety_streaks": self._voc_safety_streaks,
                "voc_update_count": self._voc_update_count,
                "last_voc_metrics": {
                    "loss": self._last_voc_loss,
                    "accuracy": self._last_voc_accuracy,
                    "brier": self._last_voc_brier,
                    "stop_precision": self._last_voc_stop_precision,
                    "ece": self._last_voc_ece,
                },
                "executed_budget_history": list(self._executed_budget_history),
                "model_expansion_history": list(self._model_expansion_history),
                "predicted_budget_history": list(self._predicted_budget_history),
                "requested_budget_history": list(self._requested_budget_history),
                "permitted_budget_history": list(self._permitted_budget_history),
                "executed_search_history": list(self._executed_search_history),
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
                "LatentImZero checkpoint v6 is incompatible with v11: v10.1 uses the "
                "ResearchImZero core and cannot safely migrate the deleted stochastic "
                "world-model/actor/critic architecture.",
            )
        if version not in {7, 8, 9, 10, LATENTIMZERO_CHECKPOINT_VERSION}:
            raise ValueError(
                f"Unsupported LatentImZero checkpoint version {version!r}; expected v7-v11.",
            )

        migrated_hyperparams = {**DEFAULT_HYPERPARAMS, **payload.get("hyperparams", {})}
        algo = cls(env, migrated_hyperparams, None, device)
        algo.tokenizer.load_state_dict(payload["tokenizer_state_dict"])
        algo.action_embed.load_state_dict(payload["action_embed_state_dict"])
        algo.transformer.load_state_dict(payload["transformer_state_dict"])
        algo.heads.load_state_dict(payload["heads_state_dict"])
        if "projector_state_dict" in payload:
            algo.projector.load_state_dict(payload["projector_state_dict"])
            algo.predictor.load_state_dict(payload["predictor_state_dict"])
        if "target_tokenizer_state_dict" in payload:
            algo.target_tokenizer.load_state_dict(payload["target_tokenizer_state_dict"])
            algo.target_action_embed.load_state_dict(
                payload.get("target_action_embed_state_dict", payload["action_embed_state_dict"]),
            )
            algo.target_transformer.load_state_dict(payload["target_transformer_state_dict"])
            algo.target_heads.load_state_dict(payload["target_heads_state_dict"])
        if algo.rnd and payload.get("rnd_state"):
            algo.rnd.load_checkpoint_state(payload["rnd_state"])
        if "replay_buffer_state" in payload:
            algo.buffer.load_state_dict(payload["replay_buffer_state"])
        if "loss_log_vars" in payload and payload["loss_log_vars"].shape == algo.loss_log_vars.shape:
            with torch.no_grad():
                algo.loss_log_vars.copy_(payload["loss_log_vars"].to(device))
        algo.uncertainty_probe.load_state_dict(
            payload["uncertainty_probe_state_dict"],
            strict=version >= 9,
        )
        if "optimizer_state_dict" in payload:
            algo.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "grad_scaler_state_dict" in payload:
            algo._grad_scaler.load_state_dict(payload["grad_scaler_state_dict"])
        if version >= 9 and "uncertainty_optimizer_state_dict" in payload:
            algo.uncertainty_optimizer.load_state_dict(
                payload["uncertainty_optimizer_state_dict"],
            )
        if version >= LATENTIMZERO_CHECKPOINT_VERSION:
            if payload.get("voc_label_schema_version") == VOC_LABEL_SCHEMA_VERSION:
                algo.voc_classifier.load_state_dict(payload["voc_classifier_state_dict"])
                if "voc_optimizer_state_dict" in payload:
                    algo.voc_optimizer.load_state_dict(payload["voc_optimizer_state_dict"])
                algo._voc_replay.extend(payload.get("voc_replay", []))
                algo._voc_audits.extend(payload.get("voc_audits", []))
                algo._voc_shadow_mode = bool(payload.get("voc_shadow_mode", True))
                algo._voc_minimum_budget_index = min(
                    len(algo.voc_budgets) - 1,
                    int(payload.get("voc_minimum_budget_index", len(algo.voc_budgets) - 1)),
                )
                saved_streaks = payload.get("voc_safety_streaks")
                if isinstance(saved_streaks, list):
                    for stage, streak in enumerate(
                        saved_streaks[: len(algo._voc_safety_streaks)],
                    ):
                        algo._voc_safety_streaks[stage] = int(streak)
                algo._voc_update_count = int(payload.get("voc_update_count", 0))
                last_voc = payload.get("last_voc_metrics", {})
                algo._last_voc_loss = float(last_voc.get("loss", 0.0))
                algo._last_voc_accuracy = float(last_voc.get("accuracy", 0.0))
                algo._last_voc_brier = float(last_voc.get("brier", 1.0))
                algo._last_voc_stop_precision = float(last_voc.get("stop_precision", 0.0))
                algo._last_voc_ece = float(last_voc.get("ece", 1.0))
                algo._executed_budget_history.extend(
                    payload.get("executed_budget_history", []),
                )
                algo._model_expansion_history.extend(
                    payload.get("model_expansion_history", []),
                )
                algo._predicted_budget_history.extend(
                    payload.get("predicted_budget_history", []),
                )
                algo._requested_budget_history.extend(
                    payload.get("requested_budget_history", []),
                )
                algo._permitted_budget_history.extend(
                    payload.get("permitted_budget_history", []),
                )
                algo._executed_search_history.extend(
                    payload.get("executed_search_history", []),
                )
                if "voc_rng_state" in payload:
                    algo._voc_rng.bit_generator.state = payload["voc_rng_state"]
        algo._uncertainty_ema_initialized = bool(
            payload.get("uncertainty_ema_initialized", False),
        )
        algo._uncertainty_probe_loss_ema = float(
            payload.get("uncertainty_probe_loss_ema", 0.0),
        )
        algo._uncertainty_disagreement_ema = float(
            payload.get("uncertainty_disagreement_ema", 0.0),
        )
        algo._uncertainty_surprise_ema = float(
            payload.get(
                "uncertainty_surprise_ema",
                math.sqrt(max(0.0, algo._uncertainty_probe_loss_ema)),
            ),
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
        algo._adaptive_closed_loop_horizon = min(
            algo.adaptive_closed_loop_max_horizon,
            int(payload.get("adaptive_closed_loop_horizon", algo.closed_loop_horizon)),
        )
        algo._adaptive_closed_loop_stable_count = int(
            payload.get("adaptive_closed_loop_stable_count", 0),
        )
        saved_error_ema = payload.get("closed_loop_error_ema")
        if isinstance(saved_error_ema, list):
            for depth, value in enumerate(saved_error_ema[: len(algo._closed_loop_error_ema)]):
                algo._closed_loop_error_ema[depth] = float(value)
        return algo
