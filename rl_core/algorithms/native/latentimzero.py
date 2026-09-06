"""LatentImZero: stochastic Transformer imagination actor-critic.

This module is intentionally independent from the UniZero/ResearchImZero
implementations.  It reuses their well-tested tokenizer, causal Transformer,
KV cache and scale-stable categorical helpers, but owns its stochastic state,
replay, optimizers and checkpoint format.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.preprocessing import (
    action_to_env,
    obs_batch_to_array,
    obs_flat_dim,
    obs_to_array,
)
from rl_core.algorithms.native.researchimzero import (
    _ActionEmbed,
    _CausalTransformer,
    _HalvingSchedule,
    _MinMaxStats,
    _ResearchImZeroBuffer,
    _SearchNode,
    _Tokenizer,
    _TransformerCache,
    _cache_positions,
    _select_action,
    _sequential_halving,
    _transformed_completed_qs,
)
from rl_core.algorithms.vec_env import (
    action_space as venv_action_space,
    num_envs_of,
    obs_space as venv_obs_space,
    vec_reset,
    vec_step,
)
from rl_core.composite_netbuilder import (
    build_component_mlp,
    build_composite_encoder,
    dimensions_from_spec,
    validate_composite_spec,
)


DEFAULT_HYPERPARAMS = {
    "model_learning_rate": 3e-4,
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "embed_dim": 64,
    "hidden_dim": 128,
    "num_layers": 2,
    "num_heads": 4,
    "dropout": 0.0,
    "context_length": 8,
    "stoch_variables": 16,
    "stoch_classes": 16,
    "use_stochastic_state": 1,
    "use_continue": 1,
    "unimix": 0.01,
    "kl_balance": 0.8,
    "free_bits": 0.25,
    "kl_coef": 1.0,
    "observation_loss_coef": 1.0,
    "value_support_size": 100,
    "buffer_size": 2_000,
    "batch_size": 32,
    "unroll_steps": 5,
    "td_steps": 5,
    "imagination_horizon": 15,
    "imagination_lambda": 0.95,
    "gamma": 0.99,
    "entropy_coef": 1e-3,
    "model_loss_coef": 1.0,
    "actor_loss_coef": 1.0,
    "critic_loss_coef": 1.0,
    "distill_coef": 0.5,
    "fresh_distill_fraction": 0.0,
    "ema_tau": 0.01,
    "ensemble_size": 3,
    "uncertainty_scale": 2.0,
    "uncertainty_low": 0.02,
    "uncertainty_high": 0.5,
    "num_sampled_actions": 8,
    "num_simulations": 16,
    "num_simulations_initial": 2,
    "planner_horizon": 3,
    "exploration_bonus_coef": 0.1,
    "learning_starts": 500,
    "world_model_learning_starts": 512,
    "policy_learning_starts": 5_000,
    "min_context_transitions_per_lane": 16,
    "model_error_ema_decay": 0.99,
    "ensemble_bootstrap_probability": 0.7,
    "value_minmax_delta": 0.01,
    "c_visit": 50.0,
    "c_scale": 0.1,
    "num_top_actions": 8,
    "train_freq": 1,
    "train_steps_per_iter": 1,
    "reanalyze_freq": 200,
    "reanalyze_batch_size": 32,
    "max_grad_norm": 10.0,
}


def _unimix_probs(logits: torch.Tensor, amount: float) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    if amount:
        probs = (1.0 - amount) * probs + amount / probs.shape[-1]
    return probs


def _symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def _symlog_scalar_to_two_hot(x: torch.Tensor, support_size: int) -> torch.Tensor:
    """Encode raw scalars on an integer support in symlog space."""
    bins = 2 * support_size + 1
    transformed = _symlog(x).clamp(-support_size, support_size)
    lower = transformed.floor()
    fraction = (transformed - lower).unsqueeze(-1)
    lower_index = (lower + support_size).long().clamp(0, bins - 1)
    upper_index = (lower_index + 1).clamp(0, bins - 1)
    target = torch.zeros(*x.shape, bins, dtype=x.dtype, device=x.device)
    target.scatter_(-1, lower_index.unsqueeze(-1), 1.0 - fraction)
    target.scatter_add_(-1, upper_index.unsqueeze(-1), fraction)
    return target


def _symlog_logits_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    """Decode a categorical expectation in symlog space through symexp."""
    support = torch.arange(
        -support_size, support_size + 1, dtype=logits.dtype, device=logits.device,
    )
    transformed = (F.softmax(logits, dim=-1) * support).sum(-1)
    return _symexp(transformed)


def _st_categorical(logits: torch.Tensor, unimix: float, sample: bool = True) -> torch.Tensor:
    """Straight-through one-hot sample, retaining gradients of mixed probs."""
    probs = _unimix_probs(logits, unimix)
    if sample:
        index = torch.distributions.Categorical(probs=probs).sample()
    else:
        index = probs.argmax(dim=-1)
    hard = F.one_hot(index, probs.shape[-1]).to(probs.dtype)
    return hard + probs - probs.detach()


def _categorical_kl(q_logits: torch.Tensor, p_logits: torch.Tensor, unimix: float) -> torch.Tensor:
    q = _unimix_probs(q_logits, unimix)
    p = _unimix_probs(p_logits, unimix)
    return (q * (q.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(-1)


def _stochastic_usage(post_logits: torch.Tensor, unimix: float) -> torch.Tensor:
    """Information usage: zero for a uniform posterior, one for deterministic."""
    probs = _unimix_probs(post_logits, unimix)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean()
    return (1.0 - entropy / math.log(post_logits.shape[-1])).clamp(0.0, 1.0)


def _balanced_kl(
    post_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    balance: float = 0.8,
    free_bits: float = 1.0,
    unimix: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dreamer KL; ``free_bits`` is a total budget split across variables."""
    variables = max(1, post_logits.shape[-2])
    free_nats_per_variable = free_bits / variables
    dynamics = _categorical_kl(post_logits.detach(), prior_logits, unimix)
    representation = _categorical_kl(post_logits, prior_logits.detach(), unimix)
    raw = _categorical_kl(post_logits, prior_logits, unimix).sum(-1)
    loss = (
        balance * dynamics.clamp_min(free_nats_per_variable).sum(-1).mean()
        + (1.0 - balance)
        * representation.clamp_min(free_nats_per_variable).sum(-1).mean()
    )
    return loss, raw.mean().detach()


def _lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    bootstrap: torch.Tensor,
    continue_prob: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """Continue-aware TD(lambda); all inputs are ``(B, H)`` except bootstrap."""
    out = torch.zeros_like(reward)
    next_return = bootstrap
    for t in reversed(range(reward.shape[1])):
        next_value = bootstrap if t == reward.shape[1] - 1 else value[:, t + 1]
        discount = gamma * continue_prob[:, t]
        next_return = reward[:, t] + discount * (
            (1.0 - lambda_) * next_value + lambda_ * next_return
        )
        out[:, t] = next_return
    return out


class _LatentReplay(_ResearchImZeroBuffer):
    """ResearchImZero streaming PER plus explicit terminal labels."""

    @staticmethod
    def _new_episode() -> dict[str, list[Any]]:
        return {
            "obs": [], "action": [], "reward": [], "next_obs": [],
            "policy_target": [], "done": [], "priority": [],
            "search_value": [], "reanalyzed_gen": [],
        }

    def add(
        self, obs: np.ndarray, action_flat: np.ndarray, reward: float,
        next_obs: np.ndarray, policy_target: np.ndarray, done: bool, lane: int = 0,
    ) -> None:
        cur = self._cur[lane]
        cur["obs"].append(np.asarray(obs, dtype=np.float32))
        cur["action"].append(np.asarray(action_flat, dtype=np.float32))
        cur["reward"].append(float(reward))
        cur["next_obs"].append(np.asarray(next_obs, dtype=np.float32))
        cur["policy_target"].append(np.asarray(policy_target, dtype=np.float32))
        cur["done"].append(float(done))
        cur["priority"].append(self._max_priority)
        cur["search_value"].append(self._NO_SEARCH_VALUE)
        cur["reanalyzed_gen"].append(-1)
        if done:
            self._flush_episode(lane)

    def _flush_episode(self, lane: int = 0) -> None:
        cur = self._cur[lane]
        if not cur["obs"]:
            return
        episode = {
            "obs": np.stack(cur["obs"]),
            "action": np.stack(cur["action"]),
            "reward": np.asarray(cur["reward"], dtype=np.float32),
            "next_obs": np.stack(cur["next_obs"]),
            "policy_target": np.stack(cur["policy_target"]),
            "done": np.asarray(cur["done"], dtype=np.float32),
            "priority": np.asarray(cur["priority"], dtype=np.float64),
            "search_value": np.asarray(cur["search_value"], dtype=np.float32),
            "reanalyzed_gen": np.asarray(cur["reanalyzed_gen"], dtype=np.int64),
        }
        self.episodes.append(episode)
        self._episode_alpha_sum.append(float(np.sum(episode["priority"] ** self.priority_alpha)))
        self._episode_last_reanalyzed_gen.append(-1.0)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
            self._episode_alpha_sum.pop(0)
            self._episode_last_reanalyzed_gen.pop(0)
        self._cur[lane] = self._new_episode()

    def sample(self, batch_size: int, unroll_steps: int, td_steps: int, gamma: float) -> dict[str, np.ndarray]:
        batch = super().sample(batch_size, unroll_steps, td_steps, gamma)
        done = np.zeros((batch_size, unroll_steps), dtype=np.float32)
        for b, (ep_i, start) in enumerate(zip(batch["episode_idx"], batch["timestep"])):
            source = self._cur[-int(ep_i) - 1] if ep_i < 0 else self.episodes[int(ep_i)]
            n = min(unroll_steps, len(source["done"]) - int(start))
            if n > 0:
                done[b, :n] = np.asarray(source["done"][int(start) : int(start) + n])
        batch["done"] = done
        return batch


@dataclass
class _LatentState:
    h: torch.Tensor
    z: torch.Tensor
    cache: _TransformerCache
    prior_logits: torch.Tensor
    post_logits: torch.Tensor
    prior_pending: bool = False


class _LatentSearchNode(_SearchNode):
    """ResearchImZero Gumbel node with state-dependent continuation."""

    __slots__ = ("latent_state", "edge_discount", "depth")

    def __init__(self, prior: float, parent: "_LatentSearchNode | None" = None) -> None:
        super().__init__(prior, parent)
        self.latent_state: _LatentState | None = None
        self.edge_discount = 1.0
        self.depth = 0 if parent is None else parent.depth + 1

    def expand_latent(
        self,
        priors: np.ndarray,
        state: _LatentState,
        reward: float,
        edge_discount: float,
        candidate_actions: list[Any] | None = None,
    ) -> None:
        self.latent_state = state
        self.reward_value = reward
        self.edge_discount = edge_discount
        self.children = [_LatentSearchNode(float(prior), self) for prior in priors]
        if candidate_actions is not None:
            for child, action in zip(self.children, candidate_actions):
                child.candidate_action = action

    def q_of_child(self, idx: int, discount: float) -> float:
        del discount
        child = self.children[idx]
        return child.reward() + child.edge_discount * child.value()

    def v_mix(self, discount: float) -> float:
        del discount
        priors = np.asarray([child.prior for child in self.children], dtype=np.float64)
        probs = np.exp(priors - priors.max())
        probs /= probs.sum()
        visited = [(i, child) for i, child in enumerate(self.children) if child.expanded()]
        if not visited:
            return self.value()
        mass = sum(probs[i] for i, _ in visited)
        q_sum = sum(probs[i] * self.q_of_child(i, 1.0) for i, _ in visited)
        return (self.value() + self.children_visit_sum() * q_sum / max(mass, 1e-8)) / (
            1.0 + self.children_visit_sum()
        )


def _backpropagate_variable_discount(
    path: list[_LatentSearchNode], leaf_value: float, minmax: _MinMaxStats,
) -> None:
    value = leaf_value
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        value = node.reward() + node.edge_discount * value
        minmax.update(value)


class _StochasticWorldModel(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        stoch_variables: int,
        stoch_classes: int,
        support_size: int,
        unimix: float,
        ensemble_size: int,
        encoder: nn.Module | None = None,
        component_config: dict[str, dict[str, Any]] | None = None,
        ffn_multiplier: int = 4,
        rotary_emb: bool = True,
    ) -> None:
        super().__init__()
        components = component_config or {}
        self.embed_dim = embed_dim
        self.action_dim = obs_flat_dim(action_space)
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        self.stoch_variables = stoch_variables
        self.stoch_classes = stoch_classes
        self.stoch_dim = stoch_variables * stoch_classes
        self.support_size = support_size
        self.unimix = unimix
        self.tokenizer = _Tokenizer(
            observation_space, embed_dim, hidden_dim, encoder, components.get("tokenizer"),
        )
        self.action_embed = _ActionEmbed(self.action_dim, embed_dim, self.discrete)
        self.z_embed = nn.Linear(self.stoch_dim, embed_dim)
        self.transformer = _CausalTransformer(
            embed_dim, num_layers, num_heads, dropout,
            rotary_emb=rotary_emb, ffn_multiplier=ffn_multiplier,
        )
        self.prior_head = (
            build_component_mlp(embed_dim, self.stoch_dim, components["stochastic_prior"])
            if "stochastic_prior" in components else nn.Linear(embed_dim, self.stoch_dim)
        )
        self.posterior_head = (
            build_component_mlp(2 * embed_dim, self.stoch_dim, components["stochastic_posterior"])
            if "stochastic_posterior" in components else nn.Sequential(
                nn.Linear(2 * embed_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, self.stoch_dim),
            )
        )
        self.feature_norm = nn.LayerNorm(embed_dim + self.stoch_dim)
        feature_mlp = (
            build_component_mlp(
                embed_dim + self.stoch_dim, hidden_dim, components["state_feature"],
                final_activation=nn.ELU(),
            )
            if "state_feature" in components else nn.Sequential(
                nn.Linear(embed_dim + self.stoch_dim, hidden_dim), nn.ELU(),
            )
        )
        self.feature_head = nn.Sequential(feature_mlp, nn.LayerNorm(hidden_dim))
        self.feat_dim = hidden_dim
        self.observation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, embed_dim),
        )
        self.reward_head = (
            build_component_mlp(hidden_dim, 2 * support_size + 1, components["reward"])
            if "reward" in components else nn.Linear(hidden_dim, 2 * support_size + 1)
        )
        self.continue_head = (
            build_component_mlp(hidden_dim, 1, components["continue"])
            if "continue" in components else nn.Linear(hidden_dim, 1)
        )
        self.ensemble_prior = nn.ModuleList(
            [
                build_component_mlp(embed_dim, self.stoch_dim, components["ensemble_prior"])
                if "ensemble_prior" in components else nn.Linear(embed_dim, self.stoch_dim)
                for _ in range(max(1, ensemble_size))
            ]
        )
        self.ensemble_reward = nn.ModuleList(
            [
                build_component_mlp(hidden_dim, 1, components["ensemble_reward"])
                if "ensemble_reward" in components else nn.Linear(hidden_dim, 1)
                for _ in range(max(1, ensemble_size))
            ]
        )

    def _shape_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.view(*logits.shape[:-1], self.stoch_variables, self.stoch_classes)

    def feat(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_flat = z.flatten(-2)
        return self.feature_head(self.feature_norm(torch.cat([h, z_flat], dim=-1)))

    def initial_state(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.embed_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_variables, self.stoch_classes, device=device)
        return h, z

    def _action_token(self, action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Matrix multiplication preserves the straight-through actor path for
        # discrete Gumbel samples; _ActionEmbed.argmax is only suitable for data.
        action_embed = (
            action @ self.action_embed.embedding.weight
            if self.discrete else self.action_embed(action)
        )
        return action_embed + self.z_embed(z.flatten(-2))

    def _empty_cache(self) -> _TransformerCache:
        return _TransformerCache(len(self.transformer.blocks))

    def observe(
        self,
        obs: torch.Tensor,
        states: list[_LatentState | None],
        prev_action: torch.Tensor | None = None,
        sample: bool = True,
    ) -> list[_LatentState]:
        b = obs.shape[0]
        if prev_action is not None:
            active = [i for i, state in enumerate(states) if state is not None]
            if any(states[i].prior_pending for i in active):  # type: ignore[union-attr]
                raise ValueError("cannot advance an already pending prior state")
            if active:
                advanced = self.prior_step(
                    [states[i] for i in active], prev_action[active], sample=sample,  # type: ignore[list-item]
                )
                states = list(states)
                for index, state in zip(active, advanced):
                    states[index] = state
        obs_embed = self.tokenizer(obs)
        h_rows: list[torch.Tensor] = []
        caches_out: list[_TransformerCache] = []
        prior_rows: list[torch.Tensor] = []
        for state in states:
            if state is None:
                h = torch.zeros(self.embed_dim, dtype=obs_embed.dtype, device=obs.device)
                cache = self._empty_cache()
                prior = self._shape_logits(self.prior_head(h))
            else:
                h, cache, prior = state.h, state.cache, state.prior_logits
            h_rows.append(h)
            caches_out.append(cache)
            prior_rows.append(prior)
        h = torch.stack(h_rows)
        prior_logits = torch.stack(prior_rows)
        post_logits = self._shape_logits(self.posterior_head(torch.cat([h, obs_embed], dim=-1)))
        z = (
            _st_categorical(post_logits, self.unimix, sample)
            if getattr(self, "use_stochastic_state", True)
            else torch.zeros_like(post_logits)
        )
        return [
            _LatentState(h[i], z[i], caches_out[i], prior_logits[i], post_logits[i], False)
            for i in range(b)
        ]

    def prior_step(
        self, states: list[_LatentState], action: torch.Tensor, sample: bool = True,
    ) -> list[_LatentState]:
        caches = [s.cache for s in states]
        old_z = torch.stack([s.z for s in states])
        token = self._action_token(action, old_z)
        pos, epos = _cache_positions(caches, str(action.device))
        h, caches_out = self.transformer.forward_incremental_batch(token, pos, caches, epos)
        prior_logits = self._shape_logits(self.prior_head(h))
        z = (
            _st_categorical(prior_logits, self.unimix, sample)
            if getattr(self, "use_stochastic_state", True)
            else torch.zeros_like(prior_logits)
        )
        return [
            _LatentState(h[i], z[i], caches_out[i], prior_logits[i], prior_logits[i], True)
            for i in range(action.shape[0])
        ]

    def features(self, states: list[_LatentState]) -> torch.Tensor:
        return self.feat(torch.stack([s.h for s in states]), torch.stack([s.z for s in states]))

    def reward_logits(self, feat: torch.Tensor) -> torch.Tensor:
        return self.reward_head(feat)

    def reward(self, feat: torch.Tensor) -> torch.Tensor:
        return _symlog_logits_to_scalar(self.reward_logits(feat), self.support_size)

    def continue_prob(self, feat: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "use_continue", True):
            return torch.ones(feat.shape[:-1], device=feat.device, dtype=feat.dtype)
        return torch.sigmoid(self.continue_head(feat).squeeze(-1))

    def disagreement(self, states: list[_LatentState], feat: torch.Tensor | None = None) -> torch.Tensor:
        h = torch.stack([s.h for s in states])
        if feat is None:
            feat = self.features(states)
        prior_probs = torch.stack([
            _unimix_probs(self._shape_logits(head(h)), self.unimix)
            for head in self.ensemble_prior
        ])
        reward = torch.stack([head(feat).squeeze(-1) for head in self.ensemble_reward])
        return prior_probs.var(0, unbiased=False).mean(dim=(-1, -2)) + reward.var(0, unbiased=False)


class _Actor(nn.Module):
    def __init__(
        self, feat_dim: int, action_space: gym.Space, hidden_dim: int,
        component_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        self.trunk = (
            build_component_mlp(
                feat_dim, hidden_dim, component_config, final_activation=nn.ELU(),
            )
            if component_config is not None else nn.Sequential(
                nn.Linear(feat_dim, hidden_dim), nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            )
        )
        if self.discrete:
            self.action_dim = int(action_space.n)
            self.logits_head = nn.Linear(hidden_dim, self.action_dim)
        else:
            self.action_dim = int(np.prod(action_space.shape))
            self.mean_head = nn.Linear(hidden_dim, self.action_dim)
            self.log_std_head = nn.Linear(hidden_dim, self.action_dim)
            low = np.where(np.isfinite(action_space.low), action_space.low, -1.0).reshape(-1)
            high = np.where(np.isfinite(action_space.high), action_space.high, 1.0).reshape(-1)
            self.register_buffer("action_scale", torch.as_tensor((high - low) / 2, dtype=torch.float32))
            self.register_buffer("action_bias", torch.as_tensor((high + low) / 2, dtype=torch.float32))

    def distribution(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.trunk(feat)
        if self.discrete:
            return self.logits_head(h), None
        return self.mean_head(h), self.log_std_head(h).clamp(-5.0, 2.0)

    def sample(self, feat: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = self.distribution(feat)
        if self.discrete:
            probs = F.softmax(first, -1)
            action = (
                F.one_hot(first.argmax(-1), self.action_dim).float()
                if deterministic else F.gumbel_softmax(first, tau=1.0, hard=True)
            )
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1)
            return action, entropy
        assert second is not None
        raw = first if deterministic else first + second.exp() * torch.randn_like(first)
        action = torch.tanh(raw) * self.action_scale + self.action_bias
        entropy = (second + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)
        return action, entropy


class _CategoricalCritic(nn.Module):
    def __init__(
        self, feat_dim: int, hidden_dim: int, support_size: int,
        component_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.support_size = support_size
        self.net = (
            build_component_mlp(feat_dim, 2 * support_size + 1, component_config)
            if component_config is not None else nn.Sequential(
                nn.Linear(feat_dim, hidden_dim), nn.ELU(),
                nn.Linear(hidden_dim, 2 * support_size + 1),
            )
        )

    def logits(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return _symlog_logits_to_scalar(self.logits(feat), self.support_size)


class NativeLatentImZero(CustomAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self._obs_space = venv_obs_space(env)
        self._action_space = venv_action_space(env)
        self.discrete = isinstance(self._action_space, gym.spaces.Discrete)
        self.n_actions = int(self._action_space.n) if self.discrete else None
        self.action_dim = obs_flat_dim(self._action_space)
        self.support_size = max(1, int(hyperparams.get("value_support_size", 100)))
        network_spec = hyperparams.get("network_spec")
        if network_spec is not None:
            network_spec = validate_composite_spec(network_spec, "latentimzero")
            self.hyperparams["network_spec"] = network_spec
        dimensions = dimensions_from_spec(
            network_spec,
            "latentimzero",
            {
                "embed_dim": int(hyperparams.get("embed_dim", 64)),
                "hidden_dim": int(hyperparams.get("hidden_dim", 128)),
                "num_layers": int(hyperparams.get("num_layers", 2)),
                "num_heads": int(hyperparams.get("num_heads", 4)),
                "ffn_multiplier": 4,
                "dropout": float(hyperparams.get("dropout", 0.0)),
                "rotary_emb": 1,
                "stoch_variables": int(hyperparams.get("stoch_variables", 16)),
                "stoch_classes": int(hyperparams.get("stoch_classes", 16)),
            },
        )
        embed_dim = int(dimensions["embed_dim"])
        hidden_dim = int(dimensions["hidden_dim"])
        components = network_spec["components"] if network_spec is not None else {}
        encoder = build_composite_encoder(self._obs_space, network_spec) if network_spec is not None else None
        self.world_model = _StochasticWorldModel(
            self._obs_space, self._action_space, embed_dim, hidden_dim,
            int(dimensions["num_layers"]), int(dimensions["num_heads"]),
            float(dimensions["dropout"]),
            int(dimensions["stoch_variables"]), int(dimensions["stoch_classes"]),
            self.support_size, float(hyperparams.get("unimix", 0.01)),
            int(hyperparams.get("ensemble_size", 3)),
            encoder=encoder,
            component_config=components,
            ffn_multiplier=int(dimensions["ffn_multiplier"]),
            rotary_emb=bool(int(dimensions["rotary_emb"])),
        ).to(device)
        self.world_model.use_stochastic_state = bool(int(hyperparams.get("use_stochastic_state", 1)))
        self.world_model.use_continue = bool(int(hyperparams.get("use_continue", 1)))
        self.target_tokenizer = copy.deepcopy(self.world_model.tokenizer).eval()
        for p in self.target_tokenizer.parameters():
            p.requires_grad_(False)
        self.actor = _Actor(
            self.world_model.feat_dim, self._action_space, hidden_dim, components.get("actor"),
        ).to(device)
        self.critic = _CategoricalCritic(
            self.world_model.feat_dim, hidden_dim, self.support_size, components.get("critic"),
        ).to(device)
        self.ema_critic = copy.deepcopy(self.critic).eval()
        for p in self.ema_critic.parameters():
            p.requires_grad_(False)
        self.model_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=float(hyperparams.get("model_learning_rate", 3e-4)),
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(hyperparams.get("actor_learning_rate", 3e-4)),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(hyperparams.get("critic_learning_rate", 3e-4)),
        )
        self.batch_size = int(hyperparams.get("batch_size", 32))
        self.unroll_steps = max(1, int(hyperparams.get("unroll_steps", 5)))
        self.td_steps = max(1, int(hyperparams.get("td_steps", 5)))
        self.horizon = max(1, int(hyperparams.get("imagination_horizon", 15)))
        self.lambda_ = float(hyperparams.get("imagination_lambda", 0.95))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.unimix = float(hyperparams.get("unimix", 0.01))
        self.kl_balance = float(hyperparams.get("kl_balance", 0.8))
        self.free_bits = float(hyperparams.get("free_bits", 1.0))
        self.kl_coef = (
            float(hyperparams.get("kl_coef", 1.0))
            if self.world_model.use_stochastic_state else 0.0
        )
        self.observation_loss_coef = float(hyperparams.get("observation_loss_coef", 1.0))
        self.model_loss_coef = float(hyperparams.get("model_loss_coef", 1.0))
        self.actor_loss_coef = float(hyperparams.get("actor_loss_coef", 1.0))
        self.critic_loss_coef = float(hyperparams.get("critic_loss_coef", 1.0))
        self.entropy_coef = float(hyperparams.get("entropy_coef", 1e-3))
        self.distill_coef = float(hyperparams.get("distill_coef", 0.5))
        self.fresh_distill_fraction = float(hyperparams.get("fresh_distill_fraction", 0.0))
        self.ema_tau = float(hyperparams.get("ema_tau", 0.01))
        self.uncertainty_scale = float(hyperparams.get("uncertainty_scale", 2.0))
        self.uncertainty_low = float(hyperparams.get("uncertainty_low", 0.02))
        self.uncertainty_high = float(hyperparams.get("uncertainty_high", 0.5))
        self.num_sampled_actions = max(2, int(hyperparams.get("num_sampled_actions", 8)))
        self.num_simulations = max(2, int(hyperparams.get("num_simulations", 16)))
        self.num_simulations_initial = max(1, int(hyperparams.get("num_simulations_initial", 2)))
        self.num_top_actions = max(2, int(hyperparams.get("num_top_actions", 8)))
        self.c_visit = float(hyperparams.get("c_visit", 50.0))
        self.c_scale = float(hyperparams.get("c_scale", 0.1))
        self.value_minmax_delta = float(hyperparams.get("value_minmax_delta", 0.01))
        self.planner_horizon = max(1, int(hyperparams.get("planner_horizon", 3)))
        self.exploration_bonus_coef = float(hyperparams.get("exploration_bonus_coef", 0.1))
        self.learning_starts = int(hyperparams.get("learning_starts", 500))
        self.world_model_learning_starts = int(
            hyperparams.get("world_model_learning_starts", self.learning_starts),
        )
        self.policy_learning_starts = int(
            hyperparams.get("policy_learning_starts", self.learning_starts),
        )
        self.min_context_transitions_per_lane = max(
            0, int(hyperparams.get("min_context_transitions_per_lane", 16)),
        )
        self.model_error_ema_decay = float(hyperparams.get("model_error_ema_decay", 0.99))
        self.ensemble_bootstrap_probability = float(
            hyperparams.get("ensemble_bootstrap_probability", 0.7),
        )
        self.train_freq = max(1, int(hyperparams.get("train_freq", 1)))
        self.train_steps_per_iter = max(1, int(hyperparams.get("train_steps_per_iter", 1)))
        self.reanalyze_freq = max(1, int(hyperparams.get("reanalyze_freq", 200)))
        self.reanalyze_batch_size = max(0, int(hyperparams.get("reanalyze_batch_size", 32)))
        self.max_grad_norm = float(hyperparams.get("max_grad_norm", 10.0))
        sample_obs = obs_to_array(self._obs_space.sample(), self._obs_space)
        self._obs_shape = sample_obs.shape
        self.buffer = _LatentReplay(
            int(hyperparams.get("buffer_size", 2_000)), self._obs_shape, self.action_dim,
            self.n_actions if self.discrete else self.action_dim,
            int(hyperparams.get("context_length", 8)), num_lanes=num_envs_of(env),
        )
        self._lane_states: list[_LatentState | None] = [None] * num_envs_of(env)
        self._lane_prev_action: np.ndarray | None = None
        self._eval_state: _LatentState | None = None
        self._eval_prev_action: np.ndarray | None = None
        self._last_metrics: dict[str, float] = {}
        self._uncertainty_ema = self.uncertainty_high
        # Normalized error e/(1+e)=0.5 starts conservatively at the default
        # uncertainty_high, then can decay below it from actual symlog error.
        self._model_error_ema = 1.0
        self._posterior_kl_ema = self.free_bits
        self._last_planning_budget = self.num_simulations_initial
        self._last_imagination_horizon = 1

    def _effective_uncertainty(
        self, disagreement: torch.Tensor, posterior_kl: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        kl = self._posterior_kl_ema if posterior_kl is None else posterior_kl
        kl_tensor = torch.as_tensor(kl, dtype=disagreement.dtype, device=disagreement.device)
        disagreement_norm = disagreement / (disagreement + max(self.uncertainty_scale, 1e-6))
        kl_per_variable = kl_tensor / max(1, self.world_model.stoch_variables)
        kl_norm = kl_per_variable / max(math.log(self.world_model.stoch_classes), 1e-6)
        error = torch.as_tensor(
            self._model_error_ema, dtype=disagreement.dtype, device=disagreement.device,
        )
        error_norm = error / (1.0 + error)
        # The accuracy term dominates early, but all components decay smoothly:
        # no raw-reward floor can pin planning to its minimum forever.
        components = torch.stack(torch.broadcast_tensors(
            disagreement_norm.clamp(0.0, 1.0),
            kl_norm.clamp(0.0, 1.0),
            error_norm.clamp(0.0, 1.0),
        ))
        return 1.0 - torch.prod(1.0 - components, dim=0)

    def _confidence(self, uncertainty: torch.Tensor) -> torch.Tensor:
        span = max(1e-6, self.uncertainty_high - self.uncertainty_low)
        normalized = ((self.uncertainty_high - uncertainty) / span).clamp(0.0, 1.0)
        return (normalized.square() * 0.95 + 0.05).clamp(0.05, 1.0)

    def _planning_budget(self, uncertainty: float) -> int:
        span = max(1e-6, self.uncertainty_high - self.uncertainty_low)
        confidence = 1.0 - min(1.0, max(0.0, (uncertainty - self.uncertainty_low) / span))
        return int(round(self.num_simulations_initial + confidence * (self.num_simulations - self.num_simulations_initial)))

    def _gated_imagination_horizon(self, uncertainty: float) -> int:
        span = max(1e-6, self.uncertainty_high - self.uncertainty_low)
        confidence = 1.0 - min(1.0, max(0.0, (uncertainty - self.uncertainty_low) / span))
        return max(1, int(round(1 + confidence * (self.horizon - 1))))

    def _reconstruct_history(
        self,
        context_obs: np.ndarray,
        context_action: np.ndarray,
        context_valid: np.ndarray,
        current_obs: np.ndarray,
        *,
        sample: bool,
    ) -> list[_LatentState]:
        """Reconstruct all lanes batched; observations never enter the KV cache."""
        batch_size = len(current_obs)
        counts = np.asarray(context_valid, dtype=np.int64).sum(axis=1)
        states: list[_LatentState | None] = [None] * batch_size
        for t in range(int(counts.max(initial=0))):
            active = np.flatnonzero(counts > t)
            obs_t = torch.as_tensor(
                context_obs[active, t], dtype=torch.float32, device=self.device,
            )
            subset = [states[int(i)] for i in active]
            prev = None
            if t:
                prev = torch.as_tensor(
                    context_action[active, t - 1], dtype=torch.float32, device=self.device,
                )
            updated = self.world_model.observe(obs_t, subset, prev, sample=sample)
            for index, state in zip(active.tolist(), updated):
                states[index] = state
        # Grouping by history length makes one tokenizer/Transformer call per
        # distinct final transition while retaining original lane order.
        result: list[_LatentState | None] = [None] * batch_size
        for count in np.unique(counts):
            group = np.flatnonzero(counts == count)
            action_t = None if count == 0 else torch.as_tensor(
                context_action[group, int(count) - 1], dtype=torch.float32, device=self.device,
            )
            corrected = self.world_model.observe(
                torch.as_tensor(current_obs[group], dtype=torch.float32, device=self.device),
                [states[int(i)] for i in group], action_t, sample=sample,
            )
            for index, state in zip(group.tolist(), corrected):
                result[index] = state
        return [state for state in result if state is not None]

    def _candidate_actions(self, feat: torch.Tensor) -> tuple[torch.Tensor, np.ndarray, list[Any] | None]:
        first, second = self.actor.distribution(feat)
        if self.discrete:
            actions = torch.eye(self.n_actions, dtype=torch.float32, device=self.device)
            return actions, first[0].detach().cpu().numpy().astype(np.float64), None
        assert second is not None
        count = self.num_sampled_actions
        raw = first.expand(count, -1) + second.exp().expand(count, -1) * torch.randn(count, self.action_dim, device=feat.device)
        actions = torch.tanh(raw) * self.actor.action_scale + self.actor.action_bias
        log_prior = -0.5 * (((raw - first) / second.exp().clamp_min(1e-6)) ** 2 + 2 * second).sum(-1)
        candidates = [row.detach().cpu().numpy().astype(np.float32) for row in actions]
        return actions, log_prior.detach().cpu().numpy().astype(np.float64), candidates

    @torch.no_grad()
    def _plan_state(self, state: _LatentState, deterministic: bool) -> dict[str, Any]:
        feat = self.world_model.features([state])
        disagreement = self.world_model.disagreement([state], feat)
        posterior_kl = _categorical_kl(
            state.post_logits.unsqueeze(0), state.prior_logits.unsqueeze(0), self.unimix,
        ).sum(-1)
        root_uncertainty = float(self._effective_uncertainty(disagreement, posterior_kl).item())
        budget = self._planning_budget(root_uncertainty)
        self._last_planning_budget = budget
        root_actions, root_priors, root_candidates = self._candidate_actions(feat)
        root = _LatentSearchNode(1.0)
        root.expand_latent(root_priors, state, 0.0, 1.0, root_candidates)
        root.visit_count = 1
        root.value_sum = float(self.ema_critic(feat).item())
        slots = len(root.children)
        top = max(2, min(self.num_top_actions, slots, max(2, budget // 2)))
        gumbel = np.random.gumbel(size=slots)
        schedule = _HalvingSchedule(budget, top)
        minmax = _MinMaxStats(self.value_minmax_delta)
        max_depth = 0

        for simulation in range(budget):
            if simulation == 0:
                scores = gumbel + np.asarray([child.prior for child in root.children])
                root.selected_children_idx = list(np.argsort(-scores)[:top])
            node = root
            path = [root]
            while node.expanded() and node.depth < self.planner_horizon:
                child_index = _select_action(
                    node, minmax, self.gamma, self.c_visit, self.c_scale,
                )
                node = node.children[child_index]
                path.append(node)
            parent = path[-2] if len(path) > 1 else root
            if node.depth >= self.planner_horizon and node.latent_state is not None:
                leaf_value = float(self.ema_critic(self.world_model.features([node.latent_state])).item())
                _backpropagate_variable_discount(path, leaf_value, minmax)
                max_depth = max(max_depth, node.depth)
            else:
                if self.discrete:
                    index = parent.children.index(node)
                    action = F.one_hot(
                        torch.tensor([index], device=self.device), self.n_actions,
                    ).float()
                else:
                    action = torch.as_tensor(
                        np.asarray(node.candidate_action)[None], dtype=torch.float32, device=self.device,
                    )
                assert parent.latent_state is not None
                next_state = self.world_model.prior_step([parent.latent_state], action, sample=True)[0]
                next_feat = self.world_model.features([next_state])
                edge_disagreement = self.world_model.disagreement([next_state], next_feat)
                edge_uncertainty = self._effective_uncertainty(edge_disagreement)
                reward = self.world_model.reward(next_feat)
                reward_with_bonus = reward + self.exploration_bonus_coef * edge_uncertainty
                edge_discount = self.gamma * self.world_model.continue_prob(next_feat)
                next_actions, priors, candidates = self._candidate_actions(next_feat)
                del next_actions
                node.expand_latent(
                    priors, next_state, float(reward_with_bonus.item()),
                    float(edge_discount.item()), candidates,
                )
                leaf_value = float(self.ema_critic(next_feat).item())
                _backpropagate_variable_discount(path, leaf_value, minmax)
                max_depth = max(max_depth, node.depth)
            if schedule.maybe_advance(simulation):
                _sequential_halving(
                    root, gumbel, minmax, schedule.current_top,
                    self.gamma, self.c_visit, self.c_scale,
                )

        transformed = _transformed_completed_qs(
            root, minmax, self.gamma, self.c_visit, self.c_scale,
        )
        improved = root.improved_policy(transformed)
        best = (
            int(root.selected_children_idx[0])
            if root.selected_children_idx else int(np.argmax(improved))
        )
        if self.discrete:
            policy_target = improved.astype(np.float32)
            env_action: Any = int(np.argmax(improved)) if deterministic else int(
                np.random.choice(self.n_actions, p=improved),
            )
        else:
            candidates_array = np.stack([child.candidate_action for child in root.children])
            policy_target = (improved[:, None] * candidates_array).sum(0).astype(np.float32)
            env_action = candidates_array[best].astype(np.float32)
        self._uncertainty_ema = 0.99 * self._uncertainty_ema + 0.01 * root_uncertainty
        return {
            "env_action": env_action,
            "policy_target": policy_target,
            "value_target": root.value(),
            "uncertainty": root_uncertainty,
            "budget": budget,
            "simulation_count": budget,
            "max_depth": max_depth,
        }

    @torch.no_grad()
    def search(
        self,
        obs_batch: np.ndarray,
        states: list[_LatentState | None] | None = None,
        prev_action: np.ndarray | None = None,
        deterministic: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        b = obs_batch.shape[0]
        states = [None] * b if states is None else states
        action_t = None if prev_action is None else torch.as_tensor(prev_action, dtype=torch.float32, device=self.device)
        posterior = self.world_model.observe(
            torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device), states, action_t, sample=False,
        )
        det = np.zeros(b, dtype=bool) if deterministic is None else np.broadcast_to(deterministic, (b,))
        results = [self._plan_state(posterior[i], bool(det[i])) for i in range(b)]
        for result, state in zip(results, posterior):
            result["state"] = state
        return results

    def _imagine(self, states: list[_LatentState]) -> tuple[torch.Tensor, ...]:
        feats, rewards, continues, entropies, uncertainties = [], [], [], [], []
        current = states
        with torch.no_grad():
            disagreement = self.world_model.disagreement(
                current, self.world_model.features(current),
            )
            root_uncertainty = float(self._effective_uncertainty(disagreement).mean().item())
        horizon = self._gated_imagination_horizon(root_uncertainty)
        self._last_imagination_horizon = horizon
        for _ in range(horizon):
            feat = self.world_model.features(current)
            action, entropy = self.actor.sample(feat)
            current = self.world_model.prior_step(current, action)
            next_feat = self.world_model.features(current)
            feats.append(next_feat)
            rewards.append(self.world_model.reward(next_feat))
            continues.append(self.world_model.continue_prob(next_feat))
            entropies.append(entropy)
            uncertainties.append(self.world_model.disagreement(current, next_feat))
        return tuple(torch.stack(x, dim=1) for x in (feats, rewards, continues, entropies, uncertainties))

    def _update_ema_critic(self) -> None:
        with torch.no_grad():
            for target, online in zip(self.ema_critic.parameters(), self.critic.parameters()):
                target.mul_(1.0 - self.ema_tau).add_(online, alpha=self.ema_tau)
            for target, online in zip(
                self.target_tokenizer.parameters(), self.world_model.tokenizer.parameters(),
            ):
                target.mul_(1.0 - self.ema_tau).add_(online, alpha=self.ema_tau)

    @torch.no_grad()
    def _grounded_targets(
        self, batch: dict[str, np.ndarray], starts: list[_LatentState] | None = None,
    ) -> torch.Tensor:
        if starts is None:
            starts = self._reconstruct_history(
                batch["context_obs"], batch["context_action"], batch["context_valid"],
                batch["obs0"], sample=False,
            )
        states = starts
        landing_states: list[list[_LatentState]] = []
        bootstrap_action = torch.as_tensor(
            batch["bootstrap_action"], dtype=torch.float32, device=self.device,
        )
        bootstrap_next_obs = torch.as_tensor(
            batch["bootstrap_next_obs"], dtype=torch.float32, device=self.device,
        )
        for step in range(bootstrap_action.shape[1]):
            prior = self.world_model.prior_step(states, bootstrap_action[:, step], sample=False)
            states = self.world_model.observe(
                bootstrap_next_obs[:, step], prior, sample=False,
            )
            landing_states.append(states)
        values = torch.zeros(len(starts), self.unroll_steps, device=self.device)
        horizons = torch.as_tensor(batch["td_horizon"], device=self.device)
        landing = (
            torch.arange(self.unroll_steps, device=self.device)[None] + horizons - 1
        )
        if landing_states:
            all_features = torch.stack(
                [self.world_model.features(step_states) for step_states in landing_states],
            )  # (T, B, D)
            valid = (landing >= 0) & (landing < len(landing_states))
            batch_index, unroll_index = valid.nonzero(as_tuple=True)
            selected = all_features[
                landing[batch_index, unroll_index], batch_index,
            ]
            values[batch_index, unroll_index] = self.ema_critic(selected)
        td_reward = torch.as_tensor(batch["td_reward"], dtype=torch.float32, device=self.device)
        td_discount = torch.as_tensor(batch["td_discount"], dtype=torch.float32, device=self.device)
        bootstrap_mask = torch.as_tensor(
            batch["td_bootstrap_mask"], dtype=torch.float32, device=self.device,
        )
        return td_reward + td_discount * bootstrap_mask * values

    def _train_step(self, train_policy: bool = True) -> dict[str, float]:
        batch = self.buffer.sample(self.batch_size, self.unroll_steps, self.td_steps, self.gamma)
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=self.device)
        is_weight = torch.as_tensor(batch["is_weight"], dtype=torch.float32, device=self.device)
        policy_target = torch.as_tensor(batch["policy_target"], dtype=torch.float32, device=self.device)
        states = self._reconstruct_history(
            batch["context_obs"], batch["context_action"], batch["context_valid"],
            batch["obs0"], sample=True,
        )
        real_targets = self._grounded_targets(batch, states)
        reward_loss = torch.zeros((), device=self.device)
        continue_loss = torch.zeros((), device=self.device)
        kl_loss = torch.zeros((), device=self.device)
        ensemble_loss = torch.zeros((), device=self.device)
        observation_loss = torch.zeros((), device=self.device)
        raw_kl_values: list[torch.Tensor] = []
        posterior_entropies: list[torch.Tensor] = []
        prior_entropies: list[torch.Tensor] = []
        z_feature_deltas: list[torch.Tensor] = []
        model_errors: list[torch.Tensor] = []
        for t in range(self.unroll_steps):
            m = mask[:, t] * is_weight
            denom = mask[:, t].sum().clamp_min(1.0)
            prior_states = self.world_model.prior_step(states, action[:, t])
            prior_feat = self.world_model.features(prior_states)
            r_target = _symlog_scalar_to_two_hot(reward[:, t], self.support_size)
            r_each = -(r_target * F.log_softmax(self.world_model.reward_logits(prior_feat), -1)).sum(-1)
            reward_loss = reward_loss + (m * r_each).sum() / denom
            c_each = F.binary_cross_entropy_with_logits(
                self.world_model.continue_head(prior_feat).squeeze(-1), 1.0 - done[:, t], reduction="none",
            )
            continue_loss = continue_loss + (m * c_each).sum() / denom
            posterior_states = self.world_model.observe(next_obs[:, t], prior_states)
            post_logits = torch.stack([s.post_logits for s in posterior_states])
            prior_logits = torch.stack([s.prior_logits for s in prior_states])
            kl_t, raw_kl = _balanced_kl(
                post_logits, prior_logits, self.kl_balance, self.free_bits, self.unimix,
            )
            kl_loss = kl_loss + kl_t
            raw_kl_values.append(raw_kl)
            post_probs = _unimix_probs(post_logits, self.unimix)
            posterior_entropies.append(
                -(post_probs * post_probs.clamp_min(1e-8).log()).sum(-1).mean(),
            )
            prior_probs = _unimix_probs(prior_logits, self.unimix)
            prior_entropies.append(
                -(prior_probs * prior_probs.clamp_min(1e-8).log()).sum(-1).mean(),
            )
            posterior_feat = self.world_model.features(posterior_states)
            prediction = F.normalize(
                self.world_model.observation_predictor(posterior_feat), dim=-1,
            )
            with torch.no_grad():
                target_embed = F.normalize(self.target_tokenizer(next_obs[:, t]), dim=-1)
            obs_each = 1.0 - (prediction * target_embed).sum(-1)
            observation_loss = observation_loss + (m * obs_each).sum() / denom
            zero_z_feat = self.world_model.feat(
                torch.stack([s.h for s in posterior_states]),
                torch.zeros_like(torch.stack([s.z for s in posterior_states])),
            )
            z_feature_deltas.append((posterior_feat - zero_z_feat).square().mean().sqrt())
            model_errors.append(
                ((_symlog(self.world_model.reward(prior_feat)) - _symlog(reward[:, t])).abs()
                 * mask[:, t]).sum()
                / mask[:, t].sum().clamp_min(1.0),
            )
            h = torch.stack([s.h for s in prior_states])
            for prior_head, reward_head in zip(
                self.world_model.ensemble_prior, self.world_model.ensemble_reward,
            ):
                bootstrap = (
                    torch.rand(self.batch_size, device=self.device)
                    < self.ensemble_bootstrap_probability
                ).float() * mask[:, t]
                if not bool(bootstrap.any()):
                    bootstrap[torch.randint(self.batch_size, (1,), device=self.device)] = 1.0
                member_prior = self.world_model._shape_logits(prior_head(h))
                prior_error = _categorical_kl(
                    post_logits.detach(), member_prior, self.unimix,
                ).sum(-1)
                reward_error = (
                    reward_head(prior_feat).squeeze(-1) - _symlog(reward[:, t])
                ).square()
                ensemble_loss = ensemble_loss + (
                    bootstrap * (prior_error + reward_error)
                ).sum() / bootstrap.sum().clamp_min(1.0)
            states = posterior_states

        scale = float(self.unroll_steps)
        reward_loss /= scale
        continue_loss /= scale
        kl_loss /= scale
        ensemble_loss /= scale * len(self.world_model.ensemble_prior)
        observation_loss /= scale
        model_loss = (
            reward_loss + continue_loss + self.kl_coef * kl_loss
            + self.observation_loss_coef * observation_loss + 0.1 * ensemble_loss
        )
        self.model_optimizer.zero_grad()
        (self.model_loss_coef * model_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.max_grad_norm)
        self.model_optimizer.step()

        raw_kl = torch.stack(raw_kl_values).mean()
        posterior_entropy = torch.stack(posterior_entropies).mean()
        prior_entropy = torch.stack(prior_entropies).mean()
        stochastic_usage = (
            1.0 - posterior_entropy / math.log(self.world_model.stoch_classes)
        ).clamp(0.0, 1.0)
        kl_per_variable = raw_kl / self.world_model.stoch_variables
        z_feature_delta = torch.stack(z_feature_deltas).mean()
        observed_model_error = torch.stack(model_errors).mean().detach()
        decay = self.model_error_ema_decay
        self._model_error_ema = (
            decay * self._model_error_ema + (1.0 - decay) * float(observed_model_error.item())
        )
        self._posterior_kl_ema = (
            decay * self._posterior_kl_ema + (1.0 - decay) * float(raw_kl.item())
        )

        # Rebuild posterior states after the model update. These states include
        # the full replay context and anchor both real critic and imagination.
        with torch.no_grad():
            starts = self._reconstruct_history(
                batch["context_obs"], batch["context_action"], batch["context_valid"],
                batch["obs0"], sample=True,
            )
            real_states_by_step: list[list[_LatentState]] = []
            rollout_states = starts
            for t in range(self.unroll_steps):
                real_states_by_step.append(rollout_states)
                prior = self.world_model.prior_step(rollout_states, action[:, t], sample=False)
                rollout_states = self.world_model.observe(next_obs[:, t], prior, sample=True)
            real_feat = torch.stack(
                [self.world_model.features(step_states) for step_states in real_states_by_step],
                dim=1,
            )

        real_logits = self.critic.logits(real_feat.detach().reshape(-1, self.world_model.feat_dim))
        real_two_hot = _symlog_scalar_to_two_hot(real_targets.detach().reshape(-1), self.support_size)
        real_each = -(real_two_hot * F.log_softmax(real_logits, -1)).sum(-1).view(
            self.batch_size, self.unroll_steps,
        )
        real_critic_loss = (real_each * mask * is_weight[:, None]).sum() / mask.sum().clamp_min(1.0)

        actor_loss = torch.zeros((), device=self.device)
        imagined_critic_loss = torch.zeros((), device=self.device)
        distill_loss = torch.zeros((), device=self.device)
        fresh_loss = torch.zeros((), device=self.device)
        imagined_return_mean = torch.zeros((), device=self.device)
        confidence_mean = torch.zeros((), device=self.device)
        uncertainty_mean = torch.as_tensor(
            self._effective_uncertainty(torch.zeros((), device=self.device)),
        )
        model_bias = torch.zeros((), device=self.device)
        returns: torch.Tensor | None = None
        if train_policy:
            current_feat = real_feat.detach()
            first, _ = self.actor.distribution(
                current_feat.reshape(-1, self.world_model.feat_dim),
            )
            flat_policy_target = policy_target.reshape(-1, policy_target.shape[-1])
            if self.discrete:
                distill_each = -(flat_policy_target * F.log_softmax(first, -1)).sum(-1)
            else:
                mean_action = torch.tanh(first) * self.actor.action_scale + self.actor.action_bias
                distill_each = F.mse_loss(
                    mean_action, flat_policy_target, reduction="none",
                ).sum(-1)
            distill_loss = (
                distill_each.view(self.batch_size, self.unroll_steps) * mask
            ).sum() / mask.sum().clamp_min(1.0)

            for parameter in self.world_model.parameters():
                parameter.requires_grad_(False)
            imagined_feat, imagined_reward, imagined_continue, entropy, disagreement = self._imagine(starts)
            uncertainty = self._effective_uncertainty(disagreement)
            with torch.no_grad():
                target_value = self.ema_critic(imagined_feat)
                bootstrap = target_value[:, -1]
            returns = _lambda_return(
                imagined_reward, target_value, bootstrap, imagined_continue,
                self.gamma, self.lambda_,
            )
            confidence = self._confidence(uncertainty).detach()
            normalized_return = (
                (returns - returns.mean().detach())
                / returns.std(unbiased=False).detach().clamp_min(1.0)
            )
            actor_loss = (
                -(confidence * normalized_return).mean()
                - self.entropy_coef * entropy.mean()
                + self.distill_coef * distill_loss
            )
            self.actor_optimizer.zero_grad()
            (self.actor_loss_coef * actor_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            for parameter in self.world_model.parameters():
                parameter.requires_grad_(True)

            imagined_logits = self.critic.logits(
                imagined_feat.detach().reshape(-1, self.world_model.feat_dim),
            )
            imagined_target = _symlog_scalar_to_two_hot(
                returns.detach().reshape(-1), self.support_size,
            )
            imagined_each = -(
                imagined_target * F.log_softmax(imagined_logits, -1)
            ).sum(-1)
            imagined_critic_loss = (imagined_each * confidence.reshape(-1)).mean()
            imagined_return_mean = returns.mean()
            confidence_mean = confidence.mean()
            uncertainty_mean = uncertainty.mean()
            model_bias = imagined_return_mean.detach() - real_targets[:, 0].mean()

        critic_loss = real_critic_loss + 0.5 * imagined_critic_loss
        self.critic_optimizer.zero_grad()
        (self.critic_loss_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        self._update_ema_critic()

        fresh_count = (
            min(self.batch_size, int(round(self.batch_size * self.fresh_distill_fraction)))
            if train_policy else 0
        )
        if fresh_count > 0:
            with torch.no_grad():
                fresh_results = [self._plan_state(starts[i], False) for i in range(fresh_count)]
                fresh_targets = torch.as_tensor(
                    np.stack([r["policy_target"] for r in fresh_results]), dtype=torch.float32, device=self.device,
                )
            feat0 = self.world_model.features(starts[:fresh_count]).detach()
            first, _ = self.actor.distribution(feat0)
            if self.discrete:
                fresh_loss = -(fresh_targets * F.log_softmax(first, -1)).sum(-1).mean()
            else:
                mean_action = torch.tanh(first) * self.actor.action_scale + self.actor.action_bias
                fresh_loss = F.mse_loss(mean_action, fresh_targets)
            self.actor_optimizer.zero_grad()
            (self.distill_coef * fresh_loss).backward()
            self.actor_optimizer.step()
            policy_np = np.stack([r["policy_target"] for r in fresh_results])
            value_np = np.asarray([r["value_target"] for r in fresh_results], dtype=np.float32)
            self.buffer.update_reanalyzed_targets(
                batch["episode_idx"][:fresh_count], batch["timestep"][:fresh_count], policy_np, value_np,
            )
        with torch.no_grad():
            feat0 = self.world_model.features(starts)
            priority = (self.critic(feat0) - real_targets[:, 0]).abs().cpu().numpy()
        self.buffer.update_priorities(batch["episode_idx"], batch["timestep"], priority)
        return {
            "model_loss": float(model_loss.item()),
            "reward_loss": float(reward_loss.item()),
            "continue_loss": float(continue_loss.item()),
            "observation_loss": float(observation_loss.item()),
            "kl_loss": float(kl_loss.item()),
            "kl_active_loss": float(kl_loss.item()),
            "kl_raw": float(raw_kl.item()),
            "kl_per_variable": float(kl_per_variable.item()),
            "posterior_prior_kl": float(raw_kl.item()),
            "posterior_entropy": float(posterior_entropy.item()),
            "prior_entropy": float(prior_entropy.item()),
            "stochastic_usage": float(stochastic_usage.item()),
            "z_feature_delta": float(z_feature_delta.item()),
            "ensemble_loss": float(ensemble_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "real_critic_loss": float(real_critic_loss.item()),
            "imagined_critic_loss": float(imagined_critic_loss.item()),
            "real_target_mean": float(real_targets.mean().item()),
            "model_bias": float(model_bias.item()),
            "distill_loss": float((distill_loss + fresh_loss).item()),
            "imagined_return_mean": float(imagined_return_mean.item()),
            "imagined_confidence_mean": float(confidence_mean.item()),
            "uncertainty_mean": float(uncertainty_mean.item()),
            "uncertainty_disagreement": float(
                disagreement.mean().detach().item() if train_policy else 0.0
            ),
            "uncertainty_disagreement_normalized": float(
                (
                    disagreement.mean().detach()
                    / (disagreement.mean().detach() + max(self.uncertainty_scale, 1e-6))
                ).item() if train_policy else 0.0
            ),
            "uncertainty_kl": float(
                min(1.0, self._posterior_kl_ema / self.world_model.stoch_variables
                    / max(math.log(self.world_model.stoch_classes), 1e-6))
            ),
            "uncertainty_model_error": float(
                self._model_error_ema / (1.0 + self._model_error_ema)
            ),
            "model_error_ema": self._model_error_ema,
            "planning_budget": float(self._last_planning_budget),
            "imagination_horizon": float(self._last_imagination_horizon),
        }

    def _reanalyze(self) -> dict[str, float] | None:
        if self.reanalyze_batch_size <= 0 or self.buffer.num_replay_transitions == 0:
            return None
        ep_idx, timesteps, obs, ctx_obs, ctx_action, ctx_valid = self.buffer.sample_for_reanalyze(
            self.reanalyze_batch_size,
        )
        if not len(obs):
            return None
        with torch.no_grad():
            states = self._reconstruct_history(
                ctx_obs, ctx_action, ctx_valid, obs, sample=False,
            )
            results = [self._plan_state(state, False) for state in states]
        self.buffer.update_reanalyzed_targets(
            ep_idx, timesteps, np.stack([r["policy_target"] for r in results]),
            np.asarray([r["value_target"] for r in results], dtype=np.float32),
        )
        return {"reanalyze_uncertainty": float(np.mean([r["uncertainty"] for r in results]))}

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        n_envs = num_envs_of(self.env)
        obs_arr = obs_batch_to_array(vec_reset(self.env, seed=self.seed), self._obs_space)
        ep_reward = np.zeros(n_envs)
        ep_length = np.zeros(n_envs, dtype=np.int64)
        lane_context = np.zeros(n_envs, dtype=np.int64)
        num_timesteps = 0
        while num_timesteps < total_timesteps:
            policy_ready = (
                num_timesteps >= self.policy_learning_starts
                and np.all(lane_context >= self.min_context_transitions_per_lane)
            )
            if not policy_ready:
                env_actions = [action_to_env(self._action_space.sample(), self._action_space) for _ in range(n_envs)]
                policy_targets = [
                    np.full(self.n_actions, 1 / self.n_actions, dtype=np.float32)
                    if self.discrete else obs_to_array(a, self._action_space)
                    for a in env_actions
                ]
            else:
                results = self.search(
                    obs_arr, self._lane_states, prev_action=self._lane_prev_action,
                )
                env_actions = [action_to_env(r["env_action"], self._action_space) for r in results]
                policy_targets = [r["policy_target"] for r in results]
                self._lane_states = [r["state"] for r in results]
            action_flat = obs_batch_to_array(env_actions, self._action_space)
            self._lane_prev_action = action_flat.copy()
            next_obs_list, rewards, terminated, truncated, infos = vec_step(self.env, env_actions)
            dones = terminated | truncated
            next_obs_arr = obs_batch_to_array(next_obs_list, self._obs_space)
            transition_next = next_obs_arr.copy()
            for lane in np.flatnonzero(dones):
                final_obs = infos[int(lane)].get("final_obs")
                if final_obs is not None:
                    transition_next[int(lane)] = obs_to_array(final_obs, self._obs_space)
            for lane in range(n_envs):
                self.buffer.add(
                    obs_arr[lane], action_flat[lane], float(rewards[lane]), transition_next[lane],
                    policy_targets[lane], bool(dones[lane]), lane,
                )
                if dones[lane]:
                    self._lane_states[lane] = None
                    self._lane_prev_action[lane] = 0.0
                lane_context[lane] += 1
            obs_arr = next_obs_arr
            ep_reward += rewards
            ep_length += 1
            previous = num_timesteps
            num_timesteps += n_envs
            world_ready = (
                num_timesteps >= self.world_model_learning_starts
                and self.buffer.num_replay_transitions >= max(1, self.world_model_learning_starts)
            )
            if world_ready:
                if num_timesteps // self.train_freq != max(previous, self.learning_starts) // self.train_freq:
                    for _ in range(self.train_steps_per_iter):
                        self._last_metrics = self._train_step(train_policy=policy_ready)
                if (
                    policy_ready
                    and num_timesteps // self.reanalyze_freq
                    != max(previous, self.learning_starts) // self.reanalyze_freq
                ):
                    extra = self._reanalyze()
                    if extra:
                        self._last_metrics.update(extra)
                self._last_metrics["policy_ready"] = float(policy_ready)
            any_done = False
            for lane in np.flatnonzero(dones):
                any_done = True
                if not callback.on_step(
                    num_timesteps, float(ep_reward[lane]), int(ep_length[lane]), self._last_metrics,
                ):
                    return
                ep_reward[lane], ep_length[lane] = 0.0, 0
            if not any_done and not callback.on_step(num_timesteps, metrics=self._last_metrics):
                return

    def predict(self, obs: Any, deterministic: bool = True, episode_start: bool = False) -> tuple[Any, Any]:
        if episode_start:
            self._eval_state = None
            self._eval_prev_action = None
        obs_arr = obs_to_array(obs, self._obs_space)
        result = self.search(
            obs_arr[None], [self._eval_state],
            prev_action=None if self._eval_prev_action is None else self._eval_prev_action[None],
            deterministic=np.asarray([deterministic]),
        )[0]
        self._eval_state = result["state"]
        action = action_to_env(result["env_action"], self._action_space)
        self._eval_prev_action = obs_to_array(action, self._action_space)
        return action, None

    def save(self, path: Path) -> None:
        torch.save({
            "checkpoint_version": 2,
            "world_model": self.world_model.state_dict(),
            "target_tokenizer": self.target_tokenizer.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "ema_critic": self.ema_critic.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "uncertainty_ema": self._uncertainty_ema,
            "model_error_ema": self._model_error_ema,
            "posterior_kl_ema": self._posterior_kl_ema,
            "hyperparams": self.hyperparams,
        }, path)

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativeLatentImZero":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        version = int(payload.get("checkpoint_version", 1))
        if version != 2:
            raise ValueError(
                "LatentImZero checkpoint v1 used observation-leaking dynamics and "
                "cannot be migrated safely; start a new run with checkpoint v2.",
            )
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.world_model.load_state_dict(payload["world_model"])
        algo.target_tokenizer.load_state_dict(payload["target_tokenizer"])
        algo.actor.load_state_dict(payload["actor"])
        algo.critic.load_state_dict(payload["critic"])
        algo.ema_critic.load_state_dict(payload["ema_critic"])
        for name in ("model_optimizer", "actor_optimizer", "critic_optimizer"):
            if name in payload:
                getattr(algo, name).load_state_dict(payload[name])
        algo._uncertainty_ema = float(payload.get("uncertainty_ema", algo.uncertainty_high))
        algo._model_error_ema = float(payload.get("model_error_ema", 1.0))
        algo._posterior_kl_ema = float(payload.get("posterior_kl_ema", algo.free_bits))
        return algo
