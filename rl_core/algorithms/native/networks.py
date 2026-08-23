"""Small, dependency-free (no SB3) building blocks for the native
PPO/A2C/DQN implementations: an MLP or Nature-CNN feature extractor picked
from the observation space, plus the actor-critic and Q-network heads built
on top of it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from rl_core.algorithms.native.preprocessing import is_image_space, obs_flat_dim


class MLPExtractor(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.net = nn.Sequential(*layers)
        self.out_dim = last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NatureCNN(nn.Module):
    """Mnih et al. (2015) DQN conv stack. Input is (B, H, W, C) uint8/float
    (channel-last, matching Gym's raw frames) — normalized and permuted to
    (B, C, H, W) internally."""

    def __init__(self, obs_shape: tuple[int, int, int]) -> None:
        super().__init__()
        h, w, c = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, 512), nn.ReLU())
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.linear(self.conv(x))


def build_feature_extractor(observation_space: gym.Space) -> nn.Module:
    if is_image_space(observation_space):
        return NatureCNN(tuple(observation_space.shape))
    return MLPExtractor(obs_flat_dim(observation_space))


def policy_name(observation_space: gym.Space) -> str:
    """Mirrors SB3's `MlpPolicy`/`CnnPolicy` naming so the Training Monitor
    UI needs no changes to display the native runs' policy type."""
    return "CnnPolicy" if is_image_space(observation_space) else "MlpPolicy"


class ActorCriticNet(nn.Module):
    """Shared-trunk actor-critic used by both NativePPO and NativeA2C.
    Discrete action spaces get a Categorical head; continuous (`Box`) action
    spaces get a diagonal Gaussian head with a learned, state-independent
    log-std (the standard PPO/A2C setup)."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        super().__init__()
        self.features = build_feature_extractor(observation_space)
        feat_dim = self.features.out_dim
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.discrete:
            self.action_head = nn.Linear(feat_dim, int(action_space.n))
        else:
            act_dim = int(np.prod(action_space.shape))
            self.mu_head = nn.Linear(feat_dim, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value_head = nn.Linear(feat_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (distribution params tensor, value). For discrete this is
        logits; for continuous it's the Gaussian mean."""
        feat = self.features(obs)
        value = self.value_head(feat).squeeze(-1)
        if self.discrete:
            return self.action_head(feat), value
        return self.mu_head(feat), value

    def distribution(self, obs: torch.Tensor):
        params, value = self.forward(obs)
        if self.discrete:
            return torch.distributions.Categorical(logits=params), value
        std = torch.exp(self.log_std).clamp(min=1e-6)
        return torch.distributions.Normal(params, std), value

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        params, _ = self.forward(obs)
        return torch.argmax(params, dim=-1) if self.discrete else params

    @staticmethod
    def log_prob(dist, action: torch.Tensor) -> torch.Tensor:
        lp = dist.log_prob(action)
        return lp if lp.dim() == 1 else lp.sum(dim=-1)

    @staticmethod
    def entropy(dist) -> torch.Tensor:
        ent = dist.entropy()
        return ent if ent.dim() == 1 else ent.sum(dim=-1)


class QNetwork(nn.Module):
    """Plain DQN Q-network: feature extractor + a linear head over actions."""

    def __init__(self, observation_space: gym.Space, n_actions: int) -> None:
        super().__init__()
        self.features = build_feature_extractor(observation_space)
        self.q_head = nn.Linear(self.features.out_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.features(obs))
