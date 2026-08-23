"""From-scratch Advantage Actor-Critic — the default `a2c` algorithm in the
Experiment Designer (see `rl_core/algorithms/native_runner.py`). Same
actor-critic network and rollout loop as NativePPO, but a single full-batch
gradient step per rollout instead of clipped, multi-epoch minibatch updates."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import gymnasium as gym

from rl_core.algorithms.native.buffers import RolloutBuffer
from rl_core.algorithms.native.networks import ActorCriticNet
from rl_core.algorithms.native.on_policy import OnPolicyAlgorithm

DEFAULT_HYPERPARAMS = {
    "learning_rate": 7e-4,
    "n_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


class NativeA2C(OnPolicyAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        self.net = ActorCriticNet(env.observation_space, env.action_space).to(device)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=float(hyperparams.get("learning_rate", 7e-4)))
        self.n_steps = int(hyperparams.get("n_steps", 5))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.gae_lambda = float(hyperparams.get("gae_lambda", 1.0))
        self.ent_coef = float(hyperparams.get("ent_coef", 0.01))
        self.vf_coef = float(hyperparams.get("vf_coef", 0.5))
        self.max_grad_norm = float(hyperparams.get("max_grad_norm", 0.5))

    def _update(self, buf: RolloutBuffer) -> None:
        batch = buf.all()
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)

        dist, values = self.net.distribution(obs_t)
        log_probs = self.net.log_prob(dist, actions_t)
        entropy = self.net.entropy(dist).mean()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
