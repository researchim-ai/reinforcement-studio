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
from rl_core.algorithms.native.networks import ActorCriticNet, Hidden, RecurrentActorCriticNet, detach_hidden, memory_type_from_hyperparams
from rl_core.algorithms.native.on_policy import OnPolicyAlgorithm
from rl_core.netbuilder import SpecActorCriticNet

DEFAULT_HYPERPARAMS = {
    "learning_rate": 7e-4,
    "n_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    # Memory (see rl_core/algorithms/native/networks.py:RecurrentActorCriticNet).
    # 0 = no memory (plain ActorCriticNet), 1 = LSTM, 2 = GRU.
    "memory_type": 0,
    "memory_hidden_size": 128,
    "memory_num_layers": 1,
    "memory_seq_len": 32,
}


class NativeA2C(OnPolicyAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        network_spec = hyperparams.get("network_spec")
        memory_type = memory_type_from_hyperparams(hyperparams)
        if network_spec:
            self.net = SpecActorCriticNet(env.observation_space, env.action_space, network_spec)
        elif memory_type:
            self.net = RecurrentActorCriticNet(
                env.observation_space, env.action_space, memory_type,
                hidden_size=int(hyperparams.get("memory_hidden_size", 128)),
                num_layers=int(hyperparams.get("memory_num_layers", 1)),
            )
        else:
            self.net = ActorCriticNet(env.observation_space, env.action_space)
        self.net = self.net.to(device)
        self.recurrent = isinstance(self.net, RecurrentActorCriticNet)
        self.memory_seq_len = max(1, int(hyperparams.get("memory_seq_len", 32)))
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=float(hyperparams.get("learning_rate", 7e-4)))
        self.n_steps = int(hyperparams.get("n_steps", 5))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.gae_lambda = float(hyperparams.get("gae_lambda", 1.0))
        self.ent_coef = float(hyperparams.get("ent_coef", 0.01))
        self.vf_coef = float(hyperparams.get("vf_coef", 0.5))
        self.max_grad_norm = float(hyperparams.get("max_grad_norm", 0.5))

    def _update(self, buf: RolloutBuffer, rollout_hidden: Hidden | None = None) -> None:
        if self.recurrent:
            self._update_recurrent(buf, rollout_hidden)
            return
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

    def _update_recurrent(self, buf: RolloutBuffer, rollout_hidden: Hidden | None) -> None:
        """A2C wants a *single* full-batch update per rollout — chunking is
        only for bounding BPTT length, not for separate optimizer steps, so
        gradients are accumulated (via repeated `.backward()`) across every
        chunk and averaged before one `optimizer.step()`, matching the
        non-recurrent path's "one gradient step over the whole rollout"."""
        self.optimizer.zero_grad()
        hidden = rollout_hidden
        num_chunks = 0
        for batch in buf.sequences(self.memory_seq_len):
            obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device).unsqueeze(0)
            actions_t = torch.as_tensor(batch["actions"], device=self.device).unsqueeze(0)
            advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device).unsqueeze(0)
            returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_starts = torch.as_tensor(batch["episode_starts"], dtype=torch.float32, device=self.device).unsqueeze(0)

            dist, values, hidden = self.net.distribution_sequence(obs_t, hidden, episode_starts)
            log_probs = self.net.log_prob(dist, actions_t)
            entropy = self.net.entropy(dist).mean()

            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
            loss.backward()

            hidden = detach_hidden(hidden)
            num_chunks += 1

        if num_chunks > 0:
            for p in self.net.parameters():
                if p.grad is not None:
                    p.grad /= num_chunks
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
