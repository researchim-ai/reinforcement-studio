"""Dreamer-style model-based RL (Hafner et al., 2019/2020 —
https://arxiv.org/abs/1912.01603). Learns an RSSM world model
(`rl_core/world_models/rssm.py`) continuously from real experience, and
trains an actor-critic *entirely inside imagined rollouts* through it
rather than on real transitions at all — the defining payoff of a world
model: once it's decent, one real env step can be "replayed" as many cheap
imagined ones for the actor-critic to learn from, instead of needing that
many more real env steps.

Simplifications vs. the paper (kept explicit here rather than silent):
- Diagonal Gaussian stochastic latent, not DreamerV2/V3's categorical one
  (see `rssm.py`'s own docstring for why).
- The actor-critic target is a plain discounted Monte-Carlo return,
  bootstrapped by the critic at the imagination horizon (equivalent to the
  paper's own lambda-return with lambda=1), not its intermediate-lambda
  blend.
- Discrete actions use a straight-through Gumbel-softmax one-hot during
  imagination (so the rollout stays end-to-end differentiable into the
  actor, mirroring the paper's own pathwise gradient for the continuous
  case) rather than DreamerV2's more specialized discrete-action machinery.

`training.num_envs > 1` collects real experience from that many parallel
env lanes (`AsyncVectorEnv`, one subprocess worker each) — the RSSM's own
`(h, z)` posterior state is a plain `(B, dim)` tensor already, so
"advance N lanes' hidden states one real step each, in lockstep" is just
`step_posterior` called with `B=num_envs` instead of `B=1`; each lane's
transitions land in `SequenceReplayBuffer` as its own independent episode
(`add(..., lane=i)`, see that buffer's own docstring) so they never splice
into one another despite sharing one buffer instance. This is Dreamer's
single biggest real-world speedup lever: real-experience collection (one
env.step at a time) is the part of this algorithm that can't be sped up
by more GPU compute the way the imagination-based actor-critic training
already can, so running it `num_envs`-way in parallel is the main way to
get more real steps per wall-clock second.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.preprocessing import action_to_env, obs_batch_to_array, obs_to_array
from rl_core.algorithms.vec_env import (
    action_space as venv_action_space,
    num_envs_of,
    obs_space as venv_obs_space,
    vec_reset,
    vec_step,
)
from rl_core.world_models.losses import train_step_rssm
from rl_core.world_models.replay import SequenceReplayBuffer, action_flat_dim
from rl_core.world_models.spec import resolve_or_build_for_algo, save_checkpoint

DEFAULT_HYPERPARAMS = {
    "model_learning_rate": 1e-3,
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "batch_size": 32,
    "seq_len": 50,
    "imagination_horizon": 15,
    "gamma": 0.99,
    "entropy_coef": 1e-3,
    "collect_steps_per_iter": 100,
    "train_steps_per_iter": 10,
    "learning_starts": 1000,
}


class _Actor(nn.Module):
    """Both a differentiable path for imagination (`imagine_step`) and a
    plain-numpy path for stepping the real env (`act_for_env`) — see the
    module docstring for why they can't share one implementation."""

    def __init__(self, feat_dim: int, action_space: gym.Space, hidden: int = 200) -> None:
        super().__init__()
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        self.trunk = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        if self.discrete:
            self.action_dim = int(action_space.n)
            self.logits_head = nn.Linear(hidden, self.action_dim)
        else:
            self.action_dim = int(np.prod(action_space.shape))
            self.mean_head = nn.Linear(hidden, self.action_dim)
            self.log_std_head = nn.Linear(hidden, self.action_dim)
            low = np.where(np.isfinite(action_space.low), action_space.low, -1.0).reshape(-1)
            high = np.where(np.isfinite(action_space.high), action_space.high, 1.0).reshape(-1)
            self.register_buffer("action_scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
            self.register_buffer("action_bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))

    def imagine_step(self, feat: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(feat)
        if self.discrete:
            logits = self.logits_head(h)
            if sample:
                action = F.gumbel_softmax(logits, tau=1.0, hard=True)
            else:
                action = F.one_hot(torch.argmax(logits, dim=-1), self.action_dim).float()
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(-1)
            return action, entropy
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5.0, 2.0)
        std = log_std.exp()
        raw = mean + std * torch.randn_like(mean) if sample else mean
        action = torch.tanh(raw) * self.action_scale + self.action_bias
        entropy = (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)
        return action, entropy

    @torch.no_grad()
    def act_for_env(self, feat: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        h = self.trunk(feat)
        if self.discrete:
            logits = self.logits_head(h)
            idx = torch.argmax(logits, dim=-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
            return idx.cpu().numpy()
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5.0, 2.0)
        raw = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        action = torch.tanh(raw) * self.action_scale + self.action_bias
        return action.cpu().numpy()


class _Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 200) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


class NativeDreamer(CustomAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        self._obs_space = venv_obs_space(env)
        self._action_space = venv_action_space(env)

        self.world_model, self.world_model_config, self.world_model_slug = resolve_or_build_for_algo(
            hyperparams.get("world_model_spec"), "rssm", self._obs_space, self._action_space, device,
        )
        self.world_model.to(device)
        feat_dim = self.world_model.feat_dim
        self.actor = _Actor(feat_dim, self._action_space).to(device)
        self.critic = _Critic(feat_dim).to(device)

        self.model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=float(hyperparams.get("model_learning_rate", 1e-3)))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(hyperparams.get("actor_learning_rate", 3e-4)))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(hyperparams.get("critic_learning_rate", 3e-4)))

        self.batch_size = int(hyperparams.get("batch_size", 32))
        self.seq_len = int(hyperparams.get("seq_len", 50))
        self.horizon = int(hyperparams.get("imagination_horizon", 15))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.entropy_coef = float(hyperparams.get("entropy_coef", 1e-3))
        self.collect_steps_per_iter = max(1, int(hyperparams.get("collect_steps_per_iter", 100)))
        self.train_steps_per_iter = max(1, int(hyperparams.get("train_steps_per_iter", 10)))
        self.learning_starts = int(hyperparams.get("learning_starts", 1000))

        action_dim = action_flat_dim(self._action_space)
        self.buffer = SequenceReplayBuffer(50_000, self.world_model.encoder.obs_shape, action_dim, num_lanes=num_envs_of(env))
        self._last_metrics: dict[str, float] = {}
        self._predict_h: torch.Tensor | None = None
        self._predict_z: torch.Tensor | None = None
        self._predict_prev_action: torch.Tensor | None = None

    def _train_actor_critic(self) -> dict[str, float]:
        batch = self.buffer.sample(self.batch_size, self.seq_len)
        obs_seq = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        action_seq = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            observed = self.world_model.observe(obs_seq, action_seq)
        # Every real (h, z) the model just observed is a valid imagination
        # starting point — flattening (B, T, ...) -> (B*T, ...) means one
        # batch of real sequences yields `batch_size * seq_len` independent
        # imagined rollouts per actor-critic update.
        h0 = observed["h"].reshape(-1, self.world_model.deter_dim).detach()
        z0 = observed["z"].reshape(-1, self.world_model.stoch_dim).detach()

        entropies: list[torch.Tensor] = []

        def policy_fn(feat: torch.Tensor) -> torch.Tensor:
            action, entropy = self.actor.imagine_step(feat, sample=True)
            entropies.append(entropy)
            return action

        imagined = self.world_model.imagine(h0, z0, policy_fn, horizon=self.horizon)
        feat_seq, rewards, continues = imagined["feat"], imagined["reward"], imagined["continue"]

        with torch.no_grad():
            bootstrap = self.critic(feat_seq[:, -1])
        returns = torch.zeros_like(rewards)
        next_return = bootstrap
        for t in reversed(range(self.horizon)):
            next_return = rewards[:, t] + self.gamma * continues[:, t] * next_return
            returns[:, t] = next_return

        entropy_seq = torch.stack(entropies, dim=1)
        # Gradient into the actor flows entirely through `rewards`/
        # `continues` here (both functions of the imagined trajectory,
        # which is itself a function of every sampled action) — `returns`
        # never touches the critic at all except at the (detached)
        # bootstrap, exactly like the paper's own actor objective.
        actor_loss = -returns.mean() - self.entropy_coef * entropy_seq.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        values = self.critic(feat_seq.detach())
        critic_loss = F.mse_loss(values, returns.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "imagined_return_mean": float(returns.mean().item()),
        }

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        n_envs = num_envs_of(self.env)
        obs_arr = obs_batch_to_array(vec_reset(self.env, seed=self.seed), self._obs_space)
        h, z = self.world_model.initial_state(n_envs, self.device)
        prev_action = torch.zeros(n_envs, self.world_model.action_dim, device=self.device)
        ep_reward = np.zeros(n_envs, dtype=np.float64)
        ep_length = np.zeros(n_envs, dtype=np.int64)
        num_timesteps = 0

        while num_timesteps < total_timesteps:
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                embed = self.world_model.encoder(obs_t)
                h, z, _post, _prior = self.world_model.step_posterior(h, z, prev_action, embed)
                feat = self.world_model.feat(h, z)
                if num_timesteps < self.learning_starts:
                    env_actions = [self._action_space.sample() for _ in range(n_envs)]
                else:
                    actions_np = self.actor.act_for_env(feat, deterministic=False)
                    env_actions = [action_to_env(actions_np[i], self._action_space) for i in range(n_envs)]

            action_arr = obs_batch_to_array(env_actions, self._action_space)
            prev_action = torch.as_tensor(action_arr, dtype=torch.float32, device=self.device)

            next_obs_list, rewards, terminated, truncated, _infos = vec_step(self.env, env_actions)
            dones = terminated | truncated
            next_obs_arr = obs_batch_to_array(next_obs_list, self._obs_space)
            for i in range(n_envs):
                self.buffer.add(obs_arr[i], action_arr[i], float(rewards[i]), bool(dones[i]), lane=i)
            ep_reward += rewards
            ep_length += 1
            prev_num_timesteps = num_timesteps
            num_timesteps += n_envs
            obs_arr = next_obs_arr

            if (
                num_timesteps >= self.learning_starts
                and num_timesteps // self.collect_steps_per_iter != prev_num_timesteps // self.collect_steps_per_iter
                and self.buffer.num_episodes >= 2
                and len(self.buffer) >= self.seq_len
            ):
                for _ in range(self.train_steps_per_iter):
                    model_losses = train_step_rssm(
                        self.world_model, self.model_optimizer, self.buffer, self.batch_size, self.seq_len, self.device,
                    )
                    ac_losses = self._train_actor_critic()
                self._last_metrics = {**model_losses, **ac_losses}

            any_finished = False
            for i in range(n_envs):
                if not dones[i]:
                    continue
                any_finished = True
                keep_going = callback.on_step(num_timesteps, float(ep_reward[i]), int(ep_length[i]), self._last_metrics)
                ep_reward[i], ep_length[i] = 0.0, 0
                # This lane's next real step starts a fresh episode (the
                # vector env autoresets it — see `vec_step`'s own
                # docstring) — its RSSM carry-state must restart too,
                # rather than keep conditioning on the episode that just
                # ended.
                h0, z0 = self.world_model.initial_state(1, self.device)
                h[i : i + 1], z[i : i + 1] = h0, z0
                prev_action[i : i + 1] = 0.0
                if not keep_going:
                    return
            if not any_finished:
                keep_going = callback.on_step(num_timesteps, metrics=self._last_metrics)
                if not keep_going:
                    return

    def predict(self, obs: Any, deterministic: bool = True, episode_start: bool = False) -> tuple[Any, Any]:
        if episode_start or self._predict_h is None:
            self._predict_h, self._predict_z = self.world_model.initial_state(1, self.device)
            self._predict_prev_action = torch.zeros(1, self.world_model.action_dim, device=self.device)
        obs_arr = obs_to_array(obs, self._obs_space)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            embed = self.world_model.encoder(obs_t)
            h, z, _post, _prior = self.world_model.step_posterior(
                self._predict_h, self._predict_z, self._predict_prev_action, embed,
            )
            feat = self.world_model.feat(h, z)
            action_np = self.actor.act_for_env(feat, deterministic=deterministic)
        self._predict_h, self._predict_z = h, z
        action_env = action_to_env(action_np, self._action_space)
        self._predict_prev_action = torch.as_tensor(
            obs_to_array(action_env, self._action_space), dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        return action_env, None

    def save(self, path: Path) -> None:
        torch.save(
            {
                "world_model_state_dict": self.world_model.state_dict(),
                "world_model_config": self.world_model_config,
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "hyperparams": self.hyperparams,
            },
            path,
        )

    def save_world_model_checkpoint(self, path: Path) -> None:
        """Called by `runner_utils.py` after a finished run, if this run's
        `world_model_id` pointed at a saved spec — copies just the trained
        RSSM (not the actor/critic, which aren't part of the world model
        spec itself) back into `CUSTOM_WORLD_MODELS_DIR`."""
        save_checkpoint("rssm", self.world_model, path, self.world_model_config)

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativeDreamer":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.world_model.load_state_dict(payload["world_model_state_dict"])
        algo.actor.load_state_dict(payload["actor_state_dict"])
        algo.critic.load_state_dict(payload["critic_state_dict"])
        return algo
