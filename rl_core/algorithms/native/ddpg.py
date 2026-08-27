"""From-scratch Deep Deterministic Policy Gradient (Lillicrap et al., 2015) —
the classic off-policy continuous-control baseline that predates both SAC
and TD3. Structurally almost identical to `NativeSAC` (replay buffer, target
networks, Polyak averaging), but the policy is *deterministic* — one action
vector per observation, no learned distribution — and exploration comes from
Gaussian noise added to the action at collection time rather than from
sampling a stochastic policy or entropy maximization.

DDPG is mainly here as the simplest member of this family and as the base
`NativeTD3` (see `rl_core/algorithms/native/td3.py`) builds on: single critic
(vs. TD3's twin), no target policy smoothing, no delayed policy updates.
Known to be less stable than TD3/SAC on harder tasks (Q-value
overestimation, sensitivity to noise scale) — kept for comparison purposes
in the Designer, TD3 is the better default for new experiments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.buffers import ContinuousReplayBuffer
from rl_core.algorithms.native.networks import DeterministicPolicy, QCritic
from rl_core.algorithms.native.preprocessing import action_to_env, obs_batch_to_array, obs_to_array
from rl_core.algorithms.vec_env import action_space, num_envs_of, obs_space, vec_reset, vec_step

DEFAULT_HYPERPARAMS = {
    "learning_rate": 1e-3,
    "buffer_size": 100_000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "exploration_noise": 0.1,
    "learning_starts": 1_000,
    "train_freq": 1,
}


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


class NativeDDPG(CustomAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        obs_sp, act_sp = obs_space(env), action_space(env)
        if not isinstance(act_sp, gym.spaces.Box):
            raise ValueError("NativeDDPG only supports continuous (Box) action spaces")

        self.action_dim = int(np.prod(act_sp.shape))
        low = np.asarray(act_sp.low, dtype=np.float32).reshape(-1)
        high = np.asarray(act_sp.high, dtype=np.float32).reshape(-1)
        # Used to scale the *exploration noise* (a fraction of the action
        # range makes `exploration_noise` mean roughly the same thing across
        # envs with wildly different action scales, e.g. [-1,1] vs [-1,0,0]..[1,1,1]).
        self._action_range = np.where(np.isfinite(high - low), high - low, 2.0)

        self.actor = DeterministicPolicy(obs_sp, low, high).to(device)
        self.critic = QCritic(obs_sp, self.action_dim).to(device)
        self.actor_target = DeterministicPolicy(obs_sp, low, high).to(device)
        self.critic_target = QCritic(obs_sp, self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        lr = float(hyperparams.get("learning_rate", 1e-3))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer_size = int(hyperparams.get("buffer_size", 100_000))
        self.batch_size = int(hyperparams.get("batch_size", 256))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.tau = float(hyperparams.get("tau", 0.005))
        self.exploration_noise = float(hyperparams.get("exploration_noise", 0.1))
        self.learning_starts = int(hyperparams.get("learning_starts", 1_000))
        self.train_freq = max(1, int(hyperparams.get("train_freq", 1)))
        self.replay_buffer: ContinuousReplayBuffer | None = None
        self._last_metrics: dict[str, float] = {}
        self._obs_space = obs_sp
        self._action_space = act_sp

    def _obs_arr(self, obs: Any) -> np.ndarray:
        return obs_to_array(obs, self._obs_space)

    def _sample_actions(self, obs_arr: np.ndarray, deterministic: bool) -> np.ndarray:
        """`obs_arr`: `(num_envs, ...)` -> `(num_envs, action_dim)`."""
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions = self.actor(obs_t).cpu().numpy()
        if not deterministic:
            actions = actions + np.random.normal(0.0, self.exploration_noise * self._action_range, size=actions.shape)
            actions = np.clip(actions, self._action_space.low, self._action_space.high)
        return actions.astype(np.float32)

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        n_envs = num_envs_of(self.env)
        obs_arr = obs_batch_to_array(vec_reset(self.env, seed=self.seed), self._obs_space)
        if self.replay_buffer is None:
            self.replay_buffer = ContinuousReplayBuffer(self.buffer_size, obs_arr.shape[1:], self.action_dim)
        ep_reward = np.zeros(n_envs, dtype=np.float64)
        ep_length = np.zeros(n_envs, dtype=np.int64)
        num_timesteps = 0

        while num_timesteps < total_timesteps:
            if len(self.replay_buffer) < self.learning_starts:
                actions = np.stack(
                    [np.asarray(self._action_space.sample(), dtype=np.float32).reshape(-1) for _ in range(n_envs)],
                )
            else:
                actions = self._sample_actions(obs_arr, deterministic=False)

            env_actions = [action_to_env(actions[i], self._action_space) for i in range(n_envs)]
            next_obs_list, rewards, terminated, truncated, _infos = vec_step(self.env, env_actions)
            dones = terminated | truncated
            next_obs_arr = obs_batch_to_array(next_obs_list, self._obs_space)
            for i in range(n_envs):
                self.replay_buffer.add(obs_arr[i], actions[i], float(rewards[i]), next_obs_arr[i], bool(dones[i]))
            obs_arr = next_obs_arr
            ep_reward += rewards
            ep_length += 1
            prev_num_timesteps = num_timesteps
            num_timesteps += n_envs

            if len(self.replay_buffer) >= max(self.learning_starts, self.batch_size) and (
                num_timesteps // self.train_freq != prev_num_timesteps // self.train_freq
            ):
                self._train_step()

            any_finished = False
            for i in range(n_envs):
                if dones[i]:
                    any_finished = True
                    keep_going = callback.on_step(num_timesteps, float(ep_reward[i]), int(ep_length[i]), self._last_metrics)
                    ep_reward[i], ep_length[i] = 0.0, 0
                    if not keep_going:
                        return
            if not any_finished:
                keep_going = callback.on_step(num_timesteps, metrics=self._last_metrics)
                if not keep_going:
                    return

    def _train_step(self) -> None:
        batch = self.replay_buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action = self.actor_target(next_obs_t)
            target_q = self.critic_target(next_obs_t, next_action)
            target = rewards_t + self.gamma * (1.0 - dones_t) * target_q

        current_q = self.critic(obs_t, actions_t)
        critic_loss = F.mse_loss(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Deterministic policy gradient: nudge the actor towards whatever
        # action the critic currently thinks is best at each state — the
        # critic must be reasonably accurate for this to be useful, which is
        # exactly why DDPG/TD3 alternate critic and actor updates instead of
        # training the actor from scratch against a random critic.
        actor_loss = -self.critic(obs_t, self.actor(obs_t)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        _soft_update(self.actor_target, self.actor, self.tau)
        _soft_update(self.critic_target, self.critic, self.tau)

        self._last_metrics = {
            "actor_loss": float(actor_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
        }

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]:
        obs_arr = self._obs_arr(obs)
        action = self._sample_actions(obs_arr[np.newaxis, ...], deterministic=deterministic)[0]
        return action.reshape(self._action_space.shape), None

    def save(self, path: Path) -> None:
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "hyperparams": self.hyperparams,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativeDDPG":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.actor.load_state_dict(payload["actor_state_dict"])
        algo.critic.load_state_dict(payload["critic_state_dict"])
        algo.actor_target.load_state_dict(payload["actor_state_dict"])
        algo.critic_target.load_state_dict(payload["critic_state_dict"])
        if payload.get("actor_optimizer_state_dict"):
            algo.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        if payload.get("critic_optimizer_state_dict"):
            algo.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        return algo
