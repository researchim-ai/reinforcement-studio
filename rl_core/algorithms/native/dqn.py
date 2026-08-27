"""From-scratch Deep Q-Network — the default `dqn` algorithm in the
Experiment Designer (see `rl_core/algorithms/native_runner.py`). Discrete
actions only (enforced by the environment/algorithm compatibility table in
`rl_core/envs/registry.py`, same as before)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.buffers import EpisodeSequenceReplayBuffer, ReplayBuffer
from rl_core.algorithms.native.exploration import (
    RNDModule,
    linear_schedule,
    reset_noise,
    set_noise_enabled,
    uses_noisy_net,
    uses_rnd,
)
from rl_core.algorithms.native.networks import Hidden, QNetwork, RecurrentQNetwork, mask_hidden_for_dones, memory_type_from_hyperparams
from rl_core.algorithms.native.preprocessing import obs_batch_to_array, obs_to_array
from rl_core.algorithms.vec_env import action_space, num_envs_of, obs_space, vec_reset, vec_step
from rl_core.netbuilder import SpecQNetwork

DEFAULT_HYPERPARAMS = {
    "learning_rate": 1e-3,
    "buffer_size": 50_000,
    "batch_size": 64,
    "gamma": 0.99,
    "exploration_fraction": 0.2,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "action_exploration": 0,
    "noisy_sigma0": 0.5,
    "intrinsic_exploration": 0,
    "rnd_bonus_coef": 0.1,
    "rnd_learning_rate": 1e-4,
    "rnd_feature_dim": 128,
    "rnd_hidden_dim": 128,
    "rnd_bonus_clip": 5.0,
    "learning_starts": 1_000,
    "train_freq": 4,
    "target_update_interval": 1_000,
    # Memory (see rl_core/algorithms/native/networks.py:RecurrentQNetwork —
    # this is a DRQN when enabled). 0 = no memory (plain QNetwork), 1 =
    # LSTM, 2 = GRU. `memory_seq_len` is the length of the episode windows
    # sampled for BPTT (see `EpisodeSequenceReplayBuffer` in buffers.py).
    "memory_type": 0,
    "memory_hidden_size": 128,
    "memory_num_layers": 1,
    "memory_seq_len": 20,
}


class NativeDQN(CustomAlgorithm):
    net: QNetwork | SpecQNetwork | RecurrentQNetwork

    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        obs_sp, act_sp = obs_space(env), action_space(env)
        if not isinstance(act_sp, gym.spaces.Discrete):
            raise ValueError("NativeDQN only supports Discrete action spaces")

        self.n_actions = int(act_sp.n)
        network_spec = hyperparams.get("network_spec")
        memory_type = memory_type_from_hyperparams(hyperparams)
        self.recurrent = bool(memory_type) and not network_spec
        self.noisy = uses_noisy_net(hyperparams)
        if self.noisy and network_spec:
            raise ValueError("NoisyNet is unavailable with a custom NetworkSpec; use the standard Q-network")
        noisy_sigma0 = float(hyperparams.get("noisy_sigma0", 0.5))

        def _build_q_net() -> nn.Module:
            if network_spec:
                return SpecQNetwork(obs_sp, self.n_actions, network_spec)
            if memory_type:
                return RecurrentQNetwork(
                    obs_sp, self.n_actions, memory_type,
                    hidden_size=int(hyperparams.get("memory_hidden_size", 128)),
                    num_layers=int(hyperparams.get("memory_num_layers", 1)),
                    noisy=self.noisy, noisy_sigma0=noisy_sigma0,
                )
            return QNetwork(obs_sp, self.n_actions, noisy=self.noisy, noisy_sigma0=noisy_sigma0)

        self.q_net = _build_q_net().to(device)
        self.target_net = _build_q_net().to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(hyperparams.get("learning_rate", 1e-3)))

        self.buffer_size = int(hyperparams.get("buffer_size", 50_000))
        self.batch_size = int(hyperparams.get("batch_size", 64))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.exploration_fraction = float(hyperparams.get("exploration_fraction", 0.2))
        self.exploration_initial_eps = float(hyperparams.get("exploration_initial_eps", 1.0))
        self.exploration_final_eps = float(hyperparams.get("exploration_final_eps", 0.05))
        self.learning_starts = int(hyperparams.get("learning_starts", 1_000))
        self.train_freq = max(1, int(hyperparams.get("train_freq", 4)))
        self.target_update_interval = max(1, int(hyperparams.get("target_update_interval", 1_000)))
        self.memory_seq_len = max(1, int(hyperparams.get("memory_seq_len", 20)))
        self.rnd_bonus_coef = float(hyperparams.get("rnd_bonus_coef", 0.1))
        self.rnd = RNDModule(
            obs_sp,
            device,
            feature_dim=int(hyperparams.get("rnd_feature_dim", 128)),
            hidden_dim=int(hyperparams.get("rnd_hidden_dim", 128)),
            learning_rate=float(hyperparams.get("rnd_learning_rate", 1e-4)),
            bonus_clip=float(hyperparams.get("rnd_bonus_clip", 5.0)),
        ) if uses_rnd(hyperparams) else None
        self.replay_buffer: ReplayBuffer | EpisodeSequenceReplayBuffer | None = None
        self._hidden: Hidden | None = None
        self._predict_hidden: Hidden | None = None
        self._last_td_loss: float | None = None
        self._obs_space = obs_sp
        self._action_space = act_sp

    def _obs_arr(self, obs: Any) -> np.ndarray:
        return obs_to_array(obs, self._obs_space)

    def _epsilon(self, num_timesteps: int, total_timesteps: int) -> float:
        if self.noisy:
            return 0.0
        return linear_schedule(
            num_timesteps, total_timesteps, self.exploration_fraction,
            self.exploration_initial_eps, self.exploration_final_eps,
        )

    def _metrics(
        self,
        epsilon: float,
        episode_extrinsic: float | None = None,
        episode_intrinsic: float | None = None,
    ) -> dict[str, float]:
        metrics = {"exploration_epsilon": float(epsilon)}
        if self._last_td_loss is not None:
            metrics["td_loss"] = self._last_td_loss
        if self.rnd:
            metrics["rnd_bonus_mean"] = float(self.rnd.bonus_mean)
            metrics["rnd_predictor_loss"] = float(self.rnd.last_loss)
        if episode_extrinsic is not None:
            metrics["episode_extrinsic_reward"] = float(episode_extrinsic)
        if self.rnd and episode_intrinsic is not None:
            metrics["episode_intrinsic_reward"] = float(episode_intrinsic)
        return metrics

    def _step_q_batch(self, obs_arr: np.ndarray) -> torch.Tensor:
        """Runs the Q-network forward exactly one vec-step, advancing
        `self._hidden` if recurrent. `obs_arr`: `(num_envs, ...)`. Always
        called — even when ε-greedy ends up picking a random action for
        every lane — because the recurrent core needs to see *every*
        observation to build a useful hidden state, regardless of which
        policy chose the action that produced it."""
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.recurrent:
                q, self._hidden = self.q_net.step(obs_t, self._hidden)
            else:
                q = self.q_net(obs_t)
        return q

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        n_envs = num_envs_of(self.env)
        obs_arr = obs_batch_to_array(vec_reset(self.env, seed=self.seed), self._obs_space)
        if self.replay_buffer is None:
            self.replay_buffer = (
                EpisodeSequenceReplayBuffer(self.buffer_size, obs_arr.shape[1:], num_envs=n_envs)
                if self.recurrent
                else ReplayBuffer(self.buffer_size, obs_arr.shape[1:])
            )
        if self.recurrent:
            self._hidden = self.q_net.initial_state(n_envs, self.device)
        ep_reward = np.zeros(n_envs, dtype=np.float64)
        ep_extrinsic = np.zeros(n_envs, dtype=np.float64)
        ep_intrinsic = np.zeros(n_envs, dtype=np.float64)
        ep_length = np.zeros(n_envs, dtype=np.int64)
        num_timesteps = 0

        while num_timesteps < total_timesteps:
            eps = self._epsilon(num_timesteps, total_timesteps)
            if self.noisy:
                reset_noise(self.q_net)
            q = self._step_q_batch(obs_arr)
            greedy = torch.argmax(q, dim=-1).cpu().numpy()
            actions = np.array(
                [
                    int(self._action_space.sample())
                    if len(self.replay_buffer) < self.learning_starts or np.random.rand() < eps
                    else int(greedy[i])
                    for i in range(n_envs)
                ],
                dtype=np.int64,
            )

            next_obs_list, rewards, terminated, truncated, _infos = vec_step(self.env, list(actions))
            dones = terminated | truncated
            next_obs_arr = obs_batch_to_array(next_obs_list, self._obs_space)
            extrinsic_reward = rewards.astype(np.float64)
            intrinsic_reward = (
                np.array([self.rnd_bonus_coef * self.rnd.bonus(next_obs_arr[i]) for i in range(n_envs)])
                if self.rnd
                else np.zeros(n_envs)
            )
            total_reward = extrinsic_reward + intrinsic_reward
            for i in range(n_envs):
                if self.recurrent:
                    self.replay_buffer.add(obs_arr[i], int(actions[i]), float(total_reward[i]), next_obs_arr[i], bool(dones[i]), lane=i)
                else:
                    self.replay_buffer.add(obs_arr[i], int(actions[i]), float(total_reward[i]), next_obs_arr[i], bool(dones[i]))
            obs_arr = next_obs_arr
            ep_reward += total_reward
            ep_extrinsic += extrinsic_reward
            ep_intrinsic += intrinsic_reward
            ep_length += 1
            prev_num_timesteps = num_timesteps
            num_timesteps += n_envs

            if len(self.replay_buffer) >= max(self.learning_starts, self.batch_size) and (
                num_timesteps // self.train_freq != prev_num_timesteps // self.train_freq
            ):
                self._train_step()
            if num_timesteps // self.target_update_interval != prev_num_timesteps // self.target_update_interval:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if self.recurrent:
                self._hidden = mask_hidden_for_dones(self._hidden, dones)

            any_finished = False
            for i in range(n_envs):
                if dones[i]:
                    any_finished = True
                    keep_going = callback.on_step(
                        num_timesteps, float(ep_reward[i]), int(ep_length[i]),
                        self._metrics(eps, float(ep_extrinsic[i]), float(ep_intrinsic[i])),
                    )
                    ep_reward[i], ep_extrinsic[i], ep_intrinsic[i], ep_length[i] = 0.0, 0.0, 0.0, 0
                    if not keep_going:
                        return
            if not any_finished:
                keep_going = callback.on_step(num_timesteps, metrics=self._metrics(eps))
                if not keep_going:
                    return

    def _train_step(self) -> None:
        if self.recurrent:
            self._train_step_recurrent()
            return
        batch = self.replay_buffer.sample(self.batch_size)
        if self.noisy:
            reset_noise(self.q_net)
            reset_noise(self.target_net)
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(dim=-1)[0]
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q
        current_q = self.q_net(obs_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self._last_td_loss = float(loss.detach().item())
        if self.rnd:
            self.rnd.update_predictor(batch["next_obs"])

    def _train_step_recurrent(self) -> None:
        """DRQN update: samples fixed-length episode windows (see
        `EpisodeSequenceReplayBuffer.sample`) instead of independent
        transitions, unrolls both networks over the whole window from a
        zero initial hidden state, and masks the loss so padding past a
        short episode's end (or, for `next_q`, past its very last real
        step) never contributes a gradient."""
        batch = self.replay_buffer.sample(self.batch_size, self.memory_seq_len)
        if self.noisy:
            reset_noise(self.q_net)
            reset_noise(self.target_net)
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(batch["mask"], dtype=torch.float32, device=self.device)
        batch_size = obs_t.shape[0]

        with torch.no_grad():
            h0_target = self.target_net.initial_state(batch_size, self.device)
            next_q, _ = self.target_net.forward_sequence(next_obs_t, h0_target)
            next_q_max = next_q.max(dim=-1)[0]
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q_max

        h0 = self.q_net.initial_state(batch_size, self.device)
        current_q, _ = self.q_net.forward_sequence(obs_t, h0)
        current_q_a = current_q.gather(-1, actions_t.unsqueeze(-1)).squeeze(-1)
        per_step_loss = F.smooth_l1_loss(current_q_a, target, reduction="none")
        loss = (per_step_loss * mask_t).sum() / mask_t.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self._last_td_loss = float(loss.detach().item())
        if self.rnd:
            flat_next_obs = batch["next_obs"].reshape(-1, *batch["next_obs"].shape[2:])
            self.rnd.update_predictor(flat_next_obs, mask=batch["mask"].reshape(-1))

    def predict(self, obs: Any, deterministic: bool = True, episode_start: bool = False) -> tuple[Any, Any]:
        obs_arr = self._obs_arr(obs)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.noisy:
            set_noise_enabled(self.q_net, not deterministic)
            if not deterministic:
                reset_noise(self.q_net)
        with torch.no_grad():
            if self.recurrent:
                if episode_start or self._predict_hidden is None:
                    self._predict_hidden = self.q_net.initial_state(1, self.device)
                q, self._predict_hidden = self.q_net.step(obs_t, self._predict_hidden)
            else:
                q = self.q_net(obs_t)
        if self.noisy:
            set_noise_enabled(self.q_net, True)
        if not deterministic and not self.noisy and np.random.rand() < self.exploration_final_eps:
            return int(self._action_space.sample()), None
        return int(torch.argmax(q, dim=-1).item()), None

    def save(self, path: Path) -> None:
        payload = {
            "state_dict": self.q_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hyperparams": self.hyperparams,
        }
        if self.rnd:
            payload["rnd_state"] = self.rnd.checkpoint_state()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativeDQN":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.q_net.load_state_dict(payload["state_dict"])
        algo.target_net.load_state_dict(payload["state_dict"])
        if payload.get("optimizer_state_dict"):
            algo.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if algo.rnd and payload.get("rnd_state"):
            algo.rnd.load_checkpoint_state(payload["rnd_state"])
        return algo
