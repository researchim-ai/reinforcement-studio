"""Shared rollout-collection loop for the two on-policy algorithms
(NativePPO, NativeA2C) — they only differ in how `_update()` turns a filled
`RolloutBuffer` into a gradient step.

Collection is written for `num_envs >= 1` parallel lanes (see
`rl_core/algorithms/vec_env.py`); `num_envs=1` — the default, and the only
case a plain (non-vectorized) `gym.Env` is ever involved — degenerates to
exactly the previous single-env loop's behavior, just phrased as "a batch
of 1 lane" throughout."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.buffers import RolloutBuffer
from rl_core.algorithms.native.networks import ActorCriticNet, Hidden, RecurrentActorCriticNet, mask_hidden_for_dones
from rl_core.algorithms.native.preprocessing import action_to_env, obs_batch_to_array, obs_to_array
from rl_core.algorithms.vec_env import action_space, num_envs_of, obs_space, vec_reset, vec_step
from rl_core.netbuilder import SpecActorCriticNet


class _LossAccumulator:
    """Averages policy/value/entropy loss across every minibatch of one
    `_update()` call — PPO especially runs many (`n_epochs` × minibatches)
    per rollout, and reporting just the very last one would make the Loss
    chart jump around based on minibatch order rather than reflect the
    update as a whole."""

    def __init__(self) -> None:
        self._sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        self._count = 0

    def add(self, policy_loss: torch.Tensor, value_loss: torch.Tensor, entropy: torch.Tensor) -> None:
        self._sums["policy_loss"] += float(policy_loss.detach().item())
        self._sums["value_loss"] += float(value_loss.detach().item())
        self._sums["entropy"] += float(entropy.detach().item())
        self._count += 1

    def means(self) -> dict[str, float]:
        if self._count == 0:
            return {}
        return {key: total / self._count for key, total in self._sums.items()}


class OnPolicyAlgorithm(CustomAlgorithm):
    net: ActorCriticNet | SpecActorCriticNet | RecurrentActorCriticNet
    n_steps: int
    gamma: float
    gae_lambda: float
    # Set by subclasses' __init__ once `self.net` is built (see
    # ppo.py/a2c.py: `isinstance(self.net, RecurrentActorCriticNet)`).
    # `memory_seq_len` only matters when `recurrent` is True.
    recurrent: bool = False
    memory_seq_len: int = 32
    # Hidden state threaded across rollout-collection steps and (separately)
    # across `predict()` calls — `None` until the first reset seeds it via
    # `net.initial_state(...)`. During collection, batch dim is `num_envs`
    # (one row per lane); during `predict()` it's always 1 (a single live
    # env), regardless of how many lanes `env` was trained with.
    _hidden: Hidden | None = None
    _predict_hidden: Hidden | None = None
    # Set by `_update()`/`_update_recurrent()` (see ppo.py/a2c.py) once the
    # rollout just collected has been trained on — averaged over every
    # epoch/minibatch of that update, not just the last one, to avoid a
    # single noisy minibatch dominating the reported value. Threaded into
    # `callback.on_step(metrics=...)` below so the Training Monitor's Loss
    # chart has something to plot even though updates happen once per
    # rollout, not once per env step.
    _last_metrics: dict[str, float] = {}

    def _obs_space(self) -> gym.Space:
        return obs_space(self.env)

    def _obs_arr_batch(self, obs_list: list[Any]) -> np.ndarray:
        return obs_batch_to_array(obs_list, self._obs_space())

    def _sample_actions(self, obs_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`obs_arr`: `(num_envs, ...)`. Returns `(actions, log_probs,
        values)`, each a length-`num_envs` batch."""
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.recurrent:
                dist, value, self._hidden = self.net.distribution_step(obs_t, self._hidden)
            else:
                dist, value = self.net.distribution(obs_t)
            action_t = dist.sample()
            log_prob_t = self.net.log_prob(dist, action_t)
        return action_t.cpu().numpy(), log_prob_t.cpu().numpy(), value.cpu().numpy()

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        n_envs = num_envs_of(self.env)
        obs_sp = self._obs_space()
        act_sp = action_space(self.env)
        action_dim = 0 if self.net.discrete else int(np.prod(act_sp.shape))

        obs_arr = self._obs_arr_batch(vec_reset(self.env, seed=self.seed))
        if self.recurrent:
            self._hidden = self.net.initial_state(n_envs, self.device)
        episode_start = np.ones(n_envs, dtype=np.float32)
        ep_reward = np.zeros(n_envs, dtype=np.float64)
        ep_length = np.zeros(n_envs, dtype=np.int64)
        num_timesteps = 0
        # gSDE's exploration noise matrix (see `ActorCriticNet.reset_noise`)
        # is resampled every `sde_sample_freq` *steps* below, tracked here
        # rather than in `ActorCriticNet` itself since only the collection
        # loop knows how many steps have elapsed. `getattr` throughout this
        # method so plain (non-PPO, or PPO-without-gSDE) algorithms sharing
        # this loop never need `use_sde`/`sde_sample_freq` attributes at all.
        last_sde_reset = 0

        while num_timesteps < total_timesteps:
            remaining_lane_steps = -(-(total_timesteps - num_timesteps) // n_envs)  # ceil div
            capacity = max(1, min(self.n_steps, remaining_lane_steps))
            buf = RolloutBuffer(capacity, n_envs, obs_arr.shape[1:], action_dim, self.net.discrete)
            # Hidden state as it was *entering* this rollout — the correct
            # seed for re-running the whole chunk through the RNN during
            # `_update()` (every epoch restarts from here, not from
            # wherever collection ended up, since collection already moved
            # on to the *next* rollout by the time training happens).
            rollout_hidden = self._hidden
            dones = np.zeros(n_envs, dtype=bool)

            for _ in range(capacity):
                if getattr(self, "use_sde", False) and num_timesteps - last_sde_reset >= self.sde_sample_freq:
                    last_sde_reset = num_timesteps
                    self.net.reset_noise(n_envs)
                actions, log_probs, values = self._sample_actions(obs_arr)
                env_actions = [action_to_env(actions[i], act_sp) for i in range(n_envs)]
                next_obs_list, rewards, terminated, truncated, _infos = vec_step(self.env, env_actions)
                dones = terminated | truncated
                buf.add(obs_arr, actions, log_probs, values, rewards, dones.astype(np.float32), episode_start)

                ep_reward += rewards
                ep_length += 1
                num_timesteps += n_envs

                any_finished = False
                for i in range(n_envs):
                    if dones[i]:
                        any_finished = True
                        keep_going = callback.on_step(
                            num_timesteps, float(ep_reward[i]), int(ep_length[i]), self._last_metrics,
                        )
                        ep_reward[i], ep_length[i] = 0.0, 0
                        if not keep_going:
                            return
                if not any_finished:
                    keep_going = callback.on_step(num_timesteps, metrics=self._last_metrics)
                    if not keep_going:
                        return

                episode_start = dones.astype(np.float32)
                if self.recurrent:
                    self._hidden = mask_hidden_for_dones(self._hidden, dones)
                obs_arr = self._obs_arr_batch(next_obs_list)

            with torch.no_grad():
                obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
                if self.recurrent:
                    _, last_value, _ = self.net.step(obs_t, self._hidden)
                else:
                    _, last_value = self.net.forward(obs_t)
            buf.compute_returns_and_advantage(last_value.cpu().numpy(), dones.astype(np.float32), self.gamma, self.gae_lambda)
            if getattr(self, "lr_schedule", "constant") == "linear":
                # Linearly anneal towards 0 as training progresses — same
                # idea as SB3's `lin_<lr>` schedule strings. Recomputed once
                # per rollout (not per gradient step) since `_update` below
                # runs many minibatches/epochs against one fixed LR.
                progress = min(1.0, num_timesteps / total_timesteps)
                new_lr = self._initial_lr * (1.0 - progress)
                for group in self.optimizer.param_groups:
                    group["lr"] = new_lr
            self._update(buf, rollout_hidden)

    def _update(self, buf: RolloutBuffer, rollout_hidden: Hidden | None = None) -> None:
        raise NotImplementedError

    def predict(self, obs: Any, deterministic: bool = True, episode_start: bool = False) -> tuple[Any, Any]:
        obs_arr = obs_to_array(obs, self._obs_space())
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.recurrent:
                if episode_start or self._predict_hidden is None:
                    self._predict_hidden = self.net.initial_state(1, self.device)
                if deterministic:
                    action_t, self._predict_hidden = self.net.deterministic_action_step(obs_t, self._predict_hidden)
                else:
                    dist, _, self._predict_hidden = self.net.distribution_step(obs_t, self._predict_hidden)
                    action_t = dist.sample()
            elif deterministic:
                action_t = self.net.deterministic_action(obs_t)
            else:
                dist, _ = self.net.distribution(obs_t)
                action_t = dist.sample()
        action = action_t.squeeze(0).cpu().numpy()
        return action_to_env(action, action_space(self.env)), None

    def save(self, path: Path) -> None:
        torch.save({
            "state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hyperparams": self.hyperparams,
        }, path)

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "OnPolicyAlgorithm":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.net.load_state_dict(payload["state_dict"])
        if payload.get("optimizer_state_dict"):
            algo.optimizer.load_state_dict(payload["optimizer_state_dict"])
        return algo
