"""The original "World Models" agent (Ha & Schmidhuber, 2018 —
https://arxiv.org/abs/1803.10122): train a VAE + MDN-RNN
(`rl_core/world_models/vae_mdnrnn.py`) on random-policy rollouts, freeze
both, then evolve a tiny linear controller (Evolution Strategies, same
mirrored-sampling + centered-rank-fitness recipe as
`rl_core/algorithms/native/es.py`) acting directly on `[z, h]` — the VAE's
latent code concatenated with the MDN-RNN's own recurrent hidden state —
against real episode return. Two genuinely sequential phases within one
`learn()` call, `total_timesteps` split between them by
`world_model_phase_steps`:

1. **World model phase** — collect real transitions with a uniform-random
   policy (this module has no opinion on a better one; it's also exactly
   what the paper itself does for this phase) into a
   `SequenceReplayBuffer`, training the VAE + MDN-RNN on sampled windows
   throughout.
2. **Controller phase** — both networks frozen (`requires_grad_(False)`),
   the tiny controller evolved by ES, one full real episode per population
   member per generation (unlike `es.py`, no `num_envs`-parallel population
   evaluation here — see the module docstring below for why).

Skips the paper's optional third phase (training the controller *inside*
a fully hallucinated M-model "dream" env instead of the real one) — a
bigger undertaking on top of an already large feature, and not needed to
get the core "learn a compact world model, then act on it" idea working
end-to-end.

`training.num_envs > 1` speeds both phases up, in two different ways:
- Phase 1 collects with that many parallel env lanes (`collect_rollout_auto`
  — same `AsyncVectorEnv` lever every other native algorithm's
  `training.num_envs` already uses), each lane its own independent
  episode in `SequenceReplayBuffer`.
- Phase 2 still evaluates population members one at a time (no
  vectorized *population* batching like `es.py`'s `_batched_forward`,
  since a good chunk of this algorithm's own point is being usable with
  image observations, where that trick doesn't apply anyway — see
  `es.py`'s own docstring) — but whenever `episodes_per_eval > 1`, those
  repeats of *one* candidate now run across `num_envs` lanes at once
  (`_run_episodes_vec`) instead of strictly one after another."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.preprocessing import action_to_env, obs_to_array
from rl_core.algorithms.vec_env import (
    action_space as venv_action_space,
    is_vector_env,
    num_envs_of,
    obs_space as venv_obs_space,
    vec_reset,
    vec_step,
)
from rl_core.world_models.losses import train_step_vae_mdnrnn
from rl_core.world_models.replay import SequenceReplayBuffer, action_flat_dim, collect_rollout_auto
from rl_core.world_models.spec import resolve_or_build_for_algo, save_checkpoint

DEFAULT_HYPERPARAMS = {
    "world_model_learning_rate": 1e-3,
    "world_model_phase_steps": 10_000,
    "seq_len": 50,
    "batch_size": 32,
    "world_model_train_steps_per_iter": 20,
    "population_size": 16,
    "sigma": 0.1,
    "es_learning_rate": 0.02,
    "episodes_per_eval": 1,
}


def _centered_ranks(fitness: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(fitness)
    ranks[np.argsort(fitness)] = np.arange(len(fitness))
    return ranks / (len(fitness) - 1) - 0.5


class _Controller(nn.Module):
    """A single linear layer straight from `[z, h]` to the action — exactly
    the paper's own controller (its whole point is to show how *little*
    policy is needed once the world model itself has done the hard work)."""

    def __init__(self, in_dim: int, action_space: gym.Space) -> None:
        super().__init__()
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.discrete:
            self.action_dim = int(action_space.n)
        else:
            self.action_dim = int(np.prod(action_space.shape))
            low = np.where(np.isfinite(action_space.low), action_space.low, -1.0).reshape(-1)
            high = np.where(np.isfinite(action_space.high), action_space.high, 1.0).reshape(-1)
            self.register_buffer("action_scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
            self.register_buffer("action_bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))
        self.linear = nn.Linear(in_dim, self.action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return out if self.discrete else torch.tanh(out) * self.action_scale + self.action_bias

    def act(self, x: torch.Tensor) -> np.ndarray:
        out = self.forward(x)
        if self.discrete:
            return torch.argmax(out, dim=-1).cpu().numpy()
        return out.squeeze(0).cpu().numpy()


class NativeWorldModelsHA(CustomAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self._obs_space = venv_obs_space(env)
        self._action_space = venv_action_space(env)

        built, self.world_model_config, self.world_model_slug = resolve_or_build_for_algo(
            hyperparams.get("world_model_spec"), "vae_mdnrnn", self._obs_space, self._action_space, device,
        )
        self.vae = built["vae"].to(device)
        self.mdnrnn = built["mdnrnn"].to(device)
        self.wm_optimizer = torch.optim.Adam(
            list(self.vae.parameters()) + list(self.mdnrnn.parameters()),
            lr=float(hyperparams.get("world_model_learning_rate", 1e-3)),
        )

        controller_in_dim = self.vae.latent_dim + self.mdnrnn.hidden_size
        self.controller = _Controller(controller_in_dim, self._action_space).to(device)
        self._param_shapes = [p.shape for p in self.controller.parameters()]
        self._num_params = sum(p.numel() for p in self.controller.parameters())

        pop = int(hyperparams.get("population_size", 16))
        self.population_size = max(2, pop + 1 if pop % 2 else pop)
        self.sigma = float(hyperparams.get("sigma", 0.1))
        self.es_learning_rate = float(hyperparams.get("es_learning_rate", 0.02))
        self.episodes_per_eval = max(1, int(hyperparams.get("episodes_per_eval", 1)))
        self.es_optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.es_learning_rate)

        self.world_model_phase_steps = int(hyperparams.get("world_model_phase_steps", 10_000))
        self.seq_len = int(hyperparams.get("seq_len", 50))
        self.batch_size = int(hyperparams.get("batch_size", 32))
        self.world_model_train_steps_per_iter = max(1, int(hyperparams.get("world_model_train_steps_per_iter", 20)))

        self._last_metrics: dict[str, float] = {}
        self._frozen = False
        self._predict_hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def _freeze_world_model(self) -> None:
        if self._frozen:
            return
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.mdnrnn.parameters():
            p.requires_grad_(False)
        self.vae.eval()
        self.mdnrnn.eval()
        self._frozen = True

    def _get_flat_params(self) -> np.ndarray:
        with torch.no_grad():
            return torch.cat([p.reshape(-1) for p in self.controller.parameters()]).cpu().numpy()

    def _set_flat_params(self, flat: np.ndarray) -> None:
        flat_t = torch.as_tensor(flat, dtype=torch.float32, device=self.device)
        offset = 0
        with torch.no_grad():
            for p, shape in zip(self.controller.parameters(), self._param_shapes):
                numel = int(np.prod(shape))
                p.copy_(flat_t[offset : offset + numel].reshape(shape))
                offset += numel

    @torch.no_grad()
    def _run_episode(self) -> tuple[float, int]:
        """One full real episode, controller acting on `[z, h]` at every
        step — `z` is the VAE's *mean* encoding (not a stochastic sample)
        purely to keep ES's own fitness signal from also carrying VAE
        sampling noise on top of the controller's own parameter-space
        noise; the MDN-RNN's hidden state `h` still evolves genuinely
        conditioned on the real `(z, action)` history taken along the way."""
        obs, _info = self.env.reset()
        hidden = self.mdnrnn.initial_state(1, self.device)
        total_reward, length = 0.0, 0
        done = False
        while not done:
            obs_arr = obs_to_array(obs, self._obs_space)
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
            z, _logvar = self.vae.encode(obs_t)
            controller_input = torch.cat([z, hidden[0].squeeze(0)], dim=-1)
            action_np = self.controller.act(controller_input)
            env_action = action_to_env(action_np, self._action_space)

            next_obs, reward, terminated, truncated, _info = self.env.step(env_action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            length += 1

            action_arr = obs_to_array(env_action, self._action_space)
            action_t = torch.as_tensor(action_arr, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(1)
            _logits, _means, _log_stds, _reward, _continue_logit, hidden = self.mdnrnn(z.unsqueeze(1), action_t, hidden)
            obs = next_obs
        return total_reward, length

    @torch.no_grad()
    def _run_episodes_vec(self, num_episodes: int) -> tuple[list[float], list[int]]:
        """`training.num_envs > 1` counterpart to `_run_episode` — runs up
        to `num_envs_of(self.env)` replicate episodes of the *current*
        (shared) controller weights simultaneously, one per vector-env
        lane, instead of `_run_episode`'s one-at-a-time loop. Every lane
        uses the exact same `self.controller`/`self.vae`/`self.mdnrnn`
        (all genuinely batched over the lane dimension already — no
        per-lane distinct weights needed, unlike a "different population
        member per lane" scheme would require), so this only changes
        *how many real episodes run per wall-clock env step*, never what
        gets evaluated. Repeats in waves of `num_envs_of(self.env)` until
        `num_episodes` have been collected (e.g. `episodes_per_eval=10`
        with `num_envs=4` runs three waves: 4 + 4 + 2)."""
        num_lanes = num_envs_of(self.env)
        returns: list[float] = []
        lengths: list[int] = []
        remaining = num_episodes
        while remaining > 0:
            active = min(remaining, num_lanes)
            obs = vec_reset(self.env)
            hidden = self.mdnrnn.initial_state(num_lanes, self.device)
            ep_reward = np.zeros(num_lanes, dtype=np.float64)
            ep_length = np.zeros(num_lanes, dtype=np.int64)
            # Lanes beyond `active` (only when `num_episodes` doesn't
            # evenly divide `num_lanes`, i.e. the very last wave) still
            # need a valid action every step — marking them "done" up
            # front just excludes their reward/length from the result.
            done_mask = np.arange(num_lanes) >= active
            while not done_mask.all():
                obs_arr = np.stack([obs_to_array(o, self._obs_space) for o in obs])
                obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
                z, _logvar = self.vae.encode(obs_t)
                controller_input = torch.cat([z, hidden[0].squeeze(0)], dim=-1)
                out = self.controller.forward(controller_input)
                action_np = (torch.argmax(out, dim=-1) if self.controller.discrete else out).cpu().numpy()
                env_actions = [action_to_env(action_np[i], self._action_space) for i in range(num_lanes)]

                next_obs, rewards, terminated, truncated, _infos = vec_step(self.env, env_actions)
                action_arr = np.stack([obs_to_array(a, self._action_space) for a in env_actions])
                action_t = torch.as_tensor(action_arr, dtype=torch.float32, device=self.device).unsqueeze(1)
                _logits, _means, _log_stds, _reward, _continue_logit, hidden = self.mdnrnn(z.unsqueeze(1), action_t, hidden)

                for i in range(num_lanes):
                    if done_mask[i]:
                        continue
                    ep_reward[i] += float(rewards[i])
                    ep_length[i] += 1
                    if terminated[i] or truncated[i]:
                        done_mask[i] = True
                obs = next_obs
            returns.extend(float(x) for x in ep_reward[:active])
            lengths.extend(int(x) for x in ep_length[:active])
            remaining -= active
        return returns, lengths

    def _evaluate_candidate(self, num_episodes: int) -> tuple[list[float], int]:
        """One population member's fitness — `episodes_per_eval` full
        episodes of the *current* `self.controller` weights, averaged.
        Dispatches to `_run_episodes_vec` (parallel lanes) whenever
        `self.env` is vectorized, else `_run_episode`'s original
        one-episode-at-a-time loop."""
        if is_vector_env(self.env):
            returns, lengths = self._run_episodes_vec(num_episodes)
            return returns, int(sum(lengths))
        returns, total_length = [], 0
        for _ in range(num_episodes):
            ep_return, ep_len = self._run_episode()
            returns.append(ep_return)
            total_length += ep_len
        return returns, total_length

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        num_timesteps = 0
        obs_shape = self.vae.encoder.obs_shape
        action_dim = action_flat_dim(self._action_space)
        buffer = SequenceReplayBuffer(max(self.world_model_phase_steps, 4_000), obs_shape, action_dim, num_lanes=num_envs_of(self.env))
        obs_cursor = None
        phase1_budget = min(self.world_model_phase_steps, total_timesteps)

        while num_timesteps < phase1_budget:
            step_chunk = min(200, phase1_budget - num_timesteps)
            obs_cursor, stats = collect_rollout_auto(self.env, self._action_space, step_chunk, sequence_buffer=buffer, obs=obs_cursor)
            num_timesteps += step_chunk
            if buffer.num_episodes >= 2 and len(buffer) >= self.seq_len:
                for _ in range(self.world_model_train_steps_per_iter):
                    losses = train_step_vae_mdnrnn(
                        self.vae, self.mdnrnn, self.wm_optimizer, buffer, self.batch_size, self.seq_len, self.device,
                    )
                self._last_metrics.update(losses)
            if stats["mean_reward"] is not None:
                self._last_metrics["collect_mean_reward"] = stats["mean_reward"]
            keep_going = callback.on_step(num_timesteps, metrics=self._last_metrics)
            if not keep_going:
                return

        self._freeze_world_model()

        rng = np.random.default_rng(self.seed)
        theta = self._get_flat_params()
        half = self.population_size // 2

        while num_timesteps < total_timesteps:
            eps = rng.standard_normal((half, self._num_params)).astype(np.float32)
            fitness = np.zeros(self.population_size, dtype=np.float32)
            completed_fitness: list[float] = []

            for i in range(half):
                for sign, idx in ((1.0, i), (-1.0, half + i)):
                    self._set_flat_params(theta + sign * self.sigma * eps[i])
                    ep_returns, ep_len_total = self._evaluate_candidate(self.episodes_per_eval)
                    num_timesteps += ep_len_total
                    fitness[idx] = float(np.mean(ep_returns))
                    completed_fitness.append(fitness[idx])

                    self._set_flat_params(theta)  # restore the current mean before any live-preview render
                    self._last_metrics.update({
                        "es_mean_fitness": float(np.mean(completed_fitness)),
                        "es_best_fitness": float(np.max(completed_fitness)),
                    })
                    keep_going = callback.on_step(num_timesteps, ep_returns[-1], ep_len_total, self._last_metrics)
                    if not keep_going:
                        return

            ranks = _centered_ranks(fitness)
            grad = np.zeros(self._num_params, dtype=np.float32)
            for i in range(half):
                grad += (ranks[i] - ranks[half + i]) * eps[i]
            grad /= half * self.sigma

            self._set_flat_params(theta)
            offset = 0
            self.es_optimizer.zero_grad()
            for p, shape in zip(self.controller.parameters(), self._param_shapes):
                numel = int(np.prod(shape))
                p.grad = torch.as_tensor(-grad[offset : offset + numel].reshape(shape), dtype=torch.float32, device=self.device)
                offset += numel
            self.es_optimizer.step()
            theta = self._get_flat_params()

    def predict(self, obs: Any, deterministic: bool = True, episode_start: bool = False) -> tuple[Any, Any]:
        del deterministic  # the controller is already a deterministic function of [z, h]
        if episode_start or self._predict_hidden is None:
            self._predict_hidden = self.mdnrnn.initial_state(1, self.device)
        obs_arr = obs_to_array(obs, self._obs_space)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            z, _logvar = self.vae.encode(obs_t)
            controller_input = torch.cat([z, self._predict_hidden[0].squeeze(0)], dim=-1)
            action_np = self.controller.act(controller_input)
        env_action = action_to_env(action_np, self._action_space)
        action_arr = obs_to_array(env_action, self._action_space)
        action_t = torch.as_tensor(action_arr, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(1)
        with torch.no_grad():
            _logits, _means, _log_stds, _reward, _continue_logit, hidden = self.mdnrnn(z.unsqueeze(1), action_t, self._predict_hidden)
        self._predict_hidden = hidden
        return env_action, None

    def save(self, path: Path) -> None:
        torch.save(
            {
                "vae_state_dict": self.vae.state_dict(),
                "mdnrnn_state_dict": self.mdnrnn.state_dict(),
                "controller_state_dict": self.controller.state_dict(),
                "world_model_config": self.world_model_config,
                "hyperparams": self.hyperparams,
            },
            path,
        )

    def save_world_model_checkpoint(self, path: Path) -> None:
        save_checkpoint("vae_mdnrnn", {"vae": self.vae, "mdnrnn": self.mdnrnn}, path, self.world_model_config)

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativeWorldModelsHA":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.vae.load_state_dict(payload["vae_state_dict"])
        algo.mdnrnn.load_state_dict(payload["mdnrnn_state_dict"])
        algo.controller.load_state_dict(payload["controller_state_dict"])
        return algo
