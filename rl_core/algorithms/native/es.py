"""From-scratch Evolution Strategies — a *gradient-free* alternative to
every other algorithm in this app. Every other native algorithm
(PPO/A2C/DQN/Rainbow/SAC/DDPG/TD3) backpropagates through a differentiable
loss computed from individual transitions. ES never does: it treats the
whole episode return as a black-box fitness function of the policy's
*parameters* and estimates a gradient by finite differences over a
population of randomly perturbed parameter vectors (Salimans et al., 2017 —
often just called "OpenAI-ES"). No replay buffer, no value function, no
backprop through the environment at all — which also means it's completely
indifferent to reward sparsity/non-differentiability/episode length in a way
the other algorithms aren't, at the cost of needing many full episode
rollouts per update.

This implements the *separable* (diagonal-covariance) variant rather than
full-matrix CMA-ES: CMA-ES's O(d^2) covariance estimate is only tractable
for up to a few thousand parameters, while even `ESPolicyNet`'s small
64-unit hidden layer has more than that for anything but the tiniest
observation space. Diagonal ES scales to arbitrarily large networks (it's
exactly what let Salimans et al. train Atari-sized policies with it) at the
cost of not modeling parameter *correlations* — a reasonable trade for this
app's from-scratch setting.

Two specific tricks from the paper, both implemented below:
- **Mirrored sampling (antithetic variates)**: every noise vector epsilon is
  evaluated as both `+epsilon` and `-epsilon`, which provably reduces the
  gradient estimator's variance for free (no extra rollouts needed per
  *pair*, just population_size/2 distinct noise draws).
- **Fitness shaping via centered ranks**: raw episode returns are replaced
  by their rank in the population (linearly mapped to [-0.5, 0.5]) before
  being used as weights — makes the update invariant to the reward's scale
  and robust to a single outlier episode dominating the gradient estimate.

With `num_envs>1` (see `rl_core/algorithms/vec_env.py`), parallelism here
is *population*-parallel rather than time-parallel like every other
algorithm in this app: instead of running each perturbed parameter
vector's full episode to completion before starting the next one, up to
`num_envs` population members' episodes run *concurrently*, one per lane
(in separate subprocess workers when using the default `AsyncVectorEnv`),
stepped in lockstep — wall-clock for a batch is bounded by its single
slowest member's episode length rather than by all of them summed. Each
lane still needs its *own* forward pass through the network loaded with
that lane's own parameter vector before its action, since population
members are genuinely different weight vectors, not just different inputs
to one shared policy the way every other algorithm's `num_envs` lanes are
— but for a plain MLP policy (vector observations — the vast majority of
ES's actual use here) that's just a batched matmul per layer with a
per-lane weight matrix, not `num_envs` separate `_set_flat_params` +
`forward` calls; see `_batched_forward`/`_run_episode_batch`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.networks import ESPolicyNet, MLPExtractor
from rl_core.algorithms.native.preprocessing import obs_to_array
from rl_core.algorithms.vec_env import action_space, num_envs_of, obs_space, vec_reset, vec_step

DEFAULT_HYPERPARAMS = {
    "population_size": 32,
    "sigma": 0.1,
    "learning_rate": 0.02,
    "episodes_per_eval": 1,
}


def _centered_ranks(fitness: np.ndarray) -> np.ndarray:
    """Maps raw fitness values to ranks linearly spaced over [-0.5, 0.5] —
    the same reward-scale-invariant weighting OpenAI-ES uses instead of the
    raw returns themselves."""
    ranks = np.empty_like(fitness)
    ranks[np.argsort(fitness)] = np.arange(len(fitness))
    return ranks / (len(fitness) - 1) - 0.5


class NativeES(CustomAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        self._obs_space = obs_space(env)
        self._action_space = action_space(env)
        self.policy = ESPolicyNet(self._obs_space, self._action_space).to(device)

        # Mirrored sampling needs an even population — silently round up
        # rather than erroring, since 1 off either way makes no meaningful
        # difference to the user's intent.
        pop = int(hyperparams.get("population_size", 32))
        self.population_size = pop + 1 if pop % 2 else pop
        self.population_size = max(2, self.population_size)
        self.sigma = float(hyperparams.get("sigma", 0.1))
        self.learning_rate = float(hyperparams.get("learning_rate", 0.02))
        self.episodes_per_eval = max(1, int(hyperparams.get("episodes_per_eval", 1)))

        self._param_shapes = [p.shape for p in self.policy.parameters()]
        self._num_params = sum(p.numel() for p in self.policy.parameters())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self._last_metrics: dict[str, float] = {}

        # `_run_episode_batch` below can evaluate a whole population group
        # in a handful of batched matmuls instead of looping
        # `_set_flat_params` + a separate forward per lane — but only when
        # `self.policy` is a plain stack of `Linear`/`Tanh` layers (true for
        # every vector-observation env, which is the vast majority of ES's
        # actual use here; see the module docstring). Image observations
        # fall back to the original per-lane loop instead of also handling
        # "batched conv with a distinct kernel per sample", which is a
        # meaningfully different (and much rarer, for ES) operation.
        self._batchable = isinstance(self.policy.features, MLPExtractor)
        if self._batchable:
            self._linear_layers = [m for m in self.policy.modules() if isinstance(m, nn.Linear)]

    def _obs_arr(self, obs: Any) -> np.ndarray:
        return obs_to_array(obs, self._obs_space)

    def _get_flat_params(self) -> np.ndarray:
        with torch.no_grad():
            return torch.cat([p.reshape(-1) for p in self.policy.parameters()]).cpu().numpy()

    def _set_flat_params(self, flat: np.ndarray) -> None:
        flat_t = torch.as_tensor(flat, dtype=torch.float32, device=self.device)
        offset = 0
        with torch.no_grad():
            for p, shape in zip(self.policy.parameters(), self._param_shapes):
                numel = int(np.prod(shape))
                p.copy_(flat_t[offset : offset + numel].reshape(shape))
                offset += numel

    def _batched_forward(self, param_batch: torch.Tensor, obs_batch: torch.Tensor) -> np.ndarray:
        """Evaluates `param_batch.shape[0]` *independent* copies of
        `self.policy` — one per row, i.e. one per ES population member,
        each with its own weights — on their own observation, as one pass
        of batched matmuls (`bmm`) rather than `param_batch.shape[0]`
        separate `_set_flat_params` + `forward` calls. Mathematically
        identical to that loop (every population member really does have
        different parameters, unlike every other algorithm's `num_envs`
        lanes, which all share one policy) — just without paying full
        Python + autograd-dispatch overhead per lane per step for a
        network with a few hundred parameters.

        `param_batch`: `(G, num_params)`, flattened in the exact same
        per-parameter order as `_set_flat_params`/`_get_flat_params`.
        `obs_batch`: `(G, obs_dim)`. Returns the *raw* action (already
        argmax'd for discrete, tanh-squashed+rescaled for continuous) as a
        `(G, ...)` numpy array — matching `ESPolicyNet.act()`'s per-lane
        output stacked over the batch dim."""
        x = obs_batch
        offset = 0
        n_layers = len(self._linear_layers)
        for i, layer in enumerate(self._linear_layers):
            out_f, in_f = layer.weight.shape
            w = param_batch[:, offset : offset + out_f * in_f].reshape(-1, out_f, in_f)
            offset += out_f * in_f
            b = param_batch[:, offset : offset + out_f]
            offset += out_f
            x = torch.bmm(w, x.unsqueeze(-1)).squeeze(-1) + b
            if i < n_layers - 1:
                x = torch.tanh(x)
        if self.policy.discrete:
            return torch.argmax(x, dim=-1).cpu().numpy()
        out = torch.tanh(x) * self.policy.action_scale + self.policy.action_bias
        return out.cpu().numpy()

    def _run_episode_batch(self, param_vectors: list[np.ndarray], restore_theta: np.ndarray) -> tuple[list[float], list[int]]:
        """Runs `len(param_vectors)` population members' episodes
        *concurrently*, one per lane of `self.env` (padded with repeats of
        the last vector if there are fewer jobs than lanes — their results
        are simply discarded below). Deterministic — ES's exploration
        lives entirely in parameter space, so there's no reason to also
        add action-space noise here, which would just make the fitness
        signal noisier for no benefit. Restores `restore_theta` (the
        *current mean* policy, not any perturbed candidate) into
        `self.policy` before returning, so a live-preview episode render
        triggered by the caller's `callback.on_step` right after this
        shows what the run actually believes is its best policy right
        now — see `learn()`.

        Returns `(returns, lengths)`, one pair per input parameter vector,
        in the same order — `num_envs=1` degenerates to running exactly
        one member's one episode, byte-identical to the pre-vectorization
        version of this method."""
        n_envs = num_envs_of(self.env)
        n = len(param_vectors)
        padded = list(param_vectors) + [param_vectors[-1]] * (n_envs - n)
        obs_list = vec_reset(self.env, seed=None)
        total_reward = np.zeros(n_envs, dtype=np.float64)
        length = np.zeros(n_envs, dtype=np.int64)
        active = np.ones(n_envs, dtype=bool)

        if self._batchable:
            # Each lane's parameters are fixed for the whole episode (only
            # the observation changes step to step) — build the batch once
            # rather than re-converting it on every step.
            param_batch = torch.as_tensor(np.stack(padded), dtype=torch.float32, device=self.device)
            while active.any():
                obs_batch = torch.as_tensor(
                    np.stack([self._obs_arr(o) for o in obs_list]), dtype=torch.float32, device=self.device,
                )
                with torch.no_grad():
                    actions_arr = self._batched_forward(param_batch, obs_batch)
                if isinstance(self._action_space, gym.spaces.Box):
                    low, high, shape = self._action_space.low, self._action_space.high, self._action_space.shape
                    actions = [np.clip(actions_arr[lane], low, high).reshape(shape) for lane in range(n_envs)]
                else:
                    actions = [int(a) for a in actions_arr]
                obs_list, rewards, terminated, truncated, _infos = vec_step(self.env, actions)
                dones = terminated | truncated
                total_reward += np.where(active, rewards, 0.0)
                length += active.astype(np.int64)
                active = active & ~dones
        else:
            while active.any():
                actions = []
                for lane in range(n_envs):
                    self._set_flat_params(padded[lane])
                    obs_t = torch.as_tensor(self._obs_arr(obs_list[lane]), dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        action = self.policy.act(obs_t)
                    if isinstance(self._action_space, gym.spaces.Box):
                        action = np.clip(action, self._action_space.low, self._action_space.high).reshape(self._action_space.shape)
                    actions.append(action)
                obs_list, rewards, terminated, truncated, _infos = vec_step(self.env, actions)
                dones = terminated | truncated
                total_reward += np.where(active, rewards, 0.0)
                length += active.astype(np.int64)
                active = active & ~dones

        self._set_flat_params(restore_theta)
        return total_reward[:n].tolist(), length[:n].tolist()

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        rng = np.random.default_rng(self.seed)
        theta = self._get_flat_params()
        total_steps = 0
        half = self.population_size // 2
        n_envs = num_envs_of(self.env)

        while total_steps < total_timesteps:
            eps = rng.standard_normal((half, self._num_params)).astype(np.float32)
            fitness = np.zeros(self.population_size, dtype=np.float32)
            # Every (sign, idx) below is one population member — mirrored
            # sampling pairs member `i` (+eps[i]) with `half+i` (-eps[i]).
            members = [(sign, idx, i) for i in range(half) for sign, idx in ((1.0, i), (-1.0, half + i))]
            completed_fitness: list[float] = []

            for group_start in range(0, len(members), n_envs):
                group = members[group_start : group_start + n_envs]
                group_params = [theta + sign * self.sigma * eps[i] for sign, _idx, i in group]
                ep_rewards: list[list[float]] = [[] for _ in group]
                ep_len_total = [0] * len(group)
                for _ in range(self.episodes_per_eval):
                    batch_returns, batch_lengths = self._run_episode_batch(group_params, restore_theta=theta)
                    for g in range(len(group)):
                        ep_rewards[g].append(batch_returns[g])
                        ep_len_total[g] += batch_lengths[g]
                    total_steps += sum(batch_lengths)

                for g, (_sign, idx, _i) in enumerate(group):
                    fitness[idx] = float(np.mean(ep_rewards[g]))
                    completed_fitness.append(fitness[idx])
                    self._last_metrics = {
                        "es_mean_fitness": float(np.mean(completed_fitness)),
                        "es_best_fitness": float(np.max(completed_fitness)),
                        "es_sigma": self.sigma,
                    }
                    keep_going = callback.on_step(total_steps, ep_rewards[g][-1], ep_len_total[g], self._last_metrics)
                    if not keep_going:
                        return

            ranks = _centered_ranks(fitness)
            grad = np.zeros(self._num_params, dtype=np.float32)
            for i in range(half):
                grad += (ranks[i] - ranks[half + i]) * eps[i]
            grad /= half * self.sigma

            # No autograd anywhere in ES — but plugging the estimated
            # (ascent) gradient into Adam via `param.grad` still gives us
            # its per-parameter adaptive step sizes / momentum for free,
            # which noticeably speeds up convergence over plain SGD on the
            # same estimate. Adam *descends* along `.grad`, so the sign is
            # flipped to turn "ascend along `grad`" into "descend along
            # `-grad`".
            self._set_flat_params(theta)
            offset = 0
            self.optimizer.zero_grad()
            for p, shape in zip(self.policy.parameters(), self._param_shapes):
                numel = int(np.prod(shape))
                p.grad = torch.as_tensor(
                    -grad[offset : offset + numel].reshape(shape), dtype=torch.float32, device=self.device
                )
                offset += numel
            self.optimizer.step()
            theta = self._get_flat_params()

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]:
        obs_t = torch.as_tensor(self._obs_arr(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.policy.act(obs_t)
        if isinstance(self._action_space, gym.spaces.Box):
            return np.asarray(action, dtype=np.float32).reshape(self._action_space.shape), None
        return action, None

    def save(self, path: Path) -> None:
        torch.save({"policy_state_dict": self.policy.state_dict(), "hyperparams": self.hyperparams}, path)

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativeES":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.policy.load_state_dict(payload["policy_state_dict"])
        return algo
