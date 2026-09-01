"""Probabilistic ensemble of feed-forward dynamics models (PE-TS, Chua et
al. 2018 — https://arxiv.org/abs/1805.12114 — the model half of both MBPO
and PETS). Each member is a small MLP predicting a *distribution* over the
next-step transition, `p(Δobs, reward | obs, action) = N(mean, diag(var))`,
trained with a Gaussian negative-log-likelihood loss rather than plain MSE
— that predicted variance is exactly what lets a planner/short imagined
rollout account for how *confident* the model actually is at a given
(obs, action), instead of trusting every prediction equally.

Unlike the RSSM (`rssm.py`), there's no compressed latent state here at
all — every member operates directly on the (flattened) observation
itself, exactly the MBPO/PETS papers' own MuJoCo-continuous-control setup.
That's a deliberate scope choice: it's simple, fast, and a very strong
fit for this app's vector-observation envs (classic control / Box2D /
MuJoCo / industrial / trading), but doesn't scale gracefully to large
image observations the way the RSSM's conv encoder does — an ensemble
*can* still be built against an image `Box` space (nothing here special-
cases it away), it just means the last MLP layer's input/output width
equals the raw pixel count.

Two downstream uses, in `rl_core/algorithms/native/`:
- `mbpo.py` — short imagined rollouts from real replay states, added to a
  model-buffer that augments SAC's usual real-data updates.
- `pets.py` — no learned policy at all; `cem_plan` below searches directly
  over action sequences using this ensemble as the world model, replanning
  from scratch every real env step (standard MPC).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import gymnasium as gym

from rl_core.algorithms.native.preprocessing import is_image_space, obs_flat_dim
from rl_core.world_models.nets import action_dim_of

_LOGVAR_MIN, _LOGVAR_MAX = -10.0, 2.0


class _GroupedMLP(nn.Module):
    """`ensemble_size` independent MLPs of identical shape, evaluated as
    one batched matmul per layer (`torch.bmm`) instead of a Python `for
    model in self.models: ...` loop over members. Same total FLOPs either
    way — what changes is the number of kernel launches per forward pass:
    O(1) per layer here vs. O(ensemble_size) before, which is what
    actually dominates wall time for an ensemble this small (each member
    is just a couple of `(200, 200)` layers, so per-call Python/CUDA-
    launch overhead — not compute — was the bottleneck, especially on
    GPU). Each member still gets its own independently-initialized
    weights, exactly as `ensemble_size` separate `nn.Linear` stacks would."""

    def __init__(
        self, ensemble_size: int, in_dim: int, out_dim: int,
        hidden: tuple[int, ...] = (200, 200), activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        dims = [in_dim, *hidden, out_dim]
        self.num_layers = len(dims) - 1
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            weight = torch.empty(ensemble_size, d_out, d_in)
            bias = torch.empty(ensemble_size, d_out)
            for e in range(ensemble_size):
                # Same init `nn.Linear`'s own `reset_parameters` uses, just
                # applied per-member (a 3D `(E, out, in)` tensor doesn't
                # have `nn.init`'s usual 2D fan-in/fan-out convention).
                nn.init.kaiming_uniform_(weight[e], a=math.sqrt(5))
                bound = 1.0 / math.sqrt(d_in)
                nn.init.uniform_(bias[e], -bound, bound)
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(bias))
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(B, in_dim)` (broadcast to every member — the usual case,
        every member seeing the same inputs) or already `(E, B, in_dim)`
        (one caller-chosen row-subset per member, e.g. `nll_loss`'s
        per-member bootstrap resample) -> `(E, B, out_dim)`."""
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = torch.baddbmm(bias.unsqueeze(1), x, weight.transpose(-1, -2))
            if i < self.num_layers - 1:
                x = self.activation(x)
        return x


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space,
        ensemble_size: int = 5, hidden: tuple[int, ...] = (200, 200),
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.is_image = is_image_space(observation_space)
        self.obs_dim = obs_flat_dim(observation_space)
        self.action_dim = action_dim_of(action_space)
        self.ensemble_size = max(2, int(ensemble_size))
        self.net = _GroupedMLP(self.ensemble_size, self.obs_dim + self.action_dim, 2 * self.obs_dim + 2, hidden)

    def flat_target(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Raw `(B, ...)` obs tensor (already on the right device/dtype) ->
        `(B, obs_dim)` flat, `[0, 1]`-normalized for images — the space
        every member's `Δobs` prediction lives in."""
        t = obs_batch / 255.0 if self.is_image else obs_batch
        return t.reshape(t.shape[0], -1)

    def to_raw_obs(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """Inverse of `flat_target` — `(B, obs_dim)` flat -> `(B, *obs_shape)`
        raw scale. Used by `mbpo.py` to turn one of this model's own
        imagined next-states back into something the same shape/scale as a
        real env observation, so imagined and real transitions can share
        one replay buffer (`ContinuousReplayBuffer`) downstream."""
        shape = self.observation_space.shape
        raw = flat_obs.reshape(flat_obs.shape[0], *shape)
        return (raw * 255.0).clamp(0, 255) if self.is_image else raw

    def _split_output(self, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(..., 2*obs_dim+2)` raw net output -> `(Δobs mean, Δobs logvar,
        reward mean, reward logvar)`, whatever leading batch/ensemble dims
        `out` happens to carry."""
        delta_mean, delta_logvar, reward_mean, reward_logvar = out.split(
            [self.obs_dim, self.obs_dim, 1, 1], dim=-1,
        )
        delta_logvar = delta_logvar.clamp(_LOGVAR_MIN, _LOGVAR_MAX)
        reward_logvar = reward_logvar.clamp(_LOGVAR_MIN, _LOGVAR_MAX)
        return delta_mean, delta_logvar, reward_mean.squeeze(-1), reward_logvar.squeeze(-1)

    def _all_members_forward(self, flat_obs: torch.Tensor, action: torch.Tensor):
        """Every member sees the exact same `(flat_obs, action)` batch —
        `self.net` broadcasts it to all `ensemble_size` members and
        returns each one's own `(Δobs mean, Δobs logvar, reward mean,
        reward logvar)`, every tensor shaped `(E, B, ...)`."""
        return self._split_output(self.net(torch.cat([flat_obs, action], dim=-1)))

    def nll_loss(
        self, flat_obs: torch.Tensor, action: torch.Tensor, flat_next_obs: torch.Tensor, reward: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Gaussian NLL, averaged over every ensemble member — each member
        trains on its own bootstrap resample of the batch (with
        replacement) rather than the identical batch every other member
        sees, which is what actually makes the ensemble *disagree* in
        regions with little data (exactly the signal PETS'/MBPO's
        downstream uncertainty handling depends on). `idx`: `(E, B)`, one
        resample per member, built and gathered in one shot rather than
        the Python "one `torch.randint` + one forward pass per member"
        loop this used to be — `self.net`'s own batched-per-member forward
        (see `_GroupedMLP`) already accepts a `(E, B, ...)` input as-is.

        Returns `(total_loss, {"dynamics_loss": ..., "reward_loss": ...})`
        — the two components split out (both still averaged over every
        member's own resample) purely for separate loss-curve charting in
        the Training Monitor; `total_loss` is the only one actually
        `.backward()`-ed."""
        target_delta = flat_next_obs - flat_obs
        batch_size = flat_obs.shape[0]
        idx = torch.randint(0, batch_size, (self.ensemble_size, batch_size), device=flat_obs.device)
        x = torch.cat([flat_obs[idx], action[idx]], dim=-1)
        delta_mean, delta_logvar, reward_mean, reward_logvar = self._split_output(self.net(x))
        obs_nll = 0.5 * (((target_delta[idx] - delta_mean) ** 2) / delta_logvar.exp() + delta_logvar).sum(dim=-1)
        reward_nll = 0.5 * (((reward[idx] - reward_mean) ** 2) / reward_logvar.exp() + reward_logvar)
        obs_loss = obs_nll.mean()
        reward_loss = reward_nll.mean()
        return obs_loss + reward_loss, {"dynamics_loss": obs_loss, "reward_loss": reward_loss}

    def predict_step(
        self, flat_obs: torch.Tensor, action: torch.Tensor, model_indices: torch.Tensor | None = None, sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One imagined transition, `(B, obs_dim) -> (B, obs_dim), (B,)`.
        `model_indices` (one ensemble member index per batch row) defaults
        to a fresh random draw per call — PETS/MBPO's usual "trajectory
        sampling" (TS1): every candidate rollout is stepped through a
        *fixed* member of its own for its whole horizon (pass the same
        tensor across calls within one rollout) rather than a fresh random
        member every single step, which would average away exactly the
        epistemic uncertainty the ensemble exists to capture."""
        batch_size = flat_obs.shape[0]
        means, logvars, rmeans, rlogvars = self._all_members_forward(flat_obs, action)
        if model_indices is None:
            model_indices = torch.randint(0, self.ensemble_size, (batch_size,), device=flat_obs.device)
        obs_idx = model_indices.view(1, batch_size, 1).expand(1, batch_size, self.obs_dim)
        delta_mean = means.gather(0, obs_idx).squeeze(0)
        delta_logvar = logvars.gather(0, obs_idx).squeeze(0)
        r_idx = model_indices.view(1, batch_size)
        reward_mean = rmeans.gather(0, r_idx).squeeze(0)
        reward_logvar = rlogvars.gather(0, r_idx).squeeze(0)
        if sample:
            delta = delta_mean + torch.randn_like(delta_mean) * (0.5 * delta_logvar).exp()
            reward = reward_mean + torch.randn_like(reward_mean) * (0.5 * reward_logvar).exp()
        else:
            delta, reward = delta_mean, reward_mean
        return flat_obs + delta, reward

    def sample_model_indices(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(0, self.ensemble_size, (batch_size,), device=device)


@torch.no_grad()
def cem_plan(
    model: EnsembleDynamicsModel,
    flat_obs0: torch.Tensor,
    action_space: gym.spaces.Box,
    horizon: int = 15,
    num_candidates: int = 400,
    num_elites: int = 40,
    num_iterations: int = 5,
) -> torch.Tensor:
    """Cross-Entropy Method planning (Chua et al. 2018's PETS, or plain
    MPC with CEM more generally) — no learned policy anywhere, just direct
    search: sample `num_candidates` random action *sequences* from a
    diagonal Gaussian, roll every one through the model, keep the
    `num_elites` with the highest predicted cumulative reward, refit the
    Gaussian to those elites, repeat, and finally return only the first
    action of the best-refined mean sequence (standard MPC — replan from
    scratch next real step, throwing the rest of this plan away, since the
    model's errors compound the further ahead it predicts).

    `flat_obs0`: `(obs_dim,)` — a single current (real) state; every
    candidate in the population is evaluated starting from a copy of it.
    """
    device = flat_obs0.device
    low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device).reshape(-1)
    high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device).reshape(-1)
    action_dim = low.shape[0]
    mean = torch.zeros(horizon, action_dim, device=device)
    std = ((high - low) / 2.0).unsqueeze(0).expand(horizon, -1).clone()

    obs0_batch = flat_obs0.unsqueeze(0).expand(num_candidates, -1)
    for _ in range(max(1, num_iterations)):
        noise = torch.randn(num_candidates, horizon, action_dim, device=device)
        candidates = torch.clamp(mean.unsqueeze(0) + std.unsqueeze(0) * noise, low, high)
        # TS1 trajectory sampling: each candidate keeps the *same* ensemble
        # member for its entire rollout — see `predict_step`'s docstring.
        model_indices = model.sample_model_indices(num_candidates, device)
        obs = obs0_batch
        returns = torch.zeros(num_candidates, device=device)
        for t in range(horizon):
            obs, reward = model.predict_step(obs, candidates[:, t], model_indices=model_indices, sample=True)
            returns += reward
        elite_idx = torch.topk(returns, min(num_elites, num_candidates)).indices
        elites = candidates[elite_idx]
        mean = elites.mean(dim=0)
        std = elites.std(dim=0) + 1e-3
    return torch.clamp(mean[0], low, high)
