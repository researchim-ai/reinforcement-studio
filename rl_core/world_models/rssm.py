"""Dreamer-style Recurrent State-Space Model (Hafner et al., 2019/2020 —
https://arxiv.org/abs/1912.01603) — a latent dynamics model with *two*
state components carried through time: a deterministic GRU state `h`
(remembers everything relevant, no information bottleneck) and a
stochastic latent `z` sampled from a learned Gaussian (models the genuine
uncertainty a single forward pass can't resolve, e.g. what's behind a
door the agent hasn't opened yet).

Two ways to get `z_t` given the deterministic state `h_t`:
- **posterior** (`step_posterior`) — sees the *actual* observation at time
  `t` (via the shared encoder), used whenever real experience is
  available (fitting the model to a replay-buffer sequence).
- **prior** (`step_prior`) — predicts `z_t` from `h_t` alone, no
  observation. Used both as the KL-regularization target during training
  (so the prior learns to approximate what the posterior would have said)
  and, critically, as the *only* way to step the model forward during
  imagination (`imagine`) — there's no real observation to encode once
  you're rolling the model out on its own.

Simplification vs. the paper: a diagonal Gaussian latent (reparameterized,
`z = mean + std * eps`) instead of DreamerV2/V3's categorical one — avoids
needing a Gumbel-softmax/straight-through estimator to keep the categorical
case differentiable, at the cost of DreamerV2's argument that a categorical
latent better matches genuinely multimodal uncertainty. Reasonable for this
app's from-scratch setting; every downstream use (`imagine`, the KL loss)
only assumes "some reparameterizable distribution over `z`", not
specifically which one.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from rl_core.world_models.nets import ObsEncoder, action_dim_of, build_decoder, mlp, recon_loss as _recon_loss

_MIN_STD = 0.1


def _gaussian_kl(mean_q: torch.Tensor, std_q: torch.Tensor, mean_p: torch.Tensor, std_p: torch.Tensor) -> torch.Tensor:
    """`KL(q || p)` between two diagonal Gaussians, summed over the latent
    dimension — the standard closed form, no sampling needed."""
    var_q, var_p = std_q.pow(2), std_p.pow(2)
    kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p).pow(2)) / (2 * var_p) - 0.5
    return kl.sum(dim=-1)


class RSSM(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        deter_dim: int = 200,
        stoch_dim: int = 30,
        hidden_dim: int = 200,
    ) -> None:
        super().__init__()
        self.encoder = ObsEncoder(observation_space)
        self.action_dim = action_dim_of(action_space)
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.feat_dim = deter_dim + stoch_dim

        self.gru = nn.GRUCell(stoch_dim + self.action_dim, deter_dim)
        # Prior never sees the observation — only `h` (already a function
        # of every past observation via the GRU) — that's exactly what lets
        # it be evaluated with *no* real env access during `imagine`.
        self.prior_net = mlp(deter_dim, 2 * stoch_dim, hidden=(hidden_dim,))
        self.posterior_net = mlp(deter_dim + self.encoder.out_dim, 2 * stoch_dim, hidden=(hidden_dim,))
        self.decoder = build_decoder(self.encoder, self.feat_dim, hidden=(hidden_dim, hidden_dim))
        self.reward_head = mlp(self.feat_dim, 1, hidden=(hidden_dim,))
        self.continue_head = mlp(self.feat_dim, 1, hidden=(hidden_dim,))

    def initial_state(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    def _split(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std_raw = raw.chunk(2, dim=-1)
        return mean, F.softplus(std_raw) + _MIN_STD

    def prior(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split(self.prior_net(h))

    def posterior(self, h: torch.Tensor, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split(self.posterior_net(torch.cat([h, embed], dim=-1)))

    def step_prior(
        self, h_prev: torch.Tensor, z_prev: torch.Tensor, action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One *imagined* transition — no observation needed. Returns
        `(h, z, prior_mean, prior_std)`; `z` is a reparameterized sample
        (differentiable w.r.t. everything upstream, including `action`)."""
        h = self.gru(torch.cat([z_prev, action], dim=-1), h_prev)
        mean, std = self.prior(h)
        z = mean + std * torch.randn_like(mean)
        return h, z, mean, std

    def step_posterior(
        self, h_prev: torch.Tensor, z_prev: torch.Tensor, action: torch.Tensor, embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """One *real* transition — `embed` is this step's encoded
        observation. Returns `(h, z, (post_mean, post_std), (prior_mean,
        prior_std))`; the prior returned alongside is only ever used as the
        KL target during training, not to pick `z` itself."""
        h = self.gru(torch.cat([z_prev, action], dim=-1), h_prev)
        post_mean, post_std = self.posterior(h, embed)
        z = post_mean + post_std * torch.randn_like(post_mean)
        prior_mean, prior_std = self.prior(h)
        return h, z, (post_mean, post_std), (prior_mean, prior_std)

    def feat(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], dim=-1)

    def observe(
        self, obs_seq: torch.Tensor, action_seq: torch.Tensor, free_nats: float = 1.0, kl_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Fits the model to one batch of real `(obs, action)` sequences —
        `obs_seq`: `(B, T, *obs_shape)` already in whatever raw form the
        encoder expects (see `rl_core.world_models.nets.ObsEncoder`).
        `action_seq`: `(B, T, action_dim)` — `action_seq[:, t]` is the
        action taken *after* observing `obs_seq[:, t]` (so it's what feeds
        the transition into step `t+1`; the very first step uses an
        implicit all-zero "no action yet").

        Returns a dict with the per-step latents (`h`, `z`, `feat` —
        useful as imagination starting points) and every training loss
        term, already reduced to scalars."""
        batch_size, horizon = obs_seq.shape[0], obs_seq.shape[1]
        device = obs_seq.device
        h, z = self.initial_state(batch_size, device)
        flat_obs = obs_seq.reshape(batch_size * horizon, *obs_seq.shape[2:])
        embeds = self.encoder(flat_obs).reshape(batch_size, horizon, -1)
        prev_action = torch.zeros(batch_size, self.action_dim, device=device)

        hs, zs = [], []
        post_means, post_stds, prior_means, prior_stds = [], [], [], []
        for t in range(horizon):
            h, z, (post_mean, post_std), (prior_mean, prior_std) = self.step_posterior(h, z, prev_action, embeds[:, t])
            hs.append(h)
            zs.append(z)
            post_means.append(post_mean)
            post_stds.append(post_std)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            prev_action = action_seq[:, t]

        h_seq, z_seq = torch.stack(hs, dim=1), torch.stack(zs, dim=1)
        feat_seq = self.feat(h_seq, z_seq)
        flat_feat = feat_seq.reshape(batch_size * horizon, -1)

        recon = self.decoder(flat_feat)
        # Same `[0, 1]`-for-images / raw-scale-for-vectors convention as
        # `rl_core.world_models.nets.obs_target` — reimplemented directly on
        # the already-on-device tensor here (rather than calling that numpy
        # -> tensor helper) to avoid a pointless GPU->CPU->GPU round trip on
        # every training step.
        if self.encoder.is_image:
            target = flat_obs / 255.0
        else:
            target = flat_obs.reshape(batch_size * horizon, -1)
        recon_loss = _recon_loss(recon, target)

        kl_per_step = _gaussian_kl(
            torch.stack(post_means, dim=1), torch.stack(post_stds, dim=1),
            torch.stack(prior_means, dim=1), torch.stack(prior_stds, dim=1),
        )
        # "Free nats" (Hafner et al.) — clamps the KL loss (not the raw KL
        # value used anywhere else) so the model isn't punished for using
        # a small, harmless amount of latent capacity; without this the
        # easiest way to minimize KL is often to collapse the posterior
        # onto the prior everywhere, which also destroys the useful
        # information `z` was supposed to carry.
        kl_loss = kl_per_step.clamp(min=free_nats).mean()

        reward_pred = self.reward_head(flat_feat).squeeze(-1)
        continue_pred = self.continue_head(flat_feat).squeeze(-1)

        return {
            "h": h_seq, "z": z_seq, "feat": feat_seq,
            "recon_loss": recon_loss, "kl_loss": kl_loss,
            "kl_raw": kl_per_step.mean().detach(),
            "reward_pred": reward_pred.reshape(batch_size, horizon),
            "continue_pred": continue_pred.reshape(batch_size, horizon),
            "model_loss": recon_loss + kl_scale * kl_loss,
        }

    def imagine(
        self, h0: torch.Tensor, z0: torch.Tensor, policy_fn: Callable[[torch.Tensor], torch.Tensor], horizon: int,
    ) -> dict[str, torch.Tensor]:
        """Rolls the model forward *without* touching the real env —
        `policy_fn(feat) -> action` is called once per imagined step (a
        differentiable, reparameterized sample if the caller wants
        pathwise gradients through the model into the policy; a plain
        `.detach()`-ed sample if not — this method itself has no opinion).
        Returns per-step `h`/`z`/`feat`/`action`/`reward`/`continue`
        stacked as `(B, horizon, ...)`, *not* including the initial
        `(h0, z0)` itself."""
        h, z = h0, z0
        hs, zs, actions, rewards, continues = [], [], [], [], []
        for _ in range(horizon):
            feat = self.feat(h, z)
            action = policy_fn(feat)
            h, z, _mean, _std = self.step_prior(h, z, action)
            next_feat = self.feat(h, z)
            reward = self.reward_head(next_feat).squeeze(-1)
            continue_prob = torch.sigmoid(self.continue_head(next_feat).squeeze(-1))
            hs.append(h)
            zs.append(z)
            actions.append(action)
            rewards.append(reward)
            continues.append(continue_prob)
        h_seq, z_seq = torch.stack(hs, dim=1), torch.stack(zs, dim=1)
        return {
            "h": h_seq, "z": z_seq, "feat": self.feat(h_seq, z_seq),
            "action": torch.stack(actions, dim=1),
            "reward": torch.stack(rewards, dim=1),
            "continue": torch.stack(continues, dim=1),
        }

    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        """`feat`: `(..., feat_dim)` -> decoded observation, same
        convention as `observe`'s reconstruction (`[0, 1]`-normalized for
        images, raw scale for vectors) — used by the imagined-vs-real
        preview GIF (`rl_core.world_models.viz`) to turn an imagined
        rollout back into something displayable."""
        shape = feat.shape[:-1]
        flat = self.decoder(feat.reshape(-1, feat.shape[-1]))
        return flat.reshape(*shape, *flat.shape[1:])
