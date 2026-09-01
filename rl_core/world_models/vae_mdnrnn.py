"""The original "World Models" (Ha & Schmidhuber, 2018 —
https://arxiv.org/abs/1803.10122): a VAE compresses individual observations
to a small latent `z`, and an MDN-RNN (a recurrent net whose output at each
step is the *parameters of a Gaussian mixture*, not a single point predicts
the *distribution* of `z_{t+1}` conditioned on `(z_t, a_t)` — a genuinely
multimodal next-state predictor is a big part of why this paper's approach
generalizes better than a plain deterministic RNN would, e.g. on
CarRacing's mostly-symmetric-but-occasionally-branching track layouts, both
directions after a fork are simultaneously plausible until the agent
actually commits to one.

`rl_core/algorithms/native/world_models_ha.py` (the algorithm using this)
trains the VAE + MDN-RNN first on random-policy rollouts (this module has
no opinion on where the data came from), then freezes both and evolves a
tiny linear controller acting on `[z, h]` directly against real episode
return (Evolution Strategies, matching this app's existing `native/es.py`)
— the paper's own two-phase recipe, simplified to skip its optional third
phase (training the controller *inside* a fully hallucinated M-model
"dream" env instead of the real one), which is a bigger undertaking on top
of an already large feature and not needed to get the core "learn a
compact world model, then act on it" idea working end-to-end.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from rl_core.world_models.nets import ObsEncoder, build_decoder, mlp, recon_loss as _recon_loss

_LOG_STD_MIN, _LOG_STD_MAX = -5.0, 2.0
_LOGVAR_MIN, _LOGVAR_MAX = -10.0, 2.0


class VAE(nn.Module):
    """Encodes one observation at a time (no sequence/recurrence here at
    all — that's entirely the MDN-RNN's job) to a `latent_dim`-size `z`,
    reparameterized (`z = mean + std * eps`) so `loss()` is end-to-end
    differentiable."""

    def __init__(self, observation_space: gym.Space, latent_dim: int = 32, hidden_dim: int = 200) -> None:
        super().__init__()
        self.encoder = ObsEncoder(observation_space)
        self.latent_dim = latent_dim
        self.stats_net = mlp(self.encoder.out_dim, 2 * latent_dim, hidden=(hidden_dim,))
        self.decoder = build_decoder(self.encoder, latent_dim, hidden=(hidden_dim, hidden_dim))

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.stats_net(self.encoder(obs)).chunk(2, dim=-1)
        return mean, logvar.clamp(_LOGVAR_MIN, _LOGVAR_MAX)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + (0.5 * logvar).exp() * torch.randn_like(mean)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def loss(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, logvar = self.encode(obs)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        target = obs / 255.0 if self.encoder.is_image else obs.reshape(obs.shape[0], -1)
        recon_loss = _recon_loss(recon, target)
        kl_loss = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return {"z": z, "mean": mean, "recon_loss": recon_loss, "kl_loss": kl_loss}


class MDNRNN(nn.Module):
    """`(z_t, a_t)` sequence -> per-step Gaussian-mixture parameters for
    `z_{t+1}`, plus (unimodal, unlike `z`) reward and episode-continuation
    predictions — `num_mixtures` components, each with its own mean/log-std
    over the *full* `latent_dim`-size `z` (a diagonal covariance within
    each component, exactly like the paper's own MDN-RNN)."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_size: int = 256, num_mixtures: int = 5) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_mixtures = num_mixtures
        self.lstm = nn.LSTM(latent_dim + action_dim, hidden_size, batch_first=True)
        # mixture logits (K) + per-component (mean, log_std) over D dims
        # (2*K*D) + reward + continue-logit.
        out_dim = num_mixtures * (1 + 2 * latent_dim) + 2
        self.head = nn.Linear(hidden_size, out_dim)

    def initial_state(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h, c

    def forward(
        self, z_seq: torch.Tensor, action_seq: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """`z_seq`/`action_seq`: `(B, T, ...)`. Returns `(logits (B,T,K),
        means (B,T,K,D), log_stds (B,T,K,D), reward (B,T), continue_logit
        (B,T), final_hidden)` — every one of these describes the predicted
        distribution of `z_{t+1}`/`r_t`/`continue_t` given everything up to
        and including step `t`."""
        out, hidden = self.lstm(torch.cat([z_seq, action_seq], dim=-1), hidden)
        raw = self.head(out)
        k, d = self.num_mixtures, self.latent_dim
        logits = raw[..., :k]
        rest = raw[..., k : k + 2 * k * d].reshape(*raw.shape[:-1], k, 2, d)
        means = rest[..., 0, :]
        log_stds = rest[..., 1, :].clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        reward = raw[..., k + 2 * k * d]
        continue_logit = raw[..., k + 2 * k * d + 1]
        return logits, means, log_stds, reward, continue_logit, hidden

    @staticmethod
    def mixture_nll(logits: torch.Tensor, means: torch.Tensor, log_stds: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of `target_z` (`(B, T, D)`, the *actual*
        next-step latent from the VAE) under the predicted mixture — the
        MDN-RNN's own training loss."""
        target = target_z.unsqueeze(-2)  # (B, T, 1, D) — broadcasts against the K component axis
        log_component = -0.5 * (((target - means) / log_stds.exp()) ** 2 + 2 * log_stds + math.log(2 * math.pi))
        log_component = log_component.sum(dim=-1)  # sum over D -> (B, T, K)
        log_mix_weight = F.log_softmax(logits, dim=-1)
        log_prob = torch.logsumexp(log_mix_weight + log_component, dim=-1)  # (B, T)
        return -log_prob.mean()

    @staticmethod
    def sample_next_z(logits: torch.Tensor, means: torch.Tensor, log_stds: torch.Tensor) -> torch.Tensor:
        """Samples one `z_{t+1}` per batch row from the predicted mixture —
        `logits`/`means`/`log_stds`: `(B, K)`/`(B, K, D)`/`(B, K, D)` (a
        single timestep, unlike `mixture_nll`'s full-sequence shapes; used
        during imagined-rollout preview generation, one step at a time)."""
        component = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)  # (B,)
        idx = component.view(-1, 1, 1).expand(-1, 1, means.shape[-1])
        mean = means.gather(1, idx).squeeze(1)
        log_std = log_stds.gather(1, idx).squeeze(1)
        return mean + log_std.exp() * torch.randn_like(mean)
