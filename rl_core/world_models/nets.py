"""Shared, environment-agnostic encoder/decoder building blocks for every
world model family in this package — mirrors
`rl_core/algorithms/native/networks.py`'s `build_feature_extractor` ("MLP
for vector `Box`, small CNN for image `Box`") on the encoding side, and
adds the decoder half (reconstructing an observation from a latent vector)
that policy-only native networks never needed. Actions reuse the exact
same "just a `gym.Space`" conversion helpers `rl_core/algorithms/native/
preprocessing.py` already has for observations — a one-hot vector for
`Discrete`, the raw vector for `Box` — since nothing about that logic is
actually observation-specific.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from rl_core.algorithms.native.networks import build_feature_extractor
from rl_core.algorithms.native.preprocessing import is_image_space, obs_flat_dim


def mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...] = (200, 200), activation: type[nn.Module] = nn.ELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), activation()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def action_dim_of(action_space: gym.Space) -> int:
    """Same "flat feature count" `obs_flat_dim` already computes for
    observations — a `Discrete(n)` action becomes an n-dim one-hot, a `Box`
    action keeps its own (flattened) dimensionality; nothing here differs
    from encoding an observation of that same space."""
    return obs_flat_dim(action_space)


class ObsEncoder(nn.Module):
    """Thin wrapper around `build_feature_extractor` that also remembers
    what it's encoding — every world model needs both the feature
    extractor itself and (for the decoder / reconstruction-loss target on
    the other side) whether the observation is an image and what its exact
    shape is."""

    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.is_image = is_image_space(observation_space)
        self.net = build_feature_extractor(observation_space)
        self.out_dim = self.net.out_dim
        self.obs_shape = tuple(int(d) for d in observation_space.shape) if self.is_image else (obs_flat_dim(observation_space),)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ImageDecoder(nn.Module):
    """Latent vector -> reconstructed `(H, W, C)` image, values in `[0, 1]`
    (matching `NatureCNN`/`SmallCNN`'s own `/255.0` normalization on the
    encoder side, so the reconstruction loss compares like-for-like scales).

    Deliberately not tied to any particular input resolution: a linear
    projection lands on a small `(C0, H0, W0)` feature map, two
    `ConvTranspose2d` stages double it each time, and a final
    `F.interpolate` snaps whatever that lands on back to the *exact*
    `(H, W)` this env uses — this app's image observations range from 7x7
    egocentric crops up to 96x96 CarRacing frames, and a fixed-stride
    deconv stack sized for one of those would either not fit or leave a
    rounding-error border on most of the others."""

    def __init__(self, in_dim: int, obs_shape: tuple[int, int, int], base_channels: int = 32) -> None:
        super().__init__()
        h, w, c = obs_shape
        self.obs_shape = obs_shape
        self.h0, self.w0 = max(2, h // 4), max(2, w // 4)
        self.c0 = base_channels * 2
        self.project = nn.Linear(in_dim, self.c0 * self.h0 * self.w0)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(self.c0, base_channels, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels, c, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(-1, self.c0, self.h0, self.w0)
        x = self.deconv(x)
        if x.shape[-2] != self.obs_shape[0] or x.shape[-1] != self.obs_shape[1]:
            x = F.interpolate(x, size=(self.obs_shape[0], self.obs_shape[1]), mode="bilinear", align_corners=False)
        return torch.sigmoid(x).permute(0, 2, 3, 1)  # (B, H, W, C)


class VectorDecoder(nn.Module):
    """Latent vector -> reconstructed flat observation vector, raw scale
    (no output nonlinearity) — vector observations in this app span very
    different ranges (`[-1, 1]` continuous controls vs. hundreds for
    trading-env prices), so squashing to a fixed range would need
    per-env normalization this module doesn't have."""

    def __init__(self, in_dim: int, flat_dim: int, hidden: tuple[int, ...] = (200, 200)) -> None:
        super().__init__()
        self.net = mlp(in_dim, flat_dim, hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def build_decoder(encoder: ObsEncoder, in_dim: int, hidden: tuple[int, ...] = (200, 200)) -> nn.Module:
    if encoder.is_image:
        return ImageDecoder(in_dim, encoder.obs_shape)
    return VectorDecoder(in_dim, encoder.obs_shape[0], hidden)


def obs_target(obs_batch: np.ndarray, encoder: ObsEncoder, device: str) -> torch.Tensor:
    """`obs_batch`: `(B, ...)` raw numpy (e.g. from `obs_batch_to_array`) ->
    whatever a decoder built by `build_decoder` should reconstruct
    against — `[0, 1]`-normalized for images (undoing the same `/255.0`
    the encoder itself applies internally), the untouched flat vector
    otherwise."""
    t = torch.as_tensor(np.asarray(obs_batch), dtype=torch.float32, device=device)
    if encoder.is_image:
        return t / 255.0
    return t.reshape(t.shape[0], -1)


def recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error, summed over the observation's own dimensions and
    averaged over the batch — the usual VAE/RSSM convention of treating
    reconstruction as a (unit-variance Gaussian) log-likelihood rather
    than averaging away the effect of a larger observation having
    proportionally more pixels/features to get right."""
    return F.mse_loss(recon, target, reduction="none").flatten(1).sum(-1).mean()
