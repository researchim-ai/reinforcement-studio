"""Factorized Gaussian NoisyNet layers (Fortunato et al., 2018)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Linear layer whose weights are perturbed by learned factorized noise.

    Noise is active only while the module is in training mode and
    ``noise_enabled`` is true. Deterministic evaluation therefore uses the
    learned mean parameters without mutating the model.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma0 = float(sigma0)
        self.noise_enabled = True

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma0 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.sigma0 / math.sqrt(self.out_features))

    @staticmethod
    def _scaled_noise(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        x = torch.randn(size, device=device, dtype=dtype)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scaled_noise(self.in_features, self.weight_mu.device, self.weight_mu.dtype)
        eps_out = self._scaled_noise(self.out_features, self.weight_mu.device, self.weight_mu.dtype)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_enabled:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


def reset_noise(module: nn.Module) -> None:
    """Resample every NoisyLinear layer below ``module``."""
    for child in module.modules():
        if isinstance(child, NoisyLinear):
            child.reset_noise()


def set_noise_enabled(module: nn.Module, enabled: bool) -> None:
    """Enable/disable sampled noise without changing train/eval mode."""
    for child in module.modules():
        if isinstance(child, NoisyLinear):
            child.noise_enabled = bool(enabled)
