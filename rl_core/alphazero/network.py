"""Small AlphaZero-style dual-head network: policy logits + scalar value.

Sized for CPU training on small boards (Tic-Tac-Toe/Connect Four/Gomoku) —
a handful of conv blocks is plenty; this is meant to be understandable and
fast to iterate on, not a state-of-the-art architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        rows: int,
        cols: int,
        action_size: int,
        channels: int = 48,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.action_size = action_size

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * rows * cols, action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * rows * cols, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.blocks(x)
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x)).squeeze(-1)
        return policy_logits, value

    @torch.no_grad()
    def predict(self, encoded_state, device: str = "cpu") -> tuple["torch.Tensor", float]:
        """Single-state inference used by MCTS. Returns (policy_probs, value)."""
        self.eval()
        x = torch.as_tensor(encoded_state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, float(value.item())
