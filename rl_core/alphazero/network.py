"""AlphaZero-style dual-head ResNet: policy logits + value.

Two value heads are supported:

* ``tanh`` — the classic single scalar in [-1, 1] trained with MSE (older
  checkpoints and custom plugin networks use this);
* ``wdl`` — three logits (win / draw / loss for the side to move) trained
  with cross-entropy. `forward()` still returns a scalar value
  ``p_win − p_loss`` so MCTS / Arena code does not care which head is used.

``global_pool`` swaps the flatten→Linear value head for a mean+max pooled
one: the value of a position is a global property (material, king safety),
and pooling generalises much better than a per-square linear layer on 8×8+
boards.
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


class _PooledValueHead(nn.Module):
    def __init__(self, channels: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 32, 1)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(64, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.fc2(F.relu(self.fc1(pooled)))


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        rows: int,
        cols: int,
        action_size: int,
        channels: int = 48,
        num_blocks: int = 3,
        in_planes: int = 3,
        value_head: str = "tanh",
        global_pool: bool = False,
        policy_channels: int = 8,
    ) -> None:
        super().__init__()
        if value_head not in ("tanh", "wdl"):
            raise ValueError(f"Unknown value_head: {value_head}")
        self.rows = rows
        self.cols = cols
        self.action_size = action_size
        self.in_planes = in_planes
        self.value_head_type = value_head

        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, policy_channels, 1),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_channels * rows * cols, action_size),
        )
        out_dim = 3 if value_head == "wdl" else 1
        if global_pool:
            self.value_head = _PooledValueHead(channels, out_dim)
        else:
            self.value_head = nn.Sequential(
                nn.Conv2d(channels, 4, 1),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(4 * rows * cols, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),
            )

    def forward_heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw heads: (policy logits, value logits). Value logits are (B, 3)
        win/draw/loss for ``wdl``, or a pre-tanh (B,) scalar for ``tanh``."""
        x = self.blocks(self.stem(x))
        policy_logits = self.policy_head(x)
        value_raw = self.value_head(x)
        if self.value_head_type == "tanh":
            value_raw = value_raw.squeeze(-1)
        return policy_logits, value_raw

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits, value_raw = self.forward_heads(x)
        if self.value_head_type == "wdl":
            probs = F.softmax(value_raw.float(), dim=-1)
            value = probs[:, 0] - probs[:, 2]
        else:
            value = torch.tanh(value_raw)
        return policy_logits, value

    @torch.no_grad()
    def predict(self, encoded_state, device: str = "cpu") -> tuple["torch.Tensor", float]:
        """Single-state inference used by MCTS. Returns (policy_probs, value)."""
        self.eval()
        x = torch.as_tensor(encoded_state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, float(value.item())
