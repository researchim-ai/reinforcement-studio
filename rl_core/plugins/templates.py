"""Starter code shown when creating a new plugin script on the Plugins page."""
from __future__ import annotations

GYM_FROM_SCRATCH = '''"""Custom Gym algorithm — from-scratch REINFORCE (no SB3 dependency).

Implements rl_core.algorithms.base.CustomAlgorithm directly: you own the
whole training loop. Good starting point if you want full control (a
different network, a different update rule, ...).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback

NAME = "REINFORCE (custom)"
DESCRIPTION = "Простой policy-gradient алгоритм с нуля, на чистом PyTorch."
SUPPORTED_ACTION_KINDS = ["discrete"]
HYPERPARAMS = [
    {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
    {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
]


class PolicyNet(nn.Module):
    def __init__(self, obs_size: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, 64), nn.ReLU(), nn.Linear(64, num_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Algorithm(CustomAlgorithm):
    def __init__(self, env, hyperparams, seed, device) -> None:
        super().__init__(env, hyperparams, seed, device)
        obs_size = int(np.prod(env.observation_space.shape))
        num_actions = env.action_space.n
        self.policy = PolicyNet(obs_size, num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=hyperparams.get("learning_rate", 1e-3))
        self.gamma = hyperparams.get("gamma", 0.99)

    def predict(self, obs, deterministic: bool = True):
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
        logits = self.policy(x)
        probs = F.softmax(logits, dim=-1)
        action = int(torch.argmax(probs, dim=-1).item()) if deterministic else int(
            torch.multinomial(probs, 1).item()
        )
        return action, None

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        num_timesteps = 0
        while num_timesteps < total_timesteps:
            obs, _ = self.env.reset(seed=self.seed)
            log_probs, rewards = [], []
            done = False
            while not done and num_timesteps < total_timesteps:
                x = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
                logits = self.policy(x)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))

                obs, reward, terminated, truncated, _ = self.env.step(int(action.item()))
                rewards.append(reward)
                done = terminated or truncated
                num_timesteps += 1

                if not callback.on_step(num_timesteps):
                    return

            returns, g = [], 0.0
            for r in reversed(rewards):
                g = r + self.gamma * g
                returns.insert(0, g)
            returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            loss = -torch.stack(log_probs).squeeze(-1) @ returns_t
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            callback.on_step(num_timesteps, episode_reward=float(sum(rewards)), episode_length=float(len(rewards)))

    def save(self, path) -> None:
        torch.save(self.policy.state_dict(), path)

    @classmethod
    def load(cls, path, env):
        obj = cls(env, {}, None, "cpu")
        obj.policy.load_state_dict(torch.load(path, map_location="cpu"))
        return obj


ALGORITHM_CLASS = Algorithm
'''

GYM_SB3_SUBCLASS = '''"""Custom Gym algorithm — subclass of an existing Stable-Baselines3 algorithm.

Reuses the whole SB3 + MetricsCallback pipeline unchanged (network, rollout
collection, live-frame rendering, ...) — override only what you want to
change, e.g. the policy network or a custom `train()` step.
"""
from __future__ import annotations

from stable_baselines3 import PPO

NAME = "PPO (кастомная сеть)"
DESCRIPTION = "PPO с переопределённой политикой — отправная точка для своих модификаций."
SUPPORTED_ACTION_KINDS = ["discrete", "continuous"]
HYPERPARAMS = [
    {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 3e-4, "min": 1e-6, "max": 1e-1},
    {"key": "n_steps", "label": "Steps per update", "type": "int", "default": 2048, "min": 32, "max": 8192},
    {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
]


class Algorithm(PPO):
    """Override any PPO method here, e.g. `train()`, to change the update rule,
    or pass a custom `policy_kwargs` in __init__ to change the network shape."""
    pass


ALGORITHM_CLASS = Algorithm
'''

ALPHAZERO_CUSTOM_NETWORK = '''"""Custom AlphaZero trainer — swap only the network architecture.

Keeps the built-in self-play / MCTS / arena-gating loop (from
BuiltinAlphaZeroTrainer) and overrides just build_network().
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from rl_core.alphazero.base import BuiltinAlphaZeroTrainer

NAME = "AlphaZero (своя сеть)"
DESCRIPTION = "Тот же self-play/MCTS/arena, но с другой архитектурой сети."
HYPERPARAMS = [
    {"key": "num_simulations", "label": "MCTS simulations/move", "type": "int", "default": 25, "min": 4, "max": 400},
    {"key": "games_per_iteration", "label": "Self-play games/iteration", "type": "int", "default": 20, "min": 2, "max": 200},
    {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
    {"key": "channels", "label": "Conv channels", "type": "int", "default": 32, "min": 8, "max": 256},
]


class CustomNet(nn.Module):
    """Must expose .rows/.cols/.action_size and predict(encoded_state, device)."""

    def __init__(self, rows: int, cols: int, action_size: int, channels: int = 32, num_blocks: int = 2) -> None:
        super().__init__()
        self.rows, self.cols, self.action_size = rows, cols, action_size
        self.conv = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU())
        self.policy_head = nn.Linear(channels * rows * cols, action_size)
        self.value_head = nn.Linear(channels * rows * cols, 1)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.policy_head(x), F.tanh(self.value_head(x)).squeeze(-1)

    def predict(self, encoded_state, device: str = "cpu"):
        import torch

        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(encoded_state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = self.forward(x)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            return probs, float(value.item())


class Trainer(BuiltinAlphaZeroTrainer):
    def build_network(self):
        g = self.sample_game
        return CustomNet(g.rows, g.cols, g.action_size, self.hp.get("channels", 32))


ALGORITHM_CLASS = Trainer
'''

ALPHAZERO_FULL_LOOP = '''"""Custom AlphaZero trainer — full replacement of the training loop.

Overrides run_iteration() entirely instead of just the network. You can
still reuse the building blocks (self_play.play_self_play_game,
arena.play_match, network.AlphaZeroNet) — this template shows the minimum
needed to satisfy the contract.
"""
from __future__ import annotations

import copy

from rl_core.alphazero.arena import play_match
from rl_core.alphazero.base import AlphaZeroTrainer
from rl_core.alphazero.self_play import play_self_play_game

NAME = "AlphaZero (свой цикл)"
DESCRIPTION = "Полностью свой self-play/тренировочный цикл."
HYPERPARAMS = [
    {"key": "num_simulations", "label": "MCTS simulations/move", "type": "int", "default": 25, "min": 4, "max": 400},
    {"key": "games_per_iteration", "label": "Self-play games/iteration", "type": "int", "default": 10, "min": 2, "max": 200},
]


class Trainer(AlphaZeroTrainer):
    def __init__(self, game_cls, hyperparams, device) -> None:
        super().__init__(game_cls, hyperparams, device)
        self.best_state = copy.deepcopy(self.net.state_dict())

    def run_iteration(self, iteration: int) -> dict:
        self.net.eval()
        records = []
        for g in range(self.hp["games_per_iteration"]):
            _examples, record = play_self_play_game(
                self.game_cls, self.net, num_simulations=self.hp["num_simulations"], device=self.device,
            )
            if g < 3:
                records.append(record)

        # ... train self.net on the collected examples here ...

        match = play_match(self.game_cls, self.net, self.net, num_games=2, device=self.device)
        return {
            "arena": match,
            "board": records[-1]["final_board"] if records else None,
            "_self_play_records": records,
        }


ALGORITHM_CLASS = Trainer
'''

REWARD_FUNCTION = '''"""Custom reward-shaping function (Gym track).

Called on every env step; return the (possibly reshaped) reward. Keep it
cheap — it runs on the hot path of every training step.
"""
from __future__ import annotations

NAME = "Пример: штраф за большие действия"
DESCRIPTION = "Немного штрафует резкие/большие действия, поощряя более плавную политику."
HYPERPARAMS = []


def shape_reward(obs, action, reward, next_obs, terminated, truncated, info):
    penalty = 0.0
    try:
        import numpy as np

        penalty = 0.01 * float(np.linalg.norm(np.asarray(action, dtype=float)))
    except Exception:
        pass
    return reward - penalty


REWARD_FN = shape_reward
'''


TEMPLATES = [
    {"id": "gym-from-scratch", "label": "Gym: с нуля (pure PyTorch REINFORCE)", "kind": "gym-algorithm", "code": GYM_FROM_SCRATCH},
    {"id": "gym-sb3-subclass", "label": "Gym: наследник SB3 PPO", "kind": "gym-algorithm", "code": GYM_SB3_SUBCLASS},
    {"id": "alphazero-custom-network", "label": "AlphaZero: своя сеть", "kind": "alphazero-algorithm", "code": ALPHAZERO_CUSTOM_NETWORK},
    {"id": "alphazero-full-loop", "label": "AlphaZero: полный цикл", "kind": "alphazero-algorithm", "code": ALPHAZERO_FULL_LOOP},
    {"id": "reward-function", "label": "Reward-функция", "kind": "reward-function", "code": REWARD_FUNCTION},
]
