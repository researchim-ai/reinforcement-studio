"""Small, dependency-free (no SB3) building blocks for the native
PPO/A2C/DQN implementations: an MLP or Nature-CNN feature extractor picked
from the observation space, plus the actor-critic and Q-network heads built
on top of it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from rl_core.algorithms.native.exploration.noisy import NoisyLinear
from rl_core.algorithms.native.preprocessing import is_image_space, obs_flat_dim


class MLPExtractor(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.net = nn.Sequential(*layers)
        self.out_dim = last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NatureCNN(nn.Module):
    """Mnih et al. (2015) DQN conv stack. Input is (B, H, W, C) uint8/float
    (channel-last, matching Gym's raw frames) — normalized and permuted to
    (B, C, H, W) internally."""

    def __init__(self, obs_shape: tuple[int, int, int]) -> None:
        super().__init__()
        h, w, c = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, 512), nn.ReLU())
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.linear(self.conv(x))


class SmallCNN(nn.Module):
    """Lighter conv stack for small (roughly <50px) image observations —
    e.g. the egocentric crop in `VisualMemoryMazeEnv`. `NatureCNN`'s
    aggressive stride-4/stride-2/stride-1 stack was tuned for 84x84 Atari
    frames; on anything much smaller than that it either errors out
    (stride-3 kernel bigger than what's left of the feature map) or
    collapses to a 1x1 spatial map before the final conv layer, throwing
    away exactly the spatial structure a small egocentric view is mostly
    made of. This stack uses `padding` and gentler strides so the spatial
    size only roughly halves once, keeping meaningfully more of it."""

    def __init__(self, obs_shape: tuple[int, int, int]) -> None:
        super().__init__()
        h, w, c = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU())
        self.out_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.linear(self.conv(x))


# Smallest side length `NatureCNN`'s stride-4/stride-2/stride-1 stack can
# accept without erroring (empirically ~38px) — anything under this
# comfortably-larger threshold routes to `SmallCNN` instead. Atari/CarRacing
# frames (84px+) are always well above it either way.
_NATURE_CNN_MIN_SIDE = 50


def build_feature_extractor(observation_space: gym.Space) -> nn.Module:
    if is_image_space(observation_space):
        shape = tuple(observation_space.shape)
        if min(shape[0], shape[1]) < _NATURE_CNN_MIN_SIDE:
            return SmallCNN(shape)
        return NatureCNN(shape)
    return MLPExtractor(obs_flat_dim(observation_space))


def policy_name(observation_space: gym.Space) -> str:
    """Mirrors SB3's `MlpPolicy`/`CnnPolicy` naming so the Training Monitor
    UI needs no changes to display the native runs' policy type."""
    return "CnnPolicy" if is_image_space(observation_space) else "MlpPolicy"


# --- Recurrent ("memory") support -------------------------------------------
#
# Every algorithm below is single-environment (no vector-env batching), so a
# rollout/episode already *is* one continuous sequence — no time-shuffling
# across parallel actors to worry about, unlike most recurrent-PPO
# writeups. That simplifies things a lot: `RecurrentCore.unroll` just needs
# to walk that one sequence step by step, zeroing the carried hidden state
# right after any timestep where an episode ended (`episode_starts`) so a
# new episode never "sees" the previous one's memory, and callers are free
# to chop the full sequence into fixed-length chunks for truncated BPTT
# (detaching the hidden state between chunks) without this method needing to
# know or care — it always receives an initial `hidden` to seed from and
# returns the final one so chunk-to-chunk state threading is the caller's
# job (see `OnPolicyAlgorithm`/`buffers.py`).
Hidden = torch.Tensor | tuple[torch.Tensor, torch.Tensor]

MEMORY_TYPES = ("lstm", "gru")

# The `memory_type` hyperparam is stored as a plain int (0/1/2) rather than a
# string — every other hyperparam in this app's schema is a number (see
# `HyperparamSpec` on the frontend and `ALGORITHM_CATALOG` on the backend),
# and keeping it that way lets the Designer render it as a labeled dropdown
# (via `HyperparamSpec.options`) without widening `Record<string, number>`
# hyperparam payloads to allow strings everywhere they flow through
# (save/load, plugin editor, ExperimentConfig, ...) for the sake of one field.
MEMORY_TYPE_BY_CODE: dict[int, str | None] = {0: None, 1: "lstm", 2: "gru"}


def memory_type_from_hyperparams(hyperparams: dict) -> str | None:
    """`None` means "no memory — plain feed-forward network", matching
    code `0`. Unknown codes fall back to `None` rather than raising, so a
    stray/garbled value never hard-crashes a run over what's a purely
    architectural preference."""
    code = int(hyperparams.get("memory_type", 0) or 0)
    return MEMORY_TYPE_BY_CODE.get(code)


class RecurrentCore(nn.Module):
    """Shared LSTM/GRU wrapper used by every *Recurrent*Net class below —
    owns the actual `nn.LSTM`/`nn.GRU` module plus the step-by-step unroll
    that resets hidden state on episode boundaries."""

    def __init__(self, in_dim: int, memory_type: str, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unknown memory_type {memory_type!r}, expected one of {MEMORY_TYPES}")
        self.memory_type = memory_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        rnn_cls = nn.LSTM if memory_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(in_dim, hidden_size, num_layers=num_layers, batch_first=True)

    def initial_state(self, batch_size: int, device: torch.device | str) -> Hidden:
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if self.memory_type == "lstm":
            return h, torch.zeros_like(h)
        return h

    @staticmethod
    def _mask_hidden(hidden: Hidden, keep_mask: torch.Tensor) -> Hidden:
        """`keep_mask` is (1, B, 1): 1 to carry a batch row's hidden state
        forward, 0 to zero it out (that row's episode just ended)."""
        if isinstance(hidden, tuple):
            return hidden[0] * keep_mask, hidden[1] * keep_mask
        return hidden * keep_mask

    def unroll(self, feats: torch.Tensor, hidden: Hidden, episode_starts: torch.Tensor | None) -> tuple[torch.Tensor, Hidden]:
        """`feats`: (B, T, feat_dim) already-extracted per-timestep features.
        `episode_starts`: (B, T) — 1.0 at timestep `t` means a *new* episode
        began exactly at `t`, so whatever hidden state was carried in from
        `t-1` (or from the caller for `t=0`) must be zeroed before
        processing it. Returns `(outputs (B, T, hidden_size), final_hidden)`."""
        batch_size, num_steps, _ = feats.shape
        outputs = []
        h = hidden
        for t in range(num_steps):
            if episode_starts is not None:
                keep_mask = (1.0 - episode_starts[:, t]).view(1, batch_size, 1)
                h = self._mask_hidden(h, keep_mask)
            step_out, h = self.rnn(feats[:, t : t + 1, :], h)
            outputs.append(step_out)
        return torch.cat(outputs, dim=1), h


def detach_hidden(hidden: Hidden) -> Hidden:
    """Cuts the autograd graph at a chunk boundary during truncated BPTT —
    the *numeric* hidden state carries on into the next chunk, but no
    gradient flows back through it, bounding backprop to one chunk's
    length regardless of how long the overall rollout/episode is."""
    if isinstance(hidden, tuple):
        return hidden[0].detach(), hidden[1].detach()
    return hidden.detach()


def _flatten_time(features: nn.Module, obs_seq: torch.Tensor) -> torch.Tensor:
    """Applies a (per-timestep) feature extractor to a (B, T, *obs_shape)
    tensor by folding B and T together for one batched forward pass, then
    unfolding back to (B, T, feat_dim) — avoids a Python-level loop over T
    just for the (stateless) feature extraction part."""
    batch_size, num_steps = obs_seq.shape[0], obs_seq.shape[1]
    flat = obs_seq.reshape(batch_size * num_steps, *obs_seq.shape[2:])
    feat = features(flat)
    return feat.reshape(batch_size, num_steps, feat.shape[-1])


class ActorCriticNet(nn.Module):
    """Shared-trunk actor-critic used by both NativePPO and NativeA2C.
    Discrete action spaces get a Categorical head; continuous (`Box`) action
    spaces get a diagonal Gaussian head with a learned, state-independent
    log-std (the standard PPO/A2C setup)."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        super().__init__()
        self.features = build_feature_extractor(observation_space)
        feat_dim = self.features.out_dim
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.discrete:
            self.action_head = nn.Linear(feat_dim, int(action_space.n))
        else:
            act_dim = int(np.prod(action_space.shape))
            self.mu_head = nn.Linear(feat_dim, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value_head = nn.Linear(feat_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (distribution params tensor, value). For discrete this is
        logits; for continuous it's the Gaussian mean."""
        feat = self.features(obs)
        value = self.value_head(feat).squeeze(-1)
        if self.discrete:
            return self.action_head(feat), value
        return self.mu_head(feat), value

    def distribution(self, obs: torch.Tensor):
        params, value = self.forward(obs)
        if self.discrete:
            return torch.distributions.Categorical(logits=params), value
        std = torch.exp(self.log_std).clamp(min=1e-6)
        return torch.distributions.Normal(params, std), value

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        params, _ = self.forward(obs)
        return torch.argmax(params, dim=-1) if self.discrete else params

    @staticmethod
    def log_prob(dist, action: torch.Tensor) -> torch.Tensor:
        lp = dist.log_prob(action)
        return lp if lp.dim() == 1 else lp.sum(dim=-1)

    @staticmethod
    def entropy(dist) -> torch.Tensor:
        ent = dist.entropy()
        return ent if ent.dim() == 1 else ent.sum(dim=-1)


class RecurrentActorCriticNet(nn.Module):
    """Memory-enabled sibling of `ActorCriticNet` — same feature extractor
    and actor/critic heads, but with an LSTM/GRU sitting between them so the
    policy can integrate information across timesteps (useful whenever the
    single observation isn't Markovian on its own, e.g. partial
    observability, needing to remember something a few steps back, ...).
    Used by NativePPO/NativeA2C when `memory_type` isn't `"none"`.

    Every method takes/returns a `(B, T, ...)` sequence rather than a single
    `(B, ...)` step — `T=1` for online action sampling during rollout
    collection, `T=chunk_len` during truncated-BPTT training updates."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        memory_type: str = "lstm",
        hidden_size: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.features = build_feature_extractor(observation_space)
        self.core = RecurrentCore(self.features.out_dim, memory_type, hidden_size, num_layers)
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.discrete:
            self.action_head = nn.Linear(hidden_size, int(action_space.n))
        else:
            act_dim = int(np.prod(action_space.shape))
            self.mu_head = nn.Linear(hidden_size, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size: int, device: torch.device | str) -> Hidden:
        return self.core.initial_state(batch_size, device)

    def forward_sequence(
        self, obs_seq: torch.Tensor, hidden: Hidden, episode_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Hidden]:
        """Returns `(dist_params (B,T,...), value (B,T), final_hidden)`."""
        feats = _flatten_time(self.features, obs_seq)
        out, hidden = self.core.unroll(feats, hidden, episode_starts)
        value = self.value_head(out).squeeze(-1)
        params = self.action_head(out) if self.discrete else self.mu_head(out)
        return params, value, hidden

    def _distribution_from_params(self, params: torch.Tensor):
        if self.discrete:
            return torch.distributions.Categorical(logits=params)
        std = torch.exp(self.log_std).clamp(min=1e-6)
        return torch.distributions.Normal(params, std)

    def distribution_sequence(self, obs_seq: torch.Tensor, hidden: Hidden, episode_starts: torch.Tensor | None = None):
        params, value, hidden = self.forward_sequence(obs_seq, hidden, episode_starts)
        return self._distribution_from_params(params), value, hidden

    # --- single-step convenience wrappers, used while collecting a rollout ---
    # (`obs`: (B, *obs_shape), no explicit time dim — folded to T=1 and back
    # out here so callers see the exact same shapes as the non-recurrent
    # `ActorCriticNet`, and don't need to know whether memory is enabled.)
    def step(self, obs: torch.Tensor, hidden: Hidden) -> tuple[torch.Tensor, torch.Tensor, Hidden]:
        """No `episode_starts` here — the rollout collector already resets
        `hidden` itself whenever it resets the env, rather than going
        through the mask (which is for training on an already-collected,
        possibly multi-episode chunk)."""
        params, value, hidden = self.forward_sequence(obs.unsqueeze(1), hidden, None)
        return params.squeeze(1), value.squeeze(1), hidden

    def distribution_step(self, obs: torch.Tensor, hidden: Hidden):
        params, value, hidden = self.step(obs, hidden)
        return self._distribution_from_params(params), value, hidden

    def deterministic_action_step(self, obs: torch.Tensor, hidden: Hidden) -> tuple[torch.Tensor, Hidden]:
        params, _, hidden = self.step(obs, hidden)
        return (torch.argmax(params, dim=-1) if self.discrete else params), hidden

    # Instance (not static) methods, unlike `ActorCriticNet`'s — whether a
    # sum-over-action-dim is needed depends on discrete-vs-continuous, *not*
    # on how many leading (batch/time) dims `dist` happens to carry, and
    # this class's distributions can have either a plain (B,) batch shape
    # (single-step) or a (B,T) one (sequence/BPTT training).
    def log_prob(self, dist, action: torch.Tensor) -> torch.Tensor:
        lp = dist.log_prob(action)
        return lp if self.discrete else lp.sum(dim=-1)

    def entropy(self, dist) -> torch.Tensor:
        ent = dist.entropy()
        return ent if self.discrete else ent.sum(dim=-1)


def _q_linear(in_features: int, out_features: int, noisy: bool, sigma0: float) -> nn.Module:
    return NoisyLinear(in_features, out_features, sigma0=sigma0) if noisy else nn.Linear(in_features, out_features)


class QNetwork(nn.Module):
    """Plain DQN Q-network: feature extractor + a linear head over actions."""

    def __init__(self, observation_space: gym.Space, n_actions: int, noisy: bool = False, noisy_sigma0: float = 0.5) -> None:
        super().__init__()
        self.features = build_feature_extractor(observation_space)
        self.q_head = _q_linear(self.features.out_dim, n_actions, noisy, noisy_sigma0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.features(obs))


class RecurrentQNetwork(nn.Module):
    """Memory-enabled DQN Q-network (a.k.a. DRQN, Hausknecht & Stone 2015) —
    feature extractor -> LSTM/GRU -> linear Q-head. Used by NativeDQN when
    `memory_type` isn't `"none"`; trained on fixed-length episode windows
    from `EpisodeSequenceReplayBuffer` (see buffers.py) with a zero initial
    hidden state per sampled window — the standard, simpler alternative to
    R2D2's "burn-in" warm-start, which this app's small/fast environments
    don't need."""

    def __init__(
        self, observation_space: gym.Space, n_actions: int, memory_type: str = "lstm",
        hidden_size: int = 128, num_layers: int = 1, noisy: bool = False,
        noisy_sigma0: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = build_feature_extractor(observation_space)
        self.core = RecurrentCore(self.features.out_dim, memory_type, hidden_size, num_layers)
        self.q_head = _q_linear(hidden_size, n_actions, noisy, noisy_sigma0)

    def initial_state(self, batch_size: int, device: torch.device | str) -> Hidden:
        return self.core.initial_state(batch_size, device)

    def forward_sequence(self, obs_seq: torch.Tensor, hidden: Hidden, episode_starts: torch.Tensor | None = None) -> tuple[torch.Tensor, Hidden]:
        """`obs_seq`: (B, T, *obs_shape) -> `(q_values (B,T,n_actions), final_hidden)`."""
        feats = _flatten_time(self.features, obs_seq)
        out, hidden = self.core.unroll(feats, hidden, episode_starts)
        return self.q_head(out), hidden

    def step(self, obs: torch.Tensor, hidden: Hidden) -> tuple[torch.Tensor, Hidden]:
        """`obs`: (B, *obs_shape) single timestep -> `(q_values (B,n_actions), final_hidden)`."""
        q, hidden = self.forward_sequence(obs.unsqueeze(1), hidden, None)
        return q.squeeze(1), hidden


class DuelingQNetwork(nn.Module):
    """Dueling architecture (Wang et al., 2016) used by Rainbow DQN — splits
    the head into a scalar state-value stream `V(s)` and a per-action
    advantage stream `A(s,a)`, recombined as `Q = V + (A - mean(A))` so the
    network can learn "how good is this state" independently of "which
    action is best here", which speeds up learning a lot on envs where most
    actions barely matter in most states."""

    def __init__(
        self, observation_space: gym.Space, n_actions: int, hidden: int = 128,
        noisy: bool = False, noisy_sigma0: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = build_feature_extractor(observation_space)
        feat_dim = self.features.out_dim
        self.value_stream = nn.Sequential(
            _q_linear(feat_dim, hidden, noisy, noisy_sigma0), nn.ReLU(),
            _q_linear(hidden, 1, noisy, noisy_sigma0),
        )
        self.advantage_stream = nn.Sequential(
            _q_linear(feat_dim, hidden, noisy, noisy_sigma0), nn.ReLU(),
            _q_linear(hidden, n_actions, noisy, noisy_sigma0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.features(obs)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


class RecurrentDuelingQNetwork(nn.Module):
    """Memory-enabled Rainbow DQN network — `RecurrentQNetwork`'s LSTM/GRU
    core feeding the same dueling value/advantage streams as
    `DuelingQNetwork`. Double Q-learning and the dueling head carry over
    unchanged from plain Rainbow DQN; n-step returns and prioritized replay
    don't (see `EpisodeSequenceReplayBuffer` in buffers.py for why) — a
    documented, deliberate scope cut rather than an oversight, since
    correctly combining PER + n-step + sequence sampling (à la R2D2) is a
    meaningfully bigger undertaking than the rest of this feature."""

    def __init__(
        self, observation_space: gym.Space, n_actions: int, memory_type: str = "lstm",
        hidden_size: int = 128, num_layers: int = 1, noisy: bool = False,
        noisy_sigma0: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = build_feature_extractor(observation_space)
        self.core = RecurrentCore(self.features.out_dim, memory_type, hidden_size, num_layers)
        self.value_stream = nn.Sequential(
            _q_linear(hidden_size, hidden_size, noisy, noisy_sigma0), nn.ReLU(),
            _q_linear(hidden_size, 1, noisy, noisy_sigma0),
        )
        self.advantage_stream = nn.Sequential(
            _q_linear(hidden_size, hidden_size, noisy, noisy_sigma0), nn.ReLU(),
            _q_linear(hidden_size, n_actions, noisy, noisy_sigma0),
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> Hidden:
        return self.core.initial_state(batch_size, device)

    def forward_sequence(self, obs_seq: torch.Tensor, hidden: Hidden, episode_starts: torch.Tensor | None = None) -> tuple[torch.Tensor, Hidden]:
        feats = _flatten_time(self.features, obs_seq)
        out, hidden = self.core.unroll(feats, hidden, episode_starts)
        value = self.value_stream(out)
        advantage = self.advantage_stream(out)
        q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q, hidden

    def step(self, obs: torch.Tensor, hidden: Hidden) -> tuple[torch.Tensor, Hidden]:
        q, hidden = self.forward_sequence(obs.unsqueeze(1), hidden, None)
        return q.squeeze(1), hidden


_LOG_STD_MIN, _LOG_STD_MAX = -20.0, 2.0


class GaussianPolicy(nn.Module):
    """Squashed-Gaussian policy used by SAC — samples a raw Normal(mean,
    std), squashes it through `tanh` (bounding it to [-1, 1]) and then
    rescales to the env's actual action bounds. The `tanh` correction term
    in `log_prob` (Haarnoja et al., 2018, appendix C) keeps the returned
    log-probability exact under that change of variables, which matters
    for SAC's entropy term to mean what it's supposed to."""

    def __init__(self, observation_space: gym.Space, action_low: np.ndarray, action_high: np.ndarray, hidden: int = 256) -> None:
        super().__init__()
        self.features = build_feature_extractor(observation_space)
        feat_dim = self.features.out_dim
        action_dim = len(action_low)
        self.trunk = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)
        low = np.where(np.isfinite(action_low), action_low, -1.0)
        high = np.where(np.isfinite(action_high), action_high, 1.0)
        self.register_buffer("action_scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))

    def _mean_log_std(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.features(obs))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns `(action, log_prob, deterministic_action)` — the last one
        (tanh of the mean, no sampling noise) is what `predict(deterministic=True)`
        should use at eval/inference time."""
        mean, log_std = self._mean_log_std(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        squashed = torch.tanh(raw_action)
        action = squashed * self.action_scale + self.action_bias

        log_prob = normal.log_prob(raw_action) - torch.log(self.action_scale * (1 - squashed.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_action


class QCritic(nn.Module):
    """State-action value network for SAC — concatenates the observation
    features with the (continuous) action vector before a small MLP head.
    SAC always uses two of these (`QCritic` instances, each with its own
    feature extractor) to counter Q-value overestimation, the same trick as
    Double DQN but structural instead of using a second forward pass."""

    def __init__(self, observation_space: gym.Space, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.features = build_feature_extractor(observation_space)
        self.head = nn.Sequential(
            nn.Linear(self.features.out_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        feat = self.features(obs)
        return self.head(torch.cat([feat, action], dim=-1)).squeeze(-1)
