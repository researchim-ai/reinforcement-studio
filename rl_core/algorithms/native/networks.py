"""Small, dependency-free (no SB3) building blocks for the native
PPO/A2C/DQN implementations: an MLP or Nature-CNN feature extractor picked
from the observation space, plus the actor-critic and Q-network heads built
on top of it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def mask_hidden_for_dones(hidden: Hidden, dones) -> Hidden:
    """Zeroes out exactly the batch rows (lanes) whose episode just ended —
    used between vectorized collection steps (`num_envs>1`) so a lane that
    reset doesn't carry another lane's still-running episode's memory into
    its next hidden state, while every other lane's hidden state carries
    over untouched. `dones`: any array-like of length `batch_size` (e.g. a
    numpy bool array from `vec_env.vec_step`)."""
    ref = hidden[0] if isinstance(hidden, tuple) else hidden
    keep_mask = torch.as_tensor(
        [0.0 if d else 1.0 for d in dones], dtype=ref.dtype, device=ref.device,
    ).view(1, -1, 1)
    return RecurrentCore._mask_hidden(hidden, keep_mask)


def _flatten_time(features: nn.Module, obs_seq: torch.Tensor) -> torch.Tensor:
    """Applies a (per-timestep) feature extractor to a (B, T, *obs_shape)
    tensor by folding B and T together for one batched forward pass, then
    unfolding back to (B, T, feat_dim) — avoids a Python-level loop over T
    just for the (stateless) feature extraction part."""
    batch_size, num_steps = obs_seq.shape[0], obs_seq.shape[1]
    flat = obs_seq.reshape(batch_size * num_steps, *obs_seq.shape[2:])
    feat = features(flat)
    return feat.reshape(batch_size, num_steps, feat.shape[-1])


class _StateDependentGaussian:
    """gSDE ("generalized State-Dependent Exploration", Raffin et al. 2021 —
    https://arxiv.org/abs/2005.05719) action distribution: same interface
    (`sample`/`log_prob`/`entropy`) as `torch.distributions.Normal` so
    `ActorCriticNet.log_prob`/`.entropy` and the collection/update loops in
    `on_policy.py`/`ppo.py` don't need to know which one they got.

    Ordinary PPO/A2C resample independent per-step, per-action-dim Gaussian
    noise — fine for e.g. discrete-ish "nudge left/right" control, but for a
    continuous steering wheel it means the sampled action can flip sign
    frame to frame, so the *executed* trajectory looks jittery even once the
    mean action is good. gSDE instead treats the noise as a linear function
    of the (detached) feature vector, `noise = features @ exploration_mat`,
    and only resamples `exploration_mat` every few steps (see
    `ActorCriticNet.reset_noise`/`sde_sample_freq`) — so the noise added
    stays *correlated* across consecutive steps instead of independent,
    giving temporally smooth exploration.

    The distribution used for `log_prob`/`entropy` (needed for the PPO
    ratio/entropy bonus) is the *marginal* Gaussian obtained by integrating
    out the random `exploration_mat` — `Normal(mean, sqrt(features**2 @
    std**2))` — which only depends on the trainable `log_std` parameter, not
    on which exact `exploration_mat` was sampled during collection. That's
    exactly what lets `_update()` recompute a valid log-prob for PPO's ratio
    without needing to know/replay the specific noise matrix a given
    transition was originally collected under."""

    def __init__(
        self, mean: torch.Tensor, features: torch.Tensor, log_std: torch.Tensor,
        exploration_mat: torch.Tensor, exploration_mats: torch.Tensor,
    ) -> None:
        self.mean = mean
        self._features = features.detach()
        std = torch.exp(log_std)
        variance = (self._features ** 2) @ (std ** 2) + 1e-6
        self._normal = torch.distributions.Normal(mean, variance.sqrt())
        self._exploration_mat = exploration_mat
        self._exploration_mats = exploration_mats

    def _noise(self) -> torch.Tensor:
        feat = self._features
        # One shared matrix for a lone/mismatched batch (e.g. `predict()`
        # sampling a single live env with a batch of exploration matrices
        # sized for `num_envs` parallel training lanes) — a per-lane matrix
        # via batched matmul otherwise, so each vectorized env explores
        # along its own (temporarily fixed) noise direction.
        if feat.shape[0] == 1 or feat.shape[0] != self._exploration_mats.shape[0]:
            return feat @ self._exploration_mat
        return torch.bmm(feat.unsqueeze(1), self._exploration_mats).squeeze(1)

    def sample(self) -> torch.Tensor:
        return self.mean + self._noise()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self._normal.log_prob(action).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self._normal.entropy().sum(dim=-1)


class _BetaPolicyDistribution:
    """Beta-distribution continuous-action policy (Chou et al. 2017 —
    https://proceedings.mlr.press/v70/chou17a/chou17a.pdf; Petrazzini &
    Antonelo 2021 on CarRacing specifically —
    https://arxiv.org/abs/2111.02202, +63% success rate vs. Gaussian) — same
    `sample`/`log_prob`/`entropy` interface as `torch.distributions.Normal`
    so `ActorCriticNet.log_prob`/`.entropy` don't need to know which one
    they got.

    The usual PPO/A2C Gaussian head has *unbounded* support, so any action
    space with hard bounds (steering [-1,1], but especially one-sided gas/
    brake [0,1]) needs every sampled action hard-clipped back into range
    before it reaches `env.step()`. That clip is a biased, non-smooth
    operation the policy gradient never sees: the network can keep pushing
    the raw mean further outside the box (e.g. "gas = 1.4") without the
    loss ever reflecting that 1.4 and 1.0 execute identically, which both
    papers above identify as a real source of slower/worse-converged
    continuous control. A Beta distribution has support exactly on `[0,1]`
    by construction, so it's affinely rescaled to the action space's
    `[low, high]` box and *never* needs clipping — every raw sample is
    already a valid action.

    `alpha, beta > 1` (enforced by the softplus+1 in `ActorCriticNet`
    below, matching Chou et al.'s parameterization) keeps the density
    unimodal instead of degenerating into a bathtub/U-shape that piles
    mass at the two ends of the range — the failure mode you'd get from
    the same distribution family with `alpha` or `beta` allowed below 1."""

    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> None:
        self.alpha = alpha
        self.beta = beta
        self._dist = torch.distributions.Beta(alpha, beta)
        self._low = low
        self._high = high
        self._scale = (high - low).clamp(min=1e-6)

    @property
    def mean(self) -> torch.Tensor:
        return self._low + self._scale * self._dist.mean

    def sample(self) -> torch.Tensor:
        x = self._dist.rsample()
        return self._low + self._scale * x

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        # Constant `-log(scale)` per dim (the affine transform's Jacobian)
        # is omitted deliberately: it's identical for old and new policy at
        # a fixed collected action, so it cancels exactly in PPO's
        # `exp(new_log_prob - old_log_prob)` ratio and would only ever add
        # noise, never signal, to the loss.
        x = ((action - self._low) / self._scale).clamp(1e-6, 1.0 - 1e-6)
        return self._dist.log_prob(x).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy().sum(dim=-1)


class ActorCriticNet(nn.Module):
    """Shared-trunk actor-critic used by both NativePPO and NativeA2C.
    Discrete action spaces get a Categorical head; continuous (`Box`) action
    spaces get a diagonal Gaussian head with a learned log-std — either the
    standard PPO/A2C state-*independent* one, or (`use_sde=True`) gSDE's
    state-*dependent* one, see `_StateDependentGaussian` above.

    `head_hidden_size` optionally inserts one `Linear + GELU` layer between
    the feature extractor and each head (separate weights for policy vs.
    value, like SB3's `net_arch=dict(pi=[...], vf=[...])`) instead of
    attaching `mu_head`/`value_head` directly to the raw feature vector.
    Off (`0`) by default — matches the previous, simpler architecture for
    every env that isn't explicitly opting into more capacity — but pixel
    envs with a lot going on in one frame (CarRacing) can plateau well
    below a solved score with heads that are just one bare `Linear` layer
    on 512 CNN features; a per-head hidden layer gives the policy and value
    function each their own room to specialize instead of both being a
    single linear readout off shared features."""

    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space,
        use_sde: bool = False, sde_log_std_init: float = -2.0, head_hidden_size: int = 0,
        use_beta: bool = False,
    ) -> None:
        super().__init__()
        self.features = build_feature_extractor(observation_space)
        feat_dim = self.features.out_dim
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        # gSDE/Beta only make sense for continuous action heads — silently
        # ignored for Discrete so a stale hyperparam left over from
        # switching envs never breaks a discrete-action run. Beta wins if
        # both are somehow on at once: gSDE is specifically a *Gaussian*
        # exploration scheme, incompatible with a Beta head.
        self.use_beta = use_beta and not self.discrete
        self.use_sde = use_sde and not self.discrete and not self.use_beta
        self.head_hidden_size = max(0, int(head_hidden_size))
        if self.head_hidden_size:
            self.pi_extra = nn.Sequential(nn.Linear(feat_dim, self.head_hidden_size), nn.GELU())
            self.vf_extra = nn.Sequential(nn.Linear(feat_dim, self.head_hidden_size), nn.GELU())
            head_dim = self.head_hidden_size
        else:
            self.pi_extra = None
            self.vf_extra = None
            head_dim = feat_dim
        if self.discrete:
            self.action_head = nn.Linear(head_dim, int(action_space.n))
        elif self.use_beta:
            assert np.all(np.isfinite(action_space.low)) and np.all(np.isfinite(action_space.high)), (
                "Beta policy needs a fully bounded (finite low/high) Box action space"
            )
            act_dim = int(np.prod(action_space.shape))
            self.alpha_head = nn.Linear(head_dim, act_dim)
            self.beta_head = nn.Linear(head_dim, act_dim)
            self.register_buffer("action_low", torch.as_tensor(action_space.low, dtype=torch.float32))
            self.register_buffer("action_high", torch.as_tensor(action_space.high, dtype=torch.float32))
        else:
            act_dim = int(np.prod(action_space.shape))
            self.mu_head = nn.Linear(head_dim, act_dim)
            if self.use_sde:
                # `(head_dim, act_dim)` — "full std" gSDE (one log-std per
                # policy-latent-feature/action pair) — vs. the plain
                # `(act_dim,)` below, since the whole point is a
                # *feature-dependent* std; must match whatever `pi_extra`
                # actually feeds `mu_head` (raw CNN features if there's no
                # extra head layer).
                self.log_std = nn.Parameter(torch.ones(head_dim, act_dim) * sde_log_std_init)
                self.reset_noise(1)
            else:
                self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value_head = nn.Linear(head_dim, 1)

    def reset_noise(self, batch_size: int = 1) -> None:
        """Resamples gSDE's exploration noise matrix — called once up front
        and then every `sde_sample_freq` env steps during collection (see
        `on_policy.py::learn`) so consecutive actions share the same noise
        direction for a few steps instead of it being redrawn every step.
        A no-op when gSDE isn't in use."""
        if not self.use_sde:
            return
        std = torch.exp(self.log_std)
        weights_dist = torch.distributions.Normal(torch.zeros_like(std), std)
        self._sde_exploration_mat = weights_dist.rsample()
        self._sde_exploration_mats = weights_dist.rsample((max(1, batch_size),))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (distribution params tensor, value). For discrete this is
        logits; for continuous it's the Gaussian mean."""
        params, value, _ = self._forward_with_features(obs)
        return params, value

    def _forward_with_features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.features(obs)
        pi_latent = self.pi_extra(feat) if self.pi_extra is not None else feat
        vf_latent = self.vf_extra(feat) if self.vf_extra is not None else feat
        value = self.value_head(vf_latent).squeeze(-1)
        if self.discrete:
            params = self.action_head(pi_latent)
        elif self.use_beta:
            # `softplus(...) + 1` keeps both concentration params > 1 (see
            # `_BetaPolicyDistribution` docstring for why) — packed as a
            # tuple since a Beta head has two tensors, not one "mean".
            alpha = F.softplus(self.alpha_head(pi_latent)) + 1.0
            beta = F.softplus(self.beta_head(pi_latent)) + 1.0
            params = (alpha, beta)
        else:
            params = self.mu_head(pi_latent)
        return params, value, pi_latent

    def distribution(self, obs: torch.Tensor):
        params, value, pi_latent = self._forward_with_features(obs)
        if self.discrete:
            return torch.distributions.Categorical(logits=params), value
        if self.use_beta:
            alpha, beta = params
            return _BetaPolicyDistribution(alpha, beta, self.action_low, self.action_high), value
        if self.use_sde:
            return _StateDependentGaussian(
                params, pi_latent, self.log_std, self._sde_exploration_mat, self._sde_exploration_mats,
            ), value
        std = torch.exp(self.log_std).clamp(min=1e-6)
        return torch.distributions.Normal(params, std), value

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_beta:
            dist, _ = self.distribution(obs)
            return dist.mean
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


class QuantileDuelingQNetwork(nn.Module):
    """Distributional Rainbow DQN head — QR-DQN (Dabney et al., 2017) grafted
    onto the same dueling architecture as `DuelingQNetwork`. Instead of one
    scalar `Q(s,a)`, this predicts `num_quantiles` quantiles of the *return
    distribution* for each action; the mean over quantiles recovers the
    usual scalar Q-value for action selection, but the full distribution is
    what gets trained against (quantile Huber loss in
    `NativeRainbowDQN._train_step`), which gives the network a
    richer training signal than a single expected value — this is the
    actual "distributional RL" ingredient from the original Rainbow paper
    (that implementation used the fixed-support C51 instead of quantile
    regression; QR-DQN needs no projection step and is simpler to get right
    from scratch, at basically the same benefit).

    Value/advantage streams are duelling *per quantile*: each outputs
    `num_quantiles` (value) or `n_actions * num_quantiles` (advantage)
    numbers, recombined exactly like the scalar case but broadcasting over
    the quantile axis."""

    def __init__(
        self, observation_space: gym.Space, n_actions: int, num_quantiles: int = 51, hidden: int = 128,
        noisy: bool = False, noisy_sigma0: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.num_quantiles = num_quantiles
        self.features = build_feature_extractor(observation_space)
        feat_dim = self.features.out_dim
        self.value_stream = nn.Sequential(
            _q_linear(feat_dim, hidden, noisy, noisy_sigma0), nn.ReLU(),
            _q_linear(hidden, num_quantiles, noisy, noisy_sigma0),
        )
        self.advantage_stream = nn.Sequential(
            _q_linear(feat_dim, hidden, noisy, noisy_sigma0), nn.ReLU(),
            _q_linear(hidden, n_actions * num_quantiles, noisy, noisy_sigma0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns `(batch, n_actions, num_quantiles)` — the predicted
        quantiles of each action's return distribution."""
        feat = self.features(obs)
        value = self.value_stream(feat).unsqueeze(1)
        advantage = self.advantage_stream(feat).view(-1, self.n_actions, self.num_quantiles)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Scalar `Q(s,a)` for action selection — the mean of each action's
        quantile distribution."""
        return self.forward(obs).mean(dim=-1)


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


class DeterministicPolicy(nn.Module):
    """Tanh-squashed deterministic policy used by DDPG/TD3 — unlike
    `GaussianPolicy`, there's no distribution here at all: the network
    outputs one action vector per observation, full stop. Exploration comes
    from Gaussian noise added *outside* this module (see `DDPG`/`TD3`
    `_sample_action`), not from sampling a learned distribution — that's the
    defining difference between this family and SAC."""

    def __init__(self, observation_space: gym.Space, action_low: np.ndarray, action_high: np.ndarray, hidden: int = 256) -> None:
        super().__init__()
        self.features = build_feature_extractor(observation_space)
        feat_dim = self.features.out_dim
        action_dim = len(action_low)
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        low = np.where(np.isfinite(action_low), action_low, -1.0)
        high = np.where(np.isfinite(action_high), action_high, 1.0)
        self.register_buffer("action_scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(self.features(obs))
        return torch.tanh(h) * self.action_scale + self.action_bias


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


class ESPolicyNet(nn.Module):
    """Policy-only network for Evolution Strategies — no value head at all,
    since ES never learns a critic; every parameter here is something the
    black-box search in `rl_core/algorithms/native/es.py` directly perturbs.
    Deliberately much smaller (single 64-unit hidden layer) than the
    actor-critic nets used by the gradient-based algorithms: ES's gradient
    estimate is a Monte-Carlo average over the *population*, so its variance
    scales with the parameter count — small networks need far fewer
    episodes per generation to get a usable signal, and still do
    surprisingly well on this app's classic-control-sized tasks.

    Discrete actions -> raw logits, `act()` takes the argmax (ES supplies
    its own exploration via parameter-space noise, not action-space
    sampling, so there's no reason to add softmax noise on top).
    Continuous actions -> tanh-squashed and rescaled to the env's bounds,
    same convention as `DeterministicPolicy`.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, hidden: int = 64) -> None:
        super().__init__()
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        self.features = build_feature_extractor(observation_space)
        feat_dim = self.features.out_dim
        if self.discrete:
            out_dim = int(action_space.n)
        else:
            out_dim = int(np.prod(action_space.shape))
            low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
            low = np.where(np.isfinite(low), low, -1.0)
            high = np.where(np.isfinite(high), high, 1.0)
            self.register_buffer("action_scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
            self.register_buffer("action_bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))
        self.head = nn.Sequential(nn.Linear(feat_dim, hidden), nn.Tanh(), nn.Linear(hidden, out_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.head(self.features(obs))
        if self.discrete:
            return out
        return torch.tanh(out) * self.action_scale + self.action_bias

    def act(self, obs: torch.Tensor) -> np.ndarray:
        """Single (unbatched, `obs` already has a leading batch dim of 1)
        deterministic action — argmax for discrete, the raw (tanh-squashed)
        output for continuous."""
        out = self.forward(obs)
        if self.discrete:
            return int(torch.argmax(out, dim=-1).item())
        return out.squeeze(0).detach().cpu().numpy()
