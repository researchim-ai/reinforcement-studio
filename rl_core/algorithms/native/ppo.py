"""From-scratch Proximal Policy Optimization — the default `ppo` algorithm
in the Experiment Designer (see `rl_core/algorithms/native_runner.py`)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import gymnasium as gym

from rl_core.algorithms.native.buffers import RolloutBuffer
from rl_core.algorithms.native.networks import ActorCriticNet, Hidden, RecurrentActorCriticNet, detach_hidden, memory_type_from_hyperparams
from rl_core.algorithms.native.on_policy import OnPolicyAlgorithm, _LossAccumulator
from rl_core.algorithms.vec_env import action_space, num_envs_of, obs_space
from rl_core.netbuilder import SpecActorCriticNet

DEFAULT_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_epochs": 10,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    # 0 = constant `learning_rate` throughout, 1 = linearly decay it to 0
    # over `total_timesteps` — see `OnPolicyAlgorithm.learn`. Off by default
    # (matches the plain PPO paper setup); envs like CarRacing (pixel input,
    # already fairly-tuned late in training) turn it on via
    # `default_hyperparams` in rl_core/envs/registry.py since a fixed LR
    # tends to keep knocking a good driving policy around instead of
    # settling into it.
    "lr_schedule": 0,
    # gSDE (generalized State-Dependent Exploration) for continuous action
    # spaces — see `ActorCriticNet`/`_StateDependentGaussian` in
    # networks.py. Off by default (plain per-step Gaussian noise, the usual
    # PPO setup); ignored entirely for Discrete action spaces.
    "use_sde": 0,
    "sde_sample_freq": 4,
    "sde_log_std_init": -2.0,
    # Beta-distribution policy head for continuous action spaces instead of
    # a Gaussian (see `_BetaPolicyDistribution` in networks.py) — natively
    # bounded to the action space's box, so it never needs the clip-after-
    # sample step a Gaussian does, which literature (Chou et al. 2017,
    # Petrazzini & Antonelo 2021 on CarRacing specifically) reports as a
    # real source of bias/slower convergence for hard-bounded continuous
    # actions like one-sided gas/brake in [0, 1]. Off by default (plain
    # Gaussian, the common PPO setup and what `use_sde` builds on); wins
    # over `use_sde` if both are somehow set, since gSDE is a Gaussian-only
    # exploration scheme.
    "use_beta": 0,
    # Extra per-head hidden layer between the feature extractor and each of
    # mu/value (see `ActorCriticNet.head_hidden_size` in networks.py) — 0
    # keeps the previous "heads attached directly to raw features"
    # architecture; pixel envs with a lot of visual detail to summarize
    # into one action (CarRacing) get more out of a dedicated 256-wide
    # layer per head than out of a bigger shared CNN, matching SB3 zoo's
    # `net_arch=dict(pi=[256], vf=[256])` tuning for that env.
    "head_hidden_size": 0,
    # Memory (see rl_core/algorithms/native/networks.py:RecurrentActorCriticNet).
    # 0 = no memory (plain ActorCriticNet), 1 = LSTM, 2 = GRU.
    "memory_type": 0,
    "memory_hidden_size": 128,
    "memory_num_layers": 1,
    "memory_seq_len": 32,
}


class NativePPO(OnPolicyAlgorithm):
    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        obs_sp, act_sp = obs_space(env), action_space(env)
        network_spec = hyperparams.get("network_spec")
        memory_type = memory_type_from_hyperparams(hyperparams)
        # gSDE (below) needs `reset_noise()` on the actual net class, which
        # only `ActorCriticNet` implements — a hand-designed (Network
        # Builder) or recurrent architecture just never gets it, regardless
        # of `use_sde`, rather than erroring.
        use_sde = bool(int(hyperparams.get("use_sde", 0)))
        sde_log_std_init = float(hyperparams.get("sde_log_std_init", -2.0))
        head_hidden_size = int(hyperparams.get("head_hidden_size", 0))
        use_beta = bool(int(hyperparams.get("use_beta", 0)))
        if network_spec:
            self.net = SpecActorCriticNet(obs_sp, act_sp, network_spec)
        elif memory_type:
            self.net = RecurrentActorCriticNet(
                obs_sp, act_sp, memory_type,
                hidden_size=int(hyperparams.get("memory_hidden_size", 128)),
                num_layers=int(hyperparams.get("memory_num_layers", 1)),
            )
        else:
            self.net = ActorCriticNet(
                obs_sp, act_sp, use_sde=use_sde, sde_log_std_init=sde_log_std_init,
                head_hidden_size=head_hidden_size, use_beta=use_beta,
            )
        self.net = self.net.to(device)
        self.recurrent = isinstance(self.net, RecurrentActorCriticNet)
        self.memory_seq_len = max(1, int(hyperparams.get("memory_seq_len", 32)))
        self._initial_lr = float(hyperparams.get("learning_rate", 3e-4))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self._initial_lr)
        self.lr_schedule = "linear" if int(hyperparams.get("lr_schedule", 0)) == 1 else "constant"
        self.n_steps = int(hyperparams.get("n_steps", 2048))
        self.batch_size = int(hyperparams.get("batch_size", 64))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.gae_lambda = float(hyperparams.get("gae_lambda", 0.95))
        self.clip_range = float(hyperparams.get("clip_range", 0.2))
        self.n_epochs = int(hyperparams.get("n_epochs", 10))
        self.ent_coef = float(hyperparams.get("ent_coef", 0.0))
        self.vf_coef = float(hyperparams.get("vf_coef", 0.5))
        self.max_grad_norm = float(hyperparams.get("max_grad_norm", 0.5))
        # `getattr(self, "use_sde", False)` in on_policy.py's collection
        # loop is what actually gates the periodic `reset_noise()` calls —
        # this flag (not the raw hyperparam) reflects whether `self.net`
        # really is an SDE-capable `ActorCriticNet` with it switched on.
        self.use_sde = isinstance(self.net, ActorCriticNet) and self.net.use_sde
        self.sde_sample_freq = max(1, int(hyperparams.get("sde_sample_freq", 4)))
        if self.use_sde:
            self.net.reset_noise(num_envs_of(env))

    def _update(self, buf: RolloutBuffer, rollout_hidden: Hidden | None = None) -> None:
        if self.recurrent:
            self._update_recurrent(buf, rollout_hidden)
            return
        acc = _LossAccumulator()
        for _ in range(self.n_epochs):
            for batch in buf.minibatches(self.batch_size):
                obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
                actions_t = torch.as_tensor(batch["actions"], device=self.device)
                old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
                advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)

                dist, values = self.net.distribution(obs_t)
                log_probs = self.net.log_prob(dist, actions_t)
                entropy = self.net.entropy(dist).mean()

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                acc.add(policy_loss, value_loss, entropy)
        self._last_metrics = acc.means()

    def _update_recurrent(self, buf: RolloutBuffer, rollout_hidden: Hidden | None) -> None:
        """Same clipped PPO objective as above, but chunked over
        `buf.sequences(...)` (in time order — see `RolloutBuffer.sequences`)
        with the LSTM/GRU hidden state threaded chunk-to-chunk and detached
        in between, so each chunk only backprops through its own length
        (truncated BPTT) instead of the whole rollout. Every epoch restarts
        from `rollout_hidden` — the state the *collector* was carrying at
        the very start of this rollout — since PPO re-derives log-probs and
        values from the same fixed batch of experience `n_epochs` times."""
        acc = _LossAccumulator()
        for _ in range(self.n_epochs):
            hidden = rollout_hidden
            for batch in buf.sequences(self.memory_seq_len):
                # `batch["obs"]` etc. are already `(num_envs, chunk_len, ...)`
                # — batch dim is the lane axis, not a size-1 placeholder.
                obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
                actions_t = torch.as_tensor(batch["actions"], device=self.device)
                old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
                advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
                episode_starts = torch.as_tensor(batch["episode_starts"], dtype=torch.float32, device=self.device)

                dist, values, hidden = self.net.distribution_sequence(obs_t, hidden, episode_starts)
                log_probs = self.net.log_prob(dist, actions_t)
                entropy = self.net.entropy(dist).mean()

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                acc.add(policy_loss, value_loss, entropy)

                hidden = detach_hidden(hidden)
        self._last_metrics = acc.means()
