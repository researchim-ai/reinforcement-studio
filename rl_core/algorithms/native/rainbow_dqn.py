"""From-scratch Rainbow DQN — a stronger `dqn` sibling in the Experiment
Designer (see `rl_core/algorithms/native_runner.py`). Discrete actions only,
same as plain DQN.

Implements four of the six ingredients from Hessel et al. (2018) "Rainbow:
Combining Improvements in Deep Reinforcement Learning", picked for being the
most broadly useful (and cheapest to add correctly) without changing this
project's "small, dependency-free" style:

- **Double Q-learning** — the online network picks the greedy next action,
  the target network evaluates it, decoupling action-selection from
  action-evaluation and cutting DQN's well-known overestimation bias.
- **Dueling networks** — separate value/advantage streams (see
  `DuelingQNetwork` / `SpecDuelingQNetwork`), which learn state values
  faster in states where the choice of action barely matters.
- **Prioritized experience replay** — transitions with bigger TD-error get
  replayed more often (`NStepPrioritizedReplayBuffer`).
- **N-step returns** — bootstraps `n_step` steps ahead instead of 1,
  propagating reward signal through the buffer faster.

- **Distributional RL (QR-DQN)** — optional (`distributional=1`): the network
  predicts `num_quantiles` quantiles of each action's return *distribution*
  instead of a single scalar `Q(s,a)` (Dabney et al., 2017's quantile
  regression, rather than the original paper's fixed-support C51 — no
  projection step needed, simpler to get right from scratch, same benefit).
  Combines with Double Q-learning, Dueling, PER and n-step exactly as
  described above, and with NoisyNet/RND action/intrinsic exploration too
  — turning all of these on together is the from-scratch equivalent of the
  *complete* 6-ingredient Rainbow recipe. Only combination this doesn't
  support is memory (see below): `QuantileDuelingQNetwork` has no recurrent
  counterpart.

Noisy Nets are available as a selectable action exploration strategy; RND
can independently add an intrinsic novelty bonus.

**Memory (LSTM/GRU)**, when enabled, is Double Q-learning + Dueling on top
of `RecurrentDuelingQNetwork`, trained on episode windows from
`EpisodeSequenceReplayBuffer` — the same DRQN-style setup as recurrent
`NativeDQN`. N-step returns and prioritized replay are *not* combined with
it: n-step needs to pre-aggregate rewards across a short window before a
transition is even stored, and PER needs a per-transition priority — both
assume independently-sampleable transitions, which is exactly what episode
windows (needed for BPTT) give up. Correctly reconciling all three (à la
R2D2's sequence-level priorities) is a meaningfully bigger undertaking than
the rest of this feature, so recurrent mode intentionally trades them away
for Double+Dueling+memory instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.buffers import EpisodeSequenceReplayBuffer, NStepPrioritizedReplayBuffer
from rl_core.algorithms.native.exploration import (
    RNDModule,
    linear_schedule,
    reset_noise,
    set_noise_enabled,
    uses_noisy_net,
    uses_rnd,
)
from rl_core.algorithms.native.networks import (
    DuelingQNetwork,
    Hidden,
    QuantileDuelingQNetwork,
    RecurrentDuelingQNetwork,
    mask_hidden_for_dones,
    memory_type_from_hyperparams,
)
from rl_core.algorithms.native.preprocessing import obs_batch_to_array, obs_to_array
from rl_core.algorithms.vec_env import action_space, num_envs_of, obs_space, vec_reset, vec_step
from rl_core.netbuilder import SpecDuelingQNetwork

DEFAULT_HYPERPARAMS = {
    "learning_rate": 1e-3,
    "buffer_size": 50_000,
    "batch_size": 64,
    "gamma": 0.99,
    "n_step": 3,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "exploration_fraction": 0.2,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "action_exploration": 0,
    "noisy_sigma0": 0.5,
    "distributional": 0,
    "num_quantiles": 51,
    "intrinsic_exploration": 0,
    "rnd_bonus_coef": 0.1,
    "rnd_learning_rate": 1e-4,
    "rnd_feature_dim": 128,
    "rnd_hidden_dim": 128,
    "rnd_bonus_clip": 5.0,
    "learning_starts": 1_000,
    "train_freq": 4,
    "target_update_interval": 1_000,
    # Memory (see module docstring). 0 = no memory (plain DuelingQNetwork,
    # full Rainbow-lite recipe), 1 = LSTM, 2 = GRU (Double+Dueling only).
    "memory_type": 0,
    "memory_hidden_size": 128,
    "memory_num_layers": 1,
    "memory_seq_len": 20,
}


def _quantile_huber_loss(current_quantiles: torch.Tensor, target_quantiles: torch.Tensor, tau: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Pairwise quantile Huber loss (Dabney et al., 2017, eq. 9-10) — every
    predicted quantile `i` is compared against every target sample `j`
    (there's no correspondence between "the i-th predicted quantile" and
    "the i-th target quantile", they're just `N` unordered samples of the
    same distribution once the Bellman shift is applied), weighted by how
    far `tau_i` is from "target sample landed below/above this quantile".
    Returns one loss value per batch element, `(batch,)` — callers combine
    it with importance-sampling weights before reducing to a scalar."""
    td_error = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)  # (batch, N_current, N_target)
    abs_td = td_error.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td_error.pow(2), kappa * (abs_td - 0.5 * kappa))
    tau_b = tau.view(1, -1, 1)
    weight = (tau_b - (td_error.detach() < 0).float()).abs()
    return (weight * huber).mean(dim=2).sum(dim=1)


class NativeRainbowDQN(CustomAlgorithm):
    net: DuelingQNetwork | SpecDuelingQNetwork | RecurrentDuelingQNetwork

    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
        obs_sp, act_sp = obs_space(env), action_space(env)
        if not isinstance(act_sp, gym.spaces.Discrete):
            raise ValueError("NativeRainbowDQN only supports Discrete action spaces")

        self.n_actions = int(act_sp.n)
        network_spec = hyperparams.get("network_spec")
        memory_type = memory_type_from_hyperparams(hyperparams)
        self.recurrent = bool(memory_type) and not network_spec
        self.noisy = uses_noisy_net(hyperparams)
        if self.noisy and network_spec:
            raise ValueError("NoisyNet is unavailable with a custom NetworkSpec; use the standard dueling network")
        noisy_sigma0 = float(hyperparams.get("noisy_sigma0", 0.5))
        self.distributional = bool(int(hyperparams.get("distributional", 0) or 0))
        if self.distributional and network_spec:
            raise ValueError("Distributional (QR-DQN) is unavailable with a custom NetworkSpec; use the standard dueling network")
        if self.distributional and memory_type:
            raise ValueError("Distributional (QR-DQN) doesn't have a recurrent counterpart yet; disable memory to use it")
        self.num_quantiles = max(2, int(hyperparams.get("num_quantiles", 51)))
        self._tau = (torch.arange(self.num_quantiles, dtype=torch.float32, device=device) + 0.5) / self.num_quantiles

        def _build_q_net() -> nn.Module:
            if network_spec:
                return SpecDuelingQNetwork(obs_sp, self.n_actions, network_spec)
            if memory_type:
                return RecurrentDuelingQNetwork(
                    obs_sp, self.n_actions, memory_type,
                    hidden_size=int(hyperparams.get("memory_hidden_size", 128)),
                    num_layers=int(hyperparams.get("memory_num_layers", 1)),
                    noisy=self.noisy, noisy_sigma0=noisy_sigma0,
                )
            if self.distributional:
                return QuantileDuelingQNetwork(
                    obs_sp, self.n_actions, num_quantiles=self.num_quantiles,
                    noisy=self.noisy, noisy_sigma0=noisy_sigma0,
                )
            return DuelingQNetwork(
                obs_sp, self.n_actions,
                noisy=self.noisy, noisy_sigma0=noisy_sigma0,
            )

        self.q_net = _build_q_net().to(device)
        self.target_net = _build_q_net().to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(hyperparams.get("learning_rate", 1e-3)))

        self.buffer_size = int(hyperparams.get("buffer_size", 50_000))
        self.batch_size = int(hyperparams.get("batch_size", 64))
        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.n_step = max(1, int(hyperparams.get("n_step", 3)))
        self.per_alpha = float(hyperparams.get("per_alpha", 0.6))
        self.per_beta_start = float(hyperparams.get("per_beta_start", 0.4))
        self.exploration_fraction = float(hyperparams.get("exploration_fraction", 0.2))
        self.exploration_initial_eps = float(hyperparams.get("exploration_initial_eps", 1.0))
        self.exploration_final_eps = float(hyperparams.get("exploration_final_eps", 0.05))
        self.learning_starts = int(hyperparams.get("learning_starts", 1_000))
        self.train_freq = max(1, int(hyperparams.get("train_freq", 4)))
        self.target_update_interval = max(1, int(hyperparams.get("target_update_interval", 1_000)))
        self.memory_seq_len = max(1, int(hyperparams.get("memory_seq_len", 20)))
        self.rnd_bonus_coef = float(hyperparams.get("rnd_bonus_coef", 0.1))
        self.rnd = RNDModule(
            obs_sp,
            device,
            feature_dim=int(hyperparams.get("rnd_feature_dim", 128)),
            hidden_dim=int(hyperparams.get("rnd_hidden_dim", 128)),
            learning_rate=float(hyperparams.get("rnd_learning_rate", 1e-4)),
            bonus_clip=float(hyperparams.get("rnd_bonus_clip", 5.0)),
        ) if uses_rnd(hyperparams) else None
        self.replay_buffer: NStepPrioritizedReplayBuffer | EpisodeSequenceReplayBuffer | None = None
        self._hidden: Hidden | None = None
        self._predict_hidden: Hidden | None = None
        self._last_td_loss: float | None = None
        self._obs_space = obs_sp
        self._action_space = act_sp

    def _obs_arr(self, obs: Any) -> np.ndarray:
        return obs_to_array(obs, self._obs_space)

    def _epsilon(self, num_timesteps: int, total_timesteps: int) -> float:
        if self.noisy:
            return 0.0
        return linear_schedule(
            num_timesteps, total_timesteps, self.exploration_fraction,
            self.exploration_initial_eps, self.exploration_final_eps,
        )

    def _metrics(
        self,
        epsilon: float,
        episode_extrinsic: float | None = None,
        episode_intrinsic: float | None = None,
    ) -> dict[str, float]:
        metrics = {"exploration_epsilon": float(epsilon)}
        if self._last_td_loss is not None:
            metrics["td_loss"] = self._last_td_loss
        if self.rnd:
            metrics["rnd_bonus_mean"] = float(self.rnd.bonus_mean)
            metrics["rnd_predictor_loss"] = float(self.rnd.last_loss)
        if episode_extrinsic is not None:
            metrics["episode_extrinsic_reward"] = float(episode_extrinsic)
        if self.rnd and episode_intrinsic is not None:
            metrics["episode_intrinsic_reward"] = float(episode_intrinsic)
        return metrics

    def _beta(self, num_timesteps: int, total_timesteps: int) -> float:
        """Anneals the importance-sampling correction from `per_beta_start`
        up to 1.0 (full correction) linearly over the whole run — early on,
        when the buffer barely has any data, a little sampling bias doesn't
        matter much; by the end it should be fully corrected. Unused in
        recurrent mode (no PER there — see module docstring)."""
        progress = min(1.0, num_timesteps / max(1, total_timesteps))
        return self.per_beta_start + progress * (1.0 - self.per_beta_start)

    def _step_q_batch(self, obs_arr: np.ndarray) -> torch.Tensor:
        """See NativeDQN._step_q_batch — always runs, so the recurrent core
        sees every observation regardless of which policy picks the
        action. `obs_arr`: `(num_envs, ...)`. Returns plain scalar
        `Q(s,a)` either way — for the distributional net that means
        collapsing its per-action quantiles down to their mean, which is
        all action *selection* (as opposed to the training loss) ever
        needs."""
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.recurrent:
                q, self._hidden = self.q_net.step(obs_t, self._hidden)
            elif self.distributional:
                q = self.q_net.q_values(obs_t)
            else:
                q = self.q_net(obs_t)
        return q

    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        n_envs = num_envs_of(self.env)
        obs_arr = obs_batch_to_array(vec_reset(self.env, seed=self.seed), self._obs_space)
        if self.replay_buffer is None:
            self.replay_buffer = (
                EpisodeSequenceReplayBuffer(self.buffer_size, obs_arr.shape[1:], num_envs=n_envs)
                if self.recurrent
                else NStepPrioritizedReplayBuffer(
                    self.buffer_size, obs_arr.shape[1:], n_step=self.n_step, gamma=self.gamma,
                    alpha=self.per_alpha, num_envs=n_envs,
                )
            )
        if self.recurrent:
            self._hidden = self.q_net.initial_state(n_envs, self.device)
        ep_reward = np.zeros(n_envs, dtype=np.float64)
        ep_extrinsic = np.zeros(n_envs, dtype=np.float64)
        ep_intrinsic = np.zeros(n_envs, dtype=np.float64)
        ep_length = np.zeros(n_envs, dtype=np.int64)
        num_timesteps = 0

        while num_timesteps < total_timesteps:
            eps = self._epsilon(num_timesteps, total_timesteps)
            if self.noisy:
                reset_noise(self.q_net)
            q = self._step_q_batch(obs_arr)
            greedy = torch.argmax(q, dim=-1).cpu().numpy()
            actions = np.array(
                [
                    int(self._action_space.sample())
                    if len(self.replay_buffer) < self.learning_starts or np.random.rand() < eps
                    else int(greedy[i])
                    for i in range(n_envs)
                ],
                dtype=np.int64,
            )

            next_obs_list, rewards, terminated, truncated, _infos = vec_step(self.env, list(actions))
            dones = terminated | truncated
            next_obs_arr = obs_batch_to_array(next_obs_list, self._obs_space)
            extrinsic_reward = rewards.astype(np.float64)
            intrinsic_reward = (
                np.array([self.rnd_bonus_coef * self.rnd.bonus(next_obs_arr[i]) for i in range(n_envs)])
                if self.rnd
                else np.zeros(n_envs)
            )
            total_reward = extrinsic_reward + intrinsic_reward
            for i in range(n_envs):
                self.replay_buffer.add(obs_arr[i], int(actions[i]), float(total_reward[i]), next_obs_arr[i], bool(dones[i]), lane=i)
            obs_arr = next_obs_arr
            ep_reward += total_reward
            ep_extrinsic += extrinsic_reward
            ep_intrinsic += intrinsic_reward
            ep_length += 1
            prev_num_timesteps = num_timesteps
            num_timesteps += n_envs

            if len(self.replay_buffer) >= max(self.learning_starts, self.batch_size) and (
                num_timesteps // self.train_freq != prev_num_timesteps // self.train_freq
            ):
                self._train_step(self._beta(num_timesteps, total_timesteps))
            if num_timesteps // self.target_update_interval != prev_num_timesteps // self.target_update_interval:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if self.recurrent:
                self._hidden = mask_hidden_for_dones(self._hidden, dones)

            any_finished = False
            for i in range(n_envs):
                if dones[i]:
                    any_finished = True
                    keep_going = callback.on_step(
                        num_timesteps, float(ep_reward[i]), int(ep_length[i]),
                        self._metrics(eps, float(ep_extrinsic[i]), float(ep_intrinsic[i])),
                    )
                    ep_reward[i], ep_extrinsic[i], ep_intrinsic[i], ep_length[i] = 0.0, 0.0, 0.0, 0
                    if not keep_going:
                        return
            if not any_finished:
                keep_going = callback.on_step(num_timesteps, metrics=self._metrics(eps))
                if not keep_going:
                    return

    def _train_step(self, beta: float) -> None:
        if self.recurrent:
            self._train_step_recurrent()
            return
        batch = self.replay_buffer.sample(self.batch_size, beta)
        if self.noisy:
            reset_noise(self.q_net)
            reset_noise(self.target_net)
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)

        if self.distributional:
            td_error = self._train_step_distributional(obs_t, actions_t, rewards_t, next_obs_t, dones_t, weights_t)
        else:
            with torch.no_grad():
                # Double Q-learning: the *online* net picks the greedy next
                # action, the *target* net supplies its value — decouples
                # selection from evaluation so a lucky overestimate in one
                # network doesn't get echoed straight back into its own target.
                next_actions = torch.argmax(self.q_net(next_obs_t), dim=-1)
                next_q = self.target_net(next_obs_t).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                # Reward already sums `n_step` discounted steps (see
                # NStepPrioritizedReplayBuffer), so the bootstrap term uses
                # gamma**n_step, not gamma. `dones` is 1.0 whenever that window
                # ended in a terminal state, including short (< n_step)
                # end-of-episode windows, so the (1 - done) mask is always the
                # correct thing to zero the bootstrap with.
                target = rewards_t + (self.gamma**self.n_step) * (1.0 - dones_t) * next_q

            current_q = self.q_net(obs_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
            td_error = target - current_q
            loss = (weights_t * F.smooth_l1_loss(current_q, target, reduction="none")).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
            self.optimizer.step()
            self._last_td_loss = float(loss.detach().item())
            td_error = td_error.detach().cpu().numpy()

        self.replay_buffer.update_priorities(batch["indices"], td_error)
        if self.rnd:
            self.rnd.update_predictor(batch["next_obs"])

    def _train_step_distributional(
        self,
        obs_t: torch.Tensor,
        actions_t: torch.Tensor,
        rewards_t: torch.Tensor,
        next_obs_t: torch.Tensor,
        dones_t: torch.Tensor,
        weights_t: torch.Tensor,
    ) -> np.ndarray:
        """QR-DQN training step — same Double Q-learning + PER + n-step
        wiring as the scalar branch above, just carrying a full
        `num_quantiles`-vector per action instead of one number, and using
        the quantile Huber loss (`_quantile_huber_loss`) instead of plain
        Huber/MSE. Returns the per-sample TD-error array PER needs to
        refresh priorities with — the difference of *means* over each
        sample's target/current quantiles doubles as a perfectly fine proxy
        for "how surprising was this transition" even though the loss
        itself never collapses the distribution down to a scalar."""
        n = self.num_quantiles
        with torch.no_grad():
            next_actions = torch.argmax(self.q_net.q_values(next_obs_t), dim=-1)
            next_quantiles_all = self.target_net(next_obs_t)
            next_quantiles = next_quantiles_all.gather(
                1, next_actions.view(-1, 1, 1).expand(-1, 1, n)
            ).squeeze(1)
            target_quantiles = rewards_t.unsqueeze(-1) + (self.gamma**self.n_step) * (1.0 - dones_t).unsqueeze(-1) * next_quantiles

        current_quantiles_all = self.q_net(obs_t)
        current_quantiles = current_quantiles_all.gather(
            1, actions_t.view(-1, 1, 1).expand(-1, 1, n)
        ).squeeze(1)

        per_sample_loss = _quantile_huber_loss(current_quantiles, target_quantiles, self._tau)
        loss = (weights_t * per_sample_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self._last_td_loss = float(loss.detach().item())

        td_error = (target_quantiles.mean(dim=-1) - current_quantiles.mean(dim=-1)).detach()
        return td_error.cpu().numpy()

    def _train_step_recurrent(self) -> None:
        """Double-Q + Dueling on episode windows — see NativeDQN's
        `_train_step_recurrent` for the equivalent plain-DQN version this
        mirrors; the only difference is picking the online net's greedy
        next action before asking the target net to evaluate it."""
        batch = self.replay_buffer.sample(self.batch_size, self.memory_seq_len)
        if self.noisy:
            reset_noise(self.q_net)
            reset_noise(self.target_net)
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(batch["mask"], dtype=torch.float32, device=self.device)
        batch_size = obs_t.shape[0]

        with torch.no_grad():
            h0_online_next = self.q_net.initial_state(batch_size, self.device)
            next_q_online, _ = self.q_net.forward_sequence(next_obs_t, h0_online_next)
            next_actions = torch.argmax(next_q_online, dim=-1)

            h0_target = self.target_net.initial_state(batch_size, self.device)
            next_q_target, _ = self.target_net.forward_sequence(next_obs_t, h0_target)
            next_q = next_q_target.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        h0 = self.q_net.initial_state(batch_size, self.device)
        current_q, _ = self.q_net.forward_sequence(obs_t, h0)
        current_q_a = current_q.gather(-1, actions_t.unsqueeze(-1)).squeeze(-1)
        per_step_loss = F.smooth_l1_loss(current_q_a, target, reduction="none")
        loss = (per_step_loss * mask_t).sum() / mask_t.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self._last_td_loss = float(loss.detach().item())
        if self.rnd:
            flat_next_obs = batch["next_obs"].reshape(-1, *batch["next_obs"].shape[2:])
            self.rnd.update_predictor(flat_next_obs, mask=batch["mask"].reshape(-1))

    def predict(self, obs: Any, deterministic: bool = True, episode_start: bool = False) -> tuple[Any, Any]:
        obs_arr = self._obs_arr(obs)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.noisy:
            set_noise_enabled(self.q_net, not deterministic)
            if not deterministic:
                reset_noise(self.q_net)
        with torch.no_grad():
            if self.recurrent:
                if episode_start or self._predict_hidden is None:
                    self._predict_hidden = self.q_net.initial_state(1, self.device)
                q, self._predict_hidden = self.q_net.step(obs_t, self._predict_hidden)
            elif self.distributional:
                q = self.q_net.q_values(obs_t)
            else:
                q = self.q_net(obs_t)
        if self.noisy:
            set_noise_enabled(self.q_net, True)
        if not deterministic and not self.noisy and np.random.rand() < self.exploration_final_eps:
            return int(self._action_space.sample()), None
        return int(torch.argmax(q, dim=-1).item()), None

    def save(self, path: Path) -> None:
        payload = {
            "state_dict": self.q_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "hyperparams": self.hyperparams,
        }
        if self.rnd:
            payload["rnd_state"] = self.rnd.checkpoint_state()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativeRainbowDQN":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.q_net.load_state_dict(payload["state_dict"])
        algo.target_net.load_state_dict(payload["state_dict"])
        if payload.get("optimizer_state_dict"):
            algo.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if algo.rnd and payload.get("rnd_state"):
            algo.rnd.load_checkpoint_state(payload["rnd_state"])
        return algo
