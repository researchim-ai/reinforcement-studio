"""One training step per world model family — shared between the
standalone trainer (`trainer.py`, flow 1 in `__init__.py`'s docstring) and
whichever of the four algorithms trains its world model *jointly* with a
policy (`dreamer.py`/`mbpo.py`/`world_models_ha.py`, flow 2) — both need
the exact same "sample a batch, compute the loss terms, backprop, report
them for charting" step, and keeping it in one place means a change to how
(say) the RSSM's KL term is weighted only ever needs to happen once.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_core.algorithms.native.buffers import ContinuousReplayBuffer
from rl_core.world_models.ensemble import EnsembleDynamicsModel
from rl_core.world_models.replay import SequenceReplayBuffer
from rl_core.world_models.rssm import RSSM
from rl_core.world_models.vae_mdnrnn import MDNRNN, VAE

VAE_KL_SCALE = 0.1


def train_step_rssm(
    model: RSSM, optimizer: torch.optim.Optimizer, seq_buffer: SequenceReplayBuffer,
    batch_size: int, seq_len: int, device: str,
) -> dict[str, float]:
    batch = seq_buffer.sample(batch_size, seq_len)
    obs_seq = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    action_seq = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    reward_seq = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    continue_seq = torch.as_tensor(1.0 - batch["dones"], dtype=torch.float32, device=device)

    out = model.observe(obs_seq, action_seq)
    reward_loss = F.mse_loss(out["reward_pred"], reward_seq)
    continue_loss = F.binary_cross_entropy_with_logits(out["continue_pred"], continue_seq)
    total_loss = out["model_loss"] + reward_loss + continue_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return {
        "recon_loss": float(out["recon_loss"].item()),
        "kl_loss": float(out["kl_loss"].item()),
        "reward_loss": float(reward_loss.item()),
        "continue_loss": float(continue_loss.item()),
    }


def train_step_ensemble(
    model: EnsembleDynamicsModel, optimizer: torch.optim.Optimizer, transition_buffer: ContinuousReplayBuffer,
    batch_size: int, device: str,
) -> dict[str, float]:
    batch = transition_buffer.sample(batch_size)
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)
    action = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    reward = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)

    flat_obs = model.flat_target(obs)
    flat_next_obs = model.flat_target(next_obs)
    total_loss, parts = model.nll_loss(flat_obs, action, flat_next_obs, reward)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return {key: float(value.item()) for key, value in parts.items()}


def train_step_vae_mdnrnn(
    vae: VAE, mdnrnn: MDNRNN, optimizer: torch.optim.Optimizer, seq_buffer: SequenceReplayBuffer,
    batch_size: int, seq_len: int, device: str,
) -> dict[str, float]:
    batch = seq_buffer.sample(batch_size, seq_len)
    obs_seq = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    action_seq = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    reward_seq = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    continue_seq = torch.as_tensor(1.0 - batch["dones"], dtype=torch.float32, device=device)
    batch_dim, horizon = obs_seq.shape[0], obs_seq.shape[1]

    vae_out = vae.loss(obs_seq.reshape(batch_dim * horizon, *obs_seq.shape[2:]))
    z_seq = vae_out["z"].reshape(batch_dim, horizon, -1)

    hidden = mdnrnn.initial_state(batch_dim, device)
    logits, means, log_stds, reward_pred, continue_logit, _hidden = mdnrnn(z_seq[:, :-1].detach(), action_seq[:, :-1], hidden)
    target_z = z_seq[:, 1:].detach()
    mdn_loss = mdnrnn.mixture_nll(logits, means, log_stds, target_z)
    reward_loss = F.mse_loss(reward_pred, reward_seq[:, :-1])
    continue_loss = F.binary_cross_entropy_with_logits(continue_logit, continue_seq[:, :-1])

    total_loss = vae_out["recon_loss"] + VAE_KL_SCALE * vae_out["kl_loss"] + mdn_loss + reward_loss + continue_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return {
        "vae_recon_loss": float(vae_out["recon_loss"].item()),
        "vae_kl_loss": float(vae_out["kl_loss"].item()),
        "mdn_loss": float(mdn_loss.item()),
        "reward_loss": float(reward_loss.item()),
        "continue_loss": float(continue_loss.item()),
    }
