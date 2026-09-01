"""Standalone World Model training — no RL agent at all, just "collect
experience with a random policy, fit the model" (flow 1 in
`rl_core/world_models/__init__.py`'s module docstring). Dispatched from
`rl_core/runner.py` exactly like any other run (`kind: "world_model"`), so
it gets the *entire* existing run-tracking pipeline for free: `POST
/training/start`, live metrics over the same websocket, the Stop button,
the Training Monitor page itself — this module only needs to hold up its
end of that same `metrics.json`/`stop.flag` contract
(`rl_core/algorithms/runner_utils.py` and `metrics_callback.py` do the
identical thing for real algorithm runs).

Reuses the *exact* `StartRunRequest` shape (`backend/routes/training.py`)
every other run already uses — `algorithm.id` is one of
`rl_core.world_models.spec.WORLD_MODEL_TYPES` instead of a real algorithm
id, and `algorithm.hyperparams` is that type's config dict rather than a
normal algorithm's numeric hyperparams. This is exactly what the World
Model Builder page's "Train" button sends.

`training.num_envs > 1` collects with that many parallel env lanes
(see `run()` below) — same speedup lever as every other native
algorithm's `training.num_envs`, just wired up manually here since this
module bypasses `runner_utils.py` entirely.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from rl_core.algorithms.native.buffers import ContinuousReplayBuffer
from rl_core.algorithms.native.preprocessing import is_image_space, obs_flat_dim, obs_to_array
from rl_core.algorithms.sb3_runner import _make_env, make_monitored_env_factory
from rl_core.algorithms.vec_env import action_space as venv_action_space, make_env_or_vec, obs_space as venv_obs_space
from rl_core.device import resolve_device
from rl_core.metrics_history import append_history, json_safe
from rl_core.world_models import store as wm_store
from rl_core.world_models.ensemble import EnsembleDynamicsModel
from rl_core.world_models.losses import train_step_ensemble, train_step_rssm, train_step_vae_mdnrnn
from rl_core.world_models.replay import SequenceReplayBuffer, action_flat_dim, collect_rollout_auto
from rl_core.world_models.rssm import RSSM
from rl_core.world_models.spec import build_world_model, save_checkpoint
from rl_core.world_models.vae_mdnrnn import MDNRNN, VAE
from rl_core.world_models.viz import render_imagined_vs_real_gif, render_latent_scatter

_WRITE_EVERY_STEPS = 500
_PREVIEW_EVERY_STEPS = 4000
_PREVIEW_HORIZON = 30


def _obs_shape_of(observation_space) -> tuple[int, ...]:
    return tuple(int(d) for d in observation_space.shape) if is_image_space(observation_space) else (obs_flat_dim(observation_space),)


def _collect_open_loop_episode(env, action_space, horizon: int) -> tuple[list, list]:
    """A short random-policy episode, kept entirely in memory — the "real"
    half of the imagined-vs-real preview GIF, and the action sequence fed
    open-loop into the model's own imagination for the "imagined" half."""
    obs, _info = env.reset()
    obs_seq, action_seq = [obs], []
    for _ in range(horizon):
        action = action_space.sample()
        action_seq.append(action)
        obs, _reward, terminated, truncated, _info = env.step(action)
        obs_seq.append(obs)
        if terminated or truncated:
            break
    return obs_seq, action_seq


@torch.no_grad()
def _preview_rssm(model: RSSM, env, action_space, observation_space, device: str) -> tuple[bytes, bytes]:
    obs_seq, action_seq = _collect_open_loop_episode(env, action_space, _PREVIEW_HORIZON)
    obs_arrs = [obs_to_array(o, observation_space) for o in obs_seq]  # length T
    action_arrs = [obs_to_array(a, action_space) for a in action_seq]  # length T-1
    if not action_arrs:
        # Degenerate (episode ended on the very first `reset()`) — nothing
        # to imagine forward from; fall back to a trivial "real vs itself".
        return render_imagined_vs_real_gif(obs_arrs, obs_arrs, observation_space), render_latent_scatter(np.zeros((1, model.stoch_dim)))

    obs_t = torch.as_tensor(np.stack(obs_arrs), dtype=torch.float32, device=device).unsqueeze(0)
    # `observe()` needs one action per observed step (its very last one is
    # never actually read — see the loop in `RSSM.observe` — so any
    # placeholder value is fine); pad with one zero action to match.
    padded_actions = np.stack([*action_arrs, np.zeros_like(action_arrs[0])])
    action_t = torch.as_tensor(padded_actions, dtype=torch.float32, device=device).unsqueeze(0)

    observed = model.observe(obs_t, action_t)
    h0, z0 = observed["h"][:, 0], observed["z"][:, 0]

    real_action_t = torch.as_tensor(np.stack(action_arrs), dtype=torch.float32, device=device).unsqueeze(0)
    step_idx = {"i": 0}

    def replay_action(_feat: torch.Tensor) -> torch.Tensor:
        i = step_idx["i"]
        step_idx["i"] += 1
        return real_action_t[:, i]

    imagined = model.imagine(h0, z0, replay_action, horizon=len(action_arrs))
    imagined_obs = model.decode(imagined["feat"]).squeeze(0).detach().cpu().numpy()
    real_obs_from_step1 = obs_arrs[1 : 1 + imagined_obs.shape[0]]
    gif = render_imagined_vs_real_gif(real_obs_from_step1, list(imagined_obs), observation_space)
    latent = observed["z"].squeeze(0).detach().cpu().numpy()
    scatter = render_latent_scatter(latent)
    return gif, scatter


@torch.no_grad()
def _preview_ensemble(model: EnsembleDynamicsModel, env, action_space, observation_space, device: str) -> bytes:
    obs_seq, action_seq = _collect_open_loop_episode(env, action_space, _PREVIEW_HORIZON)
    obs_arrs = [obs_to_array(o, observation_space) for o in obs_seq]
    action_arrs = [obs_to_array(a, action_space) for a in action_seq]
    if not action_arrs:
        return render_imagined_vs_real_gif(obs_arrs, obs_arrs, observation_space)

    flat_obs = model.flat_target(torch.as_tensor(obs_arrs[0], dtype=torch.float32, device=device).unsqueeze(0))
    model_indices = model.sample_model_indices(1, device)
    imagined_obs = []
    for action_arr in action_arrs:
        action_t = torch.as_tensor(action_arr, dtype=torch.float32, device=device).unsqueeze(0)
        flat_obs, _reward = model.predict_step(flat_obs, action_t, model_indices=model_indices, sample=True)
        shaped = flat_obs.squeeze(0).detach().cpu().numpy()
        if model.is_image:
            shaped = shaped.reshape(observation_space.shape)
        imagined_obs.append(shaped)
    real_obs_from_step1 = obs_arrs[1 : 1 + len(imagined_obs)]
    return render_imagined_vs_real_gif(real_obs_from_step1, imagined_obs, observation_space)


@torch.no_grad()
def _preview_vae_mdnrnn(vae: VAE, mdnrnn: MDNRNN, env, action_space, observation_space, device: str) -> tuple[bytes, bytes]:
    obs_seq, action_seq = _collect_open_loop_episode(env, action_space, _PREVIEW_HORIZON)
    obs_arrs = [obs_to_array(o, observation_space) for o in obs_seq]
    action_arrs = [obs_to_array(a, action_space) for a in action_seq]

    obs_t = torch.as_tensor(np.stack(obs_arrs), dtype=torch.float32, device=device)
    mean, _logvar = vae.encode(obs_t)
    real_latents = mean.detach().cpu().numpy()  # (T+1, latent_dim) — used for the latent scatter regardless of imagination below

    if not action_arrs:
        gif = render_imagined_vs_real_gif(obs_arrs, obs_arrs, observation_space)
        return gif, render_latent_scatter(real_latents)

    z = mean[0:1]
    hidden = mdnrnn.initial_state(1, device)
    imagined_obs = []
    for action_arr in action_arrs:
        action_t = torch.as_tensor(action_arr, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        logits, means, log_stds, _reward, _continue_logit, hidden = mdnrnn(z.unsqueeze(1), action_t, hidden)
        z = MDNRNN.sample_next_z(logits.squeeze(1), means.squeeze(1), log_stds.squeeze(1))
        imagined_obs.append(vae.decode(z).squeeze(0).detach().cpu().numpy())
    real_obs_from_step1 = obs_arrs[1 : 1 + len(imagined_obs)]
    gif = render_imagined_vs_real_gif(real_obs_from_step1, imagined_obs, observation_space)
    return gif, render_latent_scatter(real_latents)


def run(config: dict[str, Any], run_dir: Path) -> None:
    env_cfg = config.get("environment", {})
    wm_cfg = config.get("algorithm", {})
    training_cfg = config.get("training", {})
    env_id = env_cfg["id"]
    wrapper_specs = env_cfg.get("wrappers", [])
    kind = str(wm_cfg.get("id", "rssm")).lower()
    model_config = wm_cfg.get("hyperparams") or {}
    slug = wm_cfg.get("world_model_id")

    seed = training_cfg.get("seed")
    total_steps = int(training_cfg.get("total_timesteps", 20_000))
    device = resolve_device(training_cfg)
    batch_size = int(training_cfg.get("batch_size", 32))
    seq_len = int(training_cfg.get("seq_len", 50))
    collect_per_iter = max(1, int(training_cfg.get("collect_steps_per_iter", 200)))
    train_steps_per_iter = max(1, int(training_cfg.get("train_steps_per_iter", 40)))
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    # "Run several environments in parallel with different seeds" — each
    # lane gets a subprocess worker (`vec_env.make_env_or_vec`, same
    # `AsyncVectorEnv` machinery every other native algorithm already
    # uses for `training.num_envs > 1`) and its own seed out of the box,
    # so `collect_per_iter` real transitions land in roughly `1/num_envs`
    # of the wall-clock time a single env would need. `preview_env` below
    # stays a single plain env regardless — the imagined-vs-real GIF and
    # latent scatter only ever need one open-loop episode at a time.
    num_envs = max(1, int(training_cfg.get("num_envs", 1) or 1))

    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    if num_envs > 1:
        env = make_env_or_vec(make_monitored_env_factory(env_id, wrapper_specs), num_envs=num_envs)
    else:
        env = _make_env(env_id, wrapper_specs, render=False)
    preview_env = _make_env(env_id, wrapper_specs, render=False)
    obs_space, action_space = venv_obs_space(env), venv_action_space(env)
    obs_shape = _obs_shape_of(obs_space)
    action_dim = action_flat_dim(action_space)

    built = build_world_model(kind, obs_space, action_space, model_config)
    if kind == "vae_mdnrnn":
        vae, mdnrnn = built["vae"].to(device), built["mdnrnn"].to(device)
        model_for_checkpoint: Any = {"vae": vae, "mdnrnn": mdnrnn}
        optimizer = torch.optim.Adam(list(vae.parameters()) + list(mdnrnn.parameters()), lr=learning_rate)
    else:
        model_for_checkpoint = built.to(device)
        optimizer = torch.optim.Adam(model_for_checkpoint.parameters(), lr=learning_rate)

    needs_sequences = kind in ("rssm", "vae_mdnrnn")
    seq_buffer = SequenceReplayBuffer(max(total_steps, 4_000), obs_shape, action_dim, num_lanes=num_envs) if needs_sequences else None
    transition_buffer = ContinuousReplayBuffer(max(total_steps, 10_000), obs_shape, action_dim) if kind == "ensemble" else None

    state: dict[str, Any] = {"step": 0, "last_write": 0, "last_preview": 0, "extra_metrics": {}, "last_gif": None, "last_gif_step": 0}
    start_time = time.time()

    def write_snapshot(step: int, status: str) -> None:
        elapsed = time.time() - start_time
        snapshot = {
            "run_id": run_dir.name, "kind": "world_model", "status": status,
            "algo": kind, "env_id": env_id, "step": int(step), "total_timesteps": int(total_steps),
            "fps": round(step / elapsed, 1) if elapsed > 0 else 0, "elapsed_seconds": round(elapsed, 1),
        }
        snapshot.update(state["extra_metrics"])
        if state["last_gif"]:
            snapshot["episode_gif_file"] = "episode_preview.gif"
            snapshot["episode_gif_step"] = state["last_gif_step"]
        snapshot = json_safe(snapshot)
        append_history(run_dir, snapshot)
        (run_dir / "metrics.json").write_text(json.dumps(snapshot))

    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    wm_store.write_world_model_snapshot(run_dir, {"type": kind, "config": model_config, "slug": slug})

    obs_cursor: Any = None
    write_snapshot(0, "running")
    status = "completed"
    try:
        while state["step"] < total_steps:
            obs_cursor, collect_stats = collect_rollout_auto(
                env, action_space, collect_per_iter,
                sequence_buffer=seq_buffer, transition_buffer=transition_buffer, obs=obs_cursor,
            )
            # `collect_rollout_auto` rounds down to a whole number of steps
            # *per lane* internally (see its own docstring) — mirror that
            # here so `state["step"]`/FPS reflect what was actually
            # collected rather than the raw (possibly not evenly
            # divisible by `num_envs`) request.
            state["step"] += max(1, collect_per_iter // num_envs) * num_envs

            enough_data = (
                (seq_buffer is not None and seq_buffer.num_episodes >= 2 and len(seq_buffer) >= seq_len)
                or (transition_buffer is not None and len(transition_buffer) >= max(batch_size, 256))
            )
            if enough_data:
                for _ in range(train_steps_per_iter):
                    if kind == "rssm":
                        losses = train_step_rssm(model_for_checkpoint, optimizer, seq_buffer, batch_size, seq_len, device)
                    elif kind == "ensemble":
                        losses = train_step_ensemble(model_for_checkpoint, optimizer, transition_buffer, batch_size, device)
                    else:
                        losses = train_step_vae_mdnrnn(vae, mdnrnn, optimizer, seq_buffer, batch_size, seq_len, device)
                state["extra_metrics"].update(losses)

            if collect_stats["mean_reward"] is not None:
                state["extra_metrics"]["collect_mean_reward"] = collect_stats["mean_reward"]

            if state["step"] - state["last_write"] >= _WRITE_EVERY_STEPS:
                state["last_write"] = state["step"]
                write_snapshot(state["step"], "running")

            if enough_data and state["step"] - state["last_preview"] >= _PREVIEW_EVERY_STEPS:
                state["last_preview"] = state["step"]
                try:
                    if kind == "rssm":
                        gif, scatter = _preview_rssm(model_for_checkpoint, preview_env, action_space, obs_space, device)
                        (run_dir / "latent_space.png").write_bytes(scatter)
                    elif kind == "ensemble":
                        gif = _preview_ensemble(model_for_checkpoint, preview_env, action_space, obs_space, device)
                    else:
                        gif, scatter = _preview_vae_mdnrnn(vae, mdnrnn, preview_env, action_space, obs_space, device)
                        (run_dir / "latent_space.png").write_bytes(scatter)
                    (run_dir / "episode_preview.gif").write_bytes(gif)
                    state["last_gif"] = True
                    state["last_gif_step"] = state["step"]
                except Exception:  # noqa: BLE001 - a broken preview must never take down training itself
                    pass

            if (run_dir / "stop.flag").exists():
                status = "stopped"
                break
    finally:
        save_checkpoint(kind, model_for_checkpoint, run_dir / "model.pt", model_config)
        if slug:
            try:
                wm_store.attach_checkpoint(slug, run_dir / "model.pt", source_run_id=run_dir.name)
            except FileNotFoundError:
                pass
        env.close()
        preview_env.close()
    write_snapshot(state["step"], status)
