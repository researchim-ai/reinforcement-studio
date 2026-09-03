"""Shared driver for any `CustomAlgorithm` (from-scratch, no SB3) run —
used both by `native_runner.py` (our own default PPO/DQN/A2C) and by
`custom_runner.py` (user-authored plugins that subclass `CustomAlgorithm`
instead of an SB3 `BaseAlgorithm`). Mirrors the metrics.json / stop.flag /
live-frame contract that `MetricsCallback` implements for SB3 runs, so the
Training Monitor page treats every kind of run identically.
"""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable, Union

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.metrics_callback import render_episode, persist_episode_gif
from rl_core.algorithms.resume import resolve_resume_source
from rl_core.algorithms.sb3_runner import _make_env, _space_info, make_monitored_env_factory
from rl_core.algorithms.vec_env import action_space, is_vector_env, make_env_or_vec, num_envs_of, obs_space
from rl_core.device import resolve_device
from rl_core.envs.factory import is_shared_world_env_id, make_training_env
from rl_core import scene_store
from rl_core.inspect import _count_params, _describe_layers, _find_torch_module
from rl_core.metrics_history import append_history, json_safe
from rl_core.netbuilder_store import resolve_network_spec, write_network_snapshot
from rl_core.world_models import store as wm_store

_WRITE_EVERY_STEPS = 500
_RENDER_EVERY_STEPS = 2000
# `writer()` below runs once per env step (i.e. up to thousands of times a
# second for fast/cheap envs), and its return value is the *only* signal
# `learn()` loops check to honor the Stop button — so unlike the write/
# render throttles above, this can't just be "every N calls" on a fixed
# cadence measured in wall-clock terms, but it also doesn't need checking
# every single step: a filesystem `stat()` every step measurably adds up
# over a long run, while a `stop.flag` a few dozen steps stale is still
# well under a second of extra delay for any env in this app.
_STOP_CHECK_EVERY_STEPS = 50


def run_custom_algorithm(
    cls: type[CustomAlgorithm],
    hyperparams: dict[str, Any],
    config: dict[str, Any],
    run_dir: Path,
    algo_label: str,
    policy_label: Union[str, Callable[[Any], str]] = "custom",
) -> None:
    """Drives any `CustomAlgorithm` subclass through the standard
    Monitor + metrics.json pipeline that the rest of the app expects.
    `policy_label` may be a plain string or a `(observation_space) -> str`
    callable (used by native_runner.py to report "MlpPolicy"/"CnnPolicy"
    without instantiating a second throwaway env)."""
    env_cfg = config.get("environment", {})
    training_cfg = config.get("training", {})
    env_id = env_cfg["id"]
    wrapper_specs = env_cfg.get("wrappers", [])
    total_timesteps = int(training_cfg.get("total_timesteps", 50_000))
    seed = training_cfg.get("seed")
    device = resolve_device(training_cfg)
    num_envs = max(1, int(training_cfg.get("num_envs", 1) or 1))
    is_scene = is_shared_world_env_id(env_id)

    if is_scene:
        # One shared world — lane count comes from the scene spec (or, for
        # a `petting:` benchmark, the env's own fixed agent count), not
        # `training.num_envs` (which would mean independent copies elsewhere).
        train_env = make_training_env(env_id, wrapper_specs)
        num_envs = num_envs_of(train_env)
    else:
        train_env = make_env_or_vec(
            make_monitored_env_factory(env_id, wrapper_specs), num_envs=num_envs,
        )
    if seed is not None and not is_vector_env(train_env):
        train_env.reset(seed=int(seed))

    # `network_spec` (a hand-designed architecture from the Network
    # Builder) is merged into a *copy* of hyperparams used only for
    # constructing the algorithm — kept out of `hyperparams` itself so it
    # never leaks into the "Гиперпараметры алгоритма" display (which
    # renders every key as a plain value) or gets clamped by the numeric
    # hyperparam UI. NativePPO/NativeA2C/NativeDQN check for this key
    # themselves (see rl_core/algorithms/native/{ppo,a2c,dqn}.py); a plugin
    # CustomAlgorithm subclass that doesn't know about it just ignores it.
    network_spec = resolve_network_spec(config)
    construct_hyperparams = {**hyperparams, "network_spec": network_spec} if network_spec else hyperparams

    # `world_model_spec` — same "resolve once, merge into construct-time
    # hyperparams only" convention as `network_spec` just above, for any of
    # the four algorithms (dreamer/mbpo/pets/world_models_ha, see
    # `rl_core/algorithms/native_runner.py::WORLD_MODEL_TYPE_FOR_ALGO`) that
    # can be pointed at a saved World Model via `algorithm.world_model_id`
    # (or an inline `algorithm.world_model_spec`) instead of always building
    # a fresh, untrained one of their own default architecture.
    world_model_spec = wm_store.resolve_world_model_spec(config)
    if world_model_spec:
        construct_hyperparams = {**construct_hyperparams, "world_model_spec": world_model_spec}

    # `training.resume_from` — fine-tune/continue-training from a previous
    # run's or Model Zoo checkpoint's weights instead of a fresh network
    # (see rl_core/algorithms/resume.py). The loaded algorithm keeps *its
    # own* saved hyperparams/architecture — the Designer only lets you
    # tweak `total_timesteps`/`num_envs`/`seed` while resuming, exactly
    # because the network shape can't change once weights are loaded into
    # it, so `hyperparams`/`network_spec` below are overwritten from
    # `algo.hyperparams` (what was actually saved) rather than whatever
    # this new run's own config happened to carry.
    resume_cfg = training_cfg.get("resume_from")
    if resume_cfg:
        algo_id_for_check = config.get("algorithm", {}).get("id", algo_label)
        model_path = resolve_resume_source(resume_cfg, algo_id_for_check)
        load_accepts_device = "device" in inspect.signature(cls.load).parameters
        algo = cls.load(model_path, train_env, device=device) if load_accepts_device else cls.load(model_path, train_env)
        network_spec = algo.hyperparams.get("network_spec")
        world_model_spec = algo.hyperparams.get("world_model_spec")
        hyperparams = {k: v for k, v in algo.hyperparams.items() if k not in ("network_spec", "world_model_spec")}
    else:
        algo = cls(train_env, construct_hyperparams, seed, device)
    write_network_snapshot(run_dir, config, network_spec)
    wm_store.write_world_model_snapshot(run_dir, world_model_spec)

    # Memory-enabled (LSTM/GRU) algorithms need to know when a live-preview
    # rollout starts over so they reset hidden state instead of carrying
    # one over from the last preview (see `render_episode`'s docstring).
    # `cls` here can be a user-authored plugin predating this parameter, so
    # only pass it through if the algorithm's own `predict()` actually
    # declares it — anything else keeps working exactly as before.
    _predict_accepts_episode_start = "episode_start" in inspect.signature(algo.predict).parameters

    def _predict(obs: Any, episode_start: bool) -> tuple[Any, Any]:
        if _predict_accepts_episode_start:
            return algo.predict(obs, deterministic=True, episode_start=episode_start)
        return algo.predict(obs, deterministic=True)

    resolved_policy_label = policy_label(obs_space(train_env)) if callable(policy_label) else policy_label
    static_info = {
        "policy": resolved_policy_label,
        "device": device,
        "hyperparams": hyperparams,
        "wrappers": wrapper_specs,
        "seed": seed,
        "num_envs": num_envs,
        "observation_space": _space_info(obs_space(train_env)),
        "action_space": _space_info(action_space(train_env)),
    }
    if resume_cfg:
        static_info["resumed_from"] = resume_cfg
    # Best-effort network introspection so the Training Monitor's "Схема
    # алгоритма" card can draw the actual layer shapes for native PPO/DQN/A2C
    # and from-scratch plugins too, not just the SB3 path (_run_sb3 already
    # has this) — reuses the exact same helpers the pre-run Designer
    # inspector uses (rl_core/inspect.py), just pointed at the real trained
    # module instead of a throwaway one.
    module = _find_torch_module(algo)
    if module is not None:
        total_params, _ = _count_params(module)
        static_info["total_params"] = total_params
        static_info["layers"] = _describe_layers(module)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    start_time = time.time()
    state: dict[str, Any] = {
        "last_write": 0, "last_render": 0, "last_stop_check": 0, "stop_requested": False, "step": 0, "last_gif": None,
        "last_gif_step": 0,
        # Cumulative count of episodes that have reached terminated/truncated
        # so far, across every lane — a running total (unlike
        # `episode_reward_mean`'s trailing-100 window), incremented once per
        # `writer(episode_reward=...)` call below since `on_policy.py`/every
        # native algorithm's own collection loop calls `callback.on_step`
        # exactly once per lane that finished this real step (never a batch
        # of several at once), so "one call with a non-`None` episode_reward"
        # and "one completed episode" are the same event here.
        "episodes_completed": 0,
        # Anything algorithms report via `TrainingCallback.on_step(metrics=...)`
        # that isn't one of the special episode_*_reward keys handled below —
        # exploration stats (epsilon, RND bonus/loss) as well as training
        # losses (policy_loss, value_loss, td_loss, ...) all flow through here
        # generically, so any algorithm (built-in or a user plugin) that
        # starts reporting a new metric key gets it in the snapshot/chart for
        # free, no changes needed on this end.
        "extra_metrics": {},
    }
    recent_rewards: list[float] = []
    recent_extrinsic_rewards: list[float] = []
    recent_intrinsic_rewards: list[float] = []
    recent_lengths: list[float] = []

    def write_snapshot(step: int, status: str) -> None:
        elapsed = time.time() - start_time
        just_rendered = False
        if step - state["last_render"] >= _RENDER_EVERY_STEPS:
            state["last_render"] = step
            # Only overwrite `last_gif` on a *successful* render — a
            # transient failure (env recreation hiccup, ...) should keep
            # showing the previous episode rather than blanking the preview.
            gif = render_episode(
                lambda: make_training_env(env_id, wrapper_specs, render=True) if is_scene
                else _make_env(env_id, wrapper_specs, render=True),
                _predict,
            )
            if gif:
                state["last_gif"] = gif
                state["last_gif_step"] = step
                persist_episode_gif(run_dir, gif, step)
                just_rendered = True
        snapshot = {
            "run_id": run_dir.name,
            "kind": "gym",
            "status": status,
            "algo": algo_label,
            "env_id": env_id,
            "step": int(step),
            "total_timesteps": int(total_timesteps),
            "episode_reward_mean": (sum(recent_rewards[-100:]) / len(recent_rewards[-100:])) if recent_rewards else None,
            "episode_extrinsic_reward_mean": (
                sum(recent_extrinsic_rewards[-100:]) / len(recent_extrinsic_rewards[-100:])
            ) if recent_extrinsic_rewards else None,
            "episode_intrinsic_reward_mean": (
                sum(recent_intrinsic_rewards[-100:]) / len(recent_intrinsic_rewards[-100:])
            ) if recent_intrinsic_rewards else None,
            "episode_length_mean": (sum(recent_lengths[-100:]) / len(recent_lengths[-100:])) if recent_lengths else None,
            "episodes_completed": state["episodes_completed"],
            "fps": round(step / elapsed, 1) if elapsed > 0 else 0,
            "elapsed_seconds": round(elapsed, 1),
        }
        snapshot.update(static_info)
        snapshot.update(state["extra_metrics"])
        if state["last_gif"]:
            # `episode_gif_file`/`episode_gif_step` (a plain path + int) are
            # cheap to re-attach to every snapshot — the Training Monitor
            # uses them to fetch/cache-bust `GET .../preview.gif` whenever
            # the (up to a few MB) base64 blob itself isn't present. That
            # blob is only ever included on the exact write that just
            # captured it (or the run's very last write, as a
            # belt-and-suspenders fallback in case serving the file back
            # ever fails) — every write in between would otherwise
            # re-serialize the exact same bytes into metrics.json/the
            # metrics WebSocket for no benefit, up to `_RENDER_EVERY_STEPS /
            # _WRITE_EVERY_STEPS` times as often as the episode it shows
            # actually changes.
            snapshot["episode_gif_file"] = "episode_preview.gif"
            snapshot["episode_gif_step"] = state["last_gif_step"]
            if just_rendered or status != "running":
                snapshot["episode_gif_base64"] = state["last_gif"]
        snapshot = json_safe(snapshot)
        append_history(run_dir, snapshot)
        (run_dir / "metrics.json").write_text(json.dumps(snapshot))

    def writer(
        num_timesteps: int,
        episode_reward: float | None,
        episode_length: float | None,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        state["step"] = num_timesteps
        if metrics:
            state["extra_metrics"].update({
                key: value for key, value in metrics.items()
                if key not in {"episode_extrinsic_reward", "episode_intrinsic_reward"}
            })
            if "episode_extrinsic_reward" in metrics:
                recent_extrinsic_rewards.append(float(metrics["episode_extrinsic_reward"]))
            if "episode_intrinsic_reward" in metrics:
                recent_intrinsic_rewards.append(float(metrics["episode_intrinsic_reward"]))
        if episode_reward is not None:
            recent_rewards.append(episode_reward)
            state["episodes_completed"] += 1
        if episode_length is not None:
            recent_lengths.append(episode_length)
        if num_timesteps - state["last_write"] >= _WRITE_EVERY_STEPS:
            state["last_write"] = num_timesteps
            write_snapshot(num_timesteps, "running")
        if not state["stop_requested"] and num_timesteps - state["last_stop_check"] >= _STOP_CHECK_EVERY_STEPS:
            state["last_stop_check"] = num_timesteps
            state["stop_requested"] = (run_dir / "stop.flag").exists()
        return not state["stop_requested"]

    write_snapshot(0, "running")
    status = "completed"
    try:
        algo.learn(total_timesteps=total_timesteps, callback=TrainingCallback(writer))
        if (run_dir / "stop.flag").exists():
            status = "stopped"
    finally:
        algo.save(run_dir / "model.zip")
        # If this run trained one of the four world-model algorithms
        # against a *saved* World Model (`world_model_spec["slug"]` set —
        # an inline one has none to attach to), copy the world model's own
        # trained weights (not the whole algorithm checkpoint above, which
        # also has an actor/critic/controller bolted on) back into
        # `CUSTOM_WORLD_MODELS_DIR`, mirroring what `world_models/trainer.py`
        # does for standalone `kind: "world_model"` runs. `save_world_model_
        # checkpoint` is a duck-typed extra method, not part of the
        # `CustomAlgorithm` contract itself — every plain SAC/PPO/... plugin
        # (and any world-model algorithm run without a `world_model_id`
        # at all) simply doesn't have it.
        save_world_model = getattr(algo, "save_world_model_checkpoint", None)
        if world_model_spec and world_model_spec.get("slug") and save_world_model is not None:
            try:
                wm_checkpoint_path = run_dir / "world_model.pt"
                save_world_model(wm_checkpoint_path)
                wm_store.attach_checkpoint(world_model_spec["slug"], wm_checkpoint_path, source_run_id=run_dir.name)
            except Exception:  # noqa: BLE001 - a broken attach must never fail an otherwise-successful run
                pass
        train_env.close()

    write_snapshot(state["step"], status)
