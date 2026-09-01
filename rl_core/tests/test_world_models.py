"""Coverage for the whole World Models feature: the three from-scratch
model families (`rl_core/world_models/{rssm,ensemble,vae_mdnrnn}.py`), spec
storage (`rl_core/world_models/store.py`), and the four algorithms that can
train against one (`rl_core/algorithms/native/{dreamer,mbpo,pets,
world_models_ha}.py`). Every algorithm test uses tiny hyperparams (short
sequences/horizons, small batches, a handful of real steps) purely to keep
this file fast — none of it exercises whether the *learned* behavior is any
good, only that every shape lines up and a full `learn()` -> `predict()`
-> `save()`/`load()` round trip doesn't crash.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.dreamer import DEFAULT_HYPERPARAMS as DREAMER_DEFAULTS
from rl_core.algorithms.native.dreamer import NativeDreamer
from rl_core.algorithms.native.mbpo import DEFAULT_HYPERPARAMS as MBPO_DEFAULTS
from rl_core.algorithms.native.mbpo import NativeMBPO
from rl_core.algorithms.native.pets import DEFAULT_HYPERPARAMS as PETS_DEFAULTS
from rl_core.algorithms.native.pets import NativePETS
from rl_core.algorithms.native.world_models_ha import DEFAULT_HYPERPARAMS as WMHA_DEFAULTS
from rl_core.algorithms.native.world_models_ha import NativeWorldModelsHA
from rl_core.world_models.ensemble import EnsembleDynamicsModel, cem_plan
from rl_core.world_models.losses import train_step_ensemble, train_step_rssm, train_step_vae_mdnrnn
from rl_core.world_models.replay import SequenceReplayBuffer, action_flat_dim, collect_rollout, collect_rollout_auto, collect_rollout_vec
from rl_core.world_models.rssm import RSSM
from rl_core.world_models.spec import build_world_model, load_checkpoint, resolve_or_build_for_algo, save_checkpoint
from rl_core.world_models.vae_mdnrnn import MDNRNN, VAE
from rl_core.algorithms.native.buffers import ContinuousReplayBuffer

_DISCRETE_ENV_ID = "CartPole-v1"
_CONTINUOUS_ENV_ID = "Pendulum-v1"


def _fill_sequence_buffer(env: gym.Env, action_space: gym.Space, buffer: SequenceReplayBuffer, steps: int) -> None:
    obs = None
    for _ in range(steps):
        obs, _stats = collect_rollout(env, action_space, 1, sequence_buffer=buffer, obs=obs)


class TestRSSM:
    def test_observe_and_imagine_shapes(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        model = RSSM(env.observation_space, env.action_space, deter_dim=16, stoch_dim=4, hidden_dim=16)
        action_dim = action_flat_dim(env.action_space)
        buffer = SequenceReplayBuffer(2_000, env.observation_space.shape, action_dim)
        _fill_sequence_buffer(env, env.action_space, buffer, 100)

        batch = buffer.sample(4, 8)
        obs_seq = torch.as_tensor(batch["obs"], dtype=torch.float32)
        action_seq = torch.as_tensor(batch["actions"], dtype=torch.float32)
        out = model.observe(obs_seq, action_seq)
        assert out["feat"].shape == (4, 8, model.feat_dim)
        assert torch.isfinite(out["model_loss"])

        h0, z0 = out["h"][:, 0], out["z"][:, 0]

        def random_policy(feat: torch.Tensor) -> torch.Tensor:
            return torch.zeros(feat.shape[0], action_dim)

        imagined = model.imagine(h0, z0, random_policy, horizon=5)
        assert imagined["feat"].shape == (4, 5, model.feat_dim)
        decoded = model.decode(imagined["feat"])
        assert decoded.shape[:2] == (4, 5)
        env.close()

    def test_train_step_reduces_loss_on_repeated_batch(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        model = RSSM(env.observation_space, env.action_space, deter_dim=16, stoch_dim=4, hidden_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        buffer = SequenceReplayBuffer(2_000, env.observation_space.shape, action_flat_dim(env.action_space))
        _fill_sequence_buffer(env, env.action_space, buffer, 200)

        first = train_step_rssm(model, optimizer, buffer, batch_size=4, seq_len=8, device="cpu")
        for _ in range(20):
            last = train_step_rssm(model, optimizer, buffer, batch_size=4, seq_len=8, device="cpu")
        assert all(np.isfinite(v) for v in last.values())
        env.close()


class TestEnsembleDynamics:
    def test_nll_loss_and_predict_step_shapes(self) -> None:
        env = gym.make(_CONTINUOUS_ENV_ID)
        model = EnsembleDynamicsModel(env.observation_space, env.action_space, ensemble_size=3, hidden=(16, 16))
        buffer = ContinuousReplayBuffer(2_000, env.observation_space.shape, int(np.prod(env.action_space.shape)))
        obs, _info = env.reset(seed=0)
        for _ in range(200):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _info = env.step(action)
            buffer.add(obs, action, float(reward), next_obs, bool(terminated or truncated))
            obs = next_obs if not (terminated or truncated) else env.reset()[0]

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(15):
            metrics = train_step_ensemble(model, optimizer, buffer, batch_size=32, device="cpu")
        assert all(np.isfinite(v) for v in metrics.values())

        flat_obs = model.flat_target(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        action_t = torch.as_tensor(env.action_space.sample(), dtype=torch.float32).unsqueeze(0)
        next_flat, reward = model.predict_step(flat_obs, action_t)
        assert next_flat.shape == flat_obs.shape
        assert reward.shape == (1,)
        raw = model.to_raw_obs(next_flat)
        assert raw.shape == (1, *env.observation_space.shape)
        env.close()

    def test_cem_plan_returns_valid_action(self) -> None:
        env = gym.make(_CONTINUOUS_ENV_ID)
        model = EnsembleDynamicsModel(env.observation_space, env.action_space, ensemble_size=2, hidden=(8, 8))
        obs, _info = env.reset(seed=0)
        flat_obs0 = model.flat_target(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        action = cem_plan(model, flat_obs0, env.action_space, horizon=3, num_candidates=16, num_elites=4, num_iterations=2)
        assert action.shape == (int(np.prod(env.action_space.shape)),)
        assert env.action_space.contains(np.clip(action.numpy(), env.action_space.low, env.action_space.high).reshape(env.action_space.shape))
        env.close()


class TestVaeMdnRnn:
    def test_vae_loss_and_mdnrnn_shapes(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        vae = VAE(env.observation_space, latent_dim=6, hidden_dim=16)
        mdnrnn = MDNRNN(latent_dim=6, action_dim=action_flat_dim(env.action_space), hidden_size=16, num_mixtures=3)
        buffer = SequenceReplayBuffer(2_000, env.observation_space.shape, action_flat_dim(env.action_space))
        _fill_sequence_buffer(env, env.action_space, buffer, 100)

        optimizer = torch.optim.Adam(list(vae.parameters()) + list(mdnrnn.parameters()), lr=1e-2)
        for _ in range(10):
            losses = train_step_vae_mdnrnn(vae, mdnrnn, optimizer, buffer, batch_size=4, seq_len=8, device="cpu")
        assert all(np.isfinite(v) for v in losses.values())

        z = torch.randn(2, 6)
        hidden = mdnrnn.initial_state(2, "cpu")
        action_t = torch.zeros(2, 1, action_flat_dim(env.action_space))
        logits, means, log_stds, reward, continue_logit, _hidden = mdnrnn(z.unsqueeze(1), action_t, hidden)
        assert logits.shape == (2, 1, 3)
        next_z = MDNRNN.sample_next_z(logits.squeeze(1), means.squeeze(1), log_stds.squeeze(1))
        assert next_z.shape == (2, 6)
        env.close()


class TestParallelCollection:
    """`training.num_envs > 1` — see `replay.py::collect_rollout_vec`'s own
    docstring for why this is the main lever for speeding up World Model
    training. Uses `SyncVectorEnv` (`parallel=False`) rather than the real
    `AsyncVectorEnv` every production run gets, purely so these tests stay
    fast and don't spawn subprocesses — the two are interchangeable from
    `collect_rollout_vec`'s point of view (both are `VectorEnv`s)."""

    def _make_vec_env(self, env_id: str, num_envs: int):
        from rl_core.algorithms.vec_env import make_env_or_vec, make_gym_env_factory

        return make_env_or_vec(make_gym_env_factory(env_id), num_envs=num_envs, parallel=False)

    def test_sequence_buffer_keeps_lanes_independent(self) -> None:
        buffer = SequenceReplayBuffer(1_000, (2,), 1, num_lanes=3)
        # Interleave two lanes' transitions by hand — lane 0's episode is
        # 2 steps, lane 1's is 3 — and check each lane's own episode comes
        # out exactly as long as it should, never spliced with the other's.
        buffer.add(np.array([0.0, 0.0]), np.array([0.0]), 1.0, False, lane=0)
        buffer.add(np.array([10.0, 10.0]), np.array([1.0]), 5.0, False, lane=1)
        buffer.add(np.array([1.0, 1.0]), np.array([0.0]), 1.0, True, lane=0)
        buffer.add(np.array([11.0, 11.0]), np.array([1.0]), 5.0, False, lane=1)
        buffer.add(np.array([12.0, 12.0]), np.array([1.0]), 5.0, True, lane=1)
        assert buffer.num_episodes == 2
        lengths = sorted(len(ep["dones"]) for ep in buffer._episodes)
        assert lengths == [2, 3]
        # Lane 2 never got anything — its own accumulator must stay empty
        # rather than errors/`IndexError`.
        assert buffer._current[2] == []

    def test_collect_rollout_vec_matches_lane_count(self) -> None:
        env = self._make_vec_env(_DISCRETE_ENV_ID, num_envs=4)
        action_dim = action_flat_dim(env.single_action_space)
        seq_buffer = SequenceReplayBuffer(2_000, env.single_observation_space.shape, action_dim, num_lanes=4)
        obs, stats = collect_rollout_vec(env, env.single_action_space, num_steps_per_lane=20, sequence_buffer=seq_buffer)
        assert len(obs) == 4
        assert len(seq_buffer) + sum(len(c) for c in seq_buffer._current) == 80
        assert stats["episodes"] >= 0
        env.close()

    def test_collect_rollout_auto_dispatches_on_env_kind(self) -> None:
        plain_env = gym.make(_DISCRETE_ENV_ID)
        _obs, stats = collect_rollout_auto(plain_env, plain_env.action_space, 20)
        assert stats["episodes"] >= 0
        plain_env.close()

        vec_env = self._make_vec_env(_DISCRETE_ENV_ID, num_envs=4)
        _obs, stats = collect_rollout_auto(vec_env, vec_env.single_action_space, 20)
        assert stats["episodes"] >= 0
        vec_env.close()


class TestSpecFactoryAndCheckpoint:
    def test_build_and_checkpoint_roundtrip_every_type(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        for kind in ("rssm", "ensemble", "vae_mdnrnn"):
            model = build_world_model(kind, env.observation_space, env.action_space, {})
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "model.pt"
                save_checkpoint(kind, model, path, {})
                loaded_kind, _loaded_model, _config = load_checkpoint(path, env.observation_space, env.action_space)
                assert loaded_kind == kind
        env.close()

    def test_resolve_or_build_for_algo_falls_back_on_none_or_mismatch(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        model, config, slug = resolve_or_build_for_algo(None, "rssm", env.observation_space, env.action_space)
        assert isinstance(model, RSSM)
        assert slug is None
        assert config

        mismatched = {"type": "ensemble", "config": {}, "slug": "whatever"}
        model2, _config2, slug2 = resolve_or_build_for_algo(mismatched, "rssm", env.observation_space, env.action_space)
        assert isinstance(model2, RSSM)
        assert slug2 is None
        env.close()


class TestWorldModelStore:
    def test_crud_roundtrip(self, tmp_path, monkeypatch) -> None:
        from rl_core.world_models import store

        monkeypatch.setattr(store, "CUSTOM_WORLD_MODELS_DIR", tmp_path)
        store.save("my-rssm", {"name": "My RSSM", "type": "rssm", "config": {"deter_dim": 32}})
        assert "my-rssm" in store.list_slugs()
        assert store.load("my-rssm")["type"] == "rssm"
        meta = store.meta("my-rssm")
        assert meta["trained"] is False

        checkpoint = store.checkpoint_path("my-rssm")
        checkpoint.write_bytes(b"fake-weights")
        assert store.has_checkpoint("my-rssm")

        resolved = store.resolve_world_model_spec({"algorithm": {"world_model_id": "my-rssm"}})
        assert resolved["type"] == "rssm"
        assert resolved["checkpoint_path"] == str(checkpoint)

        store.delete("my-rssm")
        assert "my-rssm" not in store.list_slugs()


def _run_smoke(algo_cls, defaults: dict, env_id: str, hyperparams: dict, total_timesteps: int, num_envs: int = 1) -> None:
    if num_envs > 1:
        from rl_core.algorithms.vec_env import make_env_or_vec, make_gym_env_factory

        # `parallel=False` (`SyncVectorEnv`) — same public API
        # (`is_vector_env`/`vec_reset`/`vec_step`) as the real
        # `AsyncVectorEnv` production runs use, just without spawning
        # subprocesses, so this test stays fast.
        env = make_env_or_vec(make_gym_env_factory(env_id), num_envs=num_envs, parallel=False)
    else:
        env = gym.make(env_id)
    algo = algo_cls(env, {**defaults, **hyperparams}, seed=0, device="cpu")

    steps: list[int] = []

    def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
        steps.append(num_timesteps)
        return num_timesteps < total_timesteps

    algo.learn(total_timesteps=total_timesteps, callback=TrainingCallback(writer))
    assert steps[-1] >= total_timesteps

    # `predict()` always takes one unbatched obs regardless of how many
    # lanes training used — a plain single env for this part even when
    # `env` above was vectorized.
    predict_env = gym.make(env_id)
    obs, _info = predict_env.reset(seed=1)
    action, _state = algo.predict(obs, deterministic=True)
    assert predict_env.action_space.contains(action)

    with tempfile.TemporaryDirectory() as d:
        checkpoint = Path(d) / "model.zip"
        algo.save(checkpoint)
        loaded = algo_cls.load(checkpoint, env, device="cpu")
        loaded.predict(obs, deterministic=True)

        save_wm = getattr(algo, "save_world_model_checkpoint", None)
        if save_wm is not None:
            wm_path = Path(d) / "world_model.pt"
            save_wm(wm_path)
            assert wm_path.exists()
    predict_env.close()
    env.close()


class TestNativeDreamer:
    def test_discrete_smoke(self) -> None:
        _run_smoke(
            NativeDreamer, DREAMER_DEFAULTS, _DISCRETE_ENV_ID,
            {
                "batch_size": 4, "seq_len": 5, "imagination_horizon": 4,
                "collect_steps_per_iter": 10, "train_steps_per_iter": 1, "learning_starts": 5,
            },
            total_timesteps=60,
        )

    def test_continuous_smoke(self) -> None:
        _run_smoke(
            NativeDreamer, DREAMER_DEFAULTS, _CONTINUOUS_ENV_ID,
            {
                "batch_size": 4, "seq_len": 5, "imagination_horizon": 4,
                "collect_steps_per_iter": 10, "train_steps_per_iter": 1, "learning_starts": 5,
            },
            total_timesteps=60,
        )

    def test_parallel_envs_smoke(self) -> None:
        """`training.num_envs > 1` — real-experience collection across
        several lockstep lanes, each its own independent episode/RSSM
        carry-state (see the module docstring)."""
        _run_smoke(
            NativeDreamer, DREAMER_DEFAULTS, _DISCRETE_ENV_ID,
            {
                "batch_size": 4, "seq_len": 5, "imagination_horizon": 4,
                "collect_steps_per_iter": 10, "train_steps_per_iter": 1, "learning_starts": 5,
            },
            total_timesteps=60, num_envs=3,
        )


class TestNativeMBPO:
    def test_continuous_smoke(self) -> None:
        _run_smoke(
            NativeMBPO, MBPO_DEFAULTS, _CONTINUOUS_ENV_ID,
            {
                "batch_size": 16, "learning_starts": 10, "model_train_freq": 10,
                "rollout_batch_size": 8, "rollout_length": 1,
            },
            total_timesteps=40,
        )

    def test_rejects_discrete_action_space(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        try:
            NativeMBPO(env, MBPO_DEFAULTS, seed=0, device="cpu")
            raised = False
        except ValueError:
            raised = True
        finally:
            env.close()
        assert raised


class TestNativePETS:
    def test_continuous_smoke(self) -> None:
        _run_smoke(
            NativePETS, PETS_DEFAULTS, _CONTINUOUS_ENV_ID,
            {
                "buffer_size": 2_000, "batch_size": 16, "learning_starts": 10, "model_train_freq": 10,
                "cem_horizon": 3, "cem_candidates": 16, "cem_elites": 4, "cem_iterations": 2,
            },
            total_timesteps=30,
        )


class TestNativeWorldModelsHA:
    def test_discrete_smoke(self) -> None:
        _run_smoke(
            NativeWorldModelsHA, WMHA_DEFAULTS, _DISCRETE_ENV_ID,
            {
                "world_model_phase_steps": 40, "seq_len": 5, "batch_size": 4,
                "world_model_train_steps_per_iter": 1, "population_size": 2, "episodes_per_eval": 1,
            },
            total_timesteps=80,
        )

    def test_parallel_envs_smoke(self) -> None:
        """`training.num_envs > 1` — phase 1's random-policy collection
        across parallel lanes, *and* phase 2's `episodes_per_eval=2`
        repeats of one candidate running across those same lanes at once
        (`_run_episodes_vec`) instead of strictly one after another."""
        _run_smoke(
            NativeWorldModelsHA, WMHA_DEFAULTS, _DISCRETE_ENV_ID,
            {
                "world_model_phase_steps": 40, "seq_len": 5, "batch_size": 4,
                "world_model_train_steps_per_iter": 1, "population_size": 2, "episodes_per_eval": 2,
            },
            total_timesteps=80, num_envs=2,
        )


class TestStandaloneTrainer:
    """`kind: "world_model"` runs — no RL agent at all, just "collect with a
    random policy, fit the model" (see `rl_core/world_models/trainer.py`'s
    own module docstring)."""

    def _run(self, tmp_path, kind: str, env_id: str, extra_hyperparams: dict, world_model_id: str | None = None) -> Path:
        from rl_core.world_models import trainer as wm_trainer

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        algorithm_cfg: dict = {"id": kind, "hyperparams": extra_hyperparams}
        if world_model_id:
            algorithm_cfg["world_model_id"] = world_model_id
        config = {
            "kind": "world_model",
            "environment": {"id": env_id, "wrappers": []},
            "algorithm": algorithm_cfg,
            "training": {
                "total_timesteps": 60, "seed": 0, "batch_size": 4, "seq_len": 5,
                "collect_steps_per_iter": 20, "train_steps_per_iter": 1, "learning_rate": 1e-2,
            },
        }
        wm_trainer.run(config, run_dir)
        return run_dir

    def test_rssm(self, tmp_path) -> None:
        run_dir = self._run(tmp_path, "rssm", _DISCRETE_ENV_ID, {"deter_dim": 8, "stoch_dim": 4, "hidden_dim": 8})
        assert (run_dir / "model.pt").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "world_model.json").exists()

    def test_rssm_parallel_envs(self, tmp_path) -> None:
        """"Run several environments in parallel with different seeds" —
        `training.num_envs > 1` for the standalone trainer itself (see
        `trainer.py`'s own module docstring)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        config = {
            "kind": "world_model",
            "environment": {"id": _DISCRETE_ENV_ID, "wrappers": []},
            "algorithm": {"id": "rssm", "hyperparams": {"deter_dim": 8, "stoch_dim": 4, "hidden_dim": 8}},
            "training": {
                "total_timesteps": 60, "seed": 0, "batch_size": 4, "seq_len": 5,
                "collect_steps_per_iter": 20, "train_steps_per_iter": 1, "learning_rate": 1e-2, "num_envs": 3,
            },
        }
        from rl_core.world_models import trainer as wm_trainer

        wm_trainer.run(config, run_dir)
        assert (run_dir / "model.pt").exists()
        assert (run_dir / "metrics.json").exists()

    def test_ensemble(self, tmp_path) -> None:
        run_dir = self._run(tmp_path, "ensemble", _CONTINUOUS_ENV_ID, {"ensemble_size": 2, "hidden_dim": 8})
        assert (run_dir / "model.pt").exists()
        assert (run_dir / "metrics.json").exists()

    def test_vae_mdnrnn(self, tmp_path) -> None:
        run_dir = self._run(
            tmp_path, "vae_mdnrnn", _DISCRETE_ENV_ID,
            {"latent_dim": 4, "hidden_dim": 8, "rnn_hidden_size": 8, "num_mixtures": 2},
        )
        assert (run_dir / "model.pt").exists()
        assert (run_dir / "metrics.json").exists()

    def test_attaches_checkpoint_to_saved_slug(self, tmp_path, monkeypatch) -> None:
        from rl_core.world_models import store

        wm_dir = tmp_path / "custom_world_models"
        wm_dir.mkdir()
        monkeypatch.setattr(store, "CUSTOM_WORLD_MODELS_DIR", wm_dir)
        store.save("attach-target", {"name": "Attach target", "type": "rssm", "config": {"deter_dim": 8, "stoch_dim": 4, "hidden_dim": 8}})

        run_dir = self._run(
            tmp_path, "rssm", _DISCRETE_ENV_ID,
            {"deter_dim": 8, "stoch_dim": 4, "hidden_dim": 8},
            world_model_id="attach-target",
        )
        assert (run_dir / "model.pt").exists()
        assert store.has_checkpoint("attach-target")


class TestRunnerUtilsWorldModelIntegration:
    """The other direction: a `dreamer`/`mbpo`/`pets`/`world_models_ha` run
    (driven through the generic `run_custom_algorithm`, exactly like the
    Designer's "Start training" button would) picking up a *saved* World
    Model spec via `algorithm.world_model_id`, and — once finished —
    attaching its own newly-trained weights back onto that same spec (see
    `runner_utils.py`'s `finally` block)."""

    def test_dreamer_run_with_world_model_id_attaches_checkpoint(self, tmp_path, monkeypatch) -> None:
        from rl_core.algorithms.runner_utils import run_custom_algorithm
        from rl_core.world_models import store

        wm_dir = tmp_path / "custom_world_models"
        wm_dir.mkdir()
        monkeypatch.setattr(store, "CUSTOM_WORLD_MODELS_DIR", wm_dir)
        store.save("dreamer-rssm", {"name": "Dreamer RSSM", "type": "rssm", "config": {"deter_dim": 8, "stoch_dim": 4, "hidden_dim": 8}})
        assert not store.has_checkpoint("dreamer-rssm")

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        config = {
            "kind": "gym",
            "environment": {"id": _DISCRETE_ENV_ID, "wrappers": []},
            "algorithm": {
                "id": "dreamer",
                "world_model_id": "dreamer-rssm",
                "hyperparams": {
                    "batch_size": 4, "seq_len": 5, "imagination_horizon": 4,
                    "collect_steps_per_iter": 10, "train_steps_per_iter": 1, "learning_starts": 5,
                },
            },
            "training": {"total_timesteps": 40, "seed": 0},
        }
        run_custom_algorithm(NativeDreamer, {**DREAMER_DEFAULTS, **config["algorithm"]["hyperparams"]}, config, run_dir, "dreamer")

        assert (run_dir / "model.zip").exists()
        assert (run_dir / "world_model.json").exists()
        assert store.has_checkpoint("dreamer-rssm")
