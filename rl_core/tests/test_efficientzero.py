"""Coverage for `rl_core/algorithms/native/efficientzero.py` (EfficientZero
V2 — MuZero-family model-based planning, see that module's own docstring).
Same "tiny hyperparams, just check shapes/no-crash, not learned quality"
convention as `test_world_models.py`."""
from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.efficientzero import DEFAULT_HYPERPARAMS, NativeEfficientZero, _EfficientZeroBuffer
from rl_core.algorithms.native.preprocessing import obs_to_array
from rl_core.algorithms.vec_env import make_env_or_vec, make_gym_env_factory

_DISCRETE_ENV_ID = "CartPole-v1"
_CONTINUOUS_ENV_ID = "Pendulum-v1"

_TINY_DISCRETE = {
    "latent_dim": 8, "hidden_dim": 16, "proj_dim": 8, "buffer_size": 200,
    "batch_size": 4, "unroll_steps": 3, "td_steps": 3, "num_sampled_actions": 4,
    "num_simulations": 6, "num_top_actions": 4, "learning_starts": 10, "train_freq": 1,
}
_TINY_CONTINUOUS = {**_TINY_DISCRETE}


def _run_smoke(env_id: str, hyperparams: dict, total_timesteps: int, num_envs: int = 1) -> None:
    if num_envs == 1:
        env = gym.make(env_id)
    else:
        env = make_env_or_vec(make_gym_env_factory(env_id), num_envs=num_envs, parallel=False)
    algo = NativeEfficientZero(env, {**DEFAULT_HYPERPARAMS, **hyperparams}, seed=0, device="cpu")

    steps: list[int] = []

    def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
        steps.append(num_timesteps)
        return num_timesteps < total_timesteps

    algo.learn(total_timesteps=total_timesteps, callback=TrainingCallback(writer))
    assert steps[-1] >= total_timesteps

    predict_env = gym.make(env_id)
    obs, _info = predict_env.reset(seed=1)
    action, _state = algo.predict(obs, deterministic=True)
    assert predict_env.action_space.contains(action)

    with tempfile.TemporaryDirectory() as d:
        checkpoint = Path(d) / "model.pt"
        algo.save(checkpoint)
        loaded = NativeEfficientZero.load(checkpoint, gym.make(env_id), device="cpu")
        loaded.predict(obs, deterministic=True)
    predict_env.close()
    env.close()


class TestEfficientZeroBuffer:
    def test_sample_shapes_and_masking(self) -> None:
        buffer = _EfficientZeroBuffer(capacity_episodes=10, obs_shape=(4,), action_dim=2, policy_target_dim=2)
        # One 3-step episode: sampling from the last step with unroll_steps=3
        # must mask out steps 1 and 2 (nothing left in the episode there).
        for t in range(3):
            done = t == 2
            buffer.add(
                obs=np.full(4, float(t)), action_flat=np.array([1.0, 0.0]), reward=1.0,
                next_obs=np.full(4, float(t + 1)), policy_target=np.array([0.5, 0.5]), done=done,
            )
        assert buffer.num_episodes == 1
        assert len(buffer) == 3

        batch = buffer.sample(batch_size=8, unroll_steps=3, td_steps=3, gamma=0.9)
        assert batch["obs0"].shape == (8, 4)
        assert batch["action"].shape == (8, 3, 2)
        assert batch["mask"].shape == (8, 3)
        # Every sampled start index is one of {0, 1, 2}; whatever mask[b]
        # looks like, it must be a monotonically non-increasing run of 1s
        # then 0s (no "hole" of a valid step after an invalid one).
        for m in batch["mask"]:
            saw_zero = False
            for v in m:
                if v == 0.0:
                    saw_zero = True
                elif saw_zero:
                    raise AssertionError(f"mask has a 1 after a 0: {m}")

    def test_terminal_step_has_no_bootstrap(self) -> None:
        buffer = _EfficientZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=1)
        buffer.add(obs=np.array([0.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([1.0]), policy_target=np.array([1.0]), done=True)
        batch = buffer.sample(batch_size=4, unroll_steps=1, td_steps=5, gamma=0.9)
        # The only real step is terminal -> no bootstrap continues past it.
        assert np.all(batch["td_bootstrap_mask"][:, 0] == 0.0)
        assert np.allclose(batch["td_reward"][:, 0], 1.0)

    def test_lanes_do_not_splice_episodes(self) -> None:
        # `num_lanes=2` — each lane's in-progress episode must stay separate
        # until *that* lane's own `done=True`, even when interleaved.
        buffer = _EfficientZeroBuffer(capacity_episodes=10, obs_shape=(1,), action_dim=1, policy_target_dim=1, num_lanes=2)
        buffer.add(obs=np.array([0.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([1.0]), policy_target=np.array([1.0]), done=False, lane=0)
        buffer.add(obs=np.array([10.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([11.0]), policy_target=np.array([1.0]), done=True, lane=1)
        buffer.add(obs=np.array([1.0]), action_flat=np.array([0.0]), reward=1.0, next_obs=np.array([2.0]), policy_target=np.array([1.0]), done=True, lane=0)
        assert buffer.num_episodes == 2
        lane1_ep = next(ep for ep in buffer.episodes if ep["obs"].shape[0] == 1)
        lane0_ep = next(ep for ep in buffer.episodes if ep["obs"].shape[0] == 2)
        assert np.allclose(lane1_ep["obs"], [[10.0]])
        assert np.allclose(lane0_ep["obs"], [[0.0], [1.0]])


class TestNativeEfficientZeroDiscrete:
    def test_search_output_shapes(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeEfficientZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)

        obs_batch = obs_to_array(obs, env.observation_space)[None]
        results = algo.search(obs_batch, deterministic=np.array([False]))
        assert len(results) == 1
        result = results[0]
        assert result["policy_target"].shape == (env.action_space.n,)
        assert np.isclose(result["policy_target"].sum(), 1.0, atol=1e-4)
        assert 0 <= result["env_action"] < env.action_space.n
        env.close()

    def test_search_is_batched_across_lanes(self) -> None:
        env = gym.make(_DISCRETE_ENV_ID)
        algo = NativeEfficientZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_DISCRETE}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)
        one = obs_to_array(obs, env.observation_space)
        results = algo.search(np.stack([one, one, one]), deterministic=np.array([True, True, True]))
        assert len(results) == 3
        for result in results:
            assert result["policy_target"].shape == (env.action_space.n,)
            assert 0 <= result["env_action"] < env.action_space.n
        env.close()

    def test_smoke(self) -> None:
        _run_smoke(_DISCRETE_ENV_ID, _TINY_DISCRETE, total_timesteps=40)

    def test_smoke_parallel_envs(self) -> None:
        _run_smoke(_DISCRETE_ENV_ID, _TINY_DISCRETE, total_timesteps=40, num_envs=3)


class TestNativeEfficientZeroContinuous:
    def test_search_output_shapes(self) -> None:
        env = gym.make(_CONTINUOUS_ENV_ID)
        algo = NativeEfficientZero(env, {**DEFAULT_HYPERPARAMS, **_TINY_CONTINUOUS}, seed=0, device="cpu")
        obs, _info = env.reset(seed=0)

        obs_batch = obs_to_array(obs, env.observation_space)[None]
        results = algo.search(obs_batch, deterministic=np.array([False]))
        assert len(results) == 1
        result = results[0]
        action_dim = int(np.prod(env.action_space.shape))
        assert result["policy_target"].shape == (action_dim,)
        assert result["env_action"].shape == (action_dim,)
        env.close()

    def test_smoke(self) -> None:
        _run_smoke(_CONTINUOUS_ENV_ID, _TINY_CONTINUOUS, total_timesteps=40)

    def test_smoke_parallel_envs(self) -> None:
        _run_smoke(_CONTINUOUS_ENV_ID, _TINY_CONTINUOUS, total_timesteps=40, num_envs=3)
