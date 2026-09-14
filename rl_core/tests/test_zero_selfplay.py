"""Self-play Gym wrapper + Zero-family legal-action search on board games."""
from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.evaluate import _run_episodes_custom_algorithm
from rl_core.algorithms.metrics_callback import render_episode
from rl_core.algorithms.native.latentimzero import NativeLatentImZero
from rl_core.algorithms.native.efficientzero import (
    DEFAULT_HYPERPARAMS,
    NativeEfficientZero,
    _EfficientZeroBuffer,
)
from rl_core.algorithms.native.preprocessing import is_image_space, obs_to_array
from rl_core.algorithms.native.researchimzero import NativeResearchImZero, _ResearchImZeroBuffer
from rl_core.algorithms.native.unizero import (
    DEFAULT_HYPERPARAMS as UNIZERO_DEFAULT_HYPERPARAMS,
    NativeUniZero,
    _UniZeroBuffer,
)
from rl_core.algorithms.native.zero_selfplay import (
    discrete_expand_slots,
    record_self_play_outcomes,
    self_play_replay_capacity,
    self_play_outcome_metrics,
)
from rl_core.envs.board_game_gym import BoardGameSelfPlayEnv, register_board_game_envs, render_board_rgb
from rl_core.envs import registry

register_board_game_envs()

_TINY = {
    "latent_dim": 8, "hidden_dim": 16, "proj_dim": 8, "buffer_size": 32,
    "batch_size": 4, "unroll_steps": 2, "td_steps": 2, "num_sampled_actions": 4,
    "num_simulations": 4, "num_top_actions": 2, "learning_starts": 4, "train_freq": 1,
    "value_support_size": 10, "reanalyze_batch_size": 0,
}


class _TerminalObservationEnv(gym.Env):
    observation_space = gym.spaces.Box(-10.0, 10.0, (1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        del action
        return np.array([9.0], dtype=np.float32), 1.0, True, False, {}


def test_board_games_list_zero_family():
    envs = {entry["id"]: entry for entry in registry.list_environments()}
    for game_id in ("tic_tac_toe", "connect_four", "gomoku"):
        compatible = envs[game_id]["compatible_algorithms"]
        assert compatible[0] == "alphazero"
        assert "efficientzero" in compatible
        assert "unizero" in compatible
        assert "researchimzero" in compatible
        assert "latentimzero" in compatible


def test_tic_tac_toe_gym_self_play_spaces_and_mask():
    env = gym.make("tic_tac_toe")
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, 3, 3)
    assert obs.dtype == np.uint8
    assert is_image_space(env.observation_space)
    mask = np.asarray(info["action_mask"])
    assert mask.shape == (9,)
    assert int(mask.sum()) == 9
    assert info["two_player"] is True
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0
    assert not terminated
    assert int(np.asarray(info["action_mask"]).sum()) == 8
    env.close()


def test_self_play_terminal_reward_is_from_mover():
    env = BoardGameSelfPlayEnv("tic_tac_toe")
    env.reset(seed=0)
    # X: 0, O: 3, X: 1, O: 4, X: 2 — X wins on the last ply.
    for action in (0, 3, 1, 4):
        _obs, reward, terminated, _trunc, _info = env.step(action)
        assert reward == 0.0
        assert not terminated
    _obs, reward, terminated, _trunc, info = env.step(2)
    assert terminated
    assert reward == 1.0
    assert info["winner"] == 1
    assert info["game_done"] is True
    env.close()


def test_self_play_outcome_metrics_separate_wins_draws_and_time_limits():
    algo = SimpleNamespace(two_player=True)
    record_self_play_outcomes(
        algo,
        [
            {"final_info": {"winner": 1}},
            {"final_info": {"winner": -1}},
            {"final_info": {"winner": 0}},
            {"final_info": {"winner": None}},
        ],
        np.array([True, True, True, False]),
        np.array([False, False, False, True]),
    )
    metrics = self_play_outcome_metrics(algo)
    assert metrics["self_play_games"] == 4
    assert metrics["self_play_decisive_games"] == 2
    assert metrics["self_play_first_player_wins"] == 1
    assert metrics["self_play_second_player_wins"] == 1
    assert metrics["self_play_draws"] == 1
    assert metrics["self_play_truncations"] == 1
    assert metrics["self_play_decisive_rate"] == 0.5


def test_chess_capture_emits_material_learning_signal():
    env = BoardGameSelfPlayEnv("chess")
    env.reset(seed=0)

    def play_san(move: str) -> float:
        state = env.game._state
        actions = {
            state.action_to_string(state.current_player(), action): action
            for action in state.legal_actions()
        }
        _obs, reward, terminated, _trunc, _info = env.step(actions[move])
        assert not terminated
        return reward

    assert play_san("e4") == 0.0
    assert play_san("d5") == 0.0
    assert play_san("exd5") == pytest.approx(1.0 / 39.0)
    env.close()


def test_illegal_action_is_a_loss_not_a_crash():
    env = BoardGameSelfPlayEnv("tic_tac_toe")
    env.reset(seed=0)
    env.step(0)
    _obs, reward, terminated, _trunc, info = env.step(0)
    assert terminated
    assert reward == -1.0
    assert info.get("illegal_action") is True
    env.close()


def test_chess_render_draws_letters_not_solid_squares():
    board = np.zeros((8, 8), dtype=np.int32)
    board[7, 4] = 6  # white king on e1
    board[0, 0] = -4  # black rook on a8
    img = render_board_rgb(board, "chess")
    assert img.shape == (320, 320, 3)
    cell = 40
    king = img[7 * cell + 8 : 8 * cell - 8, 4 * cell + 8 : 5 * cell - 8]
    # A solid white square is one color; a lettered token has ink + token fill.
    unique = np.unique(king.reshape(-1, 3), axis=0)
    assert unique.shape[0] >= 2
    assert not np.all(king == king[0, 0])
    empty = img[4 * cell + 4 : 5 * cell - 4, 4 * cell + 4 : 5 * cell - 4]
    light = np.all(empty == (232, 213, 181), axis=-1)
    dark = np.all(empty == (181, 136, 99), axis=-1)
    assert np.all(light | dark)


def test_tic_tac_toe_render_draws_round_stones():
    board = np.array([[1, 0, -1], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    img = render_board_rgb(board, "tic_tac_toe")
    cell = 52
    # Corner of an occupied cell stays board-colored; the stone is a circle.
    corner = img[1, 1]
    center = img[cell // 2, cell // 2]
    assert not np.array_equal(corner, center)


def test_chess_planes_count_as_image_space():
    space = gym.spaces.Box(0, 255, (8, 8, 20), dtype=np.uint8)
    assert is_image_space(space)


def test_discrete_expand_slots_respects_mask():
    logits = np.arange(9, dtype=np.float64)
    mask = np.zeros(9, dtype=np.int8)
    mask[[2, 5, 8]] = 1
    priors, action_ids = discrete_expand_slots(logits, action_mask=mask, as_logits=True)
    assert action_ids == [2, 5, 8]
    assert priors.tolist() == [2.0, 5.0, 8.0]


def test_efficientzero_two_player_discount_and_legal_search():
    env = gym.make("tic_tac_toe")
    algo = NativeEfficientZero(env, {**DEFAULT_HYPERPARAMS, **_TINY}, seed=0, device="cpu")
    assert algo.two_player is True
    assert algo.search_discount == -algo.gamma
    obs, info = env.reset(seed=0)
    mask = np.asarray(info["action_mask"])
    # Occupy the center so it is illegal, then search from this position.
    obs, _r, _t, _tr, info = env.step(4)
    mask = np.asarray(info["action_mask"])
    obs_batch = obs_to_array(obs, env.observation_space)[None]
    results = algo.search(obs_batch, deterministic=np.array([True]), action_masks=mask[None])
    assert mask[results[0]["env_action"]] == 1
    assert results[0]["policy_target"].shape == (9,)
    assert abs(float(results[0]["policy_target"].sum()) - 1.0) < 1e-4
    assert results[0]["policy_target"][4] == 0.0
    env.close()


def test_efficientzero_skips_reanalyze_without_historical_legal_mask():
    env = gym.make("tic_tac_toe")
    algo = NativeEfficientZero(
        env, {**DEFAULT_HYPERPARAMS, **_TINY, "reanalyze_batch_size": 1}, seed=0, device="cpu",
    )
    obs, _info = env.reset(seed=0)
    obs_arr = obs_to_array(obs, env.observation_space)
    policy = np.zeros(9, dtype=np.float32)
    policy[[1, 3, 7]] = 1.0 / 3.0
    algo.buffer.add(obs_arr, np.eye(9, dtype=np.float32)[1], 0.0, obs_arr, policy, True)
    search_called = False

    def fake_search(*_args, **_kwargs):
        nonlocal search_called
        search_called = True
        raise AssertionError("board-game reanalysis must not run without an exact historical mask")

    algo.search = fake_search
    assert algo._reanalyze() is None
    assert search_called is False
    env.close()


def test_two_player_path_consistency_uses_negative_discount(monkeypatch):
    algo = object.__new__(NativeResearchImZero)
    algo.search_discount = -0.9
    algo.support_size = 1
    algo.label_smoothing_eps = 0.0
    captured = []

    monkeypatch.setattr(
        "rl_core.algorithms.native.researchimzero._logits_to_scalar",
        lambda logits, support_size: torch.ones(logits.shape[0]),
    )

    def fake_two_hot(target, support_size, label_smoothing_eps):
        del support_size, label_smoothing_eps
        captured.append(target.detach().clone())
        return torch.full((target.shape[0], 3), 1.0 / 3.0)

    monkeypatch.setattr(
        "rl_core.algorithms.native.researchimzero._scalar_to_two_hot",
        fake_two_hot,
    )
    logits = [torch.zeros((1, 3)), torch.zeros((1, 3))]
    algo._compute_path_consistency_loss(
        logits, [torch.zeros(1)], torch.ones((1, 2)), torch.ones(1),
    )
    assert captured[0].item() == pytest.approx(-0.9)


def test_latent_uncertainty_return_uses_two_player_discount():
    algo = object.__new__(NativeLatentImZero)
    algo.search_discount = -0.5

    class Probe:
        @staticmethod
        def reward(_features):
            return torch.tensor([[1.0, 2.0]])

        @staticmethod
        def value(_features):
            return torch.tensor([[3.0, 4.0]])

    algo.uncertainty_probe = Probe()
    result = algo._return_predictions(torch.zeros(1), torch.zeros(1))
    assert result.tolist() == [[-0.5, 0.0]]


@pytest.mark.parametrize(
    "buffer",
    [
        _EfficientZeroBuffer(2, (2,), 4674, 4674),
        _UniZeroBuffer(2, (2,), 4674, 4674, context_length=2),
        _ResearchImZeroBuffer(2, (2,), 4674, 4674, context_length=2),
    ],
)
def test_large_discrete_replay_uses_compact_storage(buffer):
    obs = np.array([0.0, 1.0], dtype=np.float32)
    action = np.zeros(4674, dtype=np.float32)
    action[10] = 1.0
    policy = np.zeros(4674, dtype=np.float32)
    policy[[10, 20]] = 0.5
    buffer.add(obs, action, 0.0, obs, policy, True)
    episode = buffer.episodes[0]
    assert episode["obs"].dtype == np.float16
    assert episode["action"].dtype == np.float16
    assert episode["policy_target"].dtype == np.float16


@pytest.mark.parametrize(
    "buffer",
    [
        _EfficientZeroBuffer(4, (1,), 2, 2),
        _UniZeroBuffer(4, (1,), 2, 2, context_length=1),
        _ResearchImZeroBuffer(4, (1,), 2, 2, context_length=1),
    ],
)
def test_self_play_replay_retains_rare_decisive_games(buffer):
    transition = (
        np.zeros(1, dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        0.0,
        np.ones(1, dtype=np.float32),
        np.array([0.5, 0.5], dtype=np.float32),
        True,
    )
    buffer.add(*transition, episode_outcome=1)
    for _ in range(8):
        buffer.add(*transition, episode_outcome=0)
    assert len(buffer.episodes) == 4
    assert 1 in buffer._episode_outcomes


def test_parallel_self_play_replay_capacity_has_four_games_per_lane():
    assert self_play_replay_capacity(64, 48, True) == 192
    assert self_play_replay_capacity(256, 48, True) == 256
    assert self_play_replay_capacity(64, 48, False) == 64


def test_render_episode_passes_current_action_mask_to_predict():
    class MaskedPreviewEnv(gym.Env):
        metadata = {"render_fps": 20}
        observation_space = gym.spaces.Discrete(3)
        action_space = gym.spaces.Discrete(3)

        def __init__(self):
            self.step_index = 0

        def reset(self, *, seed=None, options=None):
            self.step_index = 0
            return 0, {"action_mask": np.array([0, 1, 0], dtype=np.int8)}

        def step(self, action):
            assert action == self.step_index + 1
            self.step_index += 1
            done = self.step_index == 2
            mask = np.array([0, 0, 1], dtype=np.int8)
            return self.step_index, 0.0, done, False, {"action_mask": mask}

        def render(self):
            return np.full((4, 4, 3), self.step_index, dtype=np.uint8)

    seen_masks: list[list[int]] = []

    def predict(_obs, _episode_start, info):
        mask = np.asarray(info["action_mask"])
        seen_masks.append(mask.tolist())
        return int(np.flatnonzero(mask)[0]), None

    assert render_episode(MaskedPreviewEnv, predict) is not None
    assert seen_masks == [[0, 1, 0], [0, 0, 1]]


def test_evaluation_passes_board_game_action_mask_to_predict():
    class MaskedAlgorithm:
        def __init__(self):
            self.masks: list[np.ndarray] = []

        def predict(self, _obs, deterministic=True, action_mask=None):
            assert deterministic
            mask = np.asarray(action_mask)
            self.masks.append(mask.copy())
            return int(np.flatnonzero(mask)[0]), None

    algo = MaskedAlgorithm()
    result = _run_episodes_custom_algorithm(
        algo, "tic_tac_toe", [], episodes=1, seed=0,
        record_gif=False, deterministic=True,
    )
    assert result["episodes"] == 1
    assert algo.masks
    assert all(int(mask.sum()) > 0 for mask in algo.masks)


@pytest.mark.parametrize(
    ("algo_cls", "defaults"),
    [
        (NativeEfficientZero, DEFAULT_HYPERPARAMS),
        (NativeUniZero, UNIZERO_DEFAULT_HYPERPARAMS),
    ],
)
def test_zero_replay_keeps_terminal_observation_not_autoreset(algo_cls, defaults):
    env = _TerminalObservationEnv()
    algo = algo_cls(
        env,
        {**defaults, **_TINY, "learning_starts": 100},
        seed=0,
        device="cpu",
    )
    algo.learn(1, TrainingCallback(lambda *_args, **_kwargs: True))
    assert algo.buffer.num_episodes == 1
    assert algo.buffer.episodes[0]["next_obs"][0].tolist() == [9.0]
    env.close()


def test_efficientzero_self_play_smoke():
    env = gym.make("tic_tac_toe")
    algo = NativeEfficientZero(env, {**DEFAULT_HYPERPARAMS, **_TINY}, seed=0, device="cpu")
    steps: list[int] = []

    def writer(num_timesteps, episode_reward=None, episode_length=None, metrics=None):
        steps.append(num_timesteps)
        return num_timesteps < 20

    algo.learn(total_timesteps=20, callback=TrainingCallback(writer))
    assert steps[-1] >= 20
    obs, _info = env.reset(seed=1)
    action, _state = algo.predict(obs, deterministic=True)
    assert env.action_space.contains(action)
    env.close()
