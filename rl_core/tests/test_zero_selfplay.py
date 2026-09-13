"""Self-play Gym wrapper + Zero-family legal-action search on board games."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl_core.algorithms.base import TrainingCallback
from rl_core.algorithms.native.efficientzero import DEFAULT_HYPERPARAMS, NativeEfficientZero
from rl_core.algorithms.native.preprocessing import is_image_space, obs_to_array
from rl_core.algorithms.native.zero_selfplay import discrete_expand_slots
from rl_core.envs.board_game_gym import BoardGameSelfPlayEnv, register_board_game_envs, render_board_rgb
from rl_core.envs import registry

register_board_game_envs()

_TINY = {
    "latent_dim": 8, "hidden_dim": 16, "proj_dim": 8, "buffer_size": 32,
    "batch_size": 4, "unroll_steps": 2, "td_steps": 2, "num_sampled_actions": 4,
    "num_simulations": 4, "num_top_actions": 2, "learning_starts": 4, "train_freq": 1,
    "value_support_size": 10, "reanalyze_batch_size": 0,
}


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
    _obs, reward, terminated, _trunc, _info = env.step(2)
    assert terminated
    assert reward == 1.0
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
