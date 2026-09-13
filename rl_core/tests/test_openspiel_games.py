"""Tests for OpenSpiel chess / Go 9×9 wrapped as AlphaZero BoardGames."""
from __future__ import annotations

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")

from rl_core.alphazero.network import AlphaZeroNet
from rl_core.envs import registry
from rl_core.games import GAME_REGISTRY, make_game
from rl_core.games.openspiel_games import Chess, Go9x9


def test_listed_as_alphazero_board_games():
    envs = {e["id"]: e for e in registry.list_environments()}
    for eid, name_substr in [("chess", "Chess"), ("go_9x9", "Go")]:
        spec = envs[eid]
        assert spec["kind"] == "alphazero"
        assert spec["available"] is True
        assert spec["compatible_algorithms"][0] == "alphazero"
        for algo_id in ("efficientzero", "unizero", "researchimzero", "latentimzero"):
            assert algo_id in spec["compatible_algorithms"]
        assert name_substr.lower() in spec["name"].lower()
        assert spec["extra_requirement"] == "open_spiel"


def test_chess_in_game_registry():
    assert "chess" in GAME_REGISTRY
    assert "go_9x9" in GAME_REGISTRY
    game = make_game("chess")
    assert isinstance(game, Chess)


def test_chess_start_position():
    game = Chess()
    enc = game.reset()
    assert enc.shape == (20, 8, 8)
    assert game.action_size == 4674
    assert len(game.legal_actions()) == 20
    assert game.current_player == 1
    board = game.board_list()
    assert len(board) == 8 and len(board[0]) == 8
    # Rank 8 (row 0) is black; rank 1 (row 7) is white.
    assert board[0][0] == -4  # black rook
    assert board[7][4] == 6   # white king
    assert board[6][4] == 1   # white pawn e2


def test_chess_step_and_clone_are_independent():
    game = Chess()
    game.reset()
    legal = game.legal_actions()
    cloned = game.clone()
    game.step(legal[0])
    assert cloned.legal_actions() == legal
    assert cloned.current_player == 1
    assert game.current_player == -1
    assert not np.array_equal(game.encode(), cloned.encode())


def test_go_9x9_pass_pass_is_terminal():
    game = Go9x9()
    enc = game.reset()
    assert enc.shape == (4, 9, 9)
    assert game.action_size == 82
    assert 81 in game.legal_actions()  # pass
    game.step(81)
    game.step(81)
    assert game.done
    # Empty board + komi 7.5 → White (second player, our −1) wins.
    assert game.winner == -1


def test_alphazero_net_accepts_chess_planes():
    game = Chess()
    enc = game.reset()
    net = AlphaZeroNet(game.rows, game.cols, game.action_size, channels=8, num_blocks=1, in_planes=enc.shape[0])
    probs, value = net.predict(enc, device="cpu")
    assert probs.shape == (game.action_size,)
    assert abs(float(np.sum(probs)) - 1.0) < 1e-4
    assert -1.0 <= value <= 1.0
