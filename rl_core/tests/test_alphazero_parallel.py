from __future__ import annotations

import threading

import numpy as np
import torch

from rl_core.alphazero.base import BuiltinAlphaZeroTrainer
from rl_core.games import make_game


def test_builtin_alphazero_uses_configured_self_play_workers(monkeypatch):
    game_cls = make_game("tic_tac_toe").__class__
    sample = game_cls()
    state = sample.encode()
    policy = np.full(sample.action_size, 1.0 / sample.action_size, dtype=np.float32)
    barrier = threading.Barrier(4)
    thread_ids: set[int] = set()
    seeds: list[int] = []
    lock = threading.Lock()

    def fake_self_play(*_args, **kwargs):
        with lock:
            thread_ids.add(threading.get_ident())
            seeds.append(kwargs["seed"])
        barrier.wait(timeout=5)
        record = {"board_history": [sample.board_list()], "winner": 0}
        return [(state.copy(), policy.copy(), 0.0)], record

    def fake_match(*_args, **kwargs):
        assert kwargs["workers"] == 4
        return {"wins_a": 0, "wins_b": 0, "draws": 2, "games": 2}

    monkeypatch.setattr("rl_core.alphazero.base.play_self_play_game", fake_self_play)
    monkeypatch.setattr("rl_core.alphazero.base.play_match", fake_match)
    trainer = BuiltinAlphaZeroTrainer(
        game_cls,
        {
            "games_per_iteration": 4,
            "self_play_workers": 4,
            "seed": 10,
            "batch_size": 64,
            "channels": 4,
            "num_blocks": 1,
            "eval_games": 2,
        },
        "cpu",
    )

    metrics = trainer.run_iteration(1)

    assert metrics["self_play_workers"] == 4
    assert len(thread_ids) == 4
    assert sorted(seeds) == [10, 11, 12, 13]


def test_alphazero_moves_only_each_minibatch_to_device(monkeypatch):
    game_cls = make_game("tic_tac_toe").__class__
    sample = game_cls()
    trainer = BuiltinAlphaZeroTrainer(
        game_cls,
        {"batch_size": 4, "epochs": 1, "channels": 4, "num_blocks": 1},
        "cpu",
    )
    examples = [
        (
            sample.encode(),
            np.full(sample.action_size, 1.0 / sample.action_size, dtype=np.float32),
            0.0,
        )
        for _ in range(10)
    ]
    original_as_tensor = torch.as_tensor
    state_batch_sizes = []

    def tracking_as_tensor(data, *args, **kwargs):
        shape = np.shape(data)
        if len(shape) == 4:
            state_batch_sizes.append(shape[0])
        return original_as_tensor(data, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", tracking_as_tensor)
    trainer._train_epochs(examples)

    assert state_batch_sizes
    assert max(state_batch_sizes) <= 4
