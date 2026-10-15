from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from rl_core.alphazero.base import BuiltinAlphaZeroTrainer
from rl_core.alphazero.batched import EloLadder, SelfPlayPool, elo_from_score, make_openings, play_match
from rl_core.alphazero.checkpoint import load_checkpoint, save_checkpoint
from rl_core.alphazero.gumbel import NetEvaluator, SearchConfig, gumbel_search
from rl_core.alphazero.network import AlphaZeroNet
from rl_core.alphazero.replay import DRAW, LOSS, WIN, ReplayBuffer
from rl_core.games import GAME_REGISTRY, make_game

TicTacToe = GAME_REGISTRY["tic_tac_toe"]


def _uniform_evaluator(action_size: int):
    def evaluate(encodings: np.ndarray, rows: np.ndarray, actions: np.ndarray):
        assert len(rows) == len(actions)
        assert (actions < action_size).all()
        return np.zeros(len(rows), dtype=np.float32), np.zeros(len(encodings), dtype=np.float32)

    return evaluate


def _position(moves: list[int]):
    game = TicTacToe()
    for a in moves:
        game.step(a)
    return game


def test_search_takes_an_immediate_win():
    # X: 0, 1 — O: 3, 4 — X to move, 2 wins on the spot.
    game = _position([0, 3, 1, 4])
    cfg = SearchConfig(num_considered=9, num_survivors=4, gumbel_scale=0.0)
    [res] = gumbel_search([game], _uniform_evaluator(9), cfg, np.random.default_rng(0))
    assert res.action == 2
    assert int(res.legal[int(np.argmax(res.policy))]) == 2
    assert math.isclose(float(res.policy.sum()), 1.0, rel_tol=1e-5)
    assert sorted(res.legal.tolist()) == sorted(game.legal_actions())


def test_search_blocks_the_opponents_win():
    # X: 0, 1 — O: 4 — O to move must block at 2 (X threatens 0-1-2).
    game = _position([0, 4, 1])
    cfg = SearchConfig(num_considered=7, num_survivors=7, gumbel_scale=0.0)
    [res] = gumbel_search([game], _uniform_evaluator(9), cfg, np.random.default_rng(0))
    assert res.action == 2


def test_search_batches_every_position_into_three_network_calls():
    calls: list[int] = []
    base = _uniform_evaluator(9)

    def counting(encodings, rows, actions):
        calls.append(len(encodings))
        return base(encodings, rows, actions)

    games = [TicTacToe() for _ in range(5)]
    results = gumbel_search(games, counting, SearchConfig(num_considered=4, num_survivors=2), np.random.default_rng(1))
    assert len(results) == 5
    assert calls == [5, 5 * 4, 5 * 2]
    for res in results:
        assert int((res.visits > 0).sum()) == 4
        assert int((res.visits == 2).sum()) == 2


def test_self_play_pool_excludes_exploration_moves_and_labels_outcomes():
    rng = np.random.default_rng(3)
    pool = SelfPlayPool(TicTacToe, 6, rng, exploration_moves=0)
    evaluate = _uniform_evaluator(9)
    cfg = SearchConfig(num_considered=4, num_survivors=2)
    samples, finished = [], []
    for _ in range(9):
        s, f = pool.play_ply(evaluate, cfg)
        samples += s
        finished += f
    assert finished
    assert len(samples) == sum(f.plies for f in finished)
    assert {outcome for *_rest, outcome in samples} <= {WIN, DRAW, LOSS}
    for f in finished:
        if f.record is not None:
            assert len(f.record["board_history"]) == f.plies + 1

    explore_pool = SelfPlayPool(TicTacToe, 6, np.random.default_rng(4), exploration_moves=5)
    switches = [slot.switch for slot in explore_pool.slots]
    assert any(switches)
    samples, finished = [], []
    for _ in range(5):
        s, f = explore_pool.play_ply(evaluate, cfg)
        samples += s
        finished += f
    # Five plies is too short for any replacement game to finish, so every
    # finished game is one of the original slots.
    assert finished
    assert len(samples) < sum(f.plies for f in finished)


def test_cached_roots_play_the_same_games_as_fresh_evaluation():
    def deterministic(encodings, rows, actions):
        flat = encodings.reshape(len(encodings), 27).astype(np.float64)
        weights = np.linspace(-1.0, 1.0, flat.shape[1])
        score = flat @ weights
        logits = np.sin(score[rows] * 3.0 + actions * 0.7).astype(np.float32)
        return logits, np.tanh(score).astype(np.float32)

    counts = {"cached": 0, "fresh": 0}

    def counted(kind):
        def evaluate(encodings, rows, actions):
            counts[kind] += len(encodings)
            return deterministic(encodings, rows, actions)
        return evaluate

    cfg = SearchConfig(num_considered=4, num_survivors=2)
    cached = SelfPlayPool(TicTacToe, 4, np.random.default_rng(7), exploration_moves=3)
    fresh = SelfPlayPool(TicTacToe, 4, np.random.default_rng(7), exploration_moves=3)
    for _ in range(12):
        a = cached.play_ply(counted("cached"), cfg)
        fresh.forget_evaluations()
        b = fresh.play_ply(counted("fresh"), cfg)
        assert [f.plies for f in a[1]] == [f.plies for f in b[1]]
        assert [s[3] for s in a[0]] == [s[3] for s in b[0]]
        for x, y in zip(a[0], b[0]):
            np.testing.assert_allclose(x[2], y[2], rtol=1e-6)
    assert [s.moves for s in cached.slots] == [s.moves for s in fresh.slots]
    assert counts["cached"] < counts["fresh"]


def test_replay_buffer_ring_and_top_k_policy():
    buf = ReplayBuffer(capacity=3, obs_shape=(3, 3, 3), action_size=9, max_policy_entries=2)
    legal = np.array([1, 4, 7])
    for i in range(5):
        buf.add(np.full((3, 3, 3), i, dtype=np.float32), legal, np.array([0.1, 0.6, 0.3], dtype=np.float32), WIN)
    assert len(buf) == 3
    assert buf.total_added == 5
    obs, idx, prob, outcome = buf.sample(16, np.random.default_rng(0))
    assert set(np.unique(obs[:, 0, 0, 0]).tolist()) <= {2.0, 3.0, 4.0}
    assert idx.shape == (16, 2)
    assert set(idx[0].tolist()) == {4, 7}
    np.testing.assert_allclose(prob.sum(axis=1), 1.0, rtol=1e-3)
    assert (outcome == WIN).all()


def test_elo_ladder_promotes_after_patience():
    net = torch.nn.Linear(2, 2)
    ladder = EloLadder(promote_score=0.75, patience=2, random_max_elo=0.0)
    assert not ladder.update(1.0, 20, net)
    assert ladder.ladder_elo == 0.0
    assert ladder.update(0.9, 20, net)
    assert ladder.rung == 1 and ladder.anchor_state is not None
    assert not ladder.update(0.75, 20, net)
    assert ladder.ladder_elo == pytest.approx(elo_from_score(0.75, 20))
    assert elo_from_score(1.0, 20) < 1000


def test_play_match_against_random_counts_every_game():
    openings = make_openings(TicTacToe, 3, 1)
    assert len({tuple(o) for o in openings}) == 3
    result = play_match(
        TicTacToe, _uniform_evaluator(9), None, openings,
        SearchConfig(num_considered=9, num_survivors=4, gumbel_scale=0.0), np.random.default_rng(0),
    )
    assert result["games"] == 6
    assert result["wins"] + result["draws"] + result["losses"] == 6
    assert 0.0 <= result["score"] <= 1.0


def test_builtin_trainer_iteration_trains_and_evaluates():
    hp = {
        "parallel_games": 8, "plies_per_iteration": 9, "batch_size": 16, "min_replay_positions": 16,
        "channels": 8, "num_blocks": 1, "eval_every": 1, "eval_games": 4, "seed": 5,
    }
    torch.manual_seed(0)
    trainer = BuiltinAlphaZeroTrainer(TicTacToe, hp, "cpu")
    assert trainer.net.value_head_type == "wdl"
    metrics = trainer.run_iteration(1)
    assert metrics["self_play_games"] >= 8
    assert metrics["buffer_size"] == metrics["new_positions"] > 0
    assert metrics["train_steps"] > 0 and math.isfinite(metrics["loss"])
    assert metrics["eval_games"] == 4
    assert "eval_ladder_elo" in metrics and "eval_score_vs_random" in metrics
    assert metrics["board_history"]
    assert metrics["_self_play_records"]

    torch.manual_seed(0)
    again = BuiltinAlphaZeroTrainer(TicTacToe, hp, "cpu")
    assert again.run_iteration(1)["new_positions"] == metrics["new_positions"]


def test_builtin_trainer_falls_back_to_mse_for_tanh_networks():
    class Trainer(BuiltinAlphaZeroTrainer):
        def build_network(self):
            g = self.sample_game
            return AlphaZeroNet(g.rows, g.cols, g.action_size, 8, 1, in_planes=g.input_planes)

    trainer = Trainer(TicTacToe, {"parallel_games": 8, "plies_per_iteration": 9, "batch_size": 16,
                                  "min_replay_positions": 16, "eval_every": 0}, "cpu")
    metrics = trainer.run_iteration(1)
    assert metrics["train_steps"] > 0 and math.isfinite(metrics["value_loss"])


def test_checkpoints_keep_old_tanh_head_and_new_wdl_head(tmp_path):
    old = AlphaZeroNet(3, 3, 9, 8, 1)
    save_checkpoint(old, tmp_path / "old.pt", {"hyperparams": {"channels": 8, "num_blocks": 1}})
    loaded, _ = load_checkpoint(tmp_path / "old.pt")
    assert loaded.value_head_type == "tanh"

    new = AlphaZeroNet(3, 3, 9, 8, 1, value_head="wdl", global_pool=True)
    hp = {"channels": 8, "num_blocks": 1, "value_head": "wdl", "global_pool": True}
    save_checkpoint(new, tmp_path / "new.pt", {"hyperparams": hp})
    loaded, _ = load_checkpoint(tmp_path / "new.pt")
    assert loaded.value_head_type == "wdl"
    probs, value = loaded.predict(TicTacToe().encode())
    assert probs.shape == (9,) and -1.0 <= value <= 1.0

    evaluate = NetEvaluator(loaded, "cpu", chunk=2)
    enc = np.stack([TicTacToe().encode()] * 3)
    rows, actions = np.array([0, 2, 2]), np.array([4, 0, 8])
    gathered, values = evaluate(enc, rows, actions)
    with torch.no_grad():
        full, ref_values = loaded(torch.from_numpy(enc))
    np.testing.assert_allclose(gathered, full.numpy()[rows, actions], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(values, ref_values.numpy(), rtol=1e-5, atol=1e-6)


def test_worker_processes_match_the_search_contract():
    from rl_core.alphazero.selfplay_workers import WorkerSelfPlay

    calls: list[int] = []
    base = _uniform_evaluator(9)

    def counting(encodings, rows, actions):
        calls.append(len(encodings))
        return base(encodings, rows, actions)

    workers = WorkerSelfPlay(TicTacToe, 6, 3, seed=1, exploration_moves=0, max_plies=0, max_considered=4)
    try:
        cfg = SearchConfig(num_considered=4, num_survivors=2)
        samples, finished = workers.play(counting, cfg, plies=9)
        assert finished and len(samples) == sum(f.plies for f in finished)
        # Two groups of workers are served alternately; each group's
        # workers are batched together (2 + 1 workers × 2 games).
        assert max(calls) <= 2 * 2 * 4
        assert any(f.record is not None for f in finished)
        samples2, _ = workers.play(counting, cfg, plies=2)
        assert isinstance(samples2, list)
    finally:
        workers.close()
    assert not workers.workers


@pytest.mark.skipif("chess" not in GAME_REGISTRY, reason="open_spiel is not installed")
def test_openspiel_trusted_step_matches_step_and_board_is_lazy():
    game = make_game("chess")
    fast = game.clone()
    for _ in range(6):
        a = game.legal_actions()[0]
        game.step(a)
        fast.step_trusted(a)
    assert fast._board_cache is None
    np.testing.assert_array_equal(fast.board, game.board)
    np.testing.assert_array_equal(fast.encode(), game.encode())
    assert fast.current_player == game.current_player
    child = fast.clone()
    child.step_trusted(child.legal_actions()[0])
    assert child.current_player != fast.current_player
