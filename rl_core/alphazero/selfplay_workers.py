"""Self-play in worker processes, network in the trainer process.

Board-game stepping and the Python side of the search are single-threaded
and dominate self-play time once the network runs on a GPU. So the
parallel games are split across worker processes, each running its own
`SelfPlayPool`, while the trainer process acts as the network server:

* a worker writes a search request (encodings + which legal-move logits it
  needs) into its shared-memory block and sends a tiny "eval" message;
* the server concatenates the requests of a whole group of workers into
  one batch, runs the network once, writes each worker's answer back into
  its block and wakes it up.

Workers are split into two groups served alternately, so while the GPU
evaluates one group's batch the other group is already doing its CPU work.

Workers never import torch (important on Windows, where every process is
started with "spawn" and would otherwise load its own copy), and the
network weights stay in the trainer process — there is nothing to sync
after a training step.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from rl_core.alphazero.batched import FinishedGame, SelfPlayPool
from rl_core.alphazero.gumbel import Evaluator, SearchConfig
from rl_core.games.base import BoardGame


class _Block:
    """Views into one shared-memory block: request encodings / logit
    indices written by the worker, answers written by the server."""

    def __init__(self, shm: shared_memory.SharedMemory, obs_shape: tuple[int, ...], pos_cap: int, idx_cap: int) -> None:
        self.shm = shm
        self.pos_cap = pos_cap
        self.idx_cap = idx_cap
        offset = 0

        def view(shape: tuple[int, ...], dtype) -> np.ndarray:
            nonlocal offset
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=offset)
            offset += arr.nbytes
            offset = (offset + 63) // 64 * 64
            return arr

        self.enc = view((pos_cap, *obs_shape), np.float16)
        self.rows = view((idx_cap,), np.int32)
        self.actions = view((idx_cap,), np.int32)
        self.logits = view((idx_cap,), np.float32)
        self.values = view((pos_cap,), np.float32)

    @staticmethod
    def nbytes(obs_shape: tuple[int, ...], pos_cap: int, idx_cap: int) -> int:
        obs = int(np.prod(obs_shape))
        return pos_cap * obs * 2 + idx_cap * 12 + pos_cap * 4 + 5 * 64


def _worker_main(
    conn, game_cls: type[BoardGame], num_games: int, seed: int, exploration_moves: int, max_plies: int,
    tracked_games: int, shm_name: str, obs_shape: tuple[int, ...], pos_cap: int, idx_cap: int,
) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    block = _Block(shm, obs_shape, pos_cap, idx_cap)
    try:
        pool = SelfPlayPool(
            game_cls, num_games, np.random.default_rng(seed), exploration_moves, max_plies, tracked_games=tracked_games,
        )
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                break
            _, plies, cfg = msg
            # The network was trained since the previous command.
            pool.forget_evaluations()
            samples: list[Any] = []
            finished: list[FinishedGame] = []
            for _ in range(plies):
                gen = pool.play_ply_steps(cfg)
                try:
                    request = next(gen)
                    while True:
                        enc, rows, actions = request
                        n, m = len(enc), len(rows)
                        if n > pos_cap or m > idx_cap:
                            raise RuntimeError(f"search request {n}×{m} exceeds shared buffer {pos_cap}×{idx_cap}")
                        block.enc[:n] = enc
                        block.rows[:m] = rows
                        block.actions[:m] = actions
                        conn.send(("eval", n, m))
                        conn.recv()
                        request = gen.send((block.logits[:m].copy(), block.values[:n].copy()))
                except StopIteration as stop:
                    ply_samples, ply_finished = stop.value
                samples += ply_samples
                finished += ply_finished
            conn.send(("done", samples, finished))
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception:  # noqa: BLE001
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:  # noqa: BLE001
            pass
    finally:
        del block
        shm.close()


class _Worker:
    def __init__(self, process, conn, shm: shared_memory.SharedMemory, block: _Block) -> None:
        self.process = process
        self.conn = conn
        self.shm = shm
        self.block = block

    def recv(self):
        try:
            msg = self.conn.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"self-play worker {self.process.pid} exited (code {self.process.exitcode})") from exc
        if msg[0] == "error":
            raise RuntimeError(f"self-play worker failed:\n{msg[1]}")
        return msg


class WorkerSelfPlay:
    """Drop-in replacement for stepping one big `SelfPlayPool`: same
    samples/finished-games output, games spread over ``num_workers``
    processes."""

    def __init__(
        self,
        game_cls: type[BoardGame],
        num_games: int,
        num_workers: int,
        seed: int,
        exploration_moves: int,
        max_plies: int,
        max_considered: int,
        tracked_games: int = 3,
    ) -> None:
        sample = game_cls()
        obs_shape = tuple(int(x) for x in sample.encode().shape)
        max_legal = min(int(sample.action_size), 320)
        num_workers = max(1, min(int(num_workers), int(num_games)))
        ctx = mp.get_context("spawn")
        self.workers: list[_Worker] = []
        base, extra = divmod(int(num_games), num_workers)
        seeds = np.random.SeedSequence(int(seed)).generate_state(num_workers)
        try:
            for w in range(num_workers):
                games = base + (1 if w < extra else 0)
                pos_cap = games * max(1, int(max_considered))
                idx_cap = pos_cap * max_legal
                shm = shared_memory.SharedMemory(create=True, size=_Block.nbytes(obs_shape, pos_cap, idx_cap))
                block = _Block(shm, obs_shape, pos_cap, idx_cap)
                parent_conn, child_conn = ctx.Pipe()
                process = ctx.Process(
                    target=_worker_main,
                    args=(child_conn, game_cls, games, int(seeds[w]), exploration_moves, max_plies,
                          tracked_games if w == 0 else 0, shm.name, obs_shape, pos_cap, idx_cap),
                    daemon=True,
                    name=f"alphazero-selfplay-{w}",
                )
                process.start()
                child_conn.close()
                self.workers.append(_Worker(process, parent_conn, shm, block))
        except Exception:
            self.close()
            raise
        self.groups = [self.workers[0::2], self.workers[1::2]] if num_workers >= 2 else [self.workers]
        self.max_considered = int(max_considered)

    def play(
        self, evaluate: Evaluator, cfg: SearchConfig, plies: int,
    ) -> tuple[list[Any], list[FinishedGame]]:
        if cfg.num_considered > self.max_considered:
            raise ValueError("num_considered grew beyond the worker buffers")
        for w in self.workers:
            w.conn.send(("play", int(plies), cfg))
        samples: list[Any] = []
        finished: list[FinishedGame] = []
        pending = [list(g) for g in self.groups]
        while any(pending):
            for gi, group in enumerate(pending):
                if not group:
                    continue
                requests = []
                for w in group:
                    msg = w.recv()
                    if msg[0] == "done":
                        samples += msg[1]
                        finished += msg[2]
                    else:
                        requests.append((w, msg[1], msg[2]))
                pending[gi] = [w for w, _n, _m in requests]
                if requests:
                    self._serve(evaluate, requests)
        return samples, finished

    @staticmethod
    def _serve(evaluate: Evaluator, requests: list[tuple[_Worker, int, int]]) -> None:
        enc = np.concatenate([w.block.enc[:n] for w, n, _m in requests])
        rows_parts, action_parts = [], []
        offset = 0
        for w, n, m in requests:
            rows_parts.append(w.block.rows[:m].astype(np.int64) + offset)
            action_parts.append(w.block.actions[:m].astype(np.int64))
            offset += n
        rows = np.concatenate(rows_parts)
        actions = np.concatenate(action_parts)
        logits, values = evaluate(enc, rows, actions)
        pos = idx = 0
        for w, n, m in requests:
            w.block.logits[:m] = logits[idx:idx + m]
            w.block.values[:n] = values[pos:pos + n]
            pos += n
            idx += m
            w.conn.send(1)

    def close(self) -> None:
        for w in self.workers:
            try:
                w.conn.send(("stop",))
            except Exception:  # noqa: BLE001
                pass
        for w in self.workers:
            w.process.join(timeout=5)
            if w.process.is_alive():
                w.process.terminate()
                w.process.join(timeout=5)
            try:
                w.conn.close()
            except Exception:  # noqa: BLE001
                pass
            del w.block
            try:
                w.shm.close()
                w.shm.unlink()
            except Exception:  # noqa: BLE001
                pass
        self.workers = []
        self.groups = []
