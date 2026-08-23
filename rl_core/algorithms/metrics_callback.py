"""SB3 callback that snapshots training progress to metrics.json.

The FastAPI backend tails this file (see backend/ws.py) and forwards each
new snapshot to the frontend over a WebSocket — same pattern used for the
AlphaZero trainer, so the Training Monitor page can treat both the same way.
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _encode_frame(frame: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class MetricsCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        env_id: str,
        algo_id: str,
        total_timesteps: int,
        make_render_env: Callable[[], Any] | None = None,
        write_every_steps: int = 500,
        render_every_steps: int = 5000,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.env_id = env_id
        self.algo_id = algo_id
        self.total_timesteps = total_timesteps
        self.make_render_env = make_render_env
        self.write_every_steps = write_every_steps
        self.render_every_steps = render_every_steps
        self.start_time = time.time()
        self._last_write = 0
        self._last_render = 0
        self._stopped_early = False

    def _stop_requested(self) -> bool:
        return (self.run_dir / "stop.flag").exists()

    def _episode_stats(self) -> tuple[float | None, float | None]:
        buf = getattr(self.model, "ep_info_buffer", None)
        if not buf:
            return None, None
        rewards = [ep["r"] for ep in buf]
        lengths = [ep["l"] for ep in buf]
        return float(np.mean(rewards)), float(np.mean(lengths))

    def _maybe_render(self) -> str | None:
        if not self.make_render_env or self.num_timesteps - self._last_render < self.render_every_steps:
            return None
        self._last_render = self.num_timesteps
        try:
            env = self.make_render_env()
            obs, _ = env.reset()
            frame = env.render()
            for _ in range(30):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                frame = env.render()
                if terminated or truncated:
                    break
            env.close()
            if frame is not None:
                return _encode_frame(np.asarray(frame))
        except Exception:
            return None
        return None

    def _write_snapshot(self, status: str, extra: dict | None = None) -> None:
        mean_reward, mean_length = self._episode_stats()
        elapsed = time.time() - self.start_time
        snapshot = {
            "run_id": self.run_dir.name,
            "kind": "gym",
            "status": status,
            "algo": self.algo_id,
            "env_id": self.env_id,
            "step": int(self.num_timesteps),
            "total_timesteps": int(self.total_timesteps),
            "episode_reward_mean": mean_reward,
            "episode_length_mean": mean_length,
            "fps": round(self.num_timesteps / elapsed, 1) if elapsed > 0 else 0,
            "elapsed_seconds": round(elapsed, 1),
        }
        frame = self._maybe_render()
        if frame:
            snapshot["frame_base64"] = frame
        if extra:
            snapshot.update(extra)
        (self.run_dir / "metrics.json").write_text(json.dumps(snapshot))

    def _on_training_start(self) -> None:
        self._write_snapshot("running")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_write >= self.write_every_steps:
            self._last_write = self.num_timesteps
            self._write_snapshot("running")
        if self._stop_requested():
            self._stopped_early = True
            return False
        return True

    def _on_training_end(self) -> None:
        status = "stopped" if self._stopped_early else "completed"
        self._write_snapshot(status)
