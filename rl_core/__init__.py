"""Reinforcement Studio core — environments, algorithms and AlphaZero self-play engine.

This package is pure Python/PyTorch and has no dependency on the desktop
app or the FastAPI layer. The backend calls into it directly (native mode)
or it runs standalone inside the Docker image; either way `rl_core.runner`
is the single subprocess entrypoint used to launch a training run.
"""
