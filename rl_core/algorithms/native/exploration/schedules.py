"""Small, deterministic schedules used by exploration policies."""
from __future__ import annotations


def linear_schedule(
    step: int,
    total_steps: int,
    fraction: float,
    initial: float,
    final: float,
) -> float:
    """Linearly interpolate from ``initial`` to ``final``.

    ``fraction`` is the portion of the full run occupied by the decay.
    A non-positive fraction means "start at the final value", which is
    useful for explicitly disabling random exploration without a special
    branch in the training loop.
    """
    if fraction <= 0.0:
        return float(final)
    decay_steps = max(1, int(float(fraction) * max(1, int(total_steps))))
    progress = min(1.0, max(0.0, float(step) / decay_steps))
    return float(initial + progress * (final - initial))
