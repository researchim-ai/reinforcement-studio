"""From-scratch PyTorch implementations of PPO/A2C/DQN.

These replace Stable-Baselines3 as the *default* engine for the built-in
Gym-track algorithms (see `rl_core/algorithms/native_runner.py`). SB3 is
still an optional dependency, used only when a user's custom algorithm
plugin explicitly subclasses `stable_baselines3.common.base_class.BaseAlgorithm`
(see `rl_core/algorithms/base.py`).

Every algorithm here is a `rl_core.algorithms.base.CustomAlgorithm` subclass,
so it's driven through the exact same harness (`run_custom_algorithm` in
`rl_core/algorithms/runner_utils.py`) as user-authored plugins.
"""
