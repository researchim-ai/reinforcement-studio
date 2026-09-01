"""Environment/wrapper/algorithm catalogs consumed by the Experiment Designer."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rl_core.envs.previews import preview_media_type, resolve_preview_file
from rl_core.envs.registry import list_environments
from rl_core.envs.wrappers import WRAPPER_CATALOG
from rl_core.plugins import loader as plugin_loader

router = APIRouter()


class InspectRequest(BaseModel):
    kind: str = "gym"  # "gym" | "alphazero"
    environment: dict[str, Any] = {}
    algorithm: dict[str, Any] = {}


# Shared "memory" hyperparam block for every algorithm that supports an
# optional LSTM/GRU recurrent core (see rl_core/algorithms/native/networks.py
# and rl_core.algorithms.native.networks.memory_type_from_hyperparams).
# `memory_type` is a plain int (0/1/2), not a string, like every other
# hyperparam in this catalog — `options` is what turns it into a labeled
# dropdown in the Designer instead of a bare number field (see
# AlgorithmNode.tsx). Ignored by the algorithm whenever a hand-designed
# network (Network Builder) is selected instead of the built-in one.
def _memory_hyperparams(default_seq_len: int) -> list[dict[str, Any]]:
    return [
        {
            "key": "memory_type", "label": "Память (рекуррентность)", "type": "int", "default": 0, "min": 0, "max": 2,
            "options": [
                {"value": 0, "label": "Нет (обычная сеть)"},
                {"value": 1, "label": "LSTM"},
                {"value": 2, "label": "GRU"},
            ],
        },
        {"key": "memory_hidden_size", "label": "Память: размер hidden state", "type": "int", "default": 128, "min": 8, "max": 512, "visibleWhen": [{"key": "memory_type", "gte": 1}]},
        {"key": "memory_num_layers", "label": "Память: число слоёв RNN", "type": "int", "default": 1, "min": 1, "max": 3, "visibleWhen": [{"key": "memory_type", "gte": 1}]},
        {"key": "memory_seq_len", "label": "Память: длина BPTT-последовательности", "type": "int", "default": default_seq_len, "min": 2, "max": 200, "visibleWhen": [{"key": "memory_type", "gte": 1}]},
    ]


def _exploration_hyperparams() -> list[dict[str, Any]]:
    """Independent action exploration and intrinsic-motivation controls."""
    epsilon = [{"key": "action_exploration", "eq": 0}]
    noisy = [{"key": "action_exploration", "eq": 1}]
    rnd = [{"key": "intrinsic_exploration", "eq": 1}]
    return [
        {
            "key": "action_exploration", "label": "Исследование действий", "type": "int", "default": 0, "min": 0, "max": 1,
            "options": [
                {"value": 0, "label": "ε-greedy"},
                {"value": 1, "label": "NoisyNet (параметрический шум)"},
            ],
        },
        {"key": "exploration_fraction", "label": "ε: доля шагов decay", "type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "visibleWhen": epsilon},
        {"key": "exploration_initial_eps", "label": "ε: начальное значение", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "visibleWhen": epsilon},
        {"key": "exploration_final_eps", "label": "ε: конечное значение", "type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "visibleWhen": epsilon},
        {"key": "noisy_sigma0", "label": "NoisyNet: начальная σ", "type": "float", "default": 0.5, "min": 0.01, "max": 1.0, "visibleWhen": noisy},
        {
            "key": "intrinsic_exploration", "label": "Intrinsic motivation", "type": "int", "default": 0, "min": 0, "max": 1,
            "options": [
                {"value": 0, "label": "Нет"},
                {"value": 1, "label": "RND (Random Network Distillation)"},
            ],
        },
        {"key": "rnd_bonus_coef", "label": "RND: коэффициент бонуса", "type": "float", "default": 0.1, "min": 0.0, "max": 10.0, "visibleWhen": rnd},
        {"key": "rnd_learning_rate", "label": "RND: learning rate", "type": "float", "default": 1e-4, "min": 1e-6, "max": 1e-2, "visibleWhen": rnd},
        {"key": "rnd_feature_dim", "label": "RND: размер признаков", "type": "int", "default": 128, "min": 8, "max": 512, "visibleWhen": rnd},
        {"key": "rnd_hidden_dim", "label": "RND: hidden size", "type": "int", "default": 128, "min": 16, "max": 1024, "visibleWhen": rnd},
        {"key": "rnd_bonus_clip", "label": "RND: clip бонуса", "type": "float", "default": 5.0, "min": 0.1, "max": 100.0, "visibleWhen": rnd},
    ]


ALGORITHM_CATALOG = [
    {
        "id": "dqn",
        "name": "DQN",
        "kind": "gym",
        "description": "Deep Q-Network — value-based, для дискретных действий.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
            {"key": "buffer_size", "label": "Replay buffer size", "type": "int", "default": 50_000, "min": 1_000, "max": 1_000_000},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 64, "min": 8, "max": 512},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            *_exploration_hyperparams(),
            *_memory_hyperparams(default_seq_len=20),
        ],
    },
    {
        "id": "rainbow_dqn",
        "name": "Rainbow DQN",
        "kind": "gym",
        "description": "DQN + Double Q-learning + Dueling-сеть + Prioritized Replay + n-step — сильнее обычного DQN на тех же дискретных средах.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
            {"key": "buffer_size", "label": "Replay buffer size", "type": "int", "default": 50_000, "min": 1_000, "max": 1_000_000},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 64, "min": 8, "max": 512},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "n_step", "label": "N-step return", "type": "int", "default": 3, "min": 1, "max": 10},
            {"key": "per_alpha", "label": "PER: приоритет (alpha)", "type": "float", "default": 0.6, "min": 0.0, "max": 1.0},
            {"key": "per_beta_start", "label": "PER: IS-коррекция (beta start)", "type": "float", "default": 0.4, "min": 0.0, "max": 1.0},
            {
                "key": "distributional", "label": "Distributional RL (QR-DQN)", "type": "int", "default": 0, "min": 0, "max": 1,
                "options": [
                    {"value": 0, "label": "Нет (скалярный Q)"},
                    {"value": 1, "label": "Да — предсказывать распределение возврата"},
                ],
                "visibleWhen": [{"key": "memory_type", "eq": 0}],
            },
            {
                "key": "num_quantiles", "label": "QR-DQN: число квантилей", "type": "int", "default": 51, "min": 3, "max": 200,
                "visibleWhen": [{"key": "distributional", "eq": 1}, {"key": "memory_type", "eq": 0}],
            },
            *_exploration_hyperparams(),
            *_memory_hyperparams(default_seq_len=20),
        ],
    },
    {
        "id": "ppo",
        "name": "PPO",
        "kind": "gym",
        "description": "Proximal Policy Optimization — стабильный policy-gradient метод, discrete и continuous.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 3e-4, "min": 1e-6, "max": 1e-1},
            {
                "key": "lr_schedule", "label": "Learning rate schedule", "type": "int", "default": 0, "min": 0, "max": 1,
                "options": [
                    {"value": 0, "label": "Постоянный"},
                    {"value": 1, "label": "Линейно убывает до 0"},
                ],
            },
            {"key": "n_steps", "label": "Steps per update", "type": "int", "default": 2048, "min": 32, "max": 8192},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 64, "min": 8, "max": 1024},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "ent_coef", "label": "Entropy coefficient", "type": "float", "default": 0.0, "min": 0.0, "max": 0.1},
            {
                "key": "use_sde", "label": "gSDE (гладкое исследование, continuous)", "type": "int", "default": 0, "min": 0, "max": 1,
                "options": [
                    {"value": 0, "label": "Нет (обычный per-step шум)"},
                    {"value": 1, "label": "Да — state-dependent exploration"},
                ],
            },
            {
                "key": "sde_sample_freq", "label": "gSDE: шагов между пересэмплингом шума", "type": "int", "default": 4, "min": 1, "max": 256,
                "visibleWhen": [{"key": "use_sde", "eq": 1}],
            },
            {
                "key": "sde_log_std_init", "label": "gSDE: начальный log_std", "type": "float", "default": -2.0, "min": -5.0, "max": 1.0,
                "visibleWhen": [{"key": "use_sde", "eq": 1}],
            },
            {
                "key": "head_hidden_size", "label": "Отдельный скрытый слой pi/value поверх признаков", "type": "int",
                "default": 0, "min": 0, "max": 512,
            },
            {
                "key": "use_beta", "label": "Beta-распределение вместо Гауссианы (continuous)", "type": "int", "default": 0, "min": 0, "max": 1,
                "options": [
                    {"value": 0, "label": "Нет (обычная Гауссиана + clip)"},
                    {"value": 1, "label": "Да — Beta, без клиппинга по границам"},
                ],
            },
            *_memory_hyperparams(default_seq_len=32),
        ],
    },
    {
        "id": "a2c",
        "name": "A2C",
        "kind": "gym",
        "description": "Advantage Actor-Critic — простой и быстрый on-policy метод.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 7e-4, "min": 1e-6, "max": 1e-1},
            {"key": "n_steps", "label": "Steps per update", "type": "int", "default": 5, "min": 1, "max": 256},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "ent_coef", "label": "Entropy coefficient", "type": "float", "default": 0.01, "min": 0.0, "max": 0.1},
            {
                "key": "use_sde", "label": "gSDE (гладкое исследование, continuous)", "type": "int", "default": 0, "min": 0, "max": 1,
                "options": [
                    {"value": 0, "label": "Нет (обычный per-step шум)"},
                    {"value": 1, "label": "Да — state-dependent exploration"},
                ],
            },
            {
                "key": "sde_sample_freq", "label": "gSDE: шагов между пересэмплингом шума", "type": "int", "default": 4, "min": 1, "max": 256,
                "visibleWhen": [{"key": "use_sde", "eq": 1}],
            },
            {
                "key": "sde_log_std_init", "label": "gSDE: начальный log_std", "type": "float", "default": -2.0, "min": -5.0, "max": 1.0,
                "visibleWhen": [{"key": "use_sde", "eq": 1}],
            },
            {
                "key": "head_hidden_size", "label": "Отдельный скрытый слой pi/value поверх признаков", "type": "int",
                "default": 0, "min": 0, "max": 512,
            },
            {
                "key": "use_beta", "label": "Beta-распределение вместо Гауссианы (continuous)", "type": "int", "default": 0, "min": 0, "max": 1,
                "options": [
                    {"value": 0, "label": "Нет (обычная Гауссиана + clip)"},
                    {"value": 1, "label": "Да — Beta, без клиппинга по границам"},
                ],
            },
            *_memory_hyperparams(default_seq_len=32),
        ],
    },
    {
        "id": "sac",
        "name": "SAC",
        "kind": "gym",
        "description": "Soft Actor-Critic — off-policy метод для непрерывных действий, обычно эффективнее по данным, чем PPO/A2C.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 3e-4, "min": 1e-6, "max": 1e-1},
            {"key": "buffer_size", "label": "Replay buffer size", "type": "int", "default": 100_000, "min": 1_000, "max": 1_000_000},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 256, "min": 8, "max": 1024},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "tau", "label": "Target soft-update (tau)", "type": "float", "default": 0.005, "min": 0.0001, "max": 0.1},
            {"key": "ent_coef", "label": "Entropy coefficient", "type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
        ],
    },
    {
        "id": "ddpg",
        "name": "DDPG",
        "kind": "gym",
        "description": "Deep Deterministic Policy Gradient — классический off-policy метод для непрерывных действий: "
                        "детерминированная политика + один critic + Gaussian-шум для исследования. Проще SAC/TD3, "
                        "но менее устойчив (переоценка Q, чувствителен к масштабу шума) — для новых экспериментов "
                        "обычно лучше сразу брать TD3.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
            {"key": "buffer_size", "label": "Replay buffer size", "type": "int", "default": 100_000, "min": 1_000, "max": 1_000_000},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 256, "min": 8, "max": 1024},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "tau", "label": "Target soft-update (tau)", "type": "float", "default": 0.005, "min": 0.0001, "max": 0.1},
            {"key": "exploration_noise", "label": "Шум исследования (доля диапазона действий)", "type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
        ],
    },
    {
        "id": "td3",
        "name": "TD3",
        "kind": "gym",
        "description": "Twin Delayed DDPG — самый надёжный из наших off-policy алгоритмов для непрерывных действий "
                        "с (почти) детерминированной динамикой: два critic'а (берём минимум, против переоценки), "
                        "отложенные обновления актора и сглаживание целевой политики шумом. Хороший дефолт для "
                        "MuJoCo-подобных задач и Car Racing.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
            {"key": "buffer_size", "label": "Replay buffer size", "type": "int", "default": 100_000, "min": 1_000, "max": 1_000_000},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 256, "min": 8, "max": 1024},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "tau", "label": "Target soft-update (tau)", "type": "float", "default": 0.005, "min": 0.0001, "max": 0.1},
            {"key": "exploration_noise", "label": "Шум исследования (доля диапазона действий)", "type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
            {"key": "policy_noise", "label": "Сглаживание целевой политики (σ)", "type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
            {"key": "noise_clip", "label": "Clip шума целевой политики", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            {"key": "policy_delay", "label": "Задержка обновления актора (шагов critic)", "type": "int", "default": 2, "min": 1, "max": 10},
        ],
    },
    {
        "id": "es",
        "name": "Evolution Strategies",
        "kind": "gym",
        "description": "Gradient-free чёрный ящик (OpenAI-ES / диагональный NES, Salimans et al. 2017) — вообще без "
                        "backprop через среду: популяция случайных возмущений параметров сети оценивается полными "
                        "эпизодами, обновление — взвешенное по рангу fitness среднее возмущений. Работает и для "
                        "дискретных, и для непрерывных действий, устойчив к разреженной/недифференцируемой награде, "
                        "но требует много эпизодов на одно обновление — на маленьких classic-control средах это "
                        "нормально, на тяжёлых pixel-средах будет намного медленнее градиентных методов.",
        "hyperparams": [
            {"key": "population_size", "label": "Размер популяции", "type": "int", "default": 32, "min": 2, "max": 512},
            {"key": "sigma", "label": "Sigma (масштаб возмущений)", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 0.02, "min": 1e-4, "max": 1.0},
            {"key": "episodes_per_eval", "label": "Эпизодов на оценку кандидата", "type": "int", "default": 1, "min": 1, "max": 10},
        ],
    },
    {
        "id": "dreamer",
        "name": "Dreamer (World Model)",
        "kind": "gym",
        "description": "Model-based RL (Hafner et al.) — учит RSSM-модель мира (rl_core/world_models/rssm.py) "
                        "на реальном опыте и обучает actor-critic целиком внутри 'воображённых' проходов через неё, "
                        "а не на реальных переходах: один настоящий шаг среды даёт много дешёвых воображаемых для "
                        "обновления политики. Работает и с discrete, и с continuous действиями, и с картиночными, и "
                        "с векторными наблюдениями. Требует world_model_id (World Model Builder) типа RSSM, либо "
                        "строит свежую модель со стандартными гиперпараметрами.",
        "world_model_type": "rssm",
        "hyperparams": [
            {"key": "model_learning_rate", "label": "World model: learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
            {"key": "actor_learning_rate", "label": "Actor: learning rate", "type": "float", "default": 3e-4, "min": 1e-6, "max": 1e-1},
            {"key": "critic_learning_rate", "label": "Critic: learning rate", "type": "float", "default": 3e-4, "min": 1e-6, "max": 1e-1},
            {"key": "batch_size", "label": "Batch size (последовательностей)", "type": "int", "default": 32, "min": 4, "max": 256},
            {"key": "seq_len", "label": "Длина последовательности обучения", "type": "int", "default": 50, "min": 5, "max": 200},
            {"key": "imagination_horizon", "label": "Горизонт воображения", "type": "int", "default": 15, "min": 3, "max": 100},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "entropy_coef", "label": "Entropy coefficient", "type": "float", "default": 1e-3, "min": 0.0, "max": 0.1},
            {"key": "collect_steps_per_iter", "label": "Реальных шагов между обновлениями", "type": "int", "default": 100, "min": 1, "max": 5000},
            {"key": "train_steps_per_iter", "label": "Шагов обучения на итерацию", "type": "int", "default": 10, "min": 1, "max": 200},
            {"key": "learning_starts", "label": "Реальных шагов до начала обучения", "type": "int", "default": 1000, "min": 0, "max": 100_000},
        ],
    },
    {
        "id": "mbpo",
        "name": "MBPO (World Model)",
        "kind": "gym",
        "description": "Model-Based Policy Optimization (Janner et al.) — SAC поверх ансамбля моделей динамики "
                        "(rl_core/world_models/ensemble.py): ансамбль непрерывно доучивается на реальных переходах, "
                        "короткие воображаемые проходы из реальных состояний дополняют реальный буфер для SAC. "
                        "Только непрерывные (Box) действия. Требует world_model_id типа Ensemble, либо строит свежий "
                        "ансамбль со стандартными гиперпараметрами.",
        "world_model_type": "ensemble",
        "hyperparams": [
            {"key": "learning_rate", "label": "SAC: learning rate", "type": "float", "default": 3e-4, "min": 1e-6, "max": 1e-1},
            {"key": "buffer_size", "label": "Real replay buffer size", "type": "int", "default": 100_000, "min": 1_000, "max": 1_000_000},
            {"key": "model_buffer_size", "label": "Model (imagined) buffer size", "type": "int", "default": 100_000, "min": 1_000, "max": 1_000_000},
            {"key": "batch_size", "label": "SAC batch size", "type": "int", "default": 256, "min": 8, "max": 1024},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "tau", "label": "Target soft-update (tau)", "type": "float", "default": 0.005, "min": 0.0001, "max": 0.1},
            {"key": "ent_coef", "label": "Entropy coefficient", "type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
            {"key": "learning_starts", "label": "Реальных шагов до начала обучения", "type": "int", "default": 1000, "min": 0, "max": 100_000},
            {"key": "model_learning_rate", "label": "Ensemble: learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
            {"key": "model_train_freq", "label": "Реальных шагов между дообучением ансамбля", "type": "int", "default": 250, "min": 1, "max": 10_000},
            {"key": "rollout_batch_size", "label": "Воображаемых проходов на дообучение", "type": "int", "default": 400, "min": 1, "max": 10_000},
            {"key": "rollout_length", "label": "Длина воображаемого прохода", "type": "int", "default": 1, "min": 1, "max": 20},
            {"key": "real_ratio", "label": "Доля реальных данных в batch SAC", "type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
        ],
    },
    {
        "id": "pets",
        "name": "PETS (World Model)",
        "kind": "gym",
        "description": "Probabilistic Ensembles with Trajectory Sampling (Chua et al.) — без обучаемой политики "
                        "вообще: каждое реальное действие — результат CEM-поиска по ансамблю моделей динамики "
                        "(rl_core/world_models/ensemble.py), заново решаемого на каждом шаге (online MPC). Только "
                        "непрерывные (Box) действия. Требует world_model_id типа Ensemble, либо строит свежий "
                        "ансамбль со стандартными гиперпараметрами.",
        "world_model_type": "ensemble",
        "hyperparams": [
            {"key": "model_learning_rate", "label": "Ensemble: learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
            {"key": "buffer_size", "label": "Replay buffer size", "type": "int", "default": 100_000, "min": 1_000, "max": 1_000_000},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 256, "min": 8, "max": 1024},
            {"key": "learning_starts", "label": "Случайных шагов до начала планирования", "type": "int", "default": 500, "min": 0, "max": 100_000},
            {"key": "model_train_freq", "label": "Реальных шагов между дообучением ансамбля", "type": "int", "default": 250, "min": 1, "max": 10_000},
            {"key": "cem_horizon", "label": "CEM: горизонт планирования", "type": "int", "default": 15, "min": 1, "max": 100},
            {"key": "cem_candidates", "label": "CEM: число кандидатов", "type": "int", "default": 400, "min": 10, "max": 5000},
            {"key": "cem_elites", "label": "CEM: число элит", "type": "int", "default": 40, "min": 1, "max": 500},
            {"key": "cem_iterations", "label": "CEM: число итераций", "type": "int", "default": 5, "min": 1, "max": 30},
        ],
    },
    {
        "id": "world_models_ha",
        "name": "World Models (VAE + MDN-RNN)",
        "kind": "gym",
        "description": "Классические 'World Models' (Ha & Schmidhuber, 2018) — VAE сжимает наблюдения в латентный "
                        "z, MDN-RNN предсказывает распределение следующего z, крошечный линейный контроллер над "
                        "[z, h] обучается Evolution Strategies уже на замороженной модели мира. Два этапа внутри "
                        "одного запуска: сбор случайных данных + обучение world model, затем ES-эволюция "
                        "контроллера. Требует world_model_id типа VAE+MDN-RNN, либо строит свежую модель со "
                        "стандартными гиперпараметрами.",
        "world_model_type": "vae_mdnrnn",
        "hyperparams": [
            {"key": "world_model_learning_rate", "label": "World model: learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
            {"key": "world_model_phase_steps", "label": "Реальных шагов на обучение world model", "type": "int", "default": 10_000, "min": 500, "max": 1_000_000},
            {"key": "seq_len", "label": "Длина последовательности обучения", "type": "int", "default": 50, "min": 5, "max": 200},
            {"key": "batch_size", "label": "Batch size (последовательностей)", "type": "int", "default": 32, "min": 4, "max": 256},
            {"key": "world_model_train_steps_per_iter", "label": "Шагов обучения world model на итерацию", "type": "int", "default": 20, "min": 1, "max": 200},
            {"key": "population_size", "label": "ES: размер популяции контроллера", "type": "int", "default": 16, "min": 2, "max": 256},
            {"key": "sigma", "label": "ES: sigma (масштаб возмущений)", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0},
            {"key": "es_learning_rate", "label": "ES: learning rate", "type": "float", "default": 0.02, "min": 1e-4, "max": 1.0},
            {"key": "episodes_per_eval", "label": "ES: эпизодов на оценку кандидата", "type": "int", "default": 1, "min": 1, "max": 10},
        ],
    },
    {
        "id": "efficientzero",
        "name": "EfficientZero V2 (MuZero-family)",
        "kind": "gym",
        "description": "Model-based планирование (Wang et al., ICML 2024 Spotlight) — учит компактные "
                        "representation/dynamics/prediction сети (в духе MuZero) и на каждом реальном шаге "
                        "заново строит настоящее дерево Gumbel-поиска (Danihelka et al., 2022): "
                        "Gumbel-Top-k + Sequential Halving среди корневых кандидатов для дискретных действий, "
                        "выборка из [текущей политики + расширенного 'prior'] для непрерывных (фирменное "
                        "отличие V2 от исходного EfficientZero) — с настоящими visit-count'ами, backup'ом "
                        "значений и 'completed Q' (v-mix) на каждом узле, а не упрощённым однолинейным "
                        "просчётом, вместо того чтобы просто исполнять выученную политику. Результат поиска "
                        "('улучшенная' softmax(prior + Q) политика/значение) — сам целевой сигнал для "
                        "обучения policy/value головы, а не сыгранное действие. Модель также учится "
                        "self-supervised consistency loss (SimSiam-style) между предсказанной и настоящей "
                        "следующей латентной, и 'value prefix' — LSTM-головой, предсказывающей накопленную "
                        "(а не единичную) награду внутри одного training-unroll'а, что снижает накопление "
                        "ошибки. Работает и с discrete, и с continuous действиями. training.num_envs > 1 "
                        "по-настоящему батчит поиск: все линии окружений планируют одним общим деревом за "
                        "раз (один batched forward pass на симуляцию), а не независимыми поисками по циклу.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 2e-4, "min": 1e-6, "max": 1e-1},
            {"key": "latent_dim", "label": "Размерность латентного состояния", "type": "int", "default": 64, "min": 8, "max": 512},
            {"key": "hidden_dim", "label": "Размерность скрытых слоёв", "type": "int", "default": 128, "min": 16, "max": 1024},
            {"key": "proj_dim", "label": "Размерность SimSiam-проекции", "type": "int", "default": 64, "min": 8, "max": 512},
            {"key": "buffer_size", "label": "Replay buffer (эпизодов)", "type": "int", "default": 2_000, "min": 10, "max": 100_000},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 64, "min": 4, "max": 1024},
            {"key": "unroll_steps", "label": "Длина training unroll'а", "type": "int", "default": 5, "min": 1, "max": 30},
            {"key": "td_steps", "label": "Горизонт n-step value target", "type": "int", "default": 5, "min": 1, "max": 50},
            {"key": "num_sampled_actions", "label": "Continuous: кандидатов-действий на узел (K)", "type": "int", "default": 8, "min": 2, "max": 64},
            {"key": "num_simulations", "label": "Симуляций поиска на реальный шаг", "type": "int", "default": 32, "min": 2, "max": 256},
            {"key": "num_top_actions", "label": "Корневых кандидатов в Sequential Halving (m)", "type": "int", "default": 8, "min": 2, "max": 64},
            {"key": "c_visit", "label": "Sigma-transform: c_visit", "type": "float", "default": 50.0, "min": 1.0, "max": 200.0},
            {"key": "c_scale", "label": "Sigma-transform: c_scale", "type": "float", "default": 0.1, "min": 0.01, "max": 5.0},
            {"key": "value_minmax_delta", "label": "Мин. эпсилон нормализации Q (MinMaxStats)", "type": "float", "default": 0.01, "min": 1e-4, "max": 1.0},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "value_loss_coef", "label": "Вес value loss", "type": "float", "default": 0.5, "min": 0.0, "max": 10.0},
            {"key": "policy_loss_coef", "label": "Вес policy loss", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            {"key": "reward_loss_coef", "label": "Вес reward (value-prefix) loss", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            {"key": "consistency_loss_coef", "label": "Вес consistency (SimSiam) loss", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            {"key": "continuous_prior_scale", "label": "Continuous: расширение std для prior-кандидатов", "type": "float", "default": 2.5, "min": 1.0, "max": 10.0},
            {"key": "policy_target_temperature", "label": "Температура шума Гумбеля в корне", "type": "float", "default": 1.0, "min": 0.01, "max": 10.0},
            {"key": "train_freq", "label": "Реальных шагов между обновлениями", "type": "int", "default": 1, "min": 1, "max": 1000},
            {"key": "train_steps_per_iter", "label": "Шагов обучения на итерацию", "type": "int", "default": 1, "min": 1, "max": 100},
            {"key": "learning_starts", "label": "Случайных шагов до начала поиска/обучения", "type": "int", "default": 500, "min": 0, "max": 100_000},
            {"key": "max_grad_norm", "label": "Max grad norm", "type": "float", "default": 5.0, "min": 0.1, "max": 100.0},
        ],
    },
    {
        "id": "ippo",
        "name": "Multi-Agent PPO (IPPO)",
        "kind": "gym",
        "description": "MARL: независимая PPO-политика на каждую *команду* сцены (Scene Builder), обучаемая "
                        "параллельно из одного общего мира — агенты внутри команды делят одну сеть "
                        "(parameter sharing), но разные команды учатся на собственных наградах и никогда не "
                        "путают чужой опыт со своим, поэтому предатор и жертва (или красная/синяя команда) "
                        "реально расходятся в поведении. Требует сцену с 2+ группами агентов с разными "
                        "значениями поля 'team' — обычная сцена с одной группой сюда не подходит, используйте "
                        "просто PPO/A2C. Готовые шаблоны с двумя командами: Predator-Prey и Team Battle "
                        "(кнопки в Scene Builder).",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 3e-4, "min": 1e-6, "max": 1e-1},
            {
                "key": "lr_schedule", "label": "Learning rate schedule", "type": "int", "default": 0, "min": 0, "max": 1,
                "options": [
                    {"value": 0, "label": "Постоянный"},
                    {"value": 1, "label": "Линейно убывает до 0"},
                ],
            },
            {"key": "n_steps", "label": "Шагов rollout перед обновлением", "type": "int", "default": 512, "min": 8, "max": 8192},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 64, "min": 8, "max": 2048},
            {"key": "n_epochs", "label": "Эпох на rollout", "type": "int", "default": 10, "min": 1, "max": 50},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "gae_lambda", "label": "GAE lambda", "type": "float", "default": 0.95, "min": 0.5, "max": 1.0},
            {"key": "clip_range", "label": "PPO clip range", "type": "float", "default": 0.2, "min": 0.05, "max": 0.5},
            {"key": "ent_coef", "label": "Entropy coefficient", "type": "float", "default": 0.0, "min": 0.0, "max": 0.1},
            {"key": "vf_coef", "label": "Value loss coefficient", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            {"key": "max_grad_norm", "label": "Max grad norm", "type": "float", "default": 0.5, "min": 0.1, "max": 5.0},
            {
                "key": "use_sde", "label": "gSDE (гладкое исследование, continuous)", "type": "int", "default": 0, "min": 0, "max": 1,
                "options": [
                    {"value": 0, "label": "Нет (обычный per-step шум)"},
                    {"value": 1, "label": "Да — state-dependent exploration"},
                ],
            },
            {
                "key": "sde_sample_freq", "label": "gSDE: шагов между пересэмплингом шума", "type": "int", "default": 4, "min": 1, "max": 256,
                "visibleWhen": [{"key": "use_sde", "eq": 1}],
            },
            {
                "key": "sde_log_std_init", "label": "gSDE: начальный log_std", "type": "float", "default": -2.0, "min": -5.0, "max": 1.0,
                "visibleWhen": [{"key": "use_sde", "eq": 1}],
            },
            {
                "key": "head_hidden_size", "label": "Отдельный скрытый слой pi/value поверх признаков", "type": "int",
                "default": 0, "min": 0, "max": 512,
            },
            {
                "key": "use_beta", "label": "Beta-распределение вместо Гауссианы (continuous)", "type": "int", "default": 0, "min": 0, "max": 1,
                "options": [
                    {"value": 0, "label": "Нет (обычная Гауссиана + clip)"},
                    {"value": 1, "label": "Да — Beta, без клиппинга по границам"},
                ],
            },
        ],
    },
    {
        "id": "qmix",
        "name": "QMIX",
        "kind": "gym",
        "description": "MARL: кооперативный value-decomposition для *команды(-ы)* агентов — все агенты одной "
                        "команды делят одну Q-сеть (parameter sharing), а их Q-значения смешиваются монотонной "
                        "mixing-сетью в одно командное Q_tot, обучаемое как обычный off-policy DQN (replay buffer + "
                        "target-сеть, Double-DQN target + Huber loss для устойчивости — см. qmix.py). В отличие от "
                        "IPPO, который учит каждого агента отдельно policy gradient'ом и требует 2+ *команд*, QMIX "
                        "работает и на одной команде 2+ агентов (это его классический сценарий) — требуется только "
                        "team_ids с 2+ агентами и дискретное движение (discrete4/discrete8); continuous-сцены сюда "
                        "не подходят (используйте 'ippo'). Готовые шаблоны/среды: Predator-Prey, Team Battle, Стая "
                        "против жертв, Командная битва 3×3 (Scene Builder), Knights Archers Zombies, MPE, RWARE, "
                        "LBForaging, SMAC(lite). Для SMAC(lite) (situational action space — не любое действие "
                        "легально в любой момент) qmix — единственный совместимый алгоритм: он маскирует и выбор "
                        "действия (ε-greedy/greedy), и Double-DQN target по get_avail_actions() среды.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 1e-4, "min": 1e-6, "max": 1e-1},
            {"key": "buffer_size", "label": "Размер replay buffer (командных переходов)", "type": "int", "default": 20_000, "min": 500, "max": 500_000},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 32, "min": 4, "max": 512},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "exploration_fraction", "label": "Доля обучения на затухание ε", "type": "float", "default": 0.3, "min": 0.01, "max": 1.0},
            {"key": "exploration_initial_eps", "label": "ε начальное", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            {"key": "exploration_final_eps", "label": "ε конечное", "type": "float", "default": 0.05, "min": 0.0, "max": 1.0},
            {"key": "learning_starts", "label": "Шагов до начала обучения", "type": "int", "default": 1_000, "min": 0, "max": 100_000},
            {"key": "train_freq", "label": "Обучаться каждые N шагов", "type": "int", "default": 4, "min": 1, "max": 100},
            {"key": "target_update_interval", "label": "Обновлять target-сети каждые N шагов", "type": "int", "default": 2_000, "min": 10, "max": 100_000},
            {"key": "mixing_embed_dim", "label": "Mixing network: embed dim", "type": "int", "default": 32, "min": 4, "max": 256},
            {"key": "max_grad_norm", "label": "Max grad norm", "type": "float", "default": 5.0, "min": 0.1, "max": 50.0},
        ],
    },
    {
        "id": "alphazero",
        "name": "AlphaZero",
        "kind": "alphazero",
        "description": "Self-play + MCTS + dual-head сеть — для настольных игр.",
        "hyperparams": [
            {"key": "num_simulations", "label": "MCTS simulations/move", "type": "int", "default": 25, "min": 4, "max": 400},
            {"key": "games_per_iteration", "label": "Self-play games/iteration", "type": "int", "default": 20, "min": 2, "max": 200},
            {"key": "epochs", "label": "Training epochs/iteration", "type": "int", "default": 4, "min": 1, "max": 20},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 64, "min": 8, "max": 512},
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
            {"key": "eval_games", "label": "Arena games/iteration", "type": "int", "default": 10, "min": 2, "max": 100},
            {"key": "win_rate_threshold", "label": "Accept win-rate", "type": "float", "default": 0.55, "min": 0.5, "max": 0.9},
            {"key": "channels", "label": "Conv channels", "type": "int", "default": 48, "min": 8, "max": 256},
            {"key": "num_blocks", "label": "Residual blocks", "type": "int", "default": 3, "min": 1, "max": 10},
        ],
    },
]


@router.get("/list")
async def list_envs():
    return {"environments": list_environments()}


@router.get("/preview")
async def env_preview(id: str = Query(..., min_length=1), thumb: bool = False):
    # resolve_preview_file() does a blocking network download (urllib) on a
    # cache miss (first time a given env's GIF is requested) — off the event
    # loop, or that one download stalls every other request (including
    # other envs' already-cached previews and WS metric traffic) for as
    # long as it takes, which reads to the user as random images failing to
    # load / flickering in and out.
    from starlette.concurrency import run_in_threadpool

    path = await run_in_threadpool(resolve_preview_file, id, thumb=thumb)
    if path is None:
        raise HTTPException(status_code=404, detail=f"No preview for {id}")
    return FileResponse(
        path,
        media_type=preview_media_type(path),
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _custom_algorithm_entries() -> list[dict]:
    """Custom Gym/AlphaZero algorithm plugins (see backend/routes/plugins.py),
    surfaced alongside the built-ins with `is_custom: true` so the
    Experiment Designer can pick them up without any UI logic changes."""
    entries: list[dict] = []
    for slug in plugin_loader.list_gym_algorithm_slugs():
        try:
            entries.append(plugin_loader.gym_algorithm_meta(slug))
        except Exception:
            continue  # broken plugin file — hidden from the catalog, still editable via /api/plugins
    for slug in plugin_loader.list_alphazero_trainer_slugs():
        try:
            entries.append(plugin_loader.alphazero_trainer_meta(slug))
        except Exception:
            continue
    return entries


def _custom_wrapper_entries() -> list[dict]:
    """One wrapper-catalog entry per custom reward-function plugin. The type
    encodes the slug directly (`custom_reward:<slug>`) so several custom
    reward scripts don't collide on the same <select> value in the
    Designer's WrapperNode."""
    entries: list[dict] = []
    for slug in plugin_loader.list_reward_fn_slugs():
        try:
            meta = plugin_loader.reward_fn_meta(slug)
        except Exception:
            continue
        entries.append({
            "type": f"custom_reward:{slug}",
            "label": f"Custom reward: {meta['name']}",
            "params": {"script_id": slug},
            "is_custom": True,
        })
    return entries


@router.get("/wrappers")
async def list_wrappers():
    return {"wrappers": WRAPPER_CATALOG + _custom_wrapper_entries()}


@router.get("/algorithms")
async def list_algorithms():
    return {"algorithms": ALGORITHM_CATALOG + _custom_algorithm_entries()}


@router.post("/inspect")
async def inspect(req: InspectRequest):
    """Live env/network preview for the Experiment Designer — builds the
    (wrapped) env and the algorithm's network with the current selection,
    without running any training, so the UI can show input/output dims,
    layer summary and parameter count while the user is still designing."""
    from rl_core import inspect as inspect_core
    from rl_core.netbuilder_store import resolve_network_spec

    env_id = req.environment.get("id")
    if not env_id:
        raise HTTPException(status_code=400, detail="environment.id is required")
    algo_id = req.algorithm.get("id", "ppo")
    hyperparams = req.algorithm.get("hyperparams") or {}

    # A saved (`network_spec_id`) or inline/unsaved (`network_spec` — the
    # Designer's quick layer editor) hand-designed architecture, resolved
    # exactly like a real run would (see `runner_utils.py`) so this preview
    # reflects it instead of always showing the algorithm's hardcoded net.
    network_spec = resolve_network_spec({"algorithm": req.algorithm})
    if network_spec:
        hyperparams = {**hyperparams, "network_spec": network_spec}

    if req.kind == "alphazero":
        return inspect_core.inspect_alphazero(env_id, algo_id, hyperparams)
    wrapper_specs = req.environment.get("wrappers") or []
    return inspect_core.inspect_gym(env_id, wrapper_specs, algo_id, hyperparams)
