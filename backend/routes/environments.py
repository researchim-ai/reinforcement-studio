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
        "description": "Model-based планирование (Wang et al., ICML 2024 Spotlight) — учит "
                        "representation/dynamics/prediction сети и на каждом шаге строит настоящее дерево "
                        "Gumbel-поиска (Danihelka et al., 2022), а не просто исполняет политику. "
                        "PER (Schaul et al., 2016) + периодический reanalyze. Discrete и continuous.",
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
            {"key": "value_support_size", "label": "Категориальный support value/reward-головы (±N бинов)", "type": "int", "default": 300, "min": 5, "max": 1000},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "value_loss_coef", "label": "Вес value loss", "type": "float", "default": 0.5, "min": 0.0, "max": 10.0},
            {"key": "policy_loss_coef", "label": "Вес policy loss", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            {"key": "reward_loss_coef", "label": "Вес reward (value-prefix) loss", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            {"key": "consistency_loss_coef", "label": "Вес consistency (SimSiam) loss", "type": "float", "default": 5.0, "min": 0.0, "max": 10.0},
            {"key": "continuous_prior_scale", "label": "Continuous: расширение std для prior-кандидатов", "type": "float", "default": 2.5, "min": 1.0, "max": 10.0},
            {"key": "policy_target_temperature", "label": "Температура шума Гумбеля в корне", "type": "float", "default": 1.0, "min": 0.01, "max": 10.0},
            {"key": "train_freq", "label": "Реальных шагов между обновлениями (суммарно по всем env)", "type": "int", "default": 1, "min": 1, "max": 1000},
            {"key": "train_steps_per_iter", "label": "Градиентных шагов за каждые train_freq реальных шагов", "type": "int", "default": 1, "min": 1, "max": 100},
            {"key": "learning_starts", "label": "Случайных шагов до начала поиска/обучения", "type": "int", "default": 500, "min": 0, "max": 100_000},
            {"key": "max_grad_norm", "label": "Max grad norm", "type": "float", "default": 5.0, "min": 0.1, "max": 100.0},
            {"key": "priority_alpha", "label": "Prioritized replay: степень приоритизации (α, 0=uniform)", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            {"key": "priority_beta", "label": "Prioritized replay: коррекция смещения (β, importance sampling)", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            {"key": "min_priority", "label": "Минимальный приоритет перехода", "type": "float", "default": 1e-6, "min": 1e-8, "max": 1.0},
            {"key": "reanalyze_freq", "label": "Реальных шагов между reanalyze-проходами", "type": "int", "default": 200, "min": 1, "max": 100_000},
            {"key": "reanalyze_batch_size", "label": "Переходов, обновляемых за один reanalyze-проход (0 = выключить)", "type": "int", "default": 64, "min": 0, "max": 1024},
        ],
    },
    {
        "id": "unizero",
        "name": "UniZero (MuZero + Transformer)",
        "kind": "gym",
        "description": "Pu, Zhao, Niu et al., ICLR 2025 (LightZero) — тот же рецепт, что и EfficientZero, но "
                        "вместо рекуррентной (LSTM/MLP) динамики — единый causal Transformer над "
                        "последовательностью токенов [obs₀, act₀, obs₁, ...] с RoPE и персистентным "
                        "KV-cache. Discrete и continuous.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 1e-4, "min": 1e-6, "max": 1e-1},
            {"key": "weight_decay", "label": "AdamW weight decay", "type": "float", "default": 1e-4, "min": 0.0, "max": 1.0},
            {"key": "embed_dim", "label": "Размерность токена (embed_dim)", "type": "int", "default": 128, "min": 8, "max": 1024},
            {"key": "num_layers", "label": "Слоёв Transformer'а", "type": "int", "default": 2, "min": 1, "max": 12},
            {"key": "num_heads", "label": "Голов self-attention", "type": "int", "default": 8, "min": 1, "max": 32},
            {"key": "dropout", "label": "Dropout внутри Transformer'а", "type": "float", "default": 0.1, "min": 0.0, "max": 0.5},
            {
                "key": "rotary_emb", "label": "Positional encoding", "type": "int", "default": 1, "min": 0, "max": 1,
                "options": [
                    {"value": 1, "label": "RoPE (быстрее — lossless O(1) KV-cache eviction)"},
                    {"value": 0, "label": "Learned absolute embedding (как в референсе, медленнее)"},
                ],
            },
            {"key": "context_length", "label": "Окно памяти (прошлых транзакций)", "type": "int", "default": 6, "min": 0, "max": 64},
            {"key": "buffer_size", "label": "Replay buffer (эпизодов)", "type": "int", "default": 2_000, "min": 10, "max": 100_000},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 256, "min": 4, "max": 1024},
            {"key": "unroll_steps", "label": "Длина training unroll'а / глубина поиска", "type": "int", "default": 10, "min": 1, "max": 30},
            {"key": "td_steps", "label": "Горизонт n-step value target", "type": "int", "default": 5, "min": 1, "max": 50},
            {"key": "num_sampled_actions", "label": "Continuous: кандидатов-действий на узел (K)", "type": "int", "default": 8, "min": 2, "max": 64},
            {"key": "num_simulations", "label": "Симуляций PUCT-поиска на реальный шаг", "type": "int", "default": 50, "min": 2, "max": 256},
            {"key": "pb_c_base", "label": "PUCT: pb_c_base", "type": "float", "default": 19652.0, "min": 1.0, "max": 100_000.0},
            {"key": "pb_c_init", "label": "PUCT: pb_c_init", "type": "float", "default": 1.25, "min": 0.01, "max": 10.0},
            {"key": "root_dirichlet_alpha", "label": "Dirichlet-шум в корне: α", "type": "float", "default": 0.3, "min": 0.01, "max": 10.0},
            {"key": "root_noise_weight", "label": "Dirichlet-шум в корне: вес", "type": "float", "default": 0.25, "min": 0.0, "max": 1.0},
            {"key": "value_minmax_delta", "label": "Мин. эпсилон нормализации Q (MinMaxStats)", "type": "float", "default": 0.01, "min": 1e-4, "max": 1.0},
            {"key": "value_support_size", "label": "Категориальный support value/reward-головы (±N бинов)", "type": "int", "default": 50, "min": 5, "max": 1000},
            {"key": "label_smoothing_eps", "label": "Label smoothing value/reward-таргетов (ε)", "type": "float", "default": 0.1, "min": 0.0, "max": 0.5},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.997, "min": 0.5, "max": 0.999},
            {"key": "value_loss_coef", "label": "Вес value loss", "type": "float", "default": 0.25, "min": 0.0, "max": 10.0},
            {"key": "policy_loss_coef", "label": "Вес policy loss", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            {"key": "reward_loss_coef", "label": "Вес reward loss", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            {"key": "consistency_loss_coef", "label": "Вес consistency (latent) loss", "type": "float", "default": 10.0, "min": 0.0, "max": 20.0},
            {"key": "policy_entropy_coef", "label": "Вес policy entropy bonus", "type": "float", "default": 5e-3, "min": 0.0, "max": 0.1},
            {"key": "target_update_theta", "label": "EMA target-модели (θ, per train-step)", "type": "float", "default": 0.05, "min": 0.0, "max": 1.0},
            {"key": "continuous_prior_scale", "label": "Continuous: расширение std для prior-кандидатов", "type": "float", "default": 2.5, "min": 1.0, "max": 10.0},
            {"key": "train_freq", "label": "Реальных шагов между обновлениями (суммарно по всем env)", "type": "int", "default": 1, "min": 1, "max": 1000},
            {"key": "train_steps_per_iter", "label": "Градиентных шагов за каждые train_freq реальных шагов", "type": "int", "default": 1, "min": 1, "max": 100},
            {"key": "learning_starts", "label": "Реальных шагов до начала обучения", "type": "int", "default": 2000, "min": 0, "max": 100_000},
            {"key": "max_grad_norm", "label": "Max grad norm", "type": "float", "default": 5.0, "min": 0.1, "max": 100.0},
            {"key": "priority_alpha", "label": "Prioritized replay: степень приоритизации (α, 0=uniform/выкл.)", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
            {"key": "priority_beta", "label": "Prioritized replay: коррекция смещения (β, importance sampling)", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            {"key": "min_priority", "label": "Минимальный приоритет перехода", "type": "float", "default": 1e-6, "min": 1e-8, "max": 1.0},
            {"key": "reanalyze_freq", "label": "Реальных шагов между reanalyze-проходами", "type": "int", "default": 200, "min": 1, "max": 100_000},
            {"key": "reanalyze_batch_size", "label": "Переходов, обновляемых за один reanalyze-проход (0 = выключить)", "type": "int", "default": 0, "min": 0, "max": 1024},
        ],
    },
    {
        "id": "researchimzero",
        "name": "ResearchImZero (наша версия)",
        "kind": "gym",
        "description": "Наш собственный алгоритм: архитектура UniZero (causal Transformer + RoPE + "
                        "персистентный KV-cache) + тренировочный рецепт EfficientZero (Gumbel-поиск, SimSiam "
                        "consistency, PER + reanalyze), плюс EMA target для bootstrap'а, адаптивные лоссы, "
                        "расписания LR/exploration и опциональный RND. Discrete и continuous.",
        "hyperparams": [
            {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 2e-4, "min": 1e-6, "max": 1e-1},
            {
                "key": "lr_schedule", "label": "LR schedule", "type": "int", "default": 1, "min": 0, "max": 1,
                "options": [
                    {"value": 1, "label": "Cosine decay после learning_starts (стабильнее к концу обучения)"},
                    {"value": 0, "label": "Постоянный learning rate"},
                ],
            },
            {
                "key": "lr_min_fraction", "label": "LR schedule: минимум (доля от базового lr)", "type": "float",
                "default": 0.05, "min": 0.0, "max": 1.0, "visibleWhen": [{"key": "lr_schedule", "eq": 1}],
            },
            {"key": "embed_dim", "label": "Размерность токена (embed_dim)", "type": "int", "default": 128, "min": 8, "max": 1024},
            {"key": "num_layers", "label": "Слоёв Transformer'а", "type": "int", "default": 2, "min": 1, "max": 12},
            {"key": "num_heads", "label": "Голов self-attention", "type": "int", "default": 4, "min": 1, "max": 32},
            {"key": "dropout", "label": "Dropout внутри Transformer'а", "type": "float", "default": 0.0, "min": 0.0, "max": 0.5},
            {
                "key": "rotary_emb", "label": "Positional encoding", "type": "int", "default": 1, "min": 0, "max": 1,
                "options": [
                    {"value": 1, "label": "RoPE (быстрее — lossless O(1) KV-cache eviction)"},
                    {"value": 0, "label": "Learned absolute embedding (медленнее)"},
                ],
            },
            {"key": "context_length", "label": "Окно памяти (прошлых транзакций)", "type": "int", "default": 6, "min": 0, "max": 64},
            {"key": "buffer_size", "label": "Replay buffer (эпизодов)", "type": "int", "default": 2_000, "min": 10, "max": 100_000},
            {"key": "batch_size", "label": "Batch size", "type": "int", "default": 64, "min": 4, "max": 1024},
            {"key": "unroll_steps", "label": "Длина training unroll'а / глубина поиска", "type": "int", "default": 5, "min": 1, "max": 30},
            {"key": "td_steps", "label": "Горизонт n-step value target", "type": "int", "default": 5, "min": 1, "max": 50},
            {"key": "num_sampled_actions", "label": "Continuous: кандидатов-действий на узел (K)", "type": "int", "default": 8, "min": 2, "max": 64},
            {"key": "num_simulations", "label": "Симуляций Gumbel-поиска на реальный шаг", "type": "int", "default": 32, "min": 2, "max": 256},
            {"key": "num_top_actions", "label": "Корневых кандидатов в Sequential Halving (m)", "type": "int", "default": 8, "min": 2, "max": 64},
            {"key": "c_visit", "label": "Sigma-transform: c_visit", "type": "float", "default": 50.0, "min": 1.0, "max": 200.0},
            {"key": "c_scale", "label": "Sigma-transform: c_scale", "type": "float", "default": 0.1, "min": 0.01, "max": 5.0},
            {"key": "policy_target_temperature", "label": "Температура шума Гумбеля в корне", "type": "float", "default": 1.0, "min": 0.01, "max": 10.0},
            {
                "key": "temperature_anneal_start_scale",
                "label": "Больше exploration/entropy в начале обучения (множитель, →1.0 к концу)",
                "type": "float", "default": 1.5, "min": 1.0, "max": 5.0,
            },
            {"key": "value_minmax_delta", "label": "Мин. эпсилон нормализации Q (MinMaxStats)", "type": "float", "default": 0.01, "min": 1e-4, "max": 1.0},
            {"key": "value_support_size", "label": "Категориальный support value/reward-головы (±N бинов)", "type": "int", "default": 300, "min": 5, "max": 1000},
            {"key": "label_smoothing_eps", "label": "Label smoothing value/reward-таргетов (ε)", "type": "float", "default": 0.0, "min": 0.0, "max": 0.5},
            {"key": "gamma", "label": "Discount (gamma)", "type": "float", "default": 0.99, "min": 0.5, "max": 0.999},
            {"key": "value_loss_coef", "label": "Вес value loss", "type": "float", "default": 0.25, "min": 0.0, "max": 10.0},
            {"key": "policy_loss_coef", "label": "Вес policy loss", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            {"key": "reward_loss_coef", "label": "Вес reward loss", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            {"key": "consistency_loss_coef", "label": "Вес consistency (SimSiam) loss", "type": "float", "default": 2.0, "min": 0.0, "max": 20.0},
            {
                "key": "adaptive_loss_weights", "label": "Адаптивное взвешивание лоссов", "type": "int", "default": 0, "min": 0, "max": 1,
                "options": [
                    {"value": 0, "label": "Нет — только статичные *_loss_coef (рекомендуется)"},
                    {"value": 1, "label": "Да — learnable uncertainty weights (Kendall et al.), риск перекоса в пользу «лёгких» задач"},
                ],
            },
            {"key": "proj_dim", "label": "SimSiam projector/predictor: размерность", "type": "int", "default": 64, "min": 8, "max": 512},
            {"key": "policy_entropy_coef", "label": "Вес policy entropy bonus", "type": "float", "default": 5e-3, "min": 0.0, "max": 0.1},
            {"key": "continuous_prior_scale", "label": "Continuous: расширение std для prior-кандидатов", "type": "float", "default": 2.5, "min": 1.0, "max": 10.0},
            {"key": "train_freq", "label": "Суммарных env-шагов между группами обновлений", "type": "int", "default": 1, "min": 1, "max": 1000},
            {"key": "train_steps_per_iter", "label": "Градиентных шагов за vector-итерацию обучения", "type": "int", "default": 1, "min": 1, "max": 100},
            {
                "key": "auto_scale_replay_ratio", "label": "Дополнительный автоскейл learner под num_envs", "type": "int", "default": 0, "min": 0, "max": 1,
                "options": [
                    {"value": 0, "label": "Нет — быстрый режим, train_steps_per_iter как задано (рекомендуется)"},
                    {"value": 1, "label": "Да — умножает обновления за vector-итерацию на num_envs//4"},
                ],
            },
            {"key": "learning_starts", "label": "Реальных шагов до начала обучения", "type": "int", "default": 500, "min": 0, "max": 100_000},
            {"key": "max_grad_norm", "label": "Max grad norm", "type": "float", "default": 5.0, "min": 0.1, "max": 100.0},
            {"key": "priority_alpha", "label": "Prioritized replay: степень приоритизации (α, 0=uniform/выкл.)", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            {"key": "priority_beta", "label": "Prioritized replay: коррекция смещения (β, importance sampling, конечное значение)", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
            {"key": "priority_beta_start", "label": "Prioritized replay: β в начале обучения (аннилится до priority_beta)", "type": "float", "default": 0.4, "min": 0.0, "max": 1.0},
            {"key": "priority_policy_weight", "label": "Приоритет: вклад ошибки policy (сверх ошибки value)", "type": "float", "default": 0.1, "min": 0.0, "max": 5.0},
            {"key": "min_priority", "label": "Минимальный приоритет перехода", "type": "float", "default": 1e-6, "min": 1e-8, "max": 1.0},
            {"key": "reanalyze_freq", "label": "Реальных шагов между reanalyze-проходами", "type": "int", "default": 200, "min": 1, "max": 100_000},
            {"key": "reanalyze_batch_size", "label": "Переходов, обновляемых за один reanalyze-проход (0 = выключить)", "type": "int", "default": 64, "min": 0, "max": 1024},
            {
                "key": "use_target_for_bootstrap", "label": "EMA target-сеть для value-bootstrap", "type": "int", "default": 1, "min": 0, "max": 1,
                "options": [
                    {"value": 1, "label": "Да — Polyak-EMA копия сети (устойчивее без самоссылочного TD-таргета)"},
                    {"value": 0, "label": "Нет — bootstrap online-сетью напрямую"},
                ],
            },
            {
                "key": "target_update_theta", "label": "EMA target: скорость обновления (θ)", "type": "float", "default": 0.02, "min": 0.001, "max": 1.0,
                "visibleWhen": [{"key": "use_target_for_bootstrap", "eq": 1}],
            },
            {"key": "search_value_max_staleness", "label": "Reanalyze: макс. \"возраст\" search_value в value target (поколений)", "type": "int", "default": 5, "min": 0, "max": 1000},
            {
                "key": "use_amp", "label": "Mixed precision (AMP) на GPU", "type": "int", "default": 1, "min": 0, "max": 1,
                "options": [
                    {"value": 1, "label": "Да — bf16/fp16 автокаст в train_step (быстрее и меньше памяти на GPU, без CPU)"},
                    {"value": 0, "label": "Нет — всё в fp32"},
                ],
            },
            {
                "key": "intrinsic_exploration", "label": "Intrinsic motivation (RND)", "type": "int", "default": 0, "min": 0, "max": 1,
                "options": [
                    {"value": 0, "label": "Нет"},
                    {"value": 1, "label": "Да — Random Network Distillation (бонус за новизну, полезно для hard-exploration сред)"},
                ],
            },
            {
                "key": "rnd_bonus_coef", "label": "RND: вес intrinsic-бонуса", "type": "float", "default": 0.1, "min": 0.0, "max": 5.0,
                "visibleWhen": [{"key": "intrinsic_exploration", "eq": 1}],
            },
            {
                "key": "rnd_learning_rate", "label": "RND: learning rate предиктора", "type": "float", "default": 1e-4, "min": 1e-6, "max": 1e-1,
                "visibleWhen": [{"key": "intrinsic_exploration", "eq": 1}],
            },
            {
                "key": "rnd_feature_dim", "label": "RND: размерность признаков", "type": "int", "default": 128, "min": 8, "max": 1024,
                "visibleWhen": [{"key": "intrinsic_exploration", "eq": 1}],
            },
            {
                "key": "rnd_hidden_dim", "label": "RND: размерность скрытого слоя", "type": "int", "default": 128, "min": 8, "max": 1024,
                "visibleWhen": [{"key": "intrinsic_exploration", "eq": 1}],
            },
            {
                "key": "rnd_bonus_clip", "label": "RND: клиппинг нормализованного бонуса", "type": "float", "default": 5.0, "min": 0.1, "max": 50.0,
                "visibleWhen": [{"key": "intrinsic_exploration", "eq": 1}],
            },
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


def _d(ru: str, en: str) -> dict[str, str]:
    return {"ru": ru, "en": en}


# Hover-tooltip text for hyperparameters, shown in the Experiment Designer
# and the Training Monitor. Keyed by hyperparam `key` — one description
# covers every algorithm that exposes a hyperparam under that name, since
# the vast majority (learning_rate, gamma, batch_size, ...) mean exactly the
# same thing everywhere. The handful of keys whose *meaning* genuinely
# differs by algorithm (not just units/scale — see e.g. "num_simulations":
# MCTS moves for AlphaZero vs. tree-search sims per real env step for
# EfficientZero) get a per-(algo_id, key) override in
# `_HP_DESCRIPTIONS_BY_ALGO` below, applied on top of (and, if present,
# instead of) this dict by `_attach_hyperparam_descriptions()`.
_HP_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "weight_decay": _d(
        "L2-регуляризация весов сети в оптимизаторе (AdamW) — штрафует слишком большие веса, снижая риск "
        "переобучения. 0 полностью выключает регуляризацию.",
        "L2 weight regularization inside the optimizer (AdamW) — penalizes overly large weights, reducing "
        "overfitting risk. 0 fully disables it.",
    ),
    "label_smoothing_eps": _d(
        "Label smoothing (ε) для категориальных value/reward-таргетов: итоговый таргет = (1-ε)×two-hot + "
        "ε/N_бинов — сглаживает распределение, снижая риск переуверенности сети в конкретном бине. 0 — обычный "
        "two-hot без сглаживания.",
        "Label smoothing (ε) for the categorical value/reward targets: final target = (1-ε)×two-hot + "
        "ε/num_bins — smooths the distribution, reducing overconfidence in a single bin. 0 is a plain two-hot "
        "with no smoothing.",
    ),
    "policy_entropy_coef": _d(
        "Вес энтропийного бонуса политики, ВЫЧИТАЕМого из общего лосса — поощряет более разнообразную "
        "(менее уверенную) политику, снижая риск преждевременной сходимости к субоптимальному действию. "
        "0 выключает бонус.",
        "Weight of the policy entropy bonus, SUBTRACTED from the total loss — encourages a more diverse (less "
        "overconfident) policy, reducing the risk of premature convergence to a suboptimal action. 0 disables "
        "the bonus.",
    ),
    "target_update_theta": _d(
        "Коэффициент EMA-обновления target-сети (используется для bootstrap value и latent-consistency таргета): "
        "target ← (1-θ)×target + θ×online, применяется на каждом шаге обучения. Меньше θ — медленнее и стабильнее "
        "обновление target-сети.",
        "EMA update coefficient for the target network (used for the bootstrap value and the latent-consistency "
        "target): target ← (1-θ)×target + θ×online, applied every training step. Smaller θ means a slower, more "
        "stable target-network update.",
    ),
    "learning_rate": _d(
        "Скорость обучения — размер шага градиентного спуска при обновлении весов сети. Слишком большое значение "
        "делает обучение нестабильным (лоссы скачут или расходятся), слишком маленькое — обучение идёт очень медленно.",
        "Learning rate — the gradient-descent step size used to update the network's weights. Too high makes "
        "training unstable (losses spike or diverge); too low makes training crawl.",
    ),
    "buffer_size": _d(
        "Максимальное число прошлых переходов (или эпизодов), которые хранятся в replay buffer для последующего "
        "переобучения. Больше — разнообразнее батчи и меньше корреляция между сэмплами, но больше памяти и старые, "
        "уже неактуальные данные дольше задерживаются в буфере.",
        "Maximum number of past transitions (or episodes) kept in the replay buffer for later re-training. Bigger "
        "means more diverse batches and less correlation between samples, but more memory use and stale data "
        "lingers longer.",
    ),
    "batch_size": _d(
        "Сколько сэмплов из буфера/rollout'а берётся за один шаг обновления сети. Больше — более гладкая, менее "
        "шумная оценка градиента, но дороже по памяти/времени на шаг; меньше — быстрее шаги, но шумнее градиент.",
        "How many samples from the buffer/rollout are used per gradient update. Bigger gives a smoother, less "
        "noisy gradient estimate but costs more memory/time per step; smaller means faster steps but noisier "
        "gradients.",
    ),
    "gamma": _d(
        "Коэффициент дисконтирования будущей награды (0..1). Ближе к 1 — агент ценит отдалённые во времени "
        "награды почти так же, как немедленные (нужно для задач с длинным горизонтом); ближе к 0 — агент почти "
        "'близорук' и оптимизирует только скорую награду.",
        "Discount factor for future reward (0..1). Closer to 1 means the agent values rewards far in the future "
        "almost as much as immediate ones (needed for long-horizon tasks); closer to 0 makes the agent "
        "near-sighted, optimizing mostly for imminent reward.",
    ),
    "action_exploration": _d(
        "Способ исследования действий: ε-greedy — со случайной вероятностью ε брать случайное действие вместо "
        "жадного максимума Q; NoisyNet — вместо этого добавляет обучаемый параметрический шум прямо в веса сети, "
        "так что исследование становится частью самой политики и убывает само по мере обучения.",
        "How action exploration is done: ε-greedy — with probability ε take a random action instead of the "
        "greedy Q-argmax; NoisyNet — instead injects learnable parametric noise directly into the network's "
        "weights, so exploration becomes part of the policy itself and naturally decays as training progresses.",
    ),
    "exploration_fraction": _d(
        "Доля от общего числа шагов обучения, за которую ε спадает от начального до конечного значения. После "
        "этой доли ε остаётся на конечном уровне до конца обучения.",
        "Fraction of total training steps over which ε decays from its initial to its final value. After that "
        "fraction, ε stays at the final level for the rest of training.",
    ),
    "exploration_initial_eps": _d(
        "Начальное значение ε (доля полностью случайных действий) в самом начале обучения.",
        "Initial value of ε (the fraction of fully random actions) at the very start of training.",
    ),
    "exploration_final_eps": _d(
        "Конечное значение ε после затухания — минимальный уровень случайности действий, который сохраняется до "
        "конца обучения (полностью убирать случайность обычно не стоит — иначе агент застревает в локально "
        "оптимальном поведении).",
        "Final value of ε after decay — the minimum amount of action randomness kept for the rest of training "
        "(dropping it to exactly zero usually isn't a good idea — the agent can get stuck in locally-optimal "
        "behaviour).",
    ),
    "noisy_sigma0": _d(
        "Начальный масштаб (σ) параметрического шума NoisyNet-слоёв — насколько сильно шум искажает веса в начале "
        "обучения (сеть сама учится уменьшать его там, где уверенность высока).",
        "Initial scale (σ) of the parametric noise in NoisyNet layers — how strongly the noise perturbs the "
        "weights at the start of training (the network itself learns to shrink it wherever it's confident).",
    ),
    "intrinsic_exploration": _d(
        "Внутренняя (intrinsic) мотивация вдобавок к награде среды: RND (Random Network Distillation) добавляет "
        "бонус за посещение состояний, которые сеть-предиктор плохо предсказывает (т.е. 'новых'/малоизученных), "
        "что помогает в средах с редкой внешней наградой.",
        "Intrinsic motivation on top of the environment reward: RND (Random Network Distillation) adds a bonus "
        "for visiting states that a predictor network still predicts poorly (i.e. 'novel'/under-explored), which "
        "helps in environments with sparse external reward.",
    ),
    "rnd_bonus_coef": _d(
        "Коэффициент, с которым intrinsic-бонус RND добавляется к внешней награде среды при обучении.",
        "Coefficient with which the RND intrinsic bonus is added to the environment's external reward during "
        "training.",
    ),
    "rnd_learning_rate": _d(
        "Learning rate отдельной сети-предиктора RND (учится приближать выход случайной фиксированной "
        "target-сети; ошибка приближения — и есть бонус новизны).",
        "Learning rate of the separate RND predictor network (it learns to approximate the output of a fixed "
        "random target network; the approximation error is exactly the novelty bonus).",
    ),
    "rnd_feature_dim": _d(
        "Размер выходного признакового вектора target- и predictor-сетей RND.",
        "Output feature-vector size of the RND target and predictor networks.",
    ),
    "rnd_hidden_dim": _d(
        "Размер скрытых слоёв target- и predictor-сетей RND.",
        "Hidden-layer size of the RND target and predictor networks.",
    ),
    "rnd_bonus_clip": _d(
        "Максимальное значение, до которого обрезается (clip) intrinsic-бонус RND за один шаг — защита от "
        "выбросов, которые иначе могли бы 'перекрыть' реальную награду среды.",
        "Maximum value the per-step RND intrinsic bonus is clipped to — guards against outliers that would "
        "otherwise drown out the environment's real reward.",
    ),
    "memory_type": _d(
        "Рекуррентное ядро сети: 'Нет' — обычная feed-forward сеть без памяти о прошлых шагах эпизода; LSTM/GRU — "
        "добавляет рекуррентный слой с hidden state, переносящимся между шагами внутри эпизода, что даёт агенту "
        "память о недавней истории (важно для частично наблюдаемых сред). Игнорируется, если выбрана "
        "самодельная архитектура (Network Builder).",
        "The network's recurrent core: 'None' — a plain feed-forward network with no memory of past steps in the "
        "episode; LSTM/GRU — adds a recurrent layer with a hidden state carried across steps within an episode, "
        "giving the agent memory of recent history (important for partially-observable environments). Ignored "
        "when a hand-designed architecture (Network Builder) is selected instead.",
    ),
    "memory_hidden_size": _d(
        "Размер hidden state рекуррентного слоя (LSTM/GRU).",
        "Hidden-state size of the recurrent (LSTM/GRU) layer.",
    ),
    "memory_num_layers": _d(
        "Число слоёв стека рекуррентной сети (LSTM/GRU) один над другим.",
        "Number of stacked recurrent (LSTM/GRU) layers on top of each other.",
    ),
    "memory_seq_len": _d(
        "Длина последовательности для Backpropagation Through Time (BPTT) — на сколько шагов назад считается "
        "градиент через рекуррентное состояние при одном обновлении. Больше — точнее учитывается длинная "
        "история, но дороже по памяти/времени на шаг обучения.",
        "Sequence length for Backpropagation Through Time (BPTT) — how many steps back the gradient is "
        "propagated through the recurrent state on a single update. Longer means longer history is accounted for "
        "more accurately, but costs more memory/time per training step.",
    ),
    "n_step": _d(
        "Горизонт n-step возврата: вместо одного шага накопленной наградой используется сумма из n реальных "
        "наград плюс дисконтированная Q-оценка на n-м шаге вперёд. Больше n — меньше смещение (bias) от неверной "
        "Q-оценки, но больше дисперсия (variance) целевого значения.",
        "Horizon of the n-step return: instead of a single-step target, the sum of n real rewards plus a "
        "discounted Q-estimate n steps ahead is used. Larger n means less bias from an inaccurate Q-estimate but "
        "more variance in the target.",
    ),
    "per_alpha": _d(
        "Prioritized Experience Replay: степень приоритизации переходов с большой TD-ошибкой при сэмплировании "
        "из буфера (0 = равномерное сэмплирование, 1 = сэмплирование строго пропорционально приоритету).",
        "Prioritized Experience Replay: how strongly transitions with a large TD-error are favored when sampling "
        "from the buffer (0 = uniform sampling, 1 = sampling strictly proportional to priority).",
    ),
    "per_beta_start": _d(
        "Prioritized Experience Replay: начальная степень importance-sampling коррекции смещения, вносимого "
        "приоритизированным (неравномерным) сэмплированием — обычно линейно растёт до 1 к концу обучения.",
        "Prioritized Experience Replay: initial strength of the importance-sampling correction for the bias "
        "introduced by prioritized (non-uniform) sampling — typically annealed up to 1 by the end of training.",
    ),
    "distributional": _d(
        "QR-DQN (Quantile Regression DQN): предсказывать не одно скалярное Q-значение, а целое распределение "
        "возможных возвратов (набор квантилей) — часто устойчивее и точнее захватывает риск/дисперсию отдачи.",
        "QR-DQN (Quantile Regression DQN): predict not a single scalar Q-value but a full distribution over "
        "possible returns (a set of quantiles) — often more stable and better captures the risk/variance of "
        "the return.",
    ),
    "num_quantiles": _d(
        "QR-DQN: число квантилей, которыми аппроксимируется распределение возврата.",
        "QR-DQN: number of quantiles used to approximate the return distribution.",
    ),
    "lr_schedule": _d(
        "Расписание learning rate за время обучения: постоянный, либо линейно убывающий от заданного значения до "
        "0 к концу обучения (часто стабилизирует финальную политику).",
        "Learning-rate schedule over the course of training: constant, or linearly decaying from the given value "
        "down to 0 by the end of training (often stabilizes the final policy).",
    ),
    "n_steps": _d(
        "Сколько шагов среды (на каждое из параллельных окружений) собирается в один rollout, прежде чем "
        "выполняется on-policy обновление сети. Больше — точнее оценка advantage, но реже обновления.",
        "How many environment steps (per parallel environment) are collected into one rollout before an "
        "on-policy update is performed. Bigger gives a more accurate advantage estimate but less frequent "
        "updates.",
    ),
    "ent_coef": _d(
        "Коэффициент энтропийного бонуса в лоссе политики — поощряет более случайную (менее детерминированную) "
        "политику, что поддерживает исследование и не даёт политике преждевременно 'схлопнуться' в одно действие.",
        "Entropy-bonus coefficient in the policy loss — rewards a more random (less deterministic) policy, which "
        "sustains exploration and keeps the policy from collapsing onto a single action too early.",
    ),
    "use_sde": _d(
        "gSDE (generalized State-Dependent Exploration) — вместо независимого шума на каждом шаге сэмплирует шум "
        "действия реже (см. sde_sample_freq) и делает его зависящим от состояния, что даёт более гладкие, "
        "физически осмысленные траектории исследования в continuous-задачах (например, движение робота).",
        "gSDE (generalized State-Dependent Exploration) — instead of independent per-step noise, samples action "
        "noise less often (see sde_sample_freq) and makes it state-dependent, giving smoother, more physically "
        "plausible exploration trajectories in continuous-action tasks (e.g. robot locomotion).",
    ),
    "sde_sample_freq": _d(
        "gSDE: через сколько шагов среды заново пересэмплируется шум исследования (а не на каждом шаге, как в "
        "обычном подходе).",
        "gSDE: after how many environment steps the exploration noise is re-sampled (rather than every single "
        "step, as with the usual approach).",
    ),
    "sde_log_std_init": _d(
        "gSDE: начальное значение log(std) для распределения шума действия — задаёт исходный масштаб "
        "исследования до того, как он подстроится обучением.",
        "gSDE: initial log(std) value for the action-noise distribution — sets the starting exploration scale "
        "before it's tuned by training.",
    ),
    "head_hidden_size": _d(
        "Размер дополнительного скрытого слоя, добавляемого поверх общих признаков отдельно перед policy- и "
        "value-головой (0 — без него, головы линейные прямо от общих признаков).",
        "Size of an extra hidden layer added on top of the shared features, separately before the policy and "
        "value heads (0 — no extra layer, the heads are linear straight off the shared features).",
    ),
    "use_beta": _d(
        "Для continuous-действий: использовать Beta-распределение (заданное на конечном отрезке, без "
        "необходимости отдельно клиппировать выход по границам диапазона действий) вместо обычной Гауссианы "
        "с последующим clip'ом.",
        "For continuous actions: use a Beta distribution (naturally bounded to a finite interval, no separate "
        "clipping of the output to the action range needed) instead of the usual Gaussian followed by clipping.",
    ),
    "tau": _d(
        "Коэффициент мягкого (soft) обновления target-сети: target = tau * online + (1 - tau) * target на каждом "
        "шаге обучения, вместо резкой периодической копии весов. Меньше — более стабильные, но медленнее "
        "'догоняющие' target-значения.",
        "Soft target-network update coefficient: target = tau * online + (1 - tau) * target on every training "
        "step, instead of an abrupt periodic weight copy. Smaller means more stable but slower-tracking target "
        "values.",
    ),
    "exploration_noise": _d(
        "Масштаб гауссовского шума, добавляемого к действию актора при сборе опыта в среде (в долях диапазона "
        "действий) — единственный источник исследования для детерминированной политики DDPG/TD3.",
        "Scale of the Gaussian noise added to the actor's action while collecting experience in the environment "
        "(as a fraction of the action range) — the only source of exploration for a deterministic DDPG/TD3 "
        "policy.",
    ),
    "policy_noise": _d(
        "TD3: масштаб шума, добавляемого к действию target-актора при вычислении target-значения (target policy "
        "smoothing) — сглаживает Q-функцию по соседним действиям, снижая переоценку на узких пиках.",
        "TD3: scale of the noise added to the target actor's action when computing the target value (target "
        "policy smoothing) — smooths the Q-function over nearby actions, reducing overestimation at sharp "
        "peaks.",
    ),
    "noise_clip": _d(
        "TD3: максимальная величина, до которой обрезается шум сглаживания целевой политики (policy_noise).",
        "TD3: the maximum magnitude the target-policy-smoothing noise (policy_noise) is clipped to.",
    ),
    "policy_delay": _d(
        "TD3: раз в сколько обновлений critic'а обновляется actor (и target-сети) — задержка снижает накопление "
        "ошибки от быстро 'убегающей' политики относительно ещё не сошедшегося critic'а.",
        "TD3: how often (in critic updates) the actor and target networks are updated — the delay reduces error "
        "accumulation from a policy racing ahead of a not-yet-converged critic.",
    ),
    "population_size": _d(
        "Evolution Strategies: число случайных возмущений параметров сети (кандидатов), оцениваемых полными "
        "эпизодами за одно обновление. Больше — точнее оценка градиента ранга fitness, но дороже (пропорционально "
        "больше эпизодов на одно обновление).",
        "Evolution Strategies: number of random parameter perturbations (candidates) evaluated with full "
        "episodes per update. Bigger gives a more accurate rank-fitness gradient estimate but is proportionally "
        "more expensive (more episodes per update).",
    ),
    "sigma": _d(
        "Evolution Strategies: масштаб (стандартное отклонение) случайных возмущений параметров сети, "
        "используемых для оценки популяции.",
        "Evolution Strategies: scale (standard deviation) of the random parameter perturbations used to evaluate "
        "the population.",
    ),
    "episodes_per_eval": _d(
        "Сколько полных эпизодов проигрывается для оценки fitness одного кандидата популяции (больше — точнее, "
        "но дороже; полезно при шумной/стохастической среде).",
        "How many full episodes are played to estimate the fitness of a single population candidate (more is "
        "more accurate but costlier; useful in a noisy/stochastic environment).",
    ),
    "model_learning_rate": _d(
        "Learning rate модели мира (world model) — динамики/энкодера, отдельно от learning rate агента "
        "(actor/critic), поверх неё обучающегося.",
        "Learning rate of the world model — the dynamics/encoder network, separate from the learning rate of the "
        "agent (actor/critic) trained on top of it.",
    ),
    "actor_learning_rate": _d(
        "Learning rate сети-актора (политики), обучаемой на 'воображённых' проходах внутри модели мира.",
        "Learning rate of the actor (policy) network, trained on imagined rollouts inside the world model.",
    ),
    "critic_learning_rate": _d(
        "Learning rate сети-критика (value function), обучаемой на 'воображённых' проходах внутри модели мира.",
        "Learning rate of the critic (value function) network, trained on imagined rollouts inside the world "
        "model.",
    ),
    "seq_len": _d(
        "Длина последовательности реальных шагов, сэмплируемой из буфера для одного обновления модели мира "
        "(и/или её проверки на реальных данных).",
        "Length of the sequence of real steps sampled from the buffer for a single world-model update (and/or "
        "its evaluation against real data).",
    ),
    "imagination_horizon": _d(
        "Dreamer: на сколько шагов вперёд 'воображается' траектория внутри модели мира (RSSM) для обучения "
        "actor-critic — без единого реального шага среды на всю эту длину.",
        "Dreamer: how many steps ahead a trajectory is 'imagined' inside the world model (RSSM) to train the "
        "actor-critic — without a single real environment step for the entire length.",
    ),
    "entropy_coef": _d(
        "Коэффициент энтропийного бонуса политики актора при обучении на воображённых траекториях — поддерживает "
        "исследование внутри модели мира.",
        "Entropy-bonus coefficient for the actor's policy when training on imagined trajectories — sustains "
        "exploration inside the world model.",
    ),
    "collect_steps_per_iter": _d(
        "Dreamer: сколько настоящих шагов среды собирается в реальный буфер между итерациями обучения модели "
        "мира и actor-critic.",
        "Dreamer: how many real environment steps are collected into the real buffer between world-model and "
        "actor-critic training iterations.",
    ),
    "train_steps_per_iter": _d(
        "Сколько градиентных обновлений выполняется за одну итерацию обучения (на заданное число собранных "
        "реальных шагов) — регулирует соотношение 'обновлений на единицу собранного опыта'.",
        "How many gradient updates are performed per training iteration (for a given amount of collected real "
        "experience) — controls the ratio of 'updates per unit of collected experience'.",
    ),
    "learning_starts": _d(
        "Сколько первых шагов среды агент проходит со случайной/неисследованной политикой перед тем как "
        "начинаются обновления сети/модели — даёт буферу накопить достаточно разнообразных данных до первого "
        "обучения.",
        "How many initial environment steps the agent takes with a random/unlearned policy before any "
        "network/model updates start — lets the buffer accumulate enough diverse data before the first training "
        "step.",
    ),
    "model_buffer_size": _d(
        "MBPO: размер отдельного буфера для 'воображённых' (сгенерированных ансамблем моделей динамики) "
        "переходов, дополняющих реальный буфер при обучении SAC.",
        "MBPO: size of the separate buffer for 'imagined' transitions (generated by the dynamics ensemble) that "
        "supplement the real buffer when training SAC.",
    ),
    "model_train_freq": _d(
        "Через сколько реальных шагов среды ансамбль моделей динамики дообучается заново на свежих данных.",
        "After how many real environment steps the dynamics ensemble is re-trained on fresh data.",
    ),
    "rollout_batch_size": _d(
        "MBPO: сколько воображаемых проходов (rollout'ов) через ансамбль генерируется за одно дообучение "
        "ансамбля, для пополнения буфера воображённых данных.",
        "MBPO: how many imagined rollouts through the ensemble are generated per ensemble re-training pass, to "
        "refill the imagined-data buffer.",
    ),
    "rollout_length": _d(
        "MBPO: длина (в шагах) каждого воображаемого прохода через ансамбль моделей динамики, начинающегося из "
        "реального состояния.",
        "MBPO: length (in steps) of each imagined rollout through the dynamics ensemble, starting from a real "
        "state.",
    ),
    "real_ratio": _d(
        "MBPO: доля настоящих (не воображённых) переходов в каждом batch, на котором обучается SAC — остальное "
        "берётся из буфера воображённых данных.",
        "MBPO: fraction of real (not imagined) transitions in every batch SAC is trained on — the rest comes "
        "from the imagined-data buffer.",
    ),
    "cem_horizon": _d(
        "PETS: горизонт планирования (число шагов вперёд), на который CEM ищет последовательность действий через "
        "ансамбль моделей динамики на каждом реальном шаге (online MPC).",
        "PETS: planning horizon (number of steps ahead) over which CEM searches for an action sequence through "
        "the dynamics ensemble at every real step (online MPC).",
    ),
    "cem_candidates": _d(
        "PETS: сколько случайных последовательностей действий-кандидатов оценивается на одной итерации CEM.",
        "PETS: how many candidate random action sequences are evaluated on a single CEM iteration.",
    ),
    "cem_elites": _d(
        "PETS: сколько лучших (по накопленной воображённой награде) кандидатов CEM отбирает на каждой итерации, "
        "чтобы по ним пересчитать распределение для следующей.",
        "PETS: how many of the top-performing (by imagined cumulative reward) candidates CEM keeps on each "
        "iteration, to re-fit the sampling distribution for the next one.",
    ),
    "cem_iterations": _d(
        "PETS: сколько итераций уточнения (пересэмплирование → оценка → отбор элит) CEM делает перед тем как "
        "выполнить первое действие спланированной последовательности в реальной среде.",
        "PETS: how many refinement iterations (re-sample → evaluate → keep elites) CEM performs before executing "
        "the first action of the planned sequence in the real environment.",
    ),
    "world_model_learning_rate": _d(
        "World Models (Ha & Schmidhuber): learning rate VAE + MDN-RNN на этапе обучения модели мира (до начала "
        "ES-эволюции контроллера).",
        "World Models (Ha & Schmidhuber): learning rate of the VAE + MDN-RNN during the world-model training "
        "phase (before controller ES evolution begins).",
    ),
    "world_model_phase_steps": _d(
        "World Models: сколько реальных шагов случайной политики собирается и используется для обучения VAE + "
        "MDN-RNN, прежде чем модель мира 'замораживается' и начинается ES-эволюция контроллера над ней.",
        "World Models: how many real steps of a random policy are collected and used to train the VAE + "
        "MDN-RNN, before the world model is frozen and controller ES evolution begins on top of it.",
    ),
    "world_model_train_steps_per_iter": _d(
        "World Models: сколько градиентных шагов обучения VAE + MDN-RNN выполняется за одну итерацию фазы "
        "обучения модели мира.",
        "World Models: how many VAE + MDN-RNN gradient steps are performed per iteration of the world-model "
        "training phase.",
    ),
    "es_learning_rate": _d(
        "World Models: learning rate обновления параметров контроллера по взвешенному рангом fitness среднему "
        "возмущений (Evolution Strategies) — уже на этапе, когда модель мира заморожена.",
        "World Models: learning rate for updating the controller's parameters via the rank-fitness-weighted "
        "average of perturbations (Evolution Strategies) — during the phase where the world model is already "
        "frozen.",
    ),
    "embed_dim": _d(
        "Размерность одного токена (наблюдения или действия) во входной последовательности Transformer-модели "
        "мира. Больше — модель может представлять более сложные наблюдения/историю, но дороже по памяти и "
        "вычислениям на каждый forward pass (который выполняется многократно: на каждый реальный шаг и на каждую "
        "симуляцию поиска).",
        "Dimensionality of a single token (an observation or an action) in the Transformer world model's input "
        "sequence. Bigger lets the model represent more complex observations/history, but costs more memory and "
        "compute per forward pass (which runs many times: once per real step and once per search simulation).",
    ),
    "num_layers": _d(
        "Количество слоёв (блоков self-attention + MLP) в Transformer-модели мира. Больше слоёв — модель может "
        "учить более сложные зависимости между токенами истории, но дороже по времени на forward pass.",
        "Number of layers (self-attention + MLP blocks) in the Transformer world model. More layers let the "
        "model learn more complex dependencies across the history's tokens, but cost more time per forward pass.",
    ),
    "num_heads": _d(
        "Количество голов self-attention в каждом слое Transformer-модели мира. Разные головы могут учиться "
        "обращать внимание на разные аспекты истории (например, одна — на последнее действие, другая — на "
        "давнее наблюдение).",
        "Number of self-attention heads per layer in the Transformer world model. Different heads can learn to "
        "attend to different aspects of the history (e.g. one to the most recent action, another to a much "
        "older observation).",
    ),
    "dropout": _d(
        "Вероятность dropout внутри Transformer-модели мира (embedding, attention, MLP) — регуляризация против "
        "переобучения на небольшом объёме собранного опыта. 0 отключает dropout полностью.",
        "Dropout probability inside the Transformer world model (embedding, attention, MLP) — regularization "
        "against overfitting on a small amount of collected experience. 0 disables dropout entirely.",
    ),
    "context_length": _d(
        "Сколько прошлых реальных (наблюдение, действие) пар подаётся в Transformer вместе с текущим "
        "наблюдением — окно 'памяти', которое модель может напрямую пронаблюдать через attention (в отличие от "
        "рекуррентных моделей мира, где вся история сжимается в один вектор состояния). Больше — модель 'помнит' "
        "дальше назад, но каждый forward pass (при поиске их много) обрабатывает вдвое больше токенов на каждую "
        "единицу этого параметра.",
        "How many past real (observation, action) pairs are fed into the Transformer alongside the current "
        "observation — the 'memory' window the model can directly attend over (unlike recurrent world models, "
        "where the whole history is squeezed into one state vector). Bigger means the model 'remembers' further "
        "back, but every forward pass (search runs many of them) processes twice as many tokens per unit of "
        "this parameter.",
    ),
    "latent_dim": _d(
        "Размерность латентного вектора состояния, в который representation-сеть сжимает наблюдение — узкое "
        "место, через которое обязана пройти вся информация, нужная для предсказаний динамики/награды/ценности. "
        "Больше — больше выразительность, но больше параметров и медленнее обучение.",
        "Dimensionality of the latent state vector that the representation network compresses the observation "
        "into — the bottleneck through which all information needed for dynamics/reward/value predictions must "
        "pass. Bigger means more expressive power but more parameters and slower training.",
    ),
    "hidden_dim": _d(
        "Размерность скрытых слоёв внутренних сетей алгоритма (dynamics/prediction и т.п.).",
        "Hidden-layer size of the algorithm's internal networks (dynamics/prediction, etc).",
    ),
    "proj_dim": _d(
        "EfficientZero: размерность выходного пространства SimSiam-style projector/predictor, используемого для "
        "self-supervised consistency loss между предсказанным и настоящим следующим латентным состоянием.",
        "EfficientZero: output dimensionality of the SimSiam-style projector/predictor used for the "
        "self-supervised consistency loss between the predicted and the actual next latent state.",
    ),
    "unroll_steps": _d(
        "EfficientZero: на сколько шагов вперёд 'разворачивается' (unroll) обучение динамики/награды/ценности от "
        "одной сэмплированной точки в буфере — модель должна предсказывать всю эту цепочку по одному начальному "
        "латентному состоянию и известной последовательности действий, без повторного кодирования наблюдений.",
        "EfficientZero: how many steps the dynamics/reward/value training is unrolled forward from a single "
        "sampled buffer point — the model must predict the entire chain from one starting latent state and a "
        "known action sequence, without re-encoding observations along the way.",
    ),
    "td_steps": _d(
        "EfficientZero: горизонт n-step возврата, используемого при построении value target (бутстрап через "
        "value-функцию на td_steps шагов вперёд плюс накопленная награда за эти шаги).",
        "EfficientZero: horizon of the n-step return used to build the value target (bootstrapping through the "
        "value function td_steps ahead, plus the accumulated reward over those steps).",
    ),
    "num_sampled_actions": _d(
        "EfficientZero, continuous-действия: сколько кандидатов-действий (K) сэмплируется из текущей политики в "
        "качестве корневых узлов Gumbel-поиска — для continuous-пространства нет конечного набора действий, "
        "поэтому вместо перебора всех действий поиск идёт по этому конечному сэмплированному набору.",
        "EfficientZero, continuous actions: how many candidate actions (K) are sampled from the current policy "
        "as root nodes of the Gumbel search — since there's no finite action set for continuous spaces, the "
        "search runs over this finite sampled set instead of enumerating every action.",
    ),
    "num_top_actions": _d(
        "EfficientZero: число корневых кандидатов-действий (m), с которых начинается Sequential Halving в "
        "Gumbel-поиске — на каждой фазе отбрасывается половина кандидатов с наименьшим 'completed Q', пока не "
        "останется один.",
        "EfficientZero: number of root candidate actions (m) that Sequential Halving in the Gumbel search starts "
        "from — each phase discards the half of the candidates with the lowest 'completed Q', down to a single "
        "one.",
    ),
    "c_visit": _d(
        "EfficientZero: константа c_visit в σ-преобразовании (Gumbel MuZero), масштабирующая вклад числа "
        "посещений узла при взвешивании Q-значений между посещёнными и непосещёнными действиями.",
        "EfficientZero: the c_visit constant in the σ-transform (Gumbel MuZero), scaling the contribution of a "
        "node's visit count when weighting Q-values between visited and unvisited actions.",
    ),
    "c_scale": _d(
        "EfficientZero: константа c_scale в σ-преобразовании (Gumbel MuZero) — масштаб самого σ-преобразования "
        "Q-значений при вычислении 'completed Q' (v-mix).",
        "EfficientZero: the c_scale constant in the σ-transform (Gumbel MuZero) — the scale of the σ-transform "
        "of Q-values when computing 'completed Q' (v-mix).",
    ),
    "value_minmax_delta": _d(
        "EfficientZero: минимальный эпсилон, добавляемый к разнице max-min при нормализации Q-значений внутри "
        "дерева поиска (MinMaxStats) — защита от деления на почти нулевой диапазон, когда все найденные Q почти "
        "одинаковы.",
        "EfficientZero: minimum epsilon added to the max-min range when normalizing Q-values inside the search "
        "tree (MinMaxStats) — guards against dividing by a near-zero range when every discovered Q is nearly "
        "identical.",
    ),
    "value_support_size": _d(
        "EfficientZero: половина ширины (±N бинов) категориального support'а value- и reward-головы — вместо "
        "одного скаляра сеть предсказывает распределение вероятностей по 2N+1 бинам в signed-hyperbolic шкале и "
        "учится через cross-entropy (MuZero Appendix F), что не даёт редким крупным наградам взрывать лосс, как "
        "это бывает со скалярной MSE-головой.",
        "EfficientZero: half-width (±N bins) of the categorical support of the value and reward heads — instead "
        "of a single scalar, the network predicts a probability distribution over 2N+1 bins on a "
        "signed-hyperbolic scale and is trained via cross-entropy (MuZero Appendix F), which keeps rare large "
        "rewards from blowing up the loss the way a scalar MSE head would.",
    ),
    "value_loss_coef": _d(
        "EfficientZero: вес слагаемого value loss в суммарном лоссе обучения.",
        "EfficientZero: weight of the value-loss term in the total training loss.",
    ),
    "policy_loss_coef": _d(
        "EfficientZero: вес слагаемого policy loss (кросс-энтропия к улучшенной поиском политике) в суммарном "
        "лоссе.",
        "EfficientZero: weight of the policy-loss term (cross-entropy toward the search-improved policy) in the "
        "total loss.",
    ),
    "reward_loss_coef": _d(
        "EfficientZero: вес слагаемого reward (value-prefix) loss в суммарном лоссе.",
        "EfficientZero: weight of the reward (value-prefix) loss term in the total loss.",
    ),
    "consistency_loss_coef": _d(
        "EfficientZero: вес self-supervised consistency loss (SimSiam-style) между спроецированным предсказанным "
        "и настоящим следующим латентным состоянием — держит представление предсказуемым вперёд по действиям, "
        "а не просто удобным для текущего шага.",
        "EfficientZero: weight of the self-supervised consistency loss (SimSiam-style) between the projected "
        "predicted and the actual next latent state — keeps the representation predictable forward through "
        "actions, not just convenient for the current step.",
    ),
    "continuous_prior_scale": _d(
        "EfficientZero, continuous-действия: во сколько раз расширяется std распределения политики при "
        "сэмплировании дополнительных 'prior'-кандидатов для корня поиска (шире исходной политики — для лучшего "
        "покрытия пространства действий при поиске).",
        "EfficientZero, continuous actions: how much the policy distribution's std is widened when sampling "
        "extra 'prior' candidates for the search root (wider than the raw policy — for better action-space "
        "coverage during search).",
    ),
    "policy_target_temperature": _d(
        "EfficientZero: температура шума Гумбеля, добавляемого к logits политики в корне поиска перед выбором "
        "Top-k кандидатов — выше температура, разнообразнее набор рассматриваемых действий-кандидатов.",
        "EfficientZero: temperature of the Gumbel noise added to the policy logits at the search root before "
        "picking the Top-k candidates — higher temperature means a more diverse set of candidate actions "
        "considered.",
    ),
    "train_freq": _d(
        "Через сколько реальных шагов среды (суммарно по всем параллельным окружениям) выполняется очередная "
        "порция градиентных обновлений сети.",
        "After how many real environment steps (summed across all parallel environments) the next batch of "
        "gradient updates is performed.",
    ),
    "max_grad_norm": _d(
        "Максимальная норма градиента при clip'е (gradient clipping) перед шагом оптимизатора — защита от "
        "редких резких скачков градиента, которые иначе могли бы дестабилизировать обучение за один шаг.",
        "Maximum gradient norm for gradient clipping before the optimizer step — guards against rare sharp "
        "gradient spikes that could otherwise destabilize training in a single step.",
    ),
    "priority_alpha": _d(
        "EfficientZero, Prioritized Experience Replay: степень приоритизации переходов с большой ошибкой "
        "|предсказание value − таргет| при сэмплировании из буфера (0 = равномерное сэмплирование).",
        "EfficientZero, Prioritized Experience Replay: how strongly transitions with a large "
        "|value-prediction − target| error are favored when sampling from the buffer (0 = uniform sampling).",
    ),
    "priority_beta": _d(
        "EfficientZero, Prioritized Experience Replay: степень importance-sampling коррекции смещения от "
        "неравномерного сэмплирования, применяемая как вес к лоссу каждого сэмпла в батче.",
        "EfficientZero, Prioritized Experience Replay: strength of the importance-sampling correction for the "
        "bias from non-uniform sampling, applied as a per-sample loss weight.",
    ),
    "min_priority": _d(
        "EfficientZero: минимальный приоритет, который может получить переход — не даёт вероятности сэмплирования "
        "упасть точно до нуля даже для переходов с идеальным (нулевым) предсказанием.",
        "EfficientZero: minimum priority a transition can receive — keeps the sampling probability from dropping "
        "to exactly zero even for transitions with a perfect (zero-error) prediction.",
    ),
    "reanalyze_freq": _d(
        "EfficientZero: через сколько реальных шагов среды выполняется очередной reanalyze-проход — повторный "
        "прогон search() текущей (уже обновившейся) сети по случайно выбранным сохранённым переходам, чтобы "
        "обновить их policy/value-таргеты свежими, а не теми, что были получены во время сбора данных.",
        "EfficientZero: after how many real environment steps the next reanalyze pass runs — re-running search() "
        "with the current (already-updated) network over randomly chosen stored transitions, to refresh their "
        "policy/value targets instead of leaving the stale ones from data-collection time.",
    ),
    "reanalyze_batch_size": _d(
        "EfficientZero: сколько сохранённых переходов обновляется за один reanalyze-проход (0 — полностью "
        "выключить reanalyze).",
        "EfficientZero: how many stored transitions are refreshed per reanalyze pass (0 — disable reanalyze "
        "entirely).",
    ),
    "n_epochs": _d(
        "PPO: сколько раз подряд весь собранный rollout проходится заново (полными эпохами) при одном обновлении "
        "— больше эпох выжимает больше из каждого собранного rollout'а, но при слишком многих клиппинг PPO "
        "начинает плохо сдерживать отход политики от той, что собирала данные.",
        "PPO: how many times the entire collected rollout is passed over again (full epochs) in a single update "
        "— more epochs squeeze more out of each collected rollout, but too many and PPO's clipping starts to "
        "poorly restrain the policy from drifting away from the one that collected the data.",
    ),
    "gae_lambda": _d(
        "Параметр λ Generalized Advantage Estimation — компромисс между смещением и дисперсией при оценке "
        "advantage: λ=0 — чисто one-step TD (мало дисперсии, много смещения), λ=1 — чисто Monte-Carlo (мало "
        "смещения, много дисперсии).",
        "The λ parameter of Generalized Advantage Estimation — a bias/variance trade-off for the advantage "
        "estimate: λ=0 is pure one-step TD (low variance, high bias), λ=1 is pure Monte-Carlo (low bias, high "
        "variance).",
    ),
    "clip_range": _d(
        "PPO: максимальное относительное отклонение новой политики от старой (той, что собирала данные), за "
        "пределами которого вклад сэмпла в лосс 'обрезается' — главный механизм, не дающий policy gradient "
        "делать слишком большие разрушительные шаги.",
        "PPO: the maximum relative deviation of the new policy from the old one (the one that collected the "
        "data), beyond which a sample's contribution to the loss is clipped — the core mechanism preventing "
        "the policy gradient from taking overly large, destructive steps.",
    ),
    "vf_coef": _d(
        "PPO: вес слагаемого value loss относительно policy loss в суммарном лоссе обновления.",
        "PPO: weight of the value-loss term relative to the policy loss in the total update loss.",
    ),
    "target_update_interval": _d(
        "QMIX: через сколько шагов обучения target-сети (Q-сеть агентов + mixing network) полностью "
        "копируются с текущих (online) сетей — редкое обновление держит цель более стабильной для Double-DQN "
        "таргета.",
        "QMIX: after how many training steps the target networks (agent Q-network + mixing network) are fully "
        "copied from the current (online) ones — infrequent updates keep the target more stable for the "
        "Double-DQN bootstrap.",
    ),
    "mixing_embed_dim": _d(
        "QMIX: размерность скрытого пространства mixing-сети, которая монотонно смешивает индивидуальные "
        "Q-значения агентов команды в одно командное Q_tot.",
        "QMIX: hidden-space dimensionality of the mixing network that monotonically combines the team's "
        "individual per-agent Q-values into a single team Q_tot.",
    ),
    "games_per_iteration": _d(
        "AlphaZero: сколько self-play партий (агент играет сам с собой, используя MCTS для выбора ходов) "
        "разыгрывается за одну итерацию перед обновлением сети.",
        "AlphaZero: how many self-play games (the agent plays against itself, using MCTS to pick moves) are "
        "played per iteration before the network is updated.",
    ),
    "epochs": _d(
        "AlphaZero: сколько эпох обучения на данных self-play партий текущей итерации выполняется перед новым "
        "циклом self-play.",
        "AlphaZero: how many training epochs over the current iteration's self-play data are run before the "
        "next self-play cycle.",
    ),
    "eval_games": _d(
        "AlphaZero: сколько партий 'арены' (новая сеть против предыдущей лучшей) играется, чтобы решить, "
        "принять ли новую сеть как текущую лучшую.",
        "AlphaZero: how many 'arena' games (the new network vs. the previous best) are played to decide whether "
        "to accept the new network as the current best.",
    ),
    "win_rate_threshold": _d(
        "AlphaZero: минимальная доля побед новой сети над предыдущей лучшей в арене, необходимая, чтобы принять "
        "новую сеть (иначе продолжаем self-play со старой лучшей).",
        "AlphaZero: minimum win rate the new network must achieve against the previous best in the arena to be "
        "accepted (otherwise self-play continues with the old best).",
    ),
    "channels": _d(
        "AlphaZero: число каналов сверточных слоёв dual-head сети (policy + value).",
        "AlphaZero: number of convolutional channels in the dual-head (policy + value) network.",
    ),
    "num_blocks": _d(
        "AlphaZero: число residual-блоков в теле сети (глубина сети).",
        "AlphaZero: number of residual blocks in the network's trunk (network depth).",
    ),
}


# Per-(algorithm_id, hyperparam key) overrides for the handful of keys whose
# *meaning* — not just phrasing/units — genuinely differs between the
# algorithms that expose them, so one shared description in
# `_HP_DESCRIPTIONS` above would be misleading for at least one of them.
_HP_DESCRIPTIONS_BY_ALGO: dict[tuple[str, str], dict[str, str]] = {
    ("alphazero", "num_simulations"): _d(
        "AlphaZero: сколько симуляций MCTS выполняется для выбора ОДНОГО хода — больше симуляций даёт более "
        "точную оценку хода, но дороже по времени на ход.",
        "AlphaZero: how many MCTS simulations are run to pick a SINGLE move — more simulations give a more "
        "accurate move estimate but cost more time per move.",
    ),
    ("efficientzero", "num_simulations"): _d(
        "EfficientZero: сколько симуляций Gumbel-поиска выполняется для выбора ОДНОГО реального действия среды "
        "(перестраивается заново на каждом шаге) — больше симуляций даёт точнее посчитанные Q/visit-статистики "
        "дерева, но дороже по времени на шаг.",
        "EfficientZero: how many Gumbel-search simulations are run to pick a SINGLE real environment action (the "
        "tree is rebuilt from scratch every step) — more simulations give more accurate tree Q/visit statistics "
        "but cost more time per step.",
    ),
    ("dreamer", "batch_size"): _d(
        "Dreamer: сколько НЕЗАВИСИМЫХ последовательностей (не отдельных переходов) сэмплируется из реального "
        "буфера за одно обновление модели мира — каждая длиной seq_len шагов.",
        "Dreamer: how many INDEPENDENT sequences (not individual transitions) are sampled from the real buffer "
        "per world-model update — each seq_len steps long.",
    ),
    ("world_models_ha", "batch_size"): _d(
        "World Models: сколько НЕЗАВИСИМЫХ последовательностей (не отдельных переходов) сэмплируется из "
        "собранных данных за одно обновление VAE + MDN-RNN — каждая длиной seq_len шагов.",
        "World Models: how many INDEPENDENT sequences (not individual transitions) are sampled from the "
        "collected data per VAE + MDN-RNN update — each seq_len steps long.",
    ),
    ("unizero", "num_simulations"): _d(
        "UniZero: сколько симуляций классического PUCT-поиска (как в AlphaZero/MuZero) выполняется для выбора "
        "ОДНОГО реального действия среды (дерево перестраивается заново на каждом шаге, каждая симуляция — два "
        "forward pass'а через Transformer) — больше симуляций даёт точнее посчитанные Q/visit-статистики "
        "дерева, но дороже по времени на шаг.",
        "UniZero: how many classic PUCT-search simulations (AlphaZero/MuZero-style) are run to pick a SINGLE "
        "real environment action (the tree is rebuilt from scratch every step, each simulation is two "
        "Transformer forward passes) — more simulations give more accurate tree Q/visit statistics but cost "
        "more time per step.",
    ),
    ("unizero", "pb_c_base"): _d(
        "UniZero, PUCT-формула: базовая константа, регулирующая, как быстро вес exploration-слагаемого растёт "
        "с числом посещений узла (чем больше pb_c_base, тем медленнее рост).",
        "UniZero, PUCT formula: base constant controlling how quickly the exploration term's weight grows with "
        "a node's visit count (larger pb_c_base = slower growth).",
    ),
    ("unizero", "pb_c_init"): _d(
        "UniZero, PUCT-формула: начальный вес exploration-слагаемого (prior × sqrt(N_родителя) / (1+N_ребёнка)) "
        "относительно нормализованного Q-значения.",
        "UniZero, PUCT formula: initial weight of the exploration term (prior × sqrt(parent visits) / "
        "(1+child visits)) relative to the normalized Q-value.",
    ),
    ("unizero", "root_dirichlet_alpha"): _d(
        "UniZero: параметр α распределения Дирихле, из которого сэмплируется шум, добавляемый к приорам "
        "действий в корне поиска — только во время сбора данных (не при eval) — для дополнительного "
        "исследования.",
        "UniZero: α parameter of the Dirichlet distribution the noise mixed into the root's action priors is "
        "sampled from — only during data collection (not eval) — for extra exploration.",
    ),
    ("unizero", "root_noise_weight"): _d(
        "UniZero: доля Dirichlet-шума, смешиваемого с приорами действий в корне поиска "
        "((1-вес)×prior + вес×шум) — 0 полностью выключает шум.",
        "UniZero: fraction of Dirichlet noise mixed into the root's action priors "
        "((1-weight)×prior + weight×noise) — 0 fully disables the noise.",
    ),
    ("unizero", "rotary_emb"): _d(
        "UniZero: способ кодирования позиции токена в Transformer'е. RoPE (по умолчанию) поворачивает "
        "query/key перед скалярным произведением — внимание зависит только от относительного смещения "
        "позиций, поэтому обрезка старых токенов из KV-кэша ничего не портит (дешёвый персистентный кэш на "
        "весь эпизод). Learned absolute embedding — как в оригинальной реализации LightZero: обучаемая "
        "таблица позиций, добавляемая к эмбеддингу токена; обрезка кэша под ней лишь приблизительно "
        "корректна (сдвиг базы позиции), а не точна, поэтому медленнее.",
        "UniZero: how the Transformer encodes a token's position. RoPE (default) rotates query/key before "
        "the dot product, so attention depends only on the relative offset between positions — trimming old "
        "tokens from the KV-cache changes nothing (a cheap cache that persists for the whole episode). "
        "Learned absolute embedding — as in the original LightZero implementation: a trainable position "
        "table added to the token embedding; cache trimming under it is only approximately correct (a "
        "position-base shift), not exact, hence slower.",
    ),
    ("qmix", "train_freq"): _d(
        "QMIX: через сколько шагов среды выполняется очередное обновление командной Q-сети/mixing network.",
        "QMIX: after how many environment steps the next update of the team Q-network/mixing network is "
        "performed.",
    ),
    ("qmix", "buffer_size"): _d(
        "QMIX: максимальное число командных переходов (joint-наблюдение + действия + суммарная награда всей "
        "команды), хранимых в replay buffer.",
        "QMIX: maximum number of team transitions (joint observation + actions + the team's total reward) kept "
        "in the replay buffer.",
    ),
}


def _attach_hyperparam_descriptions() -> None:
    """Injects a bilingual `desc: {ru, en}` field into every hyperparam spec
    in `ALGORITHM_CATALOG`, looked up by key (with a few per-algorithm
    overrides — see `_HP_DESCRIPTIONS_BY_ALGO`). Runs once at import time so
    the API response and every consumer (Designer, Sweep dialog, Training
    Monitor) always has it without each of the ~200 literal hyperparam dicts
    above needing to carry it inline."""
    for algo in ALGORITHM_CATALOG:
        for hp in algo.get("hyperparams", []):
            key = hp["key"]
            desc = _HP_DESCRIPTIONS_BY_ALGO.get((algo["id"], key)) or _HP_DESCRIPTIONS.get(key)
            if desc is not None:
                hp["desc"] = desc


_attach_hyperparam_descriptions()


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
