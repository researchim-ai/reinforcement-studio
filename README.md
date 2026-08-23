# Reinforcement Studio

Десктопное приложение для дизайна и запуска экспериментов по reinforcement
learning на своей машине. Билдится под **Linux**, **Windows**, **macOS**.

Electron-окно с React UI. Бэкенд на Python (FastAPI) запускается
автоматически при старте: через Docker в режиме `Auto`, либо нативно через
локальный Python как fallback/ручной режим — как обычные RL-эксперименты
редко требуют GPU, нативный режим по умолчанию достаточен для классики
(CartPole, Taxi, ...) и AlphaZero на маленьких досках.

Что внутри:

- **Gymnasium** классика (CartPole, MountainCar, Acrobot, Pendulum, Frozen
  Lake, Taxi, Lunar Lander, ...) + **DQN / PPO / A2C** — собственная,
  написанная с нуля на чистом PyTorch реализация (без Stable-Baselines3).
- **AlphaZero** с нуля: PUCT MCTS + dual-head CNN + self-play + arena-gating,
  для Tic-Tac-Toe, Connect Four и Gomoku.
- Визуальный **Дизайнер экспериментов** (граф env → wrappers → algo →
  training на `@xyflow/react`) — без единой строчки кода. Живая панель
  «Среда» / «Нейросеть» показывает входы/выходы и число параметров сети,
  пересчитываясь на бэкенде при любом изменении выбора (без запуска
  обучения) — см. `/api/environments/inspect` и `rl_core/inspect.py`.
- **Мониторинг** обучения в реальном времени (WebSocket): графики награды
  / win-rate, живой рендер среды, логи, остановка запуска.
- **AlphaZero Arena** — сыграй против своего агента или случайного бота.
- **Model Zoo** — сохранённые чекпоинты, независимые от истории запусков.
- **Свои плагины** — пишите собственные алгоритмы (Gym и AlphaZero) и
  reward-функции прямо в приложении (Monaco-редактор на странице
  `/plugins`), они появляются в Дизайнере рядом со встроенными. Подробности
  ниже, в разделе [«Свои плагины»](#свои-плагины).

---

## Быстрый старт

### Требования

- **Node.js** >= 20
- **Python** >= 3.10 + pip
- **Docker** — опционально, для изолированного/переносимого запуска бэкенда

### 1. Установка

```bash
npm install
```

Python ставить руками **не нужно** — только сам интерпретатор Python
3.10+ должен быть в `PATH`. При первом запуске Electron сам создаёт
приватное виртуальное окружение (`userData/pyenv`) и ставит туда
`rl_core/requirements.txt` + `backend/requirements.txt` (progress виден на
сплэш-экране). Повторные запуски это окружение переиспользуют — установка
запускается заново только если requirements изменились.

### 2. Запуск (dev mode)

```bash
npm run dev
```

Это **одна команда**. Она:
1. Запускает Vite dev server (hot-reload React)
2. Компилирует Electron `main`/`preload`
3. Открывает **настоящее Electron-окно** (десктоп-прила, не браузер)
4. Electron пробует поднять Docker backend (образ `rl-studio:latest`)
5. Если Docker недоступен — нативный режим: создаёт/переиспользует venv,
   ставит Python-зависимости при первом запуске (может занять пару минут —
   `torch` тянется в CPU-варианте, это ~200 МБ, а не гигабайты с CUDA),
   затем запускает `uvicorn backend.api:app`
6. Порт подбирается автоматически (обычно `:8000`, при конфликте `:8001+`)
7. Пока бэкенд грузится — сплэш с прогрессом (создание venv → установка
   зависимостей → старт uvicorn)
8. Когда `/api/system/health` отвечает 200 — UI подменяется на рабочий

Если что-то пошло не так, экран загрузки покажет причину падения (включая
хвост stderr процесса) и кнопку «Логи» — там полный вывод pip/uvicorn.

Статус бэкенда виден в сайдбаре (`Online` / `Offline`).

### 3. Сборка приложения

```bash
npm run build:linux   # AppImage + .deb
npm run build:win     # NSIS installer + portable .exe
npm run build:mac     # .dmg + .zip (x64 + arm64)
npm run build         # текущая платформа
```

Готовые файлы — в `release/`.

---

## Backend modes

Настраивается в **Settings → Backend**:

| Режим    | Поведение |
|----------|-----------|
| `Auto`   | Сначала Docker, если не вышло — нативный Python |
| `Docker` | Только контейнер `rl-studio:latest` (соберётся сам при первом запуске) |
| `Native` | Только локальный `python3 -m uvicorn backend.api:app` |

GPU в Docker включается тумблером в Settings (`--gpus all`, нужен NVIDIA
Container Toolkit) — полезно для AlphaZero на больших досках (Gomoku 9×9) или
PPO/DQN с более тяжёлыми сетями.

---

## Как это устроено

```
┌───────────────────────────────────────┐
│           Electron Desktop App        │
│  ┌─────────────┐   ┌────────────────┐ │
│  │ Main Process│   │ Renderer (React)│ │
│  │ Стартует     │   │ 8 страниц:      │ │
│  │ Python/Docker│   │  Dashboard      │ │
│  │ бэкенд       │   │  Designer       │ │
│  │ автоматом    │   │  Monitor        │ │
│  │              │   │  Environments   │ │
│  │              │   │  Arena          │ │
│  │              │   │  Model Zoo      │ │
│  │              │   │  Plugins        │ │
│  │              │   │  Settings       │ │
│  └──────┬───────┘   └────────┬────────┘ │
└─────────┼────────────────────┼──────────┘
          │           HTTP + WebSocket
  ┌───────▼────────────────────▼─────────┐
  │  FastAPI Backend (Python, :8000)     │
  │  /api/environments/* — каталог сред  │
  │  /api/training/*     — запуск/стоп   │
  │  /api/models/*       — Model Zoo     │
  │  /api/alphazero/*    — Arena         │
  │  /api/plugins/*      — свои плагины  │
  │  /ws/metrics/{run_id}— метрики (WS)  │
  │  subprocess → rl_core.runner         │
  └───────────────┬───────────────────────┘
                  │
        ┌─────────▼──────────┐
        │       rl_core       │
        │  envs/  games/      │
        │ algorithms/ (native)│
        │  alphazero/ (MCTS)  │
        │  plugins/ (loader)  │
        └─────────────────────┘
```

При нажатии «Запустить обучение» FastAPI спавнит `python -m rl_core.runner
<config.json> <run_dir>` как отдельный процесс → метрики пишутся в
`rl_core/.runs/{run_id}/metrics.json` → FastAPI шлёт их по WebSocket →
React рисует графики в реальном времени. Остановка — через `stop.flag` в
run dir (кооперативно) + `SIGTERM` при таймауте.

### Gym-алгоритмы с нуля

`rl_core/algorithms/native/` — собственная реализация DQN/PPO/A2C на чистом
PyTorch, без Stable-Baselines3 (это дефолтный движок для встроенных
алгоритмов, см. `native_runner.py`):

- `preprocessing.py` — приведение наблюдений к тензору: one-hot для
  скалярного `Discrete` (Taxi, FrozenLake, CliffWalking), обычный
  flatten-vector для `Box` (классика/Box2D/MuJoCo), «как есть» для
  картиночного `Box` (Atari, CarRacing).
- `networks.py` — `MLPExtractor`/`NatureCNN` (выбор автоматический по форме
  observation space) + `ActorCriticNet` (Categorical или диагональный
  Gaussian head — для discrete/continuous actions) + `QNetwork`.
- `buffers.py` — `RolloutBuffer` с GAE(λ) для PPO/A2C, `ReplayBuffer` для
  DQN.
- `ppo.py` / `a2c.py` — общий rollout-цикл в `on_policy.py`, отличаются
  только `_update()`: PPO — clipped surrogate + несколько эпох по
  минибатчам, A2C — один градиентный шаг на весь rollout, без clipping.
- `dqn.py` — target-сеть, ε-greedy с линейным затуханием, replay buffer,
  Huber-лосс.

Диспетчер (`rl_core/runner.py`) шлёт встроенные `ppo`/`dqn`/`a2c` в
`native_runner.py`. Stable-Baselines3 остаётся опциональной зависимостью —
нужна только тем, кто пишет свой алгоритм-плагин, наследуя
`stable_baselines3.common.base_class.BaseAlgorithm` (см. [«Свои
плагины»](#свои-плагины) ниже); в этом случае раннер использует
`sb3_runner.py`.

### AlphaZero с нуля

`rl_core/alphazero/` — самостоятельная, независимая от внешних RL-библиотек
реализация:

- `network.py` — маленькая CNN с dual head (policy + value), 2-4 residual
  блока — достаточно для CPU-тренировки на 3×3/6×7/9×9 досках.
- `mcts.py` — PUCT Monte-Carlo Tree Search с Dirichlet-шумом на корне.
- `self_play.py` — генерация партий текущей лучшей сетью против самой себя.
- `train.py` — цикл self-play → обучение кандидата → arena против
  предыдущей лучшей сети → принять/откатить (accept/reject gate).
- `arena.py` — используется и для accept/reject, и для интерактивной игры
  человека с агентом (`/api/alphazero/session`).

### Настольные игры

`rl_core/games/` — общий интерфейс `BoardGame` (см. `base.py`), три
реализации: `TicTacToe` (3×3), `ConnectFour` (7×6, drop-in-column),
`Gomoku` (9×9, 5 в ряд). `current_player` флипается на каждом ходе (даже
терминальном) — это упрощает знаки в MCTS backup.

---

## Свои плагины

Страница **`/plugins`** — встроенный Monaco-редактор для трёх типов
скриптов. Каждый плагин — один `.py`-файл (`<slug>.py`), сохраняется в
`ROOT/custom_algorithms/{gym,alphazero}/` или `ROOT/custom_rewards/`
(`ROOT` = `userData` в собранном приложении, `rl_core/` в dev-режиме — тот
же принцип, что у `.runs`/`checkpoints`). Кнопка «Проверить» перед
сохранением гоняет дешёвый dry-run смоук-тест (пара шагов/итераций на
CartPole или Tic-Tac-Toe) и показывает traceback прямо в UI, если контракт
не соблюдён.

> **Модель доверия.** Это локальное десктоп-приложение — код плагина
> выполняется с тем же уровнем доверия, что и сам движок (без sandbox'а),
> это эквивалентно ручному редактированию файлов в `rl_core/algorithms/`.
> Валидация — это подсказка для быстрой обратной связи, а не защита от
> злонамеренного кода.

### Gym-алгоритм

Файл в `custom_algorithms/gym/<slug>.py`, обязательные module-level поля:

```python
NAME = "..."                       # для каталога/Дизайнера
DESCRIPTION = "..."
SUPPORTED_ACTION_KINDS = ["discrete"]     # или ["discrete", "continuous"]
HYPERPARAMS = [                     # тот же формат, что у встроенных алгоритмов
    {"key": "learning_rate", "label": "Learning rate", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1},
]
ALGORITHM_CLASS = MyAlgorithm       # класс, см. ниже
```

`ALGORITHM_CLASS` — один из двух путей:

1. **Наследник `rl_core.algorithms.base.CustomAlgorithm`** — полностью свой
   цикл на чистом PyTorch, тот же контракт, что и у встроенных `ppo`/`dqn`/
   `a2c` (см. `rl_core/algorithms/native/` — неплохой пример для
   подсматривания). Обязательные методы: `learn(total_timesteps, callback:
   TrainingCallback)`, `predict(obs, deterministic)`, `save(path)`,
   `load(path, env)` (classmethod). `TrainingCallback.on_step(num_timesteps,
   episode_reward=None, episode_length=None) -> bool` — вызывайте регулярно
   из `learn()`; когда вернёт `False` — останавливайтесь (пользователь
   нажал «Стоп»).
2. **Наследник SB3** (`stable_baselines3.common.base_class.BaseAlgorithm`,
   напр. `PPO`) — если хочется переиспользовать готовую реализацию из
   Stable-Baselines3 и только переопределить кусок (сеть, `train()`, ...);
   раннер детектирует это через `isinstance(..., BaseAlgorithm)` и
   прогоняет через `sb3_runner.py` + `MetricsCallback` без изменений. SB3
   — опциональная зависимость, нужна только для этого пути.

### AlphaZero-алгоритм

Файл в `custom_algorithms/alphazero/<slug>.py`, поля `NAME`,
`DESCRIPTION`, `HYPERPARAMS` (без `SUPPORTED_ACTION_KINDS` — у настольных
игр нет понятия action kind) и `ALGORITHM_CLASS`, наследник
`rl_core.alphazero.base.AlphaZeroTrainer`:

- Override только `build_network()` — лёгкая кастомизация архитектуры,
  self-play/MCTS/arena-gating остаются встроенными (см.
  `BuiltinAlphaZeroTrainer` в `rl_core/alphazero/base.py`, от которого можно
  унаследоваться напрямую). Сеть должна иметь атрибуты `.rows/.cols/.action_size`
  и метод `predict(encoded_state, device) -> (policy_probs, value)`.
- Или override `run_iteration(iteration) -> dict` целиком — полная замена
  цикла обучения. Можно переиспользовать `self_play.play_self_play_game`,
  `arena.play_match`. Верните `"_self_play_records"` в результате, чтобы
  партии были доступны для реплея на странице Arena.

### Reward-функция

Файл в `custom_rewards/<slug>.py` (только Gym-трек — у AlphaZero
фиксированные ±1/0 правила игры):

```python
NAME = "..."
DESCRIPTION = "..."

def shape_reward(obs, action, reward, next_obs, terminated, truncated, info) -> float:
    return reward  # ваша модификация

REWARD_FN = shape_reward  # опционально, если функция названа иначе
```

Подключается как wrapper `custom_reward:<slug>` в Дизайнере (в списке
wrapper'ов появляется автоматически, отдельной записью на каждый
сохранённый reward-скрипт).

Готовые шаблоны для всех пяти вариантов (с нуля / SB3-наследник / своя
сеть AlphaZero / полный цикл AlphaZero / reward-функция) доступны прямо в
редакторе через кнопку «+» у каждого раздела.

---

## Структура проекта

```
reinforcement-studio/
├── electron/                # Electron main process
│   ├── main.ts               #   Окно, backend lifecycle, IPC
│   ├── preload.ts             #   contextBridge
│   ├── docker.ts               #   Docker management (dockerode)
│   └── config.ts                #   Backend mode config
├── src/                      # React frontend (renderer)
│   ├── pages/                 #   Dashboard, ExperimentDesigner,
│   │                              TrainingMonitor, Environments,
│   │                              AlphaZeroArena, ModelZoo, Plugins, Settings
│   ├── components/
│   │   ├── designer/           #   Кастомные ноды для @xyflow/react
│   │   ├── layout/              #   Sidebar/Header/Layout
│   │   └── ui/                   #   shadcn-стиль UI-кит
│   ├── api/                    #   REST/WS клиент + React Query hooks
│   └── stores/                  #   Zustand (docker, settings)
├── backend/                  # FastAPI Python бэкенд
│   ├── api.py                 #   Точка входа
│   ├── ws.py                   #   WebSocket метрик
│   ├── process_manager.py       #   Управление subprocess'ами запусков
│   └── routes/                   #   system, environments, training,
│                                      models, alphazero, plugins
├── rl_core/                  # RL-ядро — envs, алгоритмы, AlphaZero
│   ├── envs/                   #   Gymnasium registry + wrappers + reward_fn
│   ├── games/                   #   TicTacToe / ConnectFour / Gomoku
│   ├── algorithms/               #   base.py + native/ (PPO/A2C/DQN с нуля) +
│   │                                    native_runner/runner_utils + sb3_runner
│   │                                    (опционально, для custom-плагинов) +
│   │                                    custom_runner
│   ├── alphazero/                #   base/loop + network/mcts/self_play/train
│   ├── plugins/                    #   loader.py + templates.py (свои плагины)
│   └── runner.py                  #   Subprocess entrypoint (диспетчер custom:*)
├── Dockerfile                # Образ бэкенда (CPU по умолчанию)
├── docker-compose.yml
├── build/                    # Иконки приложения (generate_icon.py)
└── package.json
```

---

## Стек

| Слой       | Технология                              |
|------------|------------------------------------------|
| Desktop    | Electron 35                              |
| Frontend   | React 19, TypeScript, Vite 6             |
| Стили      | Tailwind CSS 4, кастомный UI-кит          |
| Граф       | @xyflow/react                             |
| Графики    | Recharts                                  |
| Редактор кода | Monaco (@monaco-editor/react)           |
| Стейт      | Zustand, React Query                      |
| i18n       | i18next (en + ru)                         |
| Backend    | FastAPI, uvicorn, WebSocket               |
| RL         | Gymnasium, PyTorch (свой PPO/DQN/A2C); Stable-Baselines3 — опционально, только для custom-плагинов |
| AlphaZero  | Собственная реализация (PUCT MCTS + CNN)   |
| Сборка     | electron-builder (Linux/Win/Mac)           |

## Лицензия

MIT
