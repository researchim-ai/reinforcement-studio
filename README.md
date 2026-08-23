# RL Studio

Десктопное приложение для дизайна и запуска экспериментов по reinforcement
learning на своей машине. Билдится под **Linux**, **Windows**, **macOS**.

Electron-окно с React UI. Бэкенд на Python (FastAPI) запускается
автоматически при старте: через Docker в режиме `Auto`, либо нативно через
локальный Python как fallback/ручной режим — как обычные RL-эксперименты
редко требуют GPU, нативный режим по умолчанию достаточен для классики
(CartPole, Taxi, ...) и AlphaZero на маленьких досках.

Что внутри:

- **Gymnasium** классика (CartPole, MountainCar, Acrobot, Pendulum, Frozen
  Lake, Taxi, Lunar Lander) + **DQN / PPO / A2C** через Stable-Baselines3.
- **AlphaZero** с нуля: PUCT MCTS + dual-head CNN + self-play + arena-gating,
  для Tic-Tac-Toe, Connect Four и Gomoku.
- Визуальный **Дизайнер экспериментов** (граф env → wrappers → algo →
  training на `@xyflow/react`) — без единой строчки кода.
- **Мониторинг** обучения в реальном времени (WebSocket): графики награды
  / win-rate, живой рендер среды, логи, остановка запуска.
- **AlphaZero Arena** — сыграй против своего агента или случайного бота.
- **Model Zoo** — сохранённые чекпоинты, независимые от истории запусков.

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
│  │ Стартует     │   │ 7 страниц:      │ │
│  │ Python/Docker│   │  Dashboard      │ │
│  │ бэкенд       │   │  Designer       │ │
│  │ автоматом    │   │  Monitor        │ │
│  │              │   │  Environments   │ │
│  │              │   │  Arena          │ │
│  │              │   │  Model Zoo      │ │
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
  │  /ws/metrics/{run_id}— метрики (WS)  │
  │  subprocess → rl_core.runner         │
  └───────────────┬───────────────────────┘
                  │
        ┌─────────▼──────────┐
        │       rl_core       │
        │  envs/  games/      │
        │  algorithms/ (SB3)  │
        │  alphazero/ (MCTS)  │
        └─────────────────────┘
```

При нажатии «Запустить обучение» FastAPI спавнит `python -m rl_core.runner
<config.json> <run_dir>` как отдельный процесс → метрики пишутся в
`rl_core/.runs/{run_id}/metrics.json` → FastAPI шлёт их по WebSocket →
React рисует графики в реальном времени. Остановка — через `stop.flag` в
run dir (кооперативно) + `SIGTERM` при таймауте.

### AlphaZero с нуля

`rl_core/alphazero/` — самостоятельная, независимая от Stable-Baselines3
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
│   │                              AlphaZeroArena, ModelZoo, Settings
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
│                                      models, alphazero
├── rl_core/                  # RL-ядро — envs, алгоритмы, AlphaZero
│   ├── envs/                   #   Gymnasium registry + wrappers
│   ├── games/                   #   TicTacToe / ConnectFour / Gomoku
│   ├── algorithms/               #   Stable-Baselines3 runner
│   ├── alphazero/                #   network / mcts / self_play / train
│   └── runner.py                  #   Subprocess entrypoint
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
| Стейт      | Zustand, React Query                      |
| i18n       | i18next (en + ru)                         |
| Backend    | FastAPI, uvicorn, WebSocket               |
| RL         | Gymnasium, Stable-Baselines3, PyTorch      |
| AlphaZero  | Собственная реализация (PUCT MCTS + CNN)   |
| Сборка     | electron-builder (Linux/Win/Mac)           |

## Лицензия

MIT
