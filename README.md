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
  Lake, Taxi, Lunar Lander, ...) + **DQN, Rainbow DQN, PPO, A2C, SAC** —
  собственная, написанная с нуля на чистом PyTorch реализация (без
  Stable-Baselines3). Rainbow DQN добавляет Double Q-learning, Dueling-сеть,
  Prioritized Replay и n-step возвраты сверху обычного DQN; SAC — off-policy
  метод для непрерывных действий (альтернатива PPO/A2C на continuous-средах).
- **AlphaZero** с нуля: PUCT MCTS + dual-head CNN + self-play + arena-gating,
  для Tic-Tac-Toe, Connect Four и Gomoku.
- **Память (LSTM/GRU)** — PPO, A2C, DQN и Rainbow DQN можно переключить в
  рекуррентный режим: между извлечением признаков и головами действия/
  ценности встаёт LSTM или GRU, настраиваются размер hidden state, число
  слоёв и длина truncated-BPTT последовательности. Подробности и границы
  применимости в разделе [«Память (LSTM/GRU)»](#память-lstmgru).
- **Exploration для DQN/Rainbow** — независимо выбираются ε-greedy или
  NoisyNet для исследования действий и опциональный RND-бонус новизны.
  Extrinsic/intrinsic reward и состояние RND показываются отдельно в
  Мониторе, а normalizers и predictor сохраняются вместе с моделью.
- **POMDP-среды** — набор частично наблюдаемых сред специально под память
  (LSTM/GRU): PO-CartPole/PO-Pendulum/PO-MountainCar/PO-Acrobot/PO-Lunar
  Lander (скрыты скорости), Flickering CartPole/Pong (кадр гасится с
  вероятностью 50% — рецепт статьи про DRQN), Memory Corridor и Memory
  Corridor (long) (нужно запомнить подсказку на 8 или 20 шагов вперёд),
  Repeat Previous / N-back (нужно непрерывно держать в памяти скользящее
  окно последних наблюдений), **RockSample(7,8)** — классический
  масштабируемый POMDP-бенчмарк из литературы по планированию (Smith &
  Simmons, 2004), где нужно копить шумные показания сенсора в уверенное
  мнение о камнях, **Visual Memory Maze** — аналог флагманского
  бенчмарка MiniGrid `MemoryEnv`: агент видит только маленькое окно
  вокруг себя и должен пронести цветную подсказку через весь лабиринт в
  памяти, а также восемь классических POMDP-бенчмарков другого типа:
  **Tiger**, **Heaven/Hell** (активный сбор информации), **Hallway**
  (state aliasing), **Battleship**/**Minesweeper (POMDP)**/**Concentration**
  (память об истории действий), **LaserTag** (движущаяся скрытая цель) и
  **Active T-Maze**/**Active T-Maze (long)** (подсказку нужно самому
  запросить). Подробности в разделе [«POMDP-среды»](#pomdp-среды).
- Визуальный **Дизайнер экспериментов** (граф env → wrappers → algo →
  training на `@xyflow/react`) — без единой строчки кода. Ноды можно
  свободно перетаскивать, а wrapper'ы вставлять в любое место цепочки (не
  только в конец) через «+» в каждом промежутке. Живая панель «Среда» /
  «Нейросеть» показывает входы/выходы и число параметров сети, пересчитываясь
  на бэкенде при любом изменении выбора (без запуска обучения) — см.
  `/api/environments/inspect` и `rl_core/inspect.py`.
- **Мониторинг** обучения в реальном времени (WebSocket): графики награды
  / win-rate, живой рендер среды, логи, остановка запуска.
- **Визуализация алгоритмов** — схема «как учится» (цикл алгоритма) и схема
  архитектуры сети показываются и при дизайне эксперимента, и во время
  обучения — для всех встроенных алгоритмов и кастомных модификаций
  (`src/components/AlgorithmDiagram.tsx`).
- **AlphaZero Arena** — сыграй против своего агента или случайного бота.
- **Model Zoo** — сохранённые чекпоинты, независимые от истории запусков.
- **Свои плагины** — пишите собственные алгоритмы (Gym и AlphaZero) и
  reward-функции прямо в приложении (Monaco-редактор на странице
  `/plugins`), они появляются в Дизайнере рядом со встроенными. Подробности
  ниже, в разделе [«Свои плагины»](#свои-плагины).
- **Конструктор архитектур сетей** (`/network-builder`) — визуально собери
  свою нейросеть (Linear/Conv2D/MaxPool/Flatten/Activation/Dropout/BatchNorm
  + головы action/value/q/advantage/policy) без единой строчки PyTorch, с
  live-превью формы каждого слоя и числа параметров. Ноды перетаскиваются, а
  новый слой можно вставить в любое место ствола/головы через «+» в любом
  промежутке — не только в конец. Готовую архитектуру можно сохранить и
  использовать в любом эксперименте той же семьи (например, спроектированной
  на Tic-Tac-Toe архитектурой для AlphaZero можно тренировать Gomoku — форма
  входа/выхода пересчитывается по фактической среде при каждом запуске), а
  не только там, где она была создана. Работает для PPO/A2C (actor_critic),
  DQN (q_network), Rainbow DQN (dueling_q) и AlphaZero (policy+value).
  Подробности в разделе [«Конструктор архитектур сетей»](#конструктор-архитектур-сетей).
- **Учебный центр** (`/academy`) — курс по RL внутри приложения: отдельный урок на
  каждый алгоритм (DQN, Rainbow DQN, A2C, PPO, SAC, AlphaZero) плюс основы RL,
  память (LSTM/GRU) и POMDP-среды — с формулами, графиками (ε-greedy decay,
  PPO clip objective, discount factor) и схемами (MCTS-дерево, replay buffer,
  dueling-сеть, ...), и итоговой шпаргалкой «что выбрать».

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

### GPU

По умолчанию всё считается на CPU — большинство сред и AlphaZero на маленьких
досках в этом не нуждаются, а на маленьких сетях GPU иногда даже медленнее
из-за накладных расходов на запуск ядер. Для тяжёлых прогонов (self-play на
больших досках вроде Gomoku 9×9, Atari/MuJoCo с CNN) GPU можно включить в двух
местах:

1. **Settings → Backend** — тумблер «GPU в Docker» или «GPU в Native-режиме».
   Это про **провижининг**: какая версия PyTorch стоит в окружении бэкенда
   (CPU-only vs CUDA-сборка). Docker пересобирает образ на `Dockerfile.gpu` +
   `rl_core/requirements-gpu.txt` и запускает контейнер с `--gpus all` (нужен
   NVIDIA Container Toolkit); Native переустанавливает venv из
   `requirements-gpu.txt` вместо `requirements.txt` (нужен просто драйвер
   NVIDIA). Оба варианта требуют перезапуска бэкенда — он сам пересоберёт
   образ/venv при первом старте после включения. Наличие CUDA после установки
   видно там же, в блоке «Система» (`torch_cuda_available`, список GPU).
2. **Дизайнер экспериментов → нода Training → «Использовать GPU»** — это про
   конкретный запуск: без CUDA-версии PyTorch (см. пункт 1) тумблер тихо
   ничего не даёт — обучение просто идёт на CPU. `training.use_gpu` в конфиге
   эксперимента дальше читает `rl_core/device.py`, который единообразно решает
   CPU/CUDA для всех треков: native PPO/DQN/A2C, custom Gym-плагины (в т.ч.
   SB3-based) и оба AlphaZero-трейнера (встроенный и custom).

При первом запуске приложение само спрашивает CPU или GPU (детектит видеокарты
через `nvidia-smi`) — до этого выбора установка не начинается, см.
`electron/gpu.ts` / фазу загрузки `awaiting-setup`.

**Важно про версию CUDA:** `rl_core/requirements-gpu.txt` не фиксирует
конкретный `--index-url` для torch — каналы вида `whl/cu121`/`whl/cu128`
периодически полностью выпиливаются PyTorch (так и случилось за время жизни
этого файла), и `pip` в этом случае молча скатывается на *текущий дефолт*
PyPI, которому может требоваться более новый драйвер, чем есть на машине —
торч ставится «успешно», но `torch.cuda.is_available()` всё равно `False`.
Поэтому канал (`cu130`/`cu126`/...) выбирается по максимальной версии CUDA,
которую реально поддерживает драйвер (`nvidia-smi`), — динамически, при каждой
установке (`electron/gpu.ts: pickTorchCudaChannel`), а не хардкодом. Если
после включения GPU в Settings всё равно видно «CUDA: нет» — там же теперь
показывается диагностика (версия PyTorch, собран ли он с CUDA) с подсказкой,
драйвер это или сам установщик.

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

`rl_core/algorithms/native/` — собственная реализация DQN/Rainbow
DQN/PPO/A2C/SAC на чистом PyTorch, без Stable-Baselines3 (это дефолтный
движок для встроенных алгоритмов, см. `native_runner.py`):

- `preprocessing.py` — приведение наблюдений к тензору: one-hot для
  скалярного `Discrete` (Taxi, FrozenLake, CliffWalking), обычный
  flatten-vector для `Box` (классика/Box2D/MuJoCo), «как есть» для
  картиночного `Box` (Atari, CarRacing).
- `networks.py` — `MLPExtractor`/`NatureCNN` (выбор автоматический по форме
  observation space) + `ActorCriticNet` (Categorical или диагональный
  Gaussian head), `QNetwork`, `DuelingQNetwork` (value+advantage стримы для
  Rainbow), `GaussianPolicy` + `QCritic` (squashed-Gaussian policy и
  state-action critic для SAC). Плюс рекуррентные варианты —
  `RecurrentActorCriticNet`, `RecurrentQNetwork`, `RecurrentDuelingQNetwork`
  (LSTM/GRU между признаками и головами, см. [«Память
  (LSTM/GRU)»](#память-lstmgru)).
- `buffers.py` — `RolloutBuffer` с GAE(λ) для PPO/A2C (плюс
  `episode_starts` и `sequences()` для рекуррентного truncated-BPTT
  обучения), `ReplayBuffer` для DQN, `NStepPrioritizedReplayBuffer` (n-step
  возвраты + priority-sampling) для Rainbow DQN, `ContinuousReplayBuffer`
  для SAC, `EpisodeSequenceReplayBuffer` — эпизодический буфер для
  рекуррентных DQN/Rainbow DQN (сэмплирует окна фиксированной длины из
  целых эпизодов вместо независимых переходов).
- `ppo.py` / `a2c.py` — общий rollout-цикл в `on_policy.py`, отличаются
  только `_update()`: PPO — clipped surrogate + несколько эпох по
  минибатчам, A2C — один градиентный шаг на весь rollout, без clipping. При
  включённой памяти оба идут по `_update_recurrent()` — то же самое, но
  по чанкам `buf.sequences(...)` с переносом/детачем hidden state между
  чанками (truncated BPTT).
- `dqn.py` — target-сеть, ε-greedy с линейным затуханием, replay buffer,
  Huber-лосс. При включённой памяти — DRQN на `EpisodeSequenceReplayBuffer`
  (окна с нулевым начальным hidden state, как в оригинальной статье
  Hausknecht & Stone, 2015, без R2D2-burn-in).
- `rainbow_dqn.py` — то же самое + Double Q-learning (online-сеть выбирает
  действие, target-сеть его оценивает), Dueling-архитектура, Prioritized
  Experience Replay (сэмплирование по `|TD-error|^alpha` + IS-коррекция) и
  n-step bootstrapping. Осознанно **не** реализованы distributional RL
  (C51) и Noisy Nets — расписано в докстринге файла, почему. При включённой
  памяти PER и n-step **отключаются** (сочетание с эпизодическими окнами
  требует R2D2-style sequence-priorities — заметно больший объём работы),
  остаются Double Q-learning + Dueling + LSTM/GRU.
- `sac.py` — Soft Actor-Critic: squashed-Gaussian policy, два Q-критика с
  target-сетями (Polyak-усреднение), максимизация reward + энтропии.
  Единственный встроенный off-policy алгоритм для continuous-действий —
  обычно эффективнее по данным, чем PPO/A2C на тех же средах.

Диспетчер (`rl_core/runner.py`) шлёт встроенные `ppo`/`dqn`/`a2c`/
`rainbow_dqn`/`sac` в `native_runner.py`. Stable-Baselines3 остаётся
опциональной зависимостью — нужна только тем, кто пишет свой
алгоритм-плагин, наследуя
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

## Конструктор архитектур сетей

Страница **`/network-builder`** — визуальный редактор сети на
`@xyflow/react`, альтернатива написанию PyTorch-кода из раздела выше для тех,
кому достаточно собрать сеть из стандартных слоёв. Архитектура — «ствол»
(последовательность слоёв: `Linear`, `Conv2D`, `MaxPool2D`, `Flatten`,
`Activation`, `Dropout`, `BatchNorm`) плюс одна или несколько «голов» —
отдельных мини-цепочек слоёв поверх ствола. Набор голов зависит от типа сети:

| Тип сети | Головы | Используется в |
|---|---|---|
| `actor_critic` | `action` (логиты/mu), `value` | PPO, A2C |
| `q_network` | `q` | DQN |
| `dueling_q` | `advantage`, `value` | Rainbow DQN |
| `alphazero` | `policy`, `value` | AlphaZero |
| `efficientzero` | typed representation, dynamics, prediction, SimSiam | EfficientZero |
| `unizero` | typed tokenizer, Transformer, reward/value/policy/latent heads | UniZero |
| `researchimzero` | UniZero-style Transformer + typed SimSiam components | ResearchImZero |

Последний слой каждой головы — «авто»: его размер (число действий,
`action_size`, `1` для value) подставляется самим движком, а не руками,
поэтому одна и та же архитектура переносится между средами/играми с разным
числом действий без правок — например, спроектированную и провалидированную
на Tic-Tac-Toe архитектуру можно тем же `network_spec_id` подключить к
эксперименту с Gomoku: формы слоёв пересчитываются под фактическую доску
(`rows`/`cols`/`action_size`) заново при каждом запуске, а не фиксируются в
момент сохранения спека. При любом изменении в редакторе бэкенд пересобирает
сеть по спеку целиком (`rl_core/netbuilder.py::preview_network`) и возвращает
форму каждого слоя + итоговое число параметров, либо чёткую ошибку
("добавь Flatten перед этим слоем" и т.п.) с указанием ровно того слоя,
где сборка остановилась — без единого traceback. Ноды на канвасе
перетаскиваются свободно, а слой можно вставить в любой промежуток ствола
или головы через «+» — не только в конец (см. `useNodePositions` в
`src/lib/useNodePositions.ts`, используется и здесь, и в Дизайнере).

Для трёх Zero-family алгоритмов та же страница автоматически переключается
в **component mode**: алгоритмически обязательные связи, Gumbel/PUCT search,
value-prefix LSTM, interleaved token protocol, KV-cache и размеры конечных
голов защищены от изменения, а observation encoder, latent/embed-размеры,
безопасные MLP-компоненты и Transformer depth/heads/FFN доступны для
настройки. Вход всегда берётся из реального observation space: `auto`
выбирает проверенный CNN/MLP, `image_cnn` и `vector_mlp` проверяют
совместимость, а custom chain обязан закончиться плоским вектором. Выходы
policy/value/reward/latent автоматически вычисляются по action space и
categorical support. Preview собирает те же классы, что реальный trainer,
поэтому валидная схема не расходится с runtime.

Архитектуры хранятся как чистый JSON (`ROOT/custom_networks/<slug>.json`,
без исполнения кода — в отличие от плагинов выше) и подключаются в узле
алгоритма в Дизайнере (селектор «Архитектура сети», виден только для
`ppo`/`a2c`/`dqn`/`rainbow_dqn`/`alphazero`/`efficientzero`/`unizero`/
`researchimzero` — кастомные плагины сами решают,
учитывать её или нет). Технически это просто ещё одно поле конфига
эксперимента (`algorithm.network_spec_id`) — раннер резолвит его в спек
(`rl_core/netbuilder_store.py::resolve_network_spec`) и подставляет
собранную сеть вместо сети по умолчанию (`ActorCriticNet`/`QNetwork`/
`DuelingQNetwork`/`AlphaZeroNet`) без изменений в самих алгоритмах обучения.

---

## Exploration: ε-greedy, NoisyNet и RND

У DQN и Rainbow DQN есть два независимых уровня исследования:

- **Исследование действий**: классический ε-greedy с настраиваемыми
  `initial_eps`, `final_eps` и долей linear decay либо NoisyNet с обучаемым
  factorized Gaussian шумом в Linear-слоях Q-головы. При deterministic
  evaluation шум отключается.
- **Intrinsic motivation**: RND (Random Network Distillation) сравнивает
  frozen random target с обучаемым predictor. Ошибка на незнакомом следующем
  состоянии нормализуется online и добавляется к reward с коэффициентом
  `rnd_bonus_coef`.

Эти механизмы комбинируются: ε-greedy + RND и NoisyNet + RND являются
валидными конфигурациями. RND совместим с LSTM/GRU и собственной
NetworkSpec, а NoisyNet — только со стандартной Q/dueling-Q сетью, поскольку
заменяет конкретные Linear-слои. Монитор всегда показывает реальный
extrinsic reward отдельно от intrinsic bonus, чтобы curiosity не выглядело
как решённая задача.

---

## Память (LSTM/GRU)

Гиперпараметр «Память (рекуррентность)» у PPO, A2C, DQN и Rainbow DQN
переключает сеть между обычной feed-forward архитектурой и версией со
встроенным LSTM/GRU между извлечением признаков и головами
действия/ценности (или Q/advantage). Полезно, когда одного наблюдения
недостаточно, чтобы принять решение — частичная наблюдаемость, нужно
помнить что-то за несколько шагов назад и т.п.

- **Гиперпараметры**: `memory_type` (Нет / LSTM / GRU), `memory_hidden_size`
  (размер hidden state), `memory_num_layers` (число слоёв RNN),
  `memory_seq_len` (длина truncated-BPTT последовательности при обучении —
  для PPO/A2C это длина чанка внутри одного rollout'а, для DQN/Rainbow DQN —
  длина сэмплируемого окна эпизода).
- **On-policy (PPO/A2C)**: поскольку среда всегда одна (без vec-env), весь
  rollout — это одна непрерывная последовательность; на обучении она режется
  на чанки по `memory_seq_len` **в исходном порядке** (не перемешивается —
  перемешивание сломало бы, зачем нужна память), hidden state переносится и
  «отрезается» (`detach`) между чанками.
- **Off-policy (DQN/Rainbow DQN)**: реплей-буфер хранит целые эпизоды
  (`EpisodeSequenceReplayBuffer`) и сэмплирует окна фиксированной длины с
  нулевым начальным hidden state (проще, чем R2D2's burn-in, и достаточно
  для тех размеров эпизодов, что есть в приложении).
- **Живое превью в предикте**: `predict(obs, episode_start=...)` — при
  `episode_start=True` алгоритм сбрасывает hidden state вместо переноса из
  прошлого вызова; используется автоматически и для рендера GIF в
  Мониторинге, и внутри самого обучения.

Осознанно **не покрыто** этой итерацией (можно предложить отдельной
задачей, если понадобится):

- **SAC и AlphaZero** — continuous control (SAC) обычно работает с
  полностью наблюдаемым состоянием, а настольные игры (AlphaZero) видят всю
  доску целиком, так что мотивация для памяти в обоих случаях намного
  слабее, чем в частично наблюдаемых Gym-средах.
- **Конструктор архитектур сетей** (`/network-builder`) — там пока нет
  LSTM/GRU как типа слоя; если выбрана своя (hand-designed) архитектура,
  встроенный переключатель памяти игнорируется (об этом явно
  предупреждает Дизайнер).
- **Rainbow DQN**: PER и n-step возвраты отключаются при включённой памяти
  (см. докстринг `rainbow_dqn.py`) — Double Q-learning + Dueling + LSTM/GRU
  остаются.

## POMDP-среды

Отдельная категория «POMDP (нужна память)» в галерее сред (`/environments`)
и в Дизайнере — среды, специально устроенные так, чтобы feed-forward
политика упиралась в потолок, а рекуррентная (см. [«Память
(LSTM/GRU)»](#память-lstmgru)) — не упиралась. Реализация — в
`rl_core/envs/pomdp.py`, регистрируется как обычные `gymnasium` env id
(`gym.register`), поэтому работает всюду, где работает любой другой env id
галереи (запуск, живой рендер в Мониторинге, `/environments/inspect`).

- **PO-CartPole / PO-Pendulum / PO-Mountain Car / PO-Acrobot / PO-Lunar
  Lander** — те же физика и рендеринг, что у обычных
  CartPole/Pendulum/MountainCar/Acrobot/LunarLander (превью — тот же
  официальный GIF), но из наблюдения убраны производные (скорости, угловые
  скорости) — стандартный приём превратить MDP-бенчмарк в POMDP (см.,
  например, RLlib's `StatelessCartPole` или Heess et al., 2015,
  "Memory-based control with recurrent neural networks"). PO-Lunar Lander —
  самый требовательный к памяти вариант: скрыты сразу все три скорости
  (vx, vy, угловая), видны только позиция, угол и контакт ног с землёй.
- **Flickering CartPole / Flickering Pong** — на каждом шаге с вероятностью
  50% всё наблюдение целиком гасится в ноль вместо настоящего кадра. Это
  ровно рецепт "flickering" из статьи Hausknecht & Stone, 2015, "Deep
  Recurrent Q-Learning for Partially Observable MDPs" (статья, которая ввела
  DRQN) — Flickering Pong воспроизводит их эксперимент буквально, Flickering
  CartPole — та же идея на среде, которая тренируется за секунды, а не за
  часы.
- **Memory Corridor / Memory Corridor (long)** — свежая, не основанная на
  обёртке среда: 8-шаговый «коридор», в самом первом наблюдении показывается
  один бит подсказки (0 или 1), дальше 7 шагов наблюдение нулевое, и награда
  (+1) начисляется только если последнее действие совпало с давно пропавшей
  подсказкой. Без памяти результат — не лучше случайного угадывания (0.5);
  с LSTM — решается идеально (проверено прямо в этом репозитории: обычный
  `NativeDQN` даёт ~0.55 средней награды за эпизод на 200 eval-эпизодах,
  `NativeDQN` с `memory_type=1` — 1.0). `Memory Corridor (long)` — та же
  задача, но задержка растянута до 20 шагов вместо 8, чтобы показать, что
  решает не любая память, а память достаточной *ёмкости*
  (`memory_seq_len`/`memory_hidden_size` должны расти вместе с задержкой).
- **Repeat Previous (N-back)** — классическая delayed-match-to-sample
  задача: каждый шаг показывается новый случайный символ из 4, а правильное
  действие — тот символ, что был показан ровно 3 шага назад. В отличие от
  Memory Corridor (одна подсказка, читается один раз в конце), здесь нужно
  непрерывно держать в памяти скользящее окно последних наблюдений и
  сверяться с ним на *каждом* шаге эпизода. Без памяти — точность не выше
  1/4 (случайное угадывание); с LSTM — решается почти идеально (проверено
  прямо в этом репозитории: `NativeDQN` без памяти держит точность 0.254 на
  200 последних эпизодах — ровно уровень шанса, — `NativeDQN` с
  `memory_type=1` — 0.961).
- **RockSample(7,8)** — классический *масштабируемый* POMDP-бенчмарк из
  литературы по планированию (Smith & Simmons, 2004, "Heuristic Search
  Value Iteration for POMDPs" — тот же бенчмарк, что и у Hallway/Hallway2 в
  той же линейке работ). 7×7-сетка, 8 камней в случайных, но известных
  агенту позициях; качество каждого камня (хороший/плохой) скрыто и
  проверяется только шумным дальнобойным сенсором (`check_i`), точность
  которого падает с расстоянием (`0.5 + 0.5·2^(-d/3)`). Здесь нужен другой
  тип памяти, чем "запомнить один чистый сигнал": несколько шумных
  проверок одного и того же камня нужно накопить в уверенное мнение,
  прежде чем тратить ход на `sample` (+10 за хороший, -10 за плохой).
- **Visual Memory Maze** — картиночный аналог флагманского бенчмарка
  MiniGrid `MemoryEnv` (того самого, на котором в документации
  `sb3-contrib`'s `RecurrentPPO` показывают, зачем нужна память): агент
  видит только окно 5×5 клеток вокруг себя, стартует в комнате с цветной
  подсказкой, идёт по одноклеточному коридору (подсказка давно выпала из
  поля зрения) и должен выбрать ту из двух комнат в конце, чей предмет
  совпадает по цвету с подсказкой. В отличие от всех остальных сред этой
  категории частичная наблюдаемость тут не искусственная маска, а
  настоящее ограниченное поле зрения — более реалистичный механизм,
  характерный для большинства настоящих POMDP.

Ещё восемь сред — прямиком из классики POMDP-*планирования* (Kaelbling,
Littman & Cassandra, 1998; Littman, Cassandra & Kaelbling, 1995) и
современных memory-RL бенчмарков (POPGym, Ni et al., 2023) — каждая
проверяет память по-своему *другому* поводу, не просто «вспомнить факт,
который был виден раньше»:

- **Tiger** — самый канонический toy-POMDP в литературе: тигр за одной из
  двух дверей, можно `listen` (85%-точная, но платная подсказка) или
  открыть дверь. Одного слушания недостаточно для уверенного решения —
  нужно интегрировать несколько независимых зашумлённых сигналов.
- **Heaven/Hell** — T-образный лабиринт с *активным* сбором информации:
  указатель, говорящий, где Небеса, стоит в тупике сбоку от главного пути,
  а не на нём. Агент без памяти, который никогда туда не заходит, обречён
  угадывать на развилке 50/50; с памятью — заглядывает один раз и потом
  уверенно идёт к цели.
- **Hallway** — компактная версия оригинального бенчмарка на *state
  aliasing*: `FORK0` и оба «двойника» `TWIN1`/`TWIN2` дают идентичное
  показание локального сенсора, но требуют противоположных действий, чтобы
  дойти до цели, а не в тупик. Любая фиксированная реакция на такое
  наблюдение гарантированно ошибается в одном из двойников — проверено
  прямо в этом репозитории скриптованной политикой (см.
  `rl_core/tests/test_more_pomdp_envs.py`): память о том, куда повернули на
  `FORK0`, даёт 100% успеха, а любой фиксированный выбор — 100% провала на
  двойнике.
- **Battleship** / **Minesweeper (POMDP)** — классические игры,
  переформулированные так, что наблюдение — только результат *последнего*
  выстрела/открытой клетки (координаты + попал/число соседних мин), а не
  рендер всей карты. Повторное действие в уже проверенной клетке
  штрафуется — без памяти обо всех прошлых ходах агент не может этого
  избежать.
- **Concentration** — карточная игра «найди пару», один из диагностических
  тестов памяти в бенчмарке POPGym: значение карты видно только в момент
  её переворота.
- **LaserTag** — впервые в этой категории скрытый факт не статичен, а
  *движется сам*: противник виден только в короткой прямой видимости,
  которую блокирует препятствие в центре арены (сам противник запрограммирован
  на побег/случайное блуждание, а не обучается — задача остаётся честным
  single-agent POMDP). Нужна не разовая память, а постоянно обновляемая
  оценка «где цель сейчас».
- **Active T-Maze / Active T-Maze (long)** — активная версия Memory
  Corridor (Ni et al., 2023, "When Do Transformers Shine in RL?"):
  подсказка не показывается сама, её нужно запросить действием `look`,
  стоя в самом начале одностороннего коридора — прошёл вперёд не
  посмотрев, подсказка потеряна навсегда. Проверяет не удержание
  информации, а то, научился ли агент понимать, когда стоит её запросить.

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
│   │                              AlphaZeroArena, ModelZoo, Plugins,
│   │                              NetworkBuilder, Academy, Settings
│   ├── components/
│   │   ├── designer/           #   Кастомные ноды для @xyflow/react
│   │   ├── networkbuilder/       #   Ноды конструктора архитектур (LayerNode, ...)
│   │   ├── academy/               #   Учебный центр — уроки (lessons.tsx) + визуалы/графики (visuals.tsx)
│   │   ├── flow/                  #   AddNode — общая "+"-нода (Designer и Network Builder)
│   │   ├── layout/              #   Sidebar/Header/Layout
│   │   └── ui/                   #   shadcn-стиль UI-кит
│   ├── lib/networkBuilder.ts       #   Дефолты/лейблы слоёв и голов сети
│   ├── lib/useNodePositions.ts       #   Драг нод с сохранением позиции между рендерами
│   ├── api/                    #   REST/WS клиент + React Query hooks
│   └── stores/                  #   Zustand (docker, settings)
├── backend/                  # FastAPI Python бэкенд
│   ├── api.py                 #   Точка входа
│   ├── ws.py                   #   WebSocket метрик
│   ├── process_manager.py       #   Управление subprocess'ами запусков
│   └── routes/                   #   system, environments, training,
│                                      models, alphazero, plugins, networks
├── rl_core/                  # RL-ядро — envs, алгоритмы, AlphaZero
│   ├── envs/                   #   Gymnasium registry + wrappers + reward_fn + pomdp.py
│   ├── games/                   #   TicTacToe / ConnectFour / Gomoku
│   ├── algorithms/               #   base.py + native/ (PPO/A2C/DQN/Rainbow/SAC с нуля) +
│   │                                    native_runner/runner_utils + sb3_runner
│   │                                    (опционально, для custom-плагинов) +
│   │                                    custom_runner
│   ├── alphazero/                #   base/loop + network/mcts/self_play/train
│   ├── plugins/                    #   loader.py + templates.py (свои плагины)
│   ├── netbuilder.py                #   Спек → torch.nn.Module (Network Builder)
│   ├── netbuilder_store.py            #   CRUD спеков сетей (custom_networks/*.json)
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
| RL         | Gymnasium, PyTorch (свой PPO/DQN/Rainbow DQN/A2C/SAC); Stable-Baselines3 — опционально, только для custom-плагинов |
| AlphaZero  | Собственная реализация (PUCT MCTS + CNN)   |
| Сборка     | electron-builder (Linux/Win/Mac)           |

## Лицензия

Apache License 2.0 — полный текст находится в файле [`LICENSE`](LICENSE).
