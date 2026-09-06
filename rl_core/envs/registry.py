"""Registry of Gymnasium environments exposed in the Environments gallery.

Every entry describes the env well enough for the frontend to render a
card and for the Experiment Designer to know which algorithms are valid
(discrete vs continuous action spaces) without importing gymnasium there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rl_core.envs.pomdp import register_pomdp_envs

register_pomdp_envs()

try:
    from rl_core.envs.minigrid_envs import register_minigrid_envs

    register_minigrid_envs()
except ImportError:
    pass

try:
    from rl_core.envs.highway_envs import register_highway_envs

    register_highway_envs()
except ImportError:
    pass

try:
    from rl_core.envs.nethack_envs import register_nethack_envs

    register_nethack_envs()
except ImportError:
    pass

try:
    from rl_core.envs.robotics_envs import register_robotics_envs

    register_robotics_envs()
except ImportError:
    pass

# No optional third-party dependency (unlike every `register_*_envs` above)
# — always registered.
from rl_core.envs.finrl_envs import register_finrl_envs
from rl_core.envs.industrial_envs import register_industrial_envs
from rl_core.envs.trading_envs import register_trading_envs

register_industrial_envs()
register_trading_envs()
register_finrl_envs()

ActionKind = Literal["discrete", "continuous"]

_DISCRETE = ["dqn", "rainbow_dqn", "ppo", "a2c", "es", "dreamer", "world_models_ha", "efficientzero", "unizero", "researchimzero", "latentimzero"]
_CONTINUOUS = ["ppo", "a2c", "sac", "ddpg", "td3", "es", "dreamer", "mbpo", "pets", "world_models_ha", "efficientzero", "unizero", "researchimzero", "latentimzero"]

@dataclass
class EnvSpec:
    id: str
    name: str
    category: str
    description: str
    action_kind: ActionKind
    compatible_algorithms: list[str] = field(default_factory=list)
    extra_requirement: str | None = None  # pip extra, e.g. "box2d" / "mujoco" / "atari"
    # Per-env overrides merged on top of the algorithm's global hyperparam
    # defaults in the Designer (see ALGORITHM_CATALOG in
    # backend/routes/environments.py). Exists because AlphaZero's defaults
    # are tuned for Tic-Tac-Toe (9 legal actions) — using the same MCTS/net
    # budget on Gomoku (81 actions, 5-in-a-row) leaves the policy stuck near
    # a uniform random baseline (loss ~= log(81), 0% win-rate vs prev) for
    # dozens of iterations, which reads as "training is broken" rather than
    # "this board needs a bigger budget".
    default_hyperparams: dict[str, float] | None = None
    # Suggested `total_timesteps` for the Designer to pre-fill when this env
    # is selected, overriding the flat 50k UI default. Exists because that
    # 50k default is a reasonable budget for CartPole-class envs but wildly
    # too small for pixel-input/hardcore envs — a user hitting "Run" with
    # the default on e.g. CarRacing only ever sees the "car spins in place"
    # stage and reasonably concludes training is broken, when it just never
    # ran long enough to show anything else. `None` keeps the flat default.
    recommended_total_timesteps: int | None = None
    # Suggested `training.num_envs` (parallel env copies, see
    # `rl_core/algorithms/vec_env.py`) for the Designer to pre-fill — set
    # for envs heavy/slow enough per-step that batching several lanes'
    # worth of experience per network update meaningfully helps (Atari,
    # Box2D, MuJoCo). `None` keeps the flat default of 1.
    recommended_num_envs: int | None = None
    # Wrapper-graph nodes to pre-fill when this env is selected (see
    # `handleEnvChange` in ExperimentDesigner.tsx) — `[{"type": ..., "params":
    # {...}}, ...]` in the exact order they should be chained. Exists for
    # pixel-input envs where the "right" preprocessing pipeline (frame
    # skip/resize/grayscale/stack) is both non-obvious and, if skipped,
    # silently leaves the CNN fed 3x more raw pixels than it needs — a user
    # dropping CarRacing onto the canvas gets the whole rl-baselines3-zoo-
    # tuned pipeline for free instead of having to know it exists.
    # `None`/`[]` leaves wrappers empty like every other env.
    recommended_wrappers: list[dict] | None = None


def _env(
    env_id: str,
    name: str,
    category: str,
    description: str,
    action_kind: ActionKind = "discrete",
    extra: str | None = None,
    recommended_total_timesteps: int | None = None,
    recommended_num_envs: int | None = None,
) -> EnvSpec:
    return EnvSpec(
        id=env_id,
        name=name,
        category=category,
        description=description,
        action_kind=action_kind,
        compatible_algorithms=_CONTINUOUS if action_kind == "continuous" else _DISCRETE,
        extra_requirement=extra,
        recommended_total_timesteps=recommended_total_timesteps,
        recommended_num_envs=recommended_num_envs,
    )


GYM_ENVIRONMENTS: list[EnvSpec] = [
    # --- Classic Control (built-in) ---
    _env("CartPole-v1", "CartPole", "classic_control",
         "Балансировка шеста на тележке. Классика для первого знакомства с RL."),
    _env("MountainCar-v0", "Mountain Car", "classic_control",
         "Разогнать машинку по инерции, чтобы заехать на гору. Разреженная награда."),
    _env("MountainCarContinuous-v0", "Mountain Car Continuous", "classic_control",
         "Та же гора, но с непрерывным газом — проверка continuous-policy.",
         action_kind="continuous"),
    _env("Acrobot-v1", "Acrobot", "classic_control",
         "Двузвенный маятник, нужно раскачать до вертикали."),
    _env("Pendulum-v1", "Pendulum", "classic_control",
         "Непрерывное управление: удержать маятник в верхнем положении.",
         action_kind="continuous"),
    # --- Toy Text ---
    _env("FrozenLake-v1", "Frozen Lake", "toy_text",
         "Сетка со льдом и лунками, стохастические переходы. Хорошо для табличных методов."),
    _env("Taxi-v4", "Taxi", "toy_text",
         "Такси должно забрать и довезти пассажира — задача с иерархией целей."),
    _env("Blackjack-v1", "Blackjack", "toy_text",
         "Карточный блэкджек против дилера. Маленькое дискретное состояние, классика табличного Q-learning. "
         "У дилера есть встроенное преимущество: даже идеальная стратегия даёт среднюю награду примерно "
         "-0.04…-0.06 за раздачу (не 0 и тем более не положительную) — если график вышел на это плато, "
         "агент уже выучил (почти) оптимальную стратегию, это не баг и не признак того, что обучение не идёт. "
         "Состояние (сумма, карта дилера, есть ли туз) полностью описывает раздачу — память (LSTM/GRU) здесь "
         "не нужна и может только замедлить обучение."),
    _env("CliffWalking-v1", "Cliff Walking", "toy_text",
         "Сетка с обрывом: кратчайший путь вдоль края vs безопасный обход. Учебник Sutton & Barto."),
    # --- Box2D ---
    _env("LunarLander-v3", "Lunar Lander", "box2d",
         "Посадить лунный модуль на площадку. Дискретные двигатели.", extra="box2d"),
    _env("LunarLanderContinuous-v3", "Lunar Lander Continuous", "box2d",
         "Та же посадка, но газ и боковые двигатели — непрерывные.",
         action_kind="continuous", extra="box2d"),
    _env("BipedalWalker-v3", "Bipedal Walker", "box2d",
         "Двуногий робот учится ходить по пересечённой местности.",
         action_kind="continuous", extra="box2d"),
    _env("BipedalWalkerHardcore-v3", "Bipedal Walker Hardcore", "box2d",
         "Ходьба с ямами, ступеньками и шипами — заметно сложнее обычного Walker.",
         action_kind="continuous", extra="box2d", recommended_total_timesteps=5_000_000),
    EnvSpec(
        id="CarRacing-v3",
        name="Car Racing",
        category="box2d",
        description="Пиксельная трасса сверху 96×96: руль, газ и тормоз — непрерывное управление по "
                     "картинке, без сигнала скорости в одном кадре. Одна из самых требовательных сред в "
                     "галерее: разреженная награда, эпизод ~1000 шагов, экшены несимметричны (газ/тормоз "
                     "в [0,1]). По умолчанию подставляются цепочка враппером Frame Skip → Resize → "
                     "Grayscale → Frame Stack и Beta-политика (без клиппинга для [0,1]-экшенов). "
                     "Рассчитывайте на 3-5 млн шагов до уверенных кругов — плато в районе 350-700 без "
                     "доп. настройки — типичная стадия обучения, а не баг.",
        action_kind="continuous",
        compatible_algorithms=_CONTINUOUS,
        extra_requirement="box2d",
        default_hyperparams={
            # SB3-zoo-tuned PPO settings for pixel CarRacing (see
            # hyperparams/ppo.yml in DLR-RM/rl-baselines3-zoo):
            # - Shorter rollouts collected more often beat the
            #   CartPole/MuJoCo-sized default of 2048 steps, since each env
            #   step here is much more expensive to simulate/render and the
            #   CNN needs many more gradient updates per environment step to
            #   make sense of raw pixels.
            "n_steps": 512,
            "batch_size": 128,
            # zoo uses a lower, linearly-decaying LR (lin_1e-4) rather than
            # the flat 3e-4 PPO default — a smaller, annealed LR keeps the
            # CNN's late-training updates from overwriting an
            # already-decent driving policy with one noisy rollout.
            "learning_rate": 1e-4,
            "lr_schedule": 1,
            # gSDE (generalized State-Dependent Exploration, Raffin et al.
            # 2021) samples one exploration noise matrix per
            # `sde_sample_freq` steps instead of independent per-step
            # Gaussian noise — for a continuous steering/gas/brake action
            # this means multi-step-coherent exploration (e.g. "steer
            # slightly left for the next few steps") instead of frame-to-
            # frame jitter, which both explores more usefully and drives
            # more smoothly. `sde_log_std_init=-2` starts exploration
            # fairly tight, matching the zoo config.
            # Beta (not gSDE) is the default exploration/action distribution
            # here: CarRacing's action box is hard-bounded and one-sided on
            # 2 of its 3 dims (gas/brake in [0,1]), and a Gaussian sampling
            # outside that box needs a hard clip that the policy gradient
            # never sees — a documented real bottleneck for exactly this
            # env (Petrazzini & Antonelo 2021, +63% success rate on
            # CarRacing swapping Gaussian -> Beta with PPO). Beta's support
            # is exactly `[0,1]` so every sample is already valid, no clip
            # needed. Mutually exclusive with gSDE in `ActorCriticNet`
            # (Beta wins) since gSDE is specifically a Gaussian scheme —
            # `use_sde` is left off rather than just unused so switching
            # this back to 0 doesn't silently re-enable it.
            "use_beta": 1,
            "use_sde": 0,
            "sde_sample_freq": 4,
            "sde_log_std_init": -2.0,
            # zoo's `net_arch=dict(pi=[256], vf=[256])` — a dedicated 256-
            # wide layer per head instead of `mu_head`/`value_head` reading
            # directly off the 512-dim CNN output (see `ActorCriticNet` in
            # networks.py). Matters more here than on most envs: one frame
            # has a lot going on (track curvature, edges, car pose) to
            # compress into 3 continuous actions, and a bare linear head
            # tends to plateau a few hundred points below a solved run.
            "head_hidden_size": 256,
            # SAC's replay buffer stores raw float32 (H, W, C) frames — at
            # 96x96x3 that's ~110KB per observation, ~220KB per transition
            # once you count obs+next_obs. The algorithm-wide default of
            # 1,000,000 transitions would need ~220GB of RAM; this keeps it
            # under ~5GB while still giving SAC a reasonably diverse buffer.
            "buffer_size": 20_000,
        },
        # Matches zoo's own `n_timesteps: 4e6` for this exact config —
        # earlier defaulted lower (2M) on the assumption that was "enough
        # to see a clear trend", but 2M in practice plateaus well below a
        # solved run (some real training runs land around ~500) even with
        # the rest of this tuning; cheaper per-step now (frame skip +
        # smaller frames) means this budget costs less wall-clock than the
        # old 2M did before this pipeline existed.
        recommended_total_timesteps=4_000_000,
        recommended_num_envs=8,
        # zoo's env_wrapper + frame_stack for CarRacing-v3, translated to
        # our wrapper catalog and chained in the order they should wrap the
        # base env (frame_skip innermost so Resize/Grayscale only ever run
        # once per skipped block, not on every raw frame) — see
        # `ChannelStackObservation` in rl_core/envs/wrappers.py for why
        # Frame Stack needs `num_stack: 2` here to end up as a proper
        # (64, 64, 2) CNN input rather than a 4-D shape nothing can read as
        # an image. 2 stacked (already frame-skipped) frames is enough to
        # recover velocity/drift; zoo uses the same `frame_stack: 2`.
        #
        # Deliberately *not* including `normalize_reward` here, unlike the
        # zoo config's `norm_reward: True`: zoo applies that via
        # `VecNormalize`, which sits *outside* `Monitor` in its wrapper
        # stack, so the logged/plotted episode reward stays the real game
        # score while only the training signal (advantage/return calc) sees
        # the normalized one. Our `normalize_reward` is just another node
        # in this same per-env chain with no such split — every consumer,
        # including the Training Monitor UI and `episode_reward_mean`, ends
        # up reading the *normalized* number instead of the actual 0-1000
        # CarRacing score, and that normalization is itself unreliable
        # early in a fresh run (gymnasium's own docs: "rewards will not be
        # scaled correctly if the wrapper was newly instantiated"). Net
        # effect if it were included: the reward chart stops meaning
        # anything close to "getting closer to 1000" and can even swing
        # negative on a perfectly fine run, which is exactly backwards from
        # what this pipeline exists to help with.
        recommended_wrappers=[
            {"type": "frame_skip", "params": {"skip": 2}},
            {"type": "resize_observation", "params": {"height": 64, "width": 64}},
            {"type": "grayscale_observation", "params": {"keep_dim": True}},
            {"type": "frame_stack", "params": {"num_stack": 2}},
        ],
    ),
    # --- MuJoCo / robotics ---
    _env("InvertedPendulum-v5", "Inverted Pendulum", "mujoco",
         "MuJoCo-версия CartPole: удержать шест непрерывным усилием.",
         action_kind="continuous", extra="mujoco"),
    _env("InvertedDoublePendulum-v5", "Inverted Double Pendulum", "mujoco",
         "Двойной перевёрнутый маятник — классика неустойчивого управления.",
         action_kind="continuous", extra="mujoco"),
    _env("Reacher-v5", "Reacher", "mujoco",
         "2D-рука: дотянуться концом до цели. Компактный continuous-бенчмарк.",
         action_kind="continuous", extra="mujoco"),
    _env("Pusher-v5", "Pusher", "mujoco",
         "3D-рука толкает объект к целевой точке.",
         action_kind="continuous", extra="mujoco"),
    _env("HalfCheetah-v5", "HalfCheetah", "mujoco",
         "Плоский гепард: бежать вперёд как можно быстрее.",
         action_kind="continuous", extra="mujoco"),
    _env("Hopper-v5", "Hopper", "mujoco",
         "Одноногий прыгун. Нужно Hop вперёд, не упав.",
         action_kind="continuous", extra="mujoco"),
    _env("Walker2d-v5", "Walker2d", "mujoco",
         "Плоский двуногий ходок. Базовый locomotion-бенчмарк.",
         action_kind="continuous", extra="mujoco"),
    _env("Swimmer-v5", "Swimmer", "mujoco",
         "Змеевидный пловец в вязкой среде.",
         action_kind="continuous", extra="mujoco"),
    _env("Ant-v5", "Ant", "mujoco",
         "3D-квадрупед: скоординировать четыре ноги и бежать вперёд.",
         action_kind="continuous", extra="mujoco"),
    _env("Humanoid-v5", "Humanoid", "mujoco",
         "3D-гуманоид: ходить/бежать, не упав. Тяжёлая задача.",
         action_kind="continuous", extra="mujoco", recommended_total_timesteps=2_000_000),
    _env("HumanoidStandup-v5", "Humanoid Standup", "mujoco",
         "Гуманоид стартует лёжа — нужно встать. Ещё сложнее обычного Humanoid.",
         action_kind="continuous", extra="mujoco", recommended_total_timesteps=2_000_000),
    # --- Atari 2600 (ALE) ---
    _env("ALE/Pong-v5", "Pong", "atari",
         "Настольный теннис. Классика для проверки pixel-DQN.", extra="atari"),
    _env("ALE/Breakout-v5", "Breakout", "atari",
         "Разбить кирпичи шариком. Нужно нажать FIRE, чтобы подать.", extra="atari"),
    _env("ALE/SpaceInvaders-v5", "Space Invaders", "atari",
         "Волны пришельцев сверху. Один из эталонных ALE-бенчмарков.", extra="atari"),
    _env("ALE/MsPacman-v5", "Ms. Pac-Man", "atari",
         "Лабиринт, точки и привидения. Хороший exploration-тест.", extra="atari"),
    _env("ALE/Qbert-v5", "Q*bert", "atari",
         "Пирамида кубиков: менять цвет, избегая врагов.", extra="atari"),
    _env("ALE/Seaquest-v5", "Seaquest", "atari",
         "Подлодка: спасать дайверов и всплывать за воздухом.", extra="atari"),
    _env("ALE/Enduro-v5", "Enduro", "atari",
         "Длинная гонка по шоссе со сменой дня и погоды.", extra="atari"),
    _env("ALE/BeamRider-v5", "Beam Rider", "atari",
         "Коридоры из лучей, волны кораблей. Быстрая реакция.", extra="atari"),
    _env("ALE/Asteroids-v5", "Asteroids", "atari",
         "Стрелять по астероидам в стиле vector-arcade.", extra="atari"),
    _env("ALE/Boxing-v5", "Boxing", "atari",
         "Вид сверху: выбить больше ударов, чем соперник.", extra="atari"),
    _env("ALE/Freeway-v5", "Freeway", "atari",
         "Перейти многополосное шоссе, не попав под машину.", extra="atari"),
    _env("ALE/Frostbite-v5", "Frostbite", "atari",
         "Собрать иглу из льдин в арктической реке.", extra="atari"),
    _env("ALE/Riverraid-v5", "River Raid", "atari",
         "Вертикальный скролл: лететь по реке, не врезавшись и не кончив топливо.", extra="atari"),
    _env("ALE/RoadRunner-v5", "Road Runner", "atari",
         "Убегать от койота и собирать зерно на шоссе.", extra="atari"),
    _env("ALE/Assault-v5", "Assault", "atari",
         "Космический шутер: волны кораблей и боковые турели.", extra="atari"),
    _env("ALE/Atlantis-v5", "Atlantis", "atari",
         "Защитить город тремя турелями от бомбардировщиков.", extra="atari"),
    _env("ALE/BattleZone-v5", "Battle Zone", "atari",
         "Псевдо-3D танковый бой. Один из первых FPS-подобных Atari.", extra="atari"),
    _env("ALE/Centipede-v5", "Centipede", "atari",
         "Стрелять по многосегментной гусенице среди грибов.", extra="atari"),
    _env("ALE/ChopperCommand-v5", "Chopper Command", "atari",
         "Вертолёт прикрывает конвой грузовиков.", extra="atari"),
    _env("ALE/CrazyClimber-v5", "Crazy Climber", "atari",
         "Залезть на небоскрёб, уворачиваясь от окон и предметов.", extra="atari"),
    _env("ALE/DemonAttack-v5", "Demon Attack", "atari",
         "Волны демонов сверху — вариация Space Invaders.", extra="atari"),
    _env("ALE/FishingDerby-v5", "Fishing Derby", "atari",
         "Соревновательная рыбалка: кто быстрее наловит.", extra="atari"),
    _env("ALE/Gopher-v5", "Gopher", "atari",
         "Защитить морковку от суслика, который роет норы.", extra="atari"),
    _env("ALE/Hero-v5", "H.E.R.O.", "atari",
         "Шахты, динамит и спасение шахтёров. Нужно планировать маршрут.", extra="atari"),
    _env("ALE/IceHockey-v5", "Ice Hockey", "atari",
         "Хоккей 2 на 2, вид сверху.", extra="atari"),
    _env("ALE/Kangaroo-v5", "Kangaroo", "atari",
         "Платформер: кенгуру спасает детёныша.", extra="atari"),
    _env("ALE/KungFuMaster-v5", "Kung-Fu Master", "atari",
         "Сайд-скролл файтинг: волны врагов на этажах пагоды.", extra="atari"),
    _env("ALE/MontezumaRevenge-v5", "Montezuma's Revenge", "atari",
         "Знаменитый hard-exploration: ключи, комнаты, почти нулевая награда.", extra="atari"),
    _env("ALE/Pitfall-v5", "Pitfall!", "atari",
         "Джунгли, ямы и сокровища. Тоже sparse-reward.", extra="atari"),
    _env("ALE/PrivateEye-v5", "Private Eye", "atari",
         "Детектив по районам города: предметы и долгие горизонты.", extra="atari"),
    _env("ALE/Skiing-v5", "Skiing", "atari",
         "Спуск по трассе между воротами на время.", extra="atari"),
    _env("ALE/Tennis-v5", "Tennis", "atari",
         "Теннис на корте. Нужно подать мяч (FIRE).", extra="atari"),
    _env("ALE/Asterix-v5", "Asterix", "atari",
         "Собирать предметы, уворачиваясь от римлян. Горизонтальный скролл.", extra="atari"),
    _env("ALE/Phoenix-v5", "Phoenix", "atari",
         "Вертикальный шутер с боссом-птицей.", extra="atari"),
    _env("ALE/TimePilot-v5", "Time Pilot", "atari",
         "Самолёт сквозь эпохи: бипланы, джеты, боссы.", extra="atari"),
    _env("ALE/UpNDown-v5", "Up'n Down", "atari",
         "Машинка по извилистой дороге: собирать флажки, таранить врагов.", extra="atari"),
    _env("ALE/VideoPinball-v5", "Video Pinball", "atari",
         "Пинбол: флипперы, бамперы и набор очков.", extra="atari"),
    _env("ALE/WizardOfWor-v5", "Wizard of Wor", "atari",
         "Лабиринт с монстрами и радаром. Кооператив/соревнование.", extra="atari"),
    _env("ALE/Zaxxon-v5", "Zaxxon", "atari",
         "Изометрический шутер: высота имеет значение.", extra="atari"),
    # --- MiniGrid (частично наблюдаемая навигация/головоломки, egocentric-вид) ---
    _env("MiniGrid-Empty-8x8-Img-v0", "MiniGrid: пустая комната", "minigrid",
         "8×8 пустая комната, дойти до угла. Самая простая — для проверки конвейера.",
         extra="minigrid"),
    _env("MiniGrid-FourRooms-Img-v0", "MiniGrid: четыре комнаты", "minigrid",
         "Четыре смежные комнаты со случайной целью — базовая навигационная задача.",
         extra="minigrid"),
    _env("MiniGrid-DoorKey-8x8-Img-v0", "MiniGrid: дверь и ключ", "minigrid",
         "8×8: сначала подобрать ключ, потом открыть дверь и дойти до цели — "
         "простейшая последовательная под-задача (subgoal).",
         extra="minigrid"),
    _env("MiniGrid-LavaCrossingS9N1-Img-v0", "MiniGrid: пересечение лавы", "minigrid",
         "Пройти через сетку с полосами лавы, не наступив — жёсткий штраф за ошибку.",
         extra="minigrid"),
    _env("MiniGrid-Dynamic-Obstacles-8x8-Img-v0", "MiniGrid: движущиеся препятствия", "minigrid",
         "8×8, препятствия сами двигаются каждый шаг — нужно реагировать, а не запоминать карту.",
         extra="minigrid"),
    _env("MiniGrid-MultiRoom-N4-S5-Img-v0", "MiniGrid: мультикомнатный лабиринт", "minigrid",
         "4 комнаты, соединённые дверными проёмами — hard-exploration задача, награда только в конце.",
         extra="minigrid"),
    _env("MiniGrid-MemoryS9-Img-v0", "MiniGrid: память", "minigrid",
         "Флагманский бенчмарк для памяти (LSTM/GRU): увидеть объект-подсказку в начале коридора, "
         "затем выбрать одну из двух дверей в конце по цвету/форме подсказки — чисто feed-forward "
         "политика упирается в потолок, см. рекуррентные сети в Network Builder.",
         extra="minigrid"),
    _env("MiniGrid-KeyCorridorS3R1-Img-v0", "MiniGrid: коридор с ключом", "minigrid",
         "Ключ спрятан в одной из комнат коридора, дверь с сокровищем — в другой. "
         "Многошаговый exploration + planning.",
         extra="minigrid"),
    # --- Highway-env (принятие решений в вождении) ---
    _env("highway-v0", "Highway: шоссе", "highway_env",
         "Многополосное шоссе с трафиком: перестроение/ускорение/торможение (5 discrete "
         "meta-действий), избегать столкновений.", extra="highway_env"),
    _env("merge-v1", "Highway: слияние полос", "highway_env",
         "Влиться в основной поток с полосы разгона, не столкнувшись — короче и плотнее, чем шоссе.",
         extra="highway_env"),
    _env("roundabout-v1", "Highway: кольцевая развязка", "highway_env",
         "Проехать кольцевую развязку среди другого трафика, выбрать нужный выезд.",
         extra="highway_env"),
    _env("intersection-v1", "Highway: перекрёсток", "highway_env",
         "Нерегулируемый перекрёсток — самая насыщенная по взаимодействиям сцена, всего "
         "3 discrete-действия (ждать/ехать/тормозить), но высокий риск столкновения.",
         extra="highway_env"),
    _env("parking-Flat-v0", "Highway: парковка", "highway_env",
         "Continuous control (руль+газ): припарковаться в размеченное место задом. "
         "Goal-conditioned (цель — часть наблюдения), сплющено в единый Box для совместимости.",
         action_kind="continuous", extra="highway_env"),
    # --- NetHack / MiniHack (процедурный dungeon-crawler, hard exploration) ---
    _env("NetHackScore-Img-v0", "NetHack: очки", "nethack",
         "Полноценный NetHack — цель максимизировать игровой score. Огромное пространство "
         "состояний и очень разреженная награда, эталонная задача для long-horizon exploration.",
         extra="nle", recommended_total_timesteps=2_000_000, recommended_num_envs=8),
    _env("NetHackEat-Img-v0", "NetHack: голод", "nethack",
         "NetHack с задачей выживания: следить за голодом и вовремя есть — простой суб-навык "
         "полной игры, короче эпизоды, чем в NetHackScore.",
         extra="nle", recommended_total_timesteps=1_000_000, recommended_num_envs=8),
    _env("MiniHack-Room-5x5-Img-v0", "MiniHack: комната 5×5", "nethack",
         "Дойти до лестницы в маленькой пустой комнате — самая простая MiniHack-задача, "
         "для проверки конвейера перед более сложными.",
         extra="minihack"),
    _env("MiniHack-Room-15x15-Img-v0", "MiniHack: комната 15×15", "nethack",
         "Та же задача, но комната в 9 раз больше — требует больше шагов исследования.",
         extra="minihack"),
    _env("MiniHack-Corridor-R2-Img-v0", "MiniHack: коридор", "nethack",
         "Найти путь по системе коридоров, соединяющих комнаты — навигация без визуального "
         "плана всего уровня (только egocentric crop).",
         extra="minihack"),
    _env("MiniHack-MazeWalk-9x9-Img-v0", "MiniHack: лабиринт 9×9", "nethack",
         "Классический лабиринт: дойти от старта до цели, избегая тупиков.",
         extra="minihack"),
    _env("MiniHack-KeyRoom-5x5-Img-v0", "MiniHack: комната с ключом", "nethack",
         "Подобрать ключ, открыть дверь, дойти до лестницы — многошаговая под-задача, "
         "как DoorKey в MiniGrid, но на движке NetHack.",
         extra="minihack"),
    _env("MiniHack-Eat-Img-v0", "MiniHack: еда", "nethack",
         "Найти и съесть предмет еды — простая задача на manipulation-команды NetHack "
         "(в отличие от чисто навигационных MiniHack-задач выше).",
         extra="minihack"),
    # --- Robotics (goal-conditioned навигация, MuJoCo-физика) ---
    _env("PointMaze-UMaze-Flat-v0", "Point Maze: U-образный", "robotics",
         "Шарик с силовым управлением (2D) должен доехать до цели в U-образном лабиринте. "
         "Goal-conditioned (наблюдение включает desired_goal) — D4RL-бенчмарк для offline/HER RL.",
         action_kind="continuous", extra="gymnasium_robotics"),
    _env("PointMaze-Medium-Flat-v0", "Point Maze: средний", "robotics",
         "Тот же шарик, но лабиринт заметно больше и запутаннее — дольше exploration.",
         action_kind="continuous", extra="gymnasium_robotics"),
    _env("AntMaze-UMaze-Flat-v0", "Ant Maze: U-образный", "robotics",
         "Четырёхногий MuJoCo-муравей должен САМ научиться ходить и одновременно "
         "навигировать по U-образному лабиринту до цели — локомоция и навигация в одной "
         "задаче, заметно сложнее, чем Point Maze или обычный Ant-v5.",
         action_kind="continuous", extra="gymnasium_robotics",
         recommended_total_timesteps=3_000_000),
    _env("AntMaze-Medium-Flat-v0", "Ant Maze: средний", "robotics",
         "Та же ходьба+навигация, но в заметно большем лабиринте.",
         action_kind="continuous", extra="gymnasium_robotics",
         recommended_total_timesteps=5_000_000),
    # --- Industrial (планирование производства, логистика) ---
    _env("JobShop-6x6-v0", "Job Shop 6×6", "industrial",
         "Классическая задача цехового планирования (Job Shop Scheduling): 6 заданий, "
         "каждое — своя последовательность из 6 операций на 6 станках, нужно расставить "
         "порядок так, чтобы минимизировать makespan (время завершения последней операции). "
         "На каждом шаге агент называет одно ещё не завершённое задание — его следующая "
         "операция встаёт в очередь на нужный станок.",
         recommended_total_timesteps=300_000),
    _env("JobShop-10x10-v0", "Job Shop 10×10", "industrial",
         "Та же задача, но 10 заданий × 10 станков — куда больше комбинаций порядка "
         "диспетчеризации, классический размер тестовых наборов Taillard.",
         recommended_total_timesteps=600_000),
    _env("BinPacking-v0", "Bin Packing (online)", "industrial",
         "Онлайн-упаковка предметов по контейнерам фиксированной ёмкости (склад/паллеты, "
         "облачный bin-packing ресурсов): предметы прибывают по одному, каждый нужно сразу "
         "положить в какой-то открытый контейнер или открыть новый — цель минимизировать "
         "число использованных контейнеров.",
         recommended_total_timesteps=300_000),
    # --- Trading (управление позицией на рынке) ---
    _env("Trading-Discrete-v0", "Trading: направление", "trading",
         "Синтетический одноактивный трейдинг: на каждом шаге выбрать шорт/флэт/лонг. "
         "Цена — regime-switching GBM (бычий/медвежий/боковой режимы со случайными "
         "переключениями), свежая случайная траектория в каждом эпизоде — учит реагировать "
         "на паттерн доходностей, а не запоминать один исторический график. Награда = "
         "доходность позиции минус издержки на смену позиции (turnover cost).",
         recommended_total_timesteps=500_000),
    _env("Trading-Continuous-v0", "Trading: размер позиции", "trading",
         "Та же задача, но действие — точная целевая позиция в [-1, 1] (доля капитала, "
         "включая шорт) вместо выбора направления — можно частично закрывать/усиливать "
         "позицию соразмерно уверенности.",
         action_kind="continuous", recommended_total_timesteps=500_000),
    _env("FinRL-StockTrading-v0", "FinRL: портфель акций", "trading",
         "В духе StockTradingEnv из FinRL: портфель из 8 синтетических (коррелированных "
         "через общий рыночный фактор) акций + денежный остаток. Действие — сколько акций "
         "каждой купить/продать (± hmax штук), с комиссией за сделку. Награда = изменение "
         "полной стоимости портфеля (кэш + позиции по текущей цене). Настоящая "
         "мультиактивная бухгалтерия (кэш, лимиты, диверсификация), а не одна позиция.",
         action_kind="continuous", recommended_total_timesteps=800_000),
    EnvSpec(
        id="FinRL-PortfolioAllocation-v0",
        name="FinRL: аллокация портфеля",
        category="trading",
        description="В духе StockPortfolioEnv из FinRL: всегда полностью инвестированы в 8 "
                     "коррелированных активов, без кэша и шорта. Действие — вектор оценок, "
                     "нормализуется softmax'ом в веса портфеля. В наблюдении — реализованная "
                     "ковариационная матрица доходностей (Markowitz-стиль). Награда = "
                     "логарифмическая доходность портфеля минус издержки на ребалансировку.",
        action_kind="continuous",
        compatible_algorithms=_CONTINUOUS,
        default_hyperparams={
            # Action space is Box(0, 1) (raw pre-softmax scores) — one-sided
            # like CarRacing's gas/brake, so Beta (native [0,1] support, no
            # clip bias) is the better default distribution here too; see
            # CarRacing-v3's own `use_beta` comment above for the full
            # rationale.
            "use_beta": 1,
        },
        recommended_total_timesteps=800_000,
    ),
    # --- POMDP (нужна память — LSTM/GRU) ---
    _env("POCartPole-v0", "PO-CartPole", "pomdp",
         "CartPole без скоростей в наблюдении (только позиция тележки и угол шеста). "
         "Классический способ сделать MDP частично наблюдаемым — feed-forward политика "
         "упирается в потолок, рекуррентная (LSTM/GRU) может восстановить скорость по истории."),
    _env("POPendulum-v0", "PO-Pendulum", "pomdp",
         "Pendulum без угловой скорости (только cos/sin угла). Continuous-аналог PO-CartPole.",
         action_kind="continuous"),
    _env("POMountainCar-v0", "PO-Mountain Car", "pomdp",
         "Mountain Car без скорости тележки в наблюдении — видна только позиция."),
    _env("POAcrobot-v0", "PO-Acrobot", "pomdp",
         "Acrobot без угловых скоростей обоих суставов (видны только cos/sin углов). "
         "Сложнее PO-CartPole: нужно восстанавливать сразу две производные по истории."),
    _env("POLunarLander-v0", "PO-Lunar Lander", "pomdp",
         "Lunar Lander без всех трёх скоростей (vx, vy, угловая) — видны только позиция, "
         "угол и контакт ног с землёй. Самая тяжёлая из masked-сред: сажать модуль, "
         "восстанавливая скорость снижения и вращения только по соседним кадрам.",
         extra="box2d"),
    _env("FlickeringCartPole-v0", "Flickering CartPole", "pomdp",
         "На каждом шаге с вероятностью 50% всё наблюдение целиком заменяется нулями — "
         "тот самый приём 'flickering' из статьи Hausknecht & Stone (2015) про DRQN, "
         "только на дешёвом CartPole вместо Atari."),
    _env("FlickeringPong-v0", "Flickering Pong", "pomdp",
         "Оригинальный DRQN-бенчмарк (Hausknecht & Stone, 2015): кадры Pong с вероятностью "
         "50% гасятся в чёрный экран — нужно держать мяч 'в памяти', пока кадр невидим.",
         extra="atari"),
    _env("MemoryCorridor-v0", "Memory Corridor", "pomdp",
         "Коридор из 8 шагов: в самом начале показывается один бит-подсказка, потом "
         "наблюдение пустое. Награда — только если последнее действие совпало с давно "
         "исчезнувшей подсказкой. Без памяти результат не лучше случайного угадывания."),
    _env("MemoryCorridorLong-v0", "Memory Corridor (long)", "pomdp",
         "Тот же Memory Corridor, но подсказку нужно продержать в памяти 20 шагов вместо "
         "8 — показывает, что решает не любая память, а память с достаточной ёмкостью "
         "(memory_seq_len/hidden_size должны расти вместе с длиной задержки)."),
    _env("RepeatPrevious-v0", "Repeat Previous (N-back)", "pomdp",
         "Классическая N-back задача: каждый шаг показывается новый случайный символ из 4, "
         "а правильное действие — тот символ, что был показан ровно 3 шага назад. В "
         "отличие от Memory Corridor, здесь нужно непрерывно держать в памяти скользящее "
         "окно последних наблюдений и сверяться с ним на каждом шаге, а не один раз в конце. "
         "Без памяти — не лучше 1/4 угадывания."),
    EnvSpec(
        id="RockSample-v0",
        name="RockSample(7,8)",
        category="pomdp",
        description="Классический масштабируемый POMDP-бенчмарк из литературы по планированию "
                     "(Smith & Simmons, 2004): 7×7-сетка, 8 камней, позиции камней известны, а вот "
                     "их качество (хороший/плохой) скрыто и проверяется только шумным дальнобойным "
                     "сенсором — точность падает с расстоянием. Чтобы решать правильно, нужно "
                     "копить несколько шумных проверок одного камня в уверенное мнение, прежде чем "
                     "тратить ход на sample — совсем другой тип памяти, чем 'запомнить один сигнал'.",
        action_kind="discrete",
        compatible_algorithms=_DISCRETE,
    ),
    EnvSpec(
        id="VisualMemoryMaze-v0",
        name="Visual Memory Maze",
        category="pomdp",
        description="Аналог флагманского бенчмарка MiniGrid MemoryEnv (тот самый, на котором в "
                     "документации sb3-contrib показывают, зачем нужен RecurrentPPO): агент видит "
                     "только маленькое окно 5×5 клеток вокруг себя (картинка), стартует в комнате с "
                     "цветной подсказкой, идёт по одноклеточному коридору (подсказка давно выпала из "
                     "поля зрения) и должен выбрать ту из двух комнат в конце, чей предмет совпадает "
                     "по цвету с подсказкой. В отличие от остальных сред этой категории, частичная "
                     "наблюдаемость тут не искусственная маска, а настоящее ограниченное поле зрения.",
        action_kind="discrete",
        compatible_algorithms=_DISCRETE,
        default_hyperparams={
            # Image observations are much heavier per transition than this
            # app's other envs — a 30x30x3 float32 obs+next_obs pair is
            # ~22KB, so the default 50k-transition replay buffer would want
            # >1GB just for observations. Scaled down to something that
            # comfortably fits without needing the user to notice/tune it.
            "buffer_size": 8_000,
            "batch_size": 32,
            "memory_seq_len": 32,
        },
    ),
    EnvSpec(
        id="Tiger-v0",
        name="Tiger",
        category="pomdp",
        description="Самый канонический toy-POMDP в литературе (Kaelbling, Littman & Cassandra, 1998): "
                     "тигр за одной из двух дверей, можно слушать (шумная подсказка, точность 85%, но "
                     "стоит очков) или открыть дверь. Одного 'слушания' недоверчиво мало — правильная "
                     "стратегия слушает несколько раз и копит подсказки в уверенность, прежде чем "
                     "рискнуть открыть дверь. Крошечная среда (2 скрытых состояния), но именно на ней "
                     "проще всего увидеть, что вообще значит 'вести себя оптимально в POMDP'.",
        action_kind="discrete",
        compatible_algorithms=_DISCRETE,
    ),
    EnvSpec(
        id="HeavenHell-v0",
        name="Heaven/Hell",
        category="pomdp",
        description="T-образный лабиринт с активным сбором информации: указатель, говорящий, где Небеса "
                     "(награда) и где Ад (штраф), стоит не на главном пути, а в тупике сбоку — нужно "
                     "самому решить туда свернуть. Агент без памяти, который никогда не заходит за "
                     "подсказкой, обречён угадывать на развилке 50/50; агент с памятью один раз "
                     "заглядывает в комнату указателя и потом уверенно идёт к цели, сколько бы шагов "
                     "между этим ни прошло.",
        action_kind="discrete",
        compatible_algorithms=_DISCRETE,
    ),
    EnvSpec(
        id="Hallway-v0",
        name="Hallway",
        category="pomdp",
        description="Компактная версия классического бенчмарка Hallway (Littman, Cassandra & Kaelbling, "
                     "1995) на state aliasing: две разные комнаты лабиринта дают абсолютно одинаковые "
                     "показания локального сенсора, но для выхода к цели из них нужны противоположные "
                     "действия. Разрешить эту неоднозначность может только память о том, в какую сторону "
                     "агент повернул раньше на развилке — любая фиксированная реакция на одинаковое "
                     "наблюдение гарантированно ошибётся в одной из двух комнат.",
        action_kind="discrete",
        compatible_algorithms=_DISCRETE,
    ),
    EnvSpec(
        id="Battleship-v0",
        name="Battleship",
        category="pomdp",
        description="Морской бой как POMDP (используется как бенчмарк планирования, напр. в статье про "
                     "POMCP): на каждом шаге видно только результат самого последнего выстрела "
                     "(координаты + попал/не попал/утопил), а не всю карту клеток. Повторный выстрел в "
                     "уже проверенную клетку штрафуется — без памяти обо всех прошлых выстрелах агент "
                     "не может этого избежать и постоянно тратит ходы впустую.",
        action_kind="discrete",
        compatible_algorithms=_DISCRETE,
    ),
    EnvSpec(
        id="MinesweeperPOMDP-v0",
        name="Minesweeper (POMDP)",
        category="pomdp",
        description="Сапёр как задача на память (один из диагностических бенчмарков POPGym): наблюдение "
                     "— снова только результат последней открытой клетки (число соседних мин), а не "
                     "картинка всего поля. Чтобы логически выводить, где мины, нужно помнить числа, "
                     "открытые много ходов назад, а повторное открытие уже известной клетки штрафуется.",
        action_kind="discrete",
        compatible_algorithms=_DISCRETE,
    ),
    EnvSpec(
        id="Concentration-v0",
        name="Concentration (память)",
        category="pomdp",
        description="Классическая игра 'найди пару': перевернуть две карты за ход, совпали — они "
                     "остаются открытыми, не совпали — переворачиваются обратно. Значение карты видно "
                     "только в момент, когда её (или её пару) перевернули — во все остальные моменты она "
                     "просто 'закрыта, неизвестно что там', так что играть лучше случайного нужно помнить, "
                     "что было под какой картой много ходов назад.",
        action_kind="discrete",
        compatible_algorithms=_DISCRETE,
    ),
    EnvSpec(
        id="LaserTag-v0",
        name="LaserTag",
        category="pomdp",
        description="Погоня за убегающим противником (бенчмарк Pineau et al., 2003) — в отличие от всех "
                     "остальных сред этой категории, скрытый факт здесь не статичен, а движется сам: "
                     "позиция противника видна только в узкой прямой видимости, которую блокирует "
                     "препятствие в центре арены. Без памяти агент реагирует только на то, что видит "
                     "прямо сейчас и часто теряет цель; с памятью — удерживает 'последнее известное "
                     "положение и направление' и предугадывает, куда цель пойдёт дальше.",
        action_kind="discrete",
        compatible_algorithms=_DISCRETE,
    ),
    EnvSpec(
        id="ActiveTMaze-v0",
        name="Active T-Maze",
        category="pomdp",
        description="Активная версия T-maze (Ni et al., 2023): в отличие от Memory Corridor, подсказка "
                     "здесь не показывается сама — её нужно запросить действием 'посмотреть', стоя в "
                     "самом начале одностороннего коридора. Прошёл вперёд не посмотрев — подсказка "
                     "потеряна навсегда, назад пути нет. Проверяет не только удержание информации в "
                     "памяти, но и то, научился ли агент вообще понимать, что в этот момент стоит "
                     "спросить.",
        action_kind="discrete",
        compatible_algorithms=_DISCRETE,
    ),
    EnvSpec(
        id="ActiveTMazeLong-v0",
        name="Active T-Maze (long)",
        category="pomdp",
        description="Тот же Active T-Maze, но коридор длиннее (25 шагов вместо 10) — проверка, что "
                     "агент не просто угадал короткую задержку, а действительно удерживает подсказку "
                     "на всём пути до развилки.",
        action_kind="discrete",
        compatible_algorithms=_DISCRETE,
    ),
]

# Atari/ALE titles are all pixel-input, sparse-reward arcade games — the
# flat 50k-step Designer default barely covers a handful of lives, let
# alone enough gradient updates for the CNN to learn anything. 5M steps is
# a conservative "long enough to see a clear trend" budget (well below the
# 40-200M frames papers use for fully-tuned scores, but far past the point
# where a first run still looks like random play).
for _spec in GYM_ENVIRONMENTS:
    if _spec.category == "atari" and _spec.recommended_total_timesteps is None:
        _spec.recommended_total_timesteps = 5_000_000

# Atari/Box2D/MuJoCo envs render/simulate real physics or an emulator per
# step, so batching several parallel copies (see
# `rl_core/algorithms/vec_env.py`, `AsyncVectorEnv` with one subprocess
# worker per lane) meaningfully cuts wall-clock — more, less-correlated
# experience per network update still trains faster in practice than the
# same total step budget from one lane. Classic control/toy text/POMDP envs
# are cheap enough per-step that this rarely matters, so they keep the flat
# default of 1.
for _spec in GYM_ENVIRONMENTS:
    if _spec.category in ("atari", "box2d", "mujoco") and _spec.recommended_num_envs is None:
        _spec.recommended_num_envs = 8


BOARD_GAMES: list[EnvSpec] = [
    EnvSpec(
        id="tic_tac_toe",
        name="Tic-Tac-Toe",
        category="board_game",
        description="3×3, классика для проверки AlphaZero — обучается до совершенной игры за минуты.",
        action_kind="discrete",
        compatible_algorithms=["alphazero"],
    ),
    EnvSpec(
        id="connect_four",
        name="Connect Four",
        category="board_game",
        description="7×6, нужно собрать 4 в ряд. Требует больше self-play итераций.",
        action_kind="discrete",
        compatible_algorithms=["alphazero"],
        default_hyperparams={
            "num_simulations": 50,
            "games_per_iteration": 24,
            "eval_num_simulations": 50,
            "channels": 64,
            "num_blocks": 4,
        },
    ),
    EnvSpec(
        id="gomoku",
        name="Gomoku",
        category="board_game",
        description="9×9, 5 в ряд. Самая тяжёлая из встроенных игр — по умолчанию считается на "
                     "заметно большем бюджете симуляций/сети, чем Tic-Tac-Toe, иначе не успевает "
                     "выучиться за разумное число итераций.",
        action_kind="discrete",
        compatible_algorithms=["alphazero"],
        default_hyperparams={
            "num_simulations": 100,
            "games_per_iteration": 30,
            "eval_num_simulations": 80,
            "buffer_size": 40_000,
            "channels": 64,
            "num_blocks": 5,
        },
    ),
]


_EXTRA_IMPORTS: dict[str, tuple[str, ...]] = {
    "box2d": ("Box2D",),
    "mujoco": ("mujoco",),
    "atari": ("ale_py",),
    "minigrid": ("minigrid",),
    "highway_env": ("highway_env",),
    "nle": ("nle",),
    "minihack": ("minihack",),
    "gymnasium_robotics": ("gymnasium_robotics",),
}

_extra_ok_cache: dict[str, bool] = {}
_avail_cache: dict[str, bool] = {}


def _extra_installed(extra: str | None) -> bool:
    if extra is None:
        return True
    if extra in _extra_ok_cache:
        return _extra_ok_cache[extra]
    modules = _EXTRA_IMPORTS.get(extra)
    if not modules:
        _extra_ok_cache[extra] = False
        return False
    try:
        import importlib

        for name in modules:
            importlib.import_module(name)
        if extra == "atari":
            import ale_py
            import gymnasium as gym

            gym.register_envs(ale_py)
        _extra_ok_cache[extra] = True
    except Exception:
        _extra_ok_cache[extra] = False
    return _extra_ok_cache[extra]


def _gym_available(spec: EnvSpec) -> bool:
    cached = _avail_cache.get(spec.id)
    if cached is not None:
        return cached
    if not _extra_installed(spec.extra_requirement):
        _avail_cache[spec.id] = False
        return False
    # Extra suites: import succeeding is enough. Instantiating Atari/MuJoCo
    # on every /list call would stall the gallery (ROMs, GL, physics).
    if spec.extra_requirement:
        _avail_cache[spec.id] = True
        return True
    try:
        import gymnasium as gym

        env = gym.make(spec.id)
        env.close()
        _avail_cache[spec.id] = True
    except Exception:
        _avail_cache[spec.id] = False
    return _avail_cache[spec.id]


def list_environments() -> list[dict]:
    """Returns gym + board game specs, each tagged with live availability."""
    from rl_core.envs.previews import preview_api_path
    from rl_core import scene_store
    from rl_core.envs import pettingzoo_envs, tuple_marl_envs

    out: list[dict] = []
    for spec in GYM_ENVIRONMENTS:
        available = _gym_available(spec)
        out.append({
            **spec.__dict__,
            "kind": "gym",
            "available": available,
            "preview_url": preview_api_path(spec.id),
            "preview_thumb_url": preview_api_path(spec.id, thumb=True),
        })
    for meta in scene_store.list_meta():
        if meta.get("broken"):
            continue
        action_kind = meta.get("action_kind", "discrete")
        team_count = meta.get("team_count", 1)
        agent_count = meta.get("agent_count", 1)
        base_algos = _CONTINUOUS if action_kind == "continuous" else _DISCRETE
        # `ippo` (Independent PPO, rl_core/algorithms/native/marl_ppo.py)
        # only makes sense once a scene actually has 2+ teams — for a
        # single-team scene it would just be a slower, needlessly-split
        # version of the existing single-policy `ppo`. `qmix`
        # (rl_core/algorithms/native/qmix.py) is gated more loosely — 2+
        # *agents* total, not teams — because a single cooperative team is
        # QMIX/VDN's own classic setting (see its module docstring): the
        # mixer still buys real credit-assignment there, unlike `ippo`.
        # Both need discrete movement (value decomposition over a joint
        # *discrete* action space).
        compatible = list(base_algos)
        if team_count >= 2:
            compatible.append("ippo")
        if agent_count >= 2 and action_kind != "continuous":
            compatible.append("qmix")
        out.append({
            "id": meta["id"],
            "name": meta.get("name") or meta["slug"],
            "category": "scene",
            "description": meta.get("description") or f"Пользовательская 3D-сцена · {meta.get('agent_count', 1)} агент(ов)",
            "action_kind": action_kind,
            "compatible_algorithms": compatible,
            "extra_requirement": None,
            "default_hyperparams": None,
            "recommended_total_timesteps": 50_000,
            "recommended_num_envs": None,
            "kind": "gym",
            "available": True,
            "preview_url": None,
            "preview_thumb_url": None,
            "scene_agent_count": meta.get("agent_count"),
            "scene_team_count": team_count,
            "scene_slug": meta.get("slug"),
        })
    for meta in pettingzoo_envs.list_meta():
        team_count = meta.get("team_count", 1)
        agent_count = meta.get("agent_count", 1)
        action_kind = meta.get("action_kind", "discrete")
        # Same `ippo` (2+ teams) / `qmix` (2+ agents, any team split) gating
        # as scenes above — plus `dqn`/`ppo`/`a2c`/`es` from `_DISCRETE`,
        # since every one of these envs is discrete-action (see
        # `rl_core/envs/pettingzoo_envs.py`'s module docstring).
        compatible = list(_DISCRETE)
        if team_count >= 2:
            compatible.append("ippo")
        if agent_count >= 2:
            compatible.append("qmix")
        out.append({
            "id": meta["id"],
            "name": meta["name"],
            "category": "marl",
            "description": meta["description"],
            "action_kind": action_kind,
            "compatible_algorithms": compatible,
            "extra_requirement": "pettingzoo" if not meta["available"] else None,
            "default_hyperparams": None,
            "recommended_total_timesteps": meta.get("recommended_total_timesteps", 100_000),
            "recommended_num_envs": None,
            "kind": "gym",
            "available": meta["available"],
            "preview_url": None,
            "preview_thumb_url": None,
            "scene_agent_count": meta.get("agent_count"),
            "scene_team_count": team_count,
            "scene_slug": meta.get("slug"),
        })
    for meta in tuple_marl_envs.list_meta():
        agent_count = meta.get("agent_count", 1)
        # Both RWARE and LBForaging are single-team by design (see
        # `rl_core/envs/tuple_marl_envs.py`'s module docstring) — `ippo`
        # never applies (needs 2+ teams), `qmix` applies once there are
        # 2+ agents (its own classic single-team case).
        if meta.get("mask_aware"):
            # `smac_*` — situational (masked) action space, only `qmix`
            # actually reads `get_avail_actions()` (see that module's
            # docstring) — everything else in `_DISCRETE` would still run
            # (the adapter clamps illegal actions rather than crashing)
            # but would be misleadingly bad, not a fair comparison.
            # Unconditional (not gated on `agent_count`, unlike below) —
            # every registered `smac_*` scenario is multi-agent by
            # construction, and `agent_count` reads as 0 when `smaclite`
            # isn't installed (see `list_meta`), which must not hide
            # `qmix` from the "what would work if you installed this"
            # listing the Environments gallery shows for every extra.
            compatible = ["qmix"]
        else:
            compatible = list(_DISCRETE)
            if agent_count >= 2:
                compatible.append("qmix")
        out.append({
            "id": meta["id"],
            "name": meta["name"],
            "category": "marl",
            "description": meta["description"],
            "action_kind": meta.get("action_kind", "discrete"),
            "compatible_algorithms": compatible,
            "extra_requirement": meta.get("extra_requirement", "rware/lbforaging") if not meta["available"] else None,
            "default_hyperparams": None,
            "recommended_total_timesteps": meta.get("recommended_total_timesteps", 150_000),
            "recommended_num_envs": None,
            "kind": "gym",
            "available": meta["available"],
            "preview_url": None,
            "preview_thumb_url": None,
            "scene_agent_count": agent_count,
            "scene_team_count": meta.get("team_count", 1),
            "scene_slug": meta.get("slug"),
        })
    for spec in BOARD_GAMES:
        out.append({
            **spec.__dict__,
            "kind": "alphazero",
            "available": True,
            "preview_url": preview_api_path(spec.id),
            "preview_thumb_url": preview_api_path(spec.id, thumb=True),
        })
    return out


def get_env_spec(env_id: str) -> EnvSpec | None:
    for spec in GYM_ENVIRONMENTS + BOARD_GAMES:
        if spec.id == env_id:
            return spec
    return None
