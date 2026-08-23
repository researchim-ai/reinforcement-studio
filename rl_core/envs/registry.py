"""Registry of Gymnasium environments exposed in the Environments gallery.

Every entry describes the env well enough for the frontend to render a
card and for the Experiment Designer to know which algorithms are valid
(discrete vs continuous action spaces) without importing gymnasium there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ActionKind = Literal["discrete", "continuous"]

_DISCRETE = ["dqn", "ppo", "a2c"]
_CONTINUOUS = ["ppo", "a2c"]


@dataclass
class EnvSpec:
    id: str
    name: str
    category: str
    description: str
    action_kind: ActionKind
    compatible_algorithms: list[str] = field(default_factory=list)
    extra_requirement: str | None = None  # pip extra, e.g. "box2d" / "mujoco" / "atari"


def _env(
    env_id: str,
    name: str,
    category: str,
    description: str,
    action_kind: ActionKind = "discrete",
    extra: str | None = None,
) -> EnvSpec:
    return EnvSpec(
        id=env_id,
        name=name,
        category=category,
        description=description,
        action_kind=action_kind,
        compatible_algorithms=_CONTINUOUS if action_kind == "continuous" else _DISCRETE,
        extra_requirement=extra,
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
         "Карточный блэкджек против дилера. Маленькое дискретное состояние, классика табличного Q-learning."),
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
         action_kind="continuous", extra="box2d"),
    _env("CarRacing-v3", "Car Racing", "box2d",
         "Пиксельная трасса сверху: руль и газ. Непрерывное управление + картинка.",
         action_kind="continuous", extra="box2d"),
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
         action_kind="continuous", extra="mujoco"),
    _env("HumanoidStandup-v5", "Humanoid Standup", "mujoco",
         "Гуманоид стартует лёжа — нужно встать. Ещё сложнее обычного Humanoid.",
         action_kind="continuous", extra="mujoco"),
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
]


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
    ),
    EnvSpec(
        id="gomoku",
        name="Gomoku",
        category="board_game",
        description="9×9, 5 в ряд. Самая тяжёлая из встроенных игр.",
        action_kind="discrete",
        compatible_algorithms=["alphazero"],
    ),
]


_EXTRA_IMPORTS: dict[str, tuple[str, ...]] = {
    "box2d": ("Box2D",),
    "mujoco": ("mujoco",),
    "atari": ("ale_py",),
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
