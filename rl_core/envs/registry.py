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

ActionKind = Literal["discrete", "continuous"]

_DISCRETE = ["dqn", "rainbow_dqn", "ppo", "a2c"]
_CONTINUOUS = ["ppo", "a2c", "sac"]


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
