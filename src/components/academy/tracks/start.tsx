import { BookOpen, GraduationCap, Play } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { Lesson } from '@/components/academy/lessonShared'
import { LessonLink } from '@/components/academy/lessonShared'
import { Callout, DesignerPipelineDiagram, PaperLink, Section, SupervisedVsRlDiagram } from '@/components/academy/visuals'

export const courseMap: Lesson = {
  id: 'fundamentals',
  title: 'Карта курса: от нуля до эксперимента',
  tagline: 'Как устроен учебный центр, в каком порядке идти и что вы сможете сделать после каждой части',
  icon: GraduationCap,
  group: 'Старт',
  badges: ['Карта курса', 'С чего начать', 'Практика в приложении'],
  content: (
    <>
      <Section title="Это не справочник алгоритмов — это путь">
        <p>
          Reinforcement learning легко выглядит как зоопарк аббревиатур: DQN, PPO, SAC, MCTS.
          Если начать с них, остаётся ощущение отрывков. Этот курс выстроен иначе: сначала — что
          вообще происходит между агентом и средой, потом — как учиться по таблице, затем — зачем
          нейросеть, и только потом конкретные алгоритмы приложения. Канонический учебник той же
          дуги —{' '}
          <PaperLink url="http://incompleteideas.net/book/RLbook2020.pdf">Sutton &amp; Barto, 2018</PaperLink>.
          Мы добавляем третью ось: всё, что можно сразу нажать в Reinforcement Studio.
        </p>
      </Section>

      <Section title="Три прохода по материалу">
        <ol className="ml-4 list-decimal space-y-2">
          <li>
            <b>Старт (этот блок).</b> Что такое RL и чем оно не является. Сразу после —{' '}
            <LessonLink id="first_experiment">первый эксперимент</LessonLink>: CartPole + PPO,
            чтобы теория не висела в воздухе.
          </li>
          <li>
            <b>Основы и табличные методы.</b> Награда, возврат, политика, ценность, Беллман,
            exploration, on/off-policy. Затем Monte Carlo, TD, SARSA и Q-learning — те кирпичи,
            из которых сделаны DQN и PPO. Без этого уроки про Rainbow и Gumbel-поиск читаются как
            магия.
          </li>
          <li>
            <b>Алгоритмы приложения и практика.</b> По одному уроку на каждый метод в Дизайнере.
            В конце — как читать графики, какие гиперпараметры крутить первыми и типичные поломки.
          </li>
        </ol>
      </Section>

      <Section title="Программа">
        <ul className="ml-4 list-disc space-y-1.5">
          <li>
            <b>Старт</b> — карта, «что такое RL», первый запуск.
          </li>
          <li>
            <b>Основы</b> — агент и среда, возврат и γ, политика, V/Q/advantage, разведка,
            on-policy vs off-policy.
          </li>
          <li>
            <b>Табличные методы</b> — динамическое программирование, Monte Carlo, TD/SARSA/Q-learning,
            зачем аппроксимация функцией.
          </li>
          <li>
            <b>Value-based</b> — <LessonLink id="dqn">DQN</LessonLink>,{' '}
            <LessonLink id="rainbow_dqn">Rainbow</LessonLink>.
          </li>
          <li>
            <b>Policy-gradient</b> — <LessonLink id="reinforce">REINFORCE</LessonLink>,{' '}
            <LessonLink id="a2c">A2C</LessonLink>, <LessonLink id="ppo">PPO</LessonLink>.
          </li>
          <li>
            <b>Continuous control</b> — SAC, DDPG, TD3, gSDE/Beta.
          </li>
          <li>
            <b>Планирование, world models, MARL</b> — когда model-free уже не тот инструмент:
            AlphaZero, Dreamer, EfficientZero/UniZero/наши Zero, IPPO/QMIX.
          </li>
          <li>
            <b>Продвинутое и практика</b> — память, POMDP, exploration, обёртки, графики,
            гиперпараметры, шпаргалка.
          </li>
        </ul>
      </Section>

      <Callout tone="good" title="Как проходить, если мало времени">
        В сайдбаре включите <b>трек новичка</b> — кнопки «Дальше» не утащат в SAC и Zero,
        пока вы не дойдёте до графиков и шпаргалки. Минимум тот же:{' '}
        <LessonLink id="what_is_rl">что такое RL</LessonLink> →{' '}
        <LessonLink id="first_experiment">первый эксперимент</LessonLink> →{' '}
        <LessonLink id="agent_env">агент и среда</LessonLink> →{' '}
        <LessonLink id="value_functions">ценность и Беллман</LessonLink> →{' '}
        <LessonLink id="td_learning">TD и Q-learning</LessonLink> →{' '}
        <LessonLink id="ppo">PPO</LessonLink> →{' '}
        <LessonLink id="reading_curves">как читать графики</LessonLink>.
        В конце уроков — упражнения с проверкой. Остальное (Rainbow, continuous, поиск) —
        справочник, его открывает выключатель «весь курс».
      </Callout>

      <Callout tone="tip" title="Схема «как учится» — не картинка из учебника">
        В уроках алгоритмов та же петля, что в Дизайнере и в Мониторинге. Если в курсе нарисован
        replay buffer, он есть в вашем запуске. Если написано «только continuous» — Дизайнер не
        даст выбрать этот алгоритм на CartPole.
      </Callout>
    </>
  ),
}

export const whatIsRl: Lesson = {
  id: 'what_is_rl',
  title: 'Что такое обучение с подкреплением',
  tagline: 'Агент учится действию из опыта, а не из правильных ответов — и за это платит шумом и отложенным сигналом',
  icon: BookOpen,
  group: 'Старт',
  badges: ['Определение', 'Supervised vs RL', 'Когда RL уместен'],
  content: (
    <>
      <Section title="Задача, которой нет в обычном ML">
        <p>
          В обучении с учителем вам дают пары (вход, правильный выход): картинка → «кошка»,
          позиция ферзя → «лучший ход» только если кто-то уже разметил миллион партий. В RL
          правильного ответа нет. Есть среда, в которой агент действует, и скаляр <b>награды</b>,
          часто приходящий не в тот же момент, когда было принято решение. Цель — научить
          <b> политику</b> π(a|s): какое действие выбирать в каком состоянии, чтобы сумма наград
          за эпизод была как можно больше.
        </p>
        <SupervisedVsRlDiagram />
      </Section>

      <Section title="Почему это сложно (три свойства, которых нет у классификации)">
        <ul className="ml-4 list-disc space-y-1.5">
          <li>
            <b>Нет учителя, есть критик.</b> Среда не говорит «надо было идти налево». Она даёт
            +1 за шаг жизни шеста или 0 почти весь FrozenLake и +1 только на финише.
          </li>
          <li>
            <b>Сигнал отложен.</b> Ход в шахматах может окупиться через тридцать плей. Агент должен
            приписать заслугу прошлым действиям (credit assignment) — это и есть ценность, Беллман,
            GAE в следующих уроках.
          </li>
          <li>
            <b>Данные зависят от политики.</b> Если агент никогда не заходит в часть карты, он не
            получит опыта оттуда. Разведка — не «регуляризация», а способ вообще увидеть задачу.
          </li>
        </ul>
      </Section>

      <Section title="Когда RL уместен — и когда нет">
        <p>
          RL имеет смысл, если вы можете смоделировать взаимодействие: симулятор, игра, робот,
          среда Gymnasium. Если у вас уже есть размеченный датасет «состояние → правильное действие»,
          imitation / supervised быстрее и стабильнее. Если награда — это «человек лайкнул через
          неделю», RL всё ещё применим, но credit assignment становится главным риском, а не
          выбором между PPO и SAC.
        </p>
        <Callout tone="info" title="В этом приложении">
          Среды — Gymnasium (CartPole, Lunar Lander, Atari, …), настольные игры для AlphaZero,
          сцены Scene Builder для нескольких агентов. Алгоритмы — от табличной идеи Q-learning,
          реализованной нейросетью (DQN), до поиска по выученной модели (EfficientZero). Курс
          учит сначала идею, потом кнопку в Дизайнере.
        </Callout>
      </Section>

      <Callout tone="tip" title="Дальше — руками">
        Следующий урок не теория:{' '}
        <LessonLink id="first_experiment">запустите CartPole с PPO</LessonLink>
        {' '}и посмотрите, как выглядит «агент научился», прежде чем разбирать формулы.
      </Callout>
    </>
  ),
}

export const firstExperiment: Lesson = {
  id: 'first_experiment',
  title: 'Первый эксперимент: CartPole + PPO',
  tagline: 'Соберите граф среда → алгоритм → training, запустите и узнайте, как выглядит «решил задачу»',
  icon: Play,
  group: 'Старт',
  badges: ['Практика', 'CartPole-v1', 'PPO', 'Дизайнер'],
  content: (
    <>
      <Section title="Что должно получиться">
        <p>
          CartPole-v1: тележка, на ней шест. Действия — толкнуть влево или вправо. Награда +1 за
          каждый шаг, пока шест не упал; эпизод обрезается на 500. Случайная политика живёт
          десятки шагов. Выученная — держит шест до потолка, <code>episode_reward_mean</code>{' '}
          около 500. PPO на этой среде сходится за минуты, не за ночь — поэтому она первый опыт,
          а не Lunar Lander и не Atari.
        </p>
        <DesignerPipelineDiagram />
      </Section>

      <Section title="Сборка в Дизайнере экспериментов">
        <ol className="ml-4 list-decimal space-y-1.5">
          <li>
            Откройте{' '}
            <Link to="/designer" className="font-medium text-primary underline decoration-primary/30 underline-offset-2">
              Дизайнер экспериментов
            </Link>
            . На холсте уже есть три узла: среда, алгоритм, training.
          </li>
          <li>
            Клик по узлу <b>среды</b> → выберите CartPole (каталог Gymnasium, classic control).
          </li>
          <li>
            Клик по узлу <b>алгоритма</b> → PPO. Для CartPole подходят ещё A2C и DQN; PPO — дефолт
            курса, его гиперпараметры прощают ошибки.
          </li>
          <li>
            Узел <b>Training</b>: хватит 50 000 шагов (дефолт) и <code>num_envs = 1</code>. Имя
            запуска — любое.
          </li>
          <li>
            Запустите. Приложение переключит вас в Мониторинг: график{' '}
            <code>episode_reward_mean</code> — главная кривая.
          </li>
        </ol>
      </Section>

      <Section title="Как понять, что получилось">
        <ul className="ml-4 list-disc space-y-1.5">
          <li>
            <b>Учится:</b> средняя награда эпизода ползёт вверх, длина эпизода тоже (агент дольше
            держит шест).
          </li>
          <li>
            <b>Решил:</b> плато около 500. Дальше крутить бессмысленно — среда просто обрезает эпизод.
          </li>
          <li>
            <b>Не смотрите только на loss.</b> У PPO value-loss может не падать к нулю, а награда
            расти. Loss — диагностика, не цель. Подробнее —{' '}
            <LessonLink id="reading_curves">урок про графики</LessonLink>.
          </li>
        </ul>
        <p>
          Ориентиры по числам, не по ощущению: случайная политика живёт примерно 20–50 шагов
          (награда того же порядка). 200 — уже не случай. 480–500 — потолок среды. Если после
          20–30k шагов среднее всё ещё ~20, это не «PPO медленный», а скорее не та среда или
          не тот запуск.
        </p>
      </Section>

      <Callout tone="warning" title="Если кривая плоская">
        Подождите хотя бы несколько тысяч шагов: первые эпизоды — почти случайные. Если после
        20–30k шагов награда всё ещё ~20, проверьте, что выбраны CartPole и PPO, а не тяжёлая
        среда. Не меняйте десять гиперпараметров сразу — сначала дочитайте основы.
      </Callout>

      <Callout tone="good" title="Что сделать вторым запуском">
        Тот же CartPole, но алгоритм <LessonLink id="dqn">DQN</LessonLink> или{' '}
        <LessonLink id="a2c">A2C</LessonLink>. Сравните кривые в Мониторинге («Сравнить»). DQN
        обычно тратит больше шагов: он off-policy и учится из буфера, но ε-greedy на старте
        почти случаен. Разница — не «кто лучше», а разные семейства, которые вы разберёте дальше.
      </Callout>

      <Callout tone="tip" title="Дальше — почему это сработало">
        PPO ещё не объяснён. Следующие уроки отвечают, что такое состояние, награда, политика и
        ценность — и почему «держать шест» можно выучить без единого размеченного кадра.{' '}
        <LessonLink id="agent_env">Агент, среда, эпизод</LessonLink>.
      </Callout>
    </>
  ),
}

export const START_LESSONS: Lesson[] = [courseMap, whatIsRl, firstExperiment]
