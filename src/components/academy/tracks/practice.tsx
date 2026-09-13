import { Activity, AlertTriangle, Wrench } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { Lesson } from '@/components/academy/lessonShared'
import { LessonLink } from '@/components/academy/lessonShared'
import { Callout, Section } from '@/components/academy/visuals'

export const readingCurves: Lesson = {
  id: 'reading_curves',
  title: 'Как читать графики обучения',
  tagline: 'Награда — цель, loss — диагностика. Шум нормален. Сравнивать нужно со случайной политикой и с потолком среды',
  icon: Activity,
  group: 'Практика',
  badges: ['Monitor', 'episode_reward_mean', 'Loss', 'Eval'],
  content: (
    <>
      <Section title="Главная кривая">
        <p>
          В Мониторинге для Gym-запусков жирная линия — <code>episode_reward_mean</code>, средняя
          сумма наград за недавние эпизоды. Это единственная кривая, которая отвечает на вопрос
          «агент стал лучше играть?». Длина эпизода часто движется вместе с ней (CartPole: дольше
          живёт → выше возврат).
        </p>
      </Section>

      <Section title="Что смотреть ещё — и чего не ждать">
        <ul className="ml-4 list-disc space-y-1.5">
          <li>
            <b>Policy / value / entropy loss (PPO, A2C).</b> Value-loss не обязан упасть к нулю:
            V учится движущейся цели. Entropy обычно падает, когда политика увереннее — хорошо на
            плато, плохо если это случилось на шаге 2k при награде случайной.
          </li>
          <li>
            <b>Q-loss (DQN).</b> Может осциллировать: target-сеть периодически прыгает. Смотрите
            награду, не «красивый убывающий MSE».
          </li>
          <li>
            <b>Шум.</b> Каждый эпизод — случай. Сглаживание в графике не отменяет разброс. Не
            останавливайте запуск из-за одного провала вниз.
          </li>
        </ul>
      </Section>

      <Section title="Масштаб: откуда знать, что 200 — это много">
        <p>
          CartPole-v1: потолок 500. Случайная политика ~20–50. 200 уже учится, 480–500 решил.
          Lunar Lander: решить ≈ 200. Pendulum: награды отрицательные, «лучше» = ближе к нулю.
          FrozenLake: среднее 0.01 после миллиона шагов может быть прогрессом (успех редкий).
          Всегда сверяйтесь с описанием среды в каталоге и со случайным бейзлайном (короткий
          запуск с крошечным бюджетом / невыученной сетью).
        </p>
      </Section>

      <Callout tone="info" title="Сравнение запусков">
        В Мониторинге кнопка «Сравнить» накладывает кривые. Меняйте один фактор: алгоритм, lr,
        память. Иначе не поймёте, что помогло. Для честного сравнения одинаковый{' '}
        <code>total_timesteps</code> и желательно несколько seed — в приложении для этого есть Sweep.
      </Callout>

      <Callout tone="tip" title="Дальше">
        Если награда живая, но хочется быстрее / стабильнее —{' '}
        <LessonLink id="hyperparams">какие ручки крутить первыми</LessonLink>.
        Если мёртвая — <LessonLink id="failure_modes">типичные поломки</LessonLink>.
      </Callout>
    </>
  ),
}

export const hyperparams: Lesson = {
  id: 'hyperparams',
  title: 'Гиперпараметры: что крутить первым',
  tagline: 'Не сетка из двадцати значений. Сначала бюджет шагов и lr, потом разведка, потом остальное',
  icon: Wrench,
  group: 'Практика',
  badges: ['learning_rate', 'γ', 'n_steps', 'ε', 'num_envs'],
  content: (
    <>
      <Section title="Порядок, который не взрывает эксперимент">
        <ol className="ml-4 list-decimal space-y-1.5">
          <li>
            <b>Бюджет шагов.</b> CartPole решается на десятках тысяч, Lunar Lander — на сотнях
            тысяч, Atari и SMAC — на миллионах. Если кривая едва оторвалась от случайной — сначала
            увеличьте <code>total_timesteps</code>, не lr.
          </li>
          <li>
            <b>Learning rate.</b> Дефолты приложения разумны. Если loss NaN / награда обрушилась
            после красивого старта — уменьшите lr в 3–10 раз. Если вообще ничего не движется на
            простой среде — можно увеличить в 2–3 раза. Не на два порядка.
          </li>
          <li>
            <b>Разведка.</b> DQN: <code>exploration_fraction</code> / финальное ε. PPO:{' '}
            <code>ent_coef</code> (слишком большой — вечный случайный агент, слишком маленький —
            коллапс).
          </li>
          <li>
            <b>Горизонт on-policy.</b> У PPO <code>n_steps</code> × <code>num_envs</code> — сколько
            свежего опыта за апдейт. Маленький n_steps на длинном credit assignment (Lunar Lander)
            часто хуже, чем чуть больший rollout.
          </li>
          <li>
            <b>γ.</b> 0.99 — дефолт. Ниже — агент близорукий (иногда помогает, если эпизоды короткие
            и награда плотная). Выше 0.999 — имеет смысл в очень длинных эпизодах, но value учится
            тяжелее.
          </li>
        </ol>
      </Section>

      <Section title="Параллельные среды">
        <p>
          <code>num_envs &gt; 1</code> ускоряет сбор опыта (векторизация). Для PPO это ещё и более
          разнообразный батч. Для отладки на CartPole оставьте 1: проще смотреть один эпизод. PETS
          параллелить нельзя — это в уроке про него.
        </p>
      </Section>

      <Callout tone="warning" title="Одна ручка за раз">
        Сетка «lr × clip × GAE × сеть × обёртки» на одном seed ничего не доказывает. Sweep в
        приложении как раз для того, чтобы менять 1–2 параметра и несколько seed. Сначала глазами
        один запуск, потом сетка.
      </Callout>

      <Callout tone="tip" title="Дальше">
        Когда даже дефолты «должны работать», а не работают —{' '}
        <LessonLink id="failure_modes">разбор поломок</LessonLink>.
      </Callout>
    </>
  ),
}

export const failureModes: Lesson = {
  id: 'failure_modes',
  title: 'Почему не учится: типичные поломки',
  tagline: 'Чеклист, прежде чем винить алгоритм. Большинство «PPO сломан» — среда, бюджет, разведка или взгляд не на ту кривую',
  icon: AlertTriangle,
  group: 'Практика',
  badges: ['Debug', 'NaN', 'Collapse', 'Sparse reward'],
  content: (
    <>
      <Section title="1. Вы смотрите не туда">
        <p>
          Loss растёт, награда растёт — для PPO это бывает. Награда на train с ε=0.2 не равна eval.
          AlphaZero: смотрите win rate, не «как DQN-награду». World models: сначала dynamics_loss,
          потом уже return воображения.
        </p>
      </Section>

      <Section title="2. Задача не того масштаба">
        <p>
          Дефолтный бюджет 50k шагов решает CartPole и не обязан решать Atari, NetHack или RWARE.
          В уроках QMIX прямо написано: нулевая награда миллионы шагов на RWARE — ожидаемо.
          Сверьтесь со шпаргалкой среды и с уроком алгоритма.
        </p>
      </Section>

      <Section title="3. Награда слишком редкая или «не та»">
        <p>
          Случайная политика никогда не видит ненулевой r → нечему учиться. Тогда:{' '}
          <LessonLink id="exploration">RND / NoisyNet</LessonLink>, проще версия среды, другой
          алгоритм, не «ещё слой в сети». Shaping награды — последнее средство: легко сломать
          постановку.
        </p>
      </Section>

      <Section title="4. Разведка умерла или не включалась">
        <p>
          PPO: энтропия ≈ 0, награда на уровне случайной. Поднимите <code>ent_coef</code> или
          уменьшите lr. DQN: ε уже 0.05, а буфер полный мусора с начала — увеличьте fraction
          или буфер не «с первого шага жадный».
        </p>
      </Section>

      <Section title="5. Численный взрыв">
        <p>
          NaN в loss, награда внезапно −10⁹: lr слишком большой, нет grad clip, continuous действие
          вылетело за границы без правильного squashing (Beta/gSDE —{' '}
          <LessonLink id="continuous_techniques">отдельный урок</LessonLink>). Откатитесь к дефолтам
          алгоритма в Дизайнере.
        </p>
      </Section>

      <Section title="6. Несовместимость, которую UI мог спрятать">
        <p>
          DQN на continuous, QMIX на одной команде vs IPPO, PETS с num_envs, память LSTM на
          алгоритме без поддержки. Каталог сред серый список алгоритмов не просто так. Если
          выбрали кастомную сеть не той семьи — превью в Конструкторе сетей ругнётся, запуск тоже.
        </p>
      </Section>

      <Callout tone="good" title="Ритуал отладки на 15 минут">
        (1) CartPole + PPO на дефолтах — если это не учится, сломан запуск, не теория.
        (2) Та же среда, второй алгоритм.
        (3) Целевая среда, дефолты, бюджет из урока/каталога.
        (4) Одна ручка.
        Не прыгайте в EfficientZero, пока (1) не зелёный.
      </Callout>

      <Callout tone="tip" title="Итог курса">
        Собрать выбор алгоритма в одну таблицу:{' '}
        <LessonLink id="cheatsheet">шпаргалка</LessonLink>.
        Новый запуск — в{' '}
        <Link to="/designer" className="font-medium text-primary underline decoration-primary/30 underline-offset-2">
          Дизайнере
        </Link>.
      </Callout>
    </>
  ),
}

export const PRACTICE_LESSONS: Lesson[] = [readingCurves, hyperparams, failureModes]
