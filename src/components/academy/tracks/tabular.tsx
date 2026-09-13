import { Grid3x3, Layers, Table2, Timer } from 'lucide-react'
import type { Lesson } from '@/components/academy/lessonShared'
import { LessonLink } from '@/components/academy/lessonShared'
import { Callout, ChainMdpDiagram, Formula, MiniTable, PaperLink, Section, TdMcCompareDiagram, WorkedExample } from '@/components/academy/visuals'

export const dynamicProgramming: Lesson = {
  id: 'dynamic_programming',
  title: 'Динамическое программирование: когда среда известна',
  tagline: 'Policy evaluation, policy iteration, value iteration — идеальный случай, от которого отталкиваются все TD-методы',
  icon: Grid3x3,
  group: 'Табличные методы',
  badges: ['Policy iteration', 'Value iteration', 'Known P(s′|s,a)'],
  content: (
    <>
      <Section title="Оракул динамики">
        <p>
          Если известны P(s′|s,a) и R — можно считать Беллмана точно, без сэмплов из среды.
          Это <b>динамическое программирование</b> в смысле Sutton &amp; Barto (гл. 4), не
          «написать рекурсию с мемоизацией». В приложении такого оракула нет: Gym не отдаёт
          матрицу переходов. Но формулы — те же, что DQN оценивает по сэмплам.
        </p>
      </Section>

      <Section title="Оценка политики (policy evaluation)">
        <p>
          Фиксируем π и итерируем V, пока не сойдётся. Для каждого s подставляем Беллмана под
          этой политикой. Получаем V^π — «насколько хороша эта политика», ещё не улучшая её.
        </p>
        <Formula>
{`V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [ r + γ V(s') ]`}
        </Formula>
      </Section>

      <Section title="Улучшение политики и value iteration">
        <p>
          Зная V, жадная политика π'(s) = argmax_a Q(s,a) не хуже старой (policy improvement
          theorem). Чередование оценки и улучшения — <b>policy iteration</b>. Можно слить в один
          sweep: сразу писать max по действиям в backup — <b>value iteration</b>, предел которой
          есть V*.
        </p>
        <Formula caption="Value iteration: каждый sweep берёт лучшее действие, а не среднее по текущей π">
{`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [ r + γ V(s') ]`}
        </Formula>
        <WorkedExample title="Посчитаем: два синхронных прохода с V = 0, γ = 0,9">
          <ChainMdpDiagram />
          <p>
            Правые части берутся из V <i>до</i> прохода (синхронное обновление). Иначе, обновив B
            раньше A, вы получите V(A)=0,9 уже в первом sweep — это тоже законный in-place вариант,
            но тогда цифра другая.
          </p>
          <MiniTable
            headers={['проход', 'V(A)', 'V(B)', 'V(цель)']}
            rows={[
              ['0 (старт)', '0', '0', '0'],
              ['1', '0 + 0,9·0 = 0', '1 + 0,9·0 = 1', '0'],
              ['2', '0 + 0,9·1 = 0,9', '1', '0'],
            ]}
            caption="Сошлось за два прохода: цепочка короткая. DQN делает тот же backup, но по одному случайному переходу вместо суммы по P."
          />
        </WorkedExample>
      </Section>

      <Callout tone="info" title="Зачем это, если у нас нет P">
        TD и Q-learning — это value iteration, где ожидание по P заменено одним случайным
        переходом из env.step. DQN — то же плюс нейросеть вместо таблицы V[s]. Когда в уроке
        про DQN написано <code>target = r + γ max Q(s′,·)</code>, вы смотрите на сэмпловую
        версию этой строки.
      </Callout>

      <Callout tone="tip" title="Дальше">
        Динамика неизвестна. Можно всё же дождаться конца эпизода и взять настоящий возврат —{' '}
        <LessonLink id="monte_carlo">Monte Carlo</LessonLink>.
      </Callout>
    </>
  ),
}

export const monteCarlo: Lesson = {
  id: 'monte_carlo',
  title: 'Monte Carlo: учиться по полному возврату',
  tagline: 'Дождаться конца эпизода, взять G, подтянуть V(s) и Q(s,a). Несмещённо и шумно',
  icon: Timer,
  group: 'Табличные методы',
  badges: ['Monte Carlo', 'First-visit', 'Unbiased'],
  content: (
    <>
      <Section title="Идея">
        <p>
          Прогоняем эпизод по текущей политике. Для каждого посещённого (s, a) целью становится
          фактический дисконтированный возврат от этого момента до конца — не bootstrap через V(s′).
          Среднее таких G сходится к Q^π. First-visit считает только первое посещение состояния в
          эпизоде; every-visit — все.
        </p>
        <Formula caption="Обновление с шагом α; без таблицы можно хранить сумму и счётчик">
{`Gₜ = rₜ₊₁ + γ rₜ₊₂ + … + γ^{T−t−1} r_T
V(sₜ) ← V(sₜ) + α (Gₜ − V(sₜ))`}
        </Formula>
        <WorkedExample title="Посчитаем: один эпизод A → B → цель, α = 1, γ = 0,9">
          <p>
            Награды: 0 из A, 1 из B. Полные возвраты считаем с конца, как MC, без V(s′):
          </p>
          <MiniTable
            headers={['состояние', 'G', 'V после эпизода (было 0)']}
            rows={[
              ['B', '1', '1'],
              ['A', '0 + 0,9·1 = 0,9', '0,9'],
            ]}
            caption="На этой игрушечной цепи один успешный эпизод уже даёт V*. На FrozenLake почти все эпизоды дадут G = 0 — среднее сходится, но медленно."
          />
        </WorkedExample>
      </Section>

      <TdMcCompareDiagram />

      <Section title="Плюсы и минусы — они перейдут в глубокий RL">
        <ul className="ml-4 list-disc space-y-1">
          <li>
            <b>+</b> Несмещённая цель: не опираемся на ошибочную V(s′).
          </li>
          <li>
            <b>−</b> Нужен конец эпизода. Для очень длинных или непрерывных задач ждать нельзя.
          </li>
          <li>
            <b>−</b> Высокая дисперсия: два эпизода CartPole с одной и той же ранней ошибкой дадут
            очень разный G.
          </li>
        </ul>
        <p>
          REINFORCE (через два блока) — Monte Carlo policy gradient: вес градиента log π — это G
          (или G минус baseline). PPO заменяет полный G на GAE, смесь MC и TD.
        </p>
      </Section>

      <Callout tone="warning" title="Разреженная награда убивает чистый MC">
        На FrozenLake почти все эпизоды дают G=0, редкие — G=1. Среднее сходится, но нужно много
        успешных эпизодов. Поэтому на разреженных задачах люди не начинают с REINFORCE без baseline
        и не ждут чуда от маленького бюджета шагов.
      </Callout>

      <Callout tone="tip" title="Дальше">
        Компромисс: обновляться после одного шага, цель = r + γ V(s′).{' '}
        <LessonLink id="td_learning">TD, SARSA, Q-learning</LessonLink>.
      </Callout>
    </>
  ),
}

export const tdLearning: Lesson = {
  id: 'td_learning',
  title: 'TD-обучение, SARSA и Q-learning',
  tagline: 'Один переход → одна цель. SARSA учит свою политику, Q-learning — жадную. DQN есть глубокий Q-learning',
  icon: Table2,
  group: 'Табличные методы',
  badges: ['TD(0)', 'SARSA', 'Q-learning', 'TD-error'],
  content: (
    <>
      <Section title="Temporal difference">
        <p>
          TD(0) подтягивает V(s) к r + γ V(s′) сразу после перехода. Цель смещена (там текущая V),
          зато дисперсия ниже, чем у MC, и не нужен конец эпизода. δ = цель − прогноз называют
          TD-ошибкой: в A2C она же становится оценкой advantage.
        </p>
        <Formula>
{`δₜ = rₜ₊₁ + γ V(sₜ₊₁) − V(sₜ)
V(sₜ) ← V(sₜ) + α δₜ`}
        </Formula>
      </Section>

      <Section title="SARSA — on-policy control">
        <p>
          Для Q: цель использует <i>фактически выбранное</i> следующее действие a′ (то, что ε-greedy
          правда сделал). Учим Q той политики, которой играем.
        </p>
        <Formula>{`Q(s,a) ← Q(s,a) + α [ r + γ Q(s′, a′) − Q(s,a) ]`}</Formula>
      </Section>

      <Section title="Q-learning — off-policy control">
        <p>
          Цель берёт max по следующему действию, даже если агент пошёл не туда из-за ε. Учим
          оптимальную Q*, собирая данные любой (достаточно исследующей) политикой.{' '}
          <PaperLink url="https://link.springer.com/article/10.1007/BF00992698">Watkins, 1989</PaperLink>.
          Это прямой предок <LessonLink id="dqn">DQN</LessonLink>: таблицу Q[s,a] заменяют сетью,
          добаляют replay и target-сеть, но backup тот же.
        </p>
        <Formula caption="В приложении нет табличного Q-learning — зато эта строка живёт в dqn.py как loss">
{`Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s′, a′) − Q(s,a) ]`}
        </Formula>
        <WorkedExample title="Посчитаем: два перехода, α = 1, γ = 0,9, все Q = 0">
          <ChainMdpDiagram />
          <MiniTable
            headers={['переход', 'цель r + γ max Q(s′,·)', 'новый Q']}
            rows={[
              ['(B, →, r=1, цель)', '1 + 0,9·0 = 1', 'Q(B,→) = 1'],
              ['(A, →, r=0, B)', '0 + 0,9·1 = 0,9', 'Q(A,→) = 0,9'],
            ]}
            caption="Порядок важен: если сначала обновить A, пока Q(B)=0, получится Q(A,→)=0, и ценность «протечёт» только со второго визита A. Bootstrap использует текущую таблицу, не истину."
          />
        </WorkedExample>
      </Section>

      <Section title="Попробуйте увидеть это на FrozenLake">
        <p>
          FrozenLake-v1 в каталоге сред: сетка, скользкий лёд, награда почти вся в клетке цели.
          DQN здесь — тот же Q-learning, только аппроксимация. Он будет долго бродить: ε должен
          успеть найти цель, replay — запомнить редкий успех. Это нормальная табличная задача,
          «упрятанная» в нейросеть. Не начинайте с неё вместо CartPole.
        </p>
      </Section>

      <Callout tone="good" title="Мост к глубокому RL">
        Если Беллман, ε-greedy и off-policy уже понятны, урок про DQN перестаёт быть набором трюков
        и становится инженерным слоем поверх этой страницы: replay (корреляции), target-сеть
        (бегущая цель), сеть (обобщение между похожими s).
      </Callout>

      <Callout tone="tip" title="Дальше">
        Таблица не влезает, когда состояний как пикселей Atari.{' '}
        <LessonLink id="function_approx">Аппроксимация функцией</LessonLink>.
      </Callout>
    </>
  ),
}

export const functionApprox: Lesson = {
  id: 'function_approx',
  title: 'Аппроксимация функцией и нейросеть',
  tagline: 'Почему таблица Q[s,a] упирается в потолок и что именно делает сеть в DQN и PPO',
  icon: Layers,
  group: 'Табличные методы',
  badges: ['Function approximation', 'Generalization', 'Deadly triad'],
  content: (
    <>
      <Section title="Таблица конечна">
        <p>
          Tabular Q-learning хранит число на каждую пару (s, a). CartPole формально с непрерывным
          наблюдением — уже бесконечно много s. Atari — пиксели. Даже дискретная доска 15×15 для
          Gomoku не живёт в таблице. Нужна функция Q_θ(s,a) с параметрами θ, которая <i>обобщает</i>
          с похожих состояний на новые.
        </p>
      </Section>

      <Section title="Что именно аппроксимируют наши алгоритмы">
        <ul className="ml-4 list-disc space-y-1">
          <li>
            <b>DQN:</b> одну голову с |A| выходами — оценки Q. Скрытые слои можно нарисовать в
            Конструкторе архитектур (семья q_network).
          </li>
          <li>
            <b>PPO/A2C:</b> общий ствол, голова политики и голова V. Семья actor_critic.
          </li>
          <li>
            <b>Rainbow:</b> dueling (V и advantage) и опционально распределение возврата, не скаляр.
          </li>
        </ul>
      </Section>

      <Section title="Deadly triad — почему «просто сеть + Q-learning» взрывается">
        <p>
          Sutton &amp; Barto предупреждают: off-policy + bootstrap + function approximation вместе
          могут расходиться. DQN живёт в этой тройке. Инженерный ответ 2015 года — replay (почти
          i.i.d. батчи) и target-сеть (медленно движущаяся цель). Когда DQN «не учится», часто
          сломана одна из этих подпорок (слишком маленький буфер, target обновляется каждый шаг),
          а не «надо больше слоёв».
        </p>
        <PaperLink url="http://incompleteideas.net/book/RLbook2020.pdf">Sutton &amp; Barto, гл. 11</PaperLink>
      </Section>

      <Callout tone="info" title="Конструктор сетей в этом приложении">
        Для PPO/DQN/Rainbow/AlphaZero архитектуру можно собрать графом. Вход и выход головы
        подставляет среда (CartPole не даст 18 действий Atari). Не начинайте кастомизацию, пока
        дефолтная сеть не решает CartPole — иначе вы отлаживаете два места сразу.
      </Callout>

      <Callout tone="tip" title="Дальше — глубокий Q-learning">
        Все кирпичи на месте.{' '}
        <LessonLink id="dqn">DQN</LessonLink>
        {' '}— первый полный алгоритм приложения, который вы уже можете объяснить по строкам.
      </Callout>
    </>
  ),
}

export const TABULAR_LESSONS: Lesson[] = [
  dynamicProgramming, monteCarlo, tdLearning, functionApprox,
]
