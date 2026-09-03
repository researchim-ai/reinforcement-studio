import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'
import {
  Aperture, Brain, Combine, Compass, Crosshair, Database, Dices, Dna, Eye, GitBranch, GitFork,
  GraduationCap, History, Layers, ListChecks, Moon, Network, Orbit, RefreshCw, Scale, Shuffle,
  SlidersHorizontal, Sparkles, Split, Swords, Target, TrendingUp, Users, Waves, Zap,
} from 'lucide-react'
import { AlgorithmDiagram } from '@/components/AlgorithmDiagram'
import {
  AgentEnvLoopDiagram, Callout, DiscountChart, EpsilonDecayChart, Formula, FlowRow, GumbelHalvingDiagram,
  ImaginationDiagram, LatentChainDiagram, MCTSTreeDiagram, MemoryTimelineDiagram, PaperLink, PomdpCompareDiagram,
  PPOClipChart, Section, TokenAttentionDiagram, TwoHotBinsChart,
} from './visuals'

export interface Lesson {
  id: string
  title: string
  tagline: string
  icon: LucideIcon
  group: string
  badges: string[]
  content: ReactNode
}

export const LESSON_GROUPS = [
  'Основы',
  'Value-based (дискретные действия)',
  'Policy-gradient (актор-критик)',
  'Continuous control',
  'Планирование и self-play',
  'World Models',
  'Multi-agent',
  'Продвинутое',
  'Итог',
] as const

// ---------------------------------------------------------------------------
// 1. Основы RL
// ---------------------------------------------------------------------------

const fundamentals: Lesson = {
  id: 'fundamentals',
  title: 'Основы RL: как агент учится',
  tagline: 'MDP, ценность, скидка на будущее, разведка vs эксплуатация — на этом стоит всё остальное в курсе',
  icon: GraduationCap,
  group: 'Основы',
  badges: ['MDP', 'Value function', 'Discount factor', 'Exploration'],
  content: (
    <>
      <Section title="Цикл агент ↔ среда">
        <p>
          Всё в RL сводится к одному циклу: агент видит состояние sₜ, выбирает действие aₜ по своей
          <b> политике</b> π(a|s), среда отвечает наградой rₜ₊₁ и новым состоянием sₜ₊₁. Повторить.
          Всё, что отличает алгоритмы друг от друга — <i>что именно</i> они запоминают из этого потока
          и <i>как</i> используют это, чтобы политика становилась лучше.
        </p>
        <AgentEnvLoopDiagram />
      </Section>

      <Section title="Марковский процесс принятия решений (MDP)">
        <p>
          Формально среда — это MDP: набор состояний S, действий A, функция переходов P(s'|s,a) и
          функция награды R(s,a). «Марковость» значит одно важное допущение: sₜ содержит{' '}
          <b>всё</b>, что нужно для принятия решения — прошлое не добавляет информации сверх текущего
          состояния. Когда это не так (агент видит меньше, чем полное состояние), получается{' '}
          <b>POMDP</b> — этому посвящён отдельный урок дальше в курсе. Каноническое изложение всей
          этой математики — учебник{' '}
          <PaperLink url="http://incompleteideas.net/book/RLbook2020.pdf">Sutton &amp; Barto, «Reinforcement Learning: An Introduction», 2018</PaperLink>{' '}
          — почти всё в этом курсе в каком-то виде восходит к нему.
        </p>
      </Section>

      <Section title="Ценность: почему одной награды недостаточно">
        <p>
          Награда за один шаг — слишком шумный сигнал: действие может быть плохим сейчас, но отличным
          в перспективе (жертва фигурой в шахматах). Поэтому вводят <b>функцию ценности</b> — ожидаемую
          сумму будущих наград:
        </p>
        <Formula caption="V(s) — «насколько хорошо быть в состоянии s»; Q(s,a) — «насколько хорошо выбрать действие a в состоянии s»">
{`V(s) = E[ rₜ₊₁ + γrₜ₊₂ + γ²rₜ₊₃ + ... | sₜ = s ]
Q(s,a) = E[ rₜ₊₁ + γrₜ₊₂ + ... | sₜ = s, aₜ = a ]`}
        </Formula>
        <p>Уравнение Беллмана переписывает это рекурсивно — ценность сейчас = награда + ценность потом:</p>
        <Formula>{`Q(s,a) = E[ r + γ · max_a' Q(s',a') ]`}</Formula>
        <p>
          Это уравнение — фундамент DQN и всех его потомков (следующие два урока). Policy-gradient
          методы (A2C, PPO, SAC) идут другим путём — учат V(s) как «подсказку», а саму политику π
          двигают напрямую по градиенту награды.
        </p>
      </Section>

      <Section title="Discount factor γ: насколько агент «терпелив»">
        <p>
          γ ∈ [0,1) определяет, сколько значит награда через t шагов: её вес — γᵗ. γ близкое к 1 —
          агент почти не дисконтирует далёкое будущее («стратег»); γ близкое к 0 — считает только
          немедленную награду («импульсивный»).
        </p>
        <DiscountChart />
      </Section>

      <Section title="Разведка vs эксплуатация (exploration vs exploitation)">
        <p>
          Если агент всегда выбирает действие, которое считает лучшим прямо сейчас, он никогда не
          узнает, что где-то есть действие лучше — оценка Q/V ещё не точна на старте обучения. Каждый
          алгоритм в этом курсе решает эту дилемму по-своему:
        </p>
        <ul className="ml-4 list-disc space-y-1">
          <li><b>DQN / Rainbow DQN</b> — ε-greedy: с вероятностью ε действие случайное, иначе жадное по Q.</li>
          <li><b>PPO / A2C</b> — политика сама по себе стохастическая (распределение над действиями) + entropy-бонус, который штрафует за слишком «уверенную» (детерминированную) политику.</li>
          <li><b>SAC</b> — разведка встроена в саму цель обучения: максимизируется награда <i>и</i> энтропия политики одновременно.</li>
          <li><b>AlphaZero</b> — MCTS с UCB-подобным бонусом за редко посещённые ходы (см. урок про AlphaZero).</li>
        </ul>
      </Section>

      <Section title="On-policy vs off-policy — почему это важно для дизайна эксперимента">
        <p>
          <b>Off-policy</b> (DQN, Rainbow DQN, SAC) может учиться на старых данных из replay-буфера —
          собранных давно устаревшей политикой — поэтому обычно сэмпл-эффективнее. <b>On-policy</b> (PPO,
          A2C) должен использовать данные, собранные (примерно) текущей политикой — буфер каждый раз
          собирается заново и выбрасывается после обновления, зато обучение стабильнее и проще
          настраивать.
        </p>
      </Section>

      <Callout tone="tip" title="Как проходить курс">
        Дальше — по одному уроку на алгоритм, от простого DQN до AlphaZero, а в конце — отдельные уроки
        про память (LSTM/GRU) и про POMDP-среды, где эта память реально нужна. Каждый урок содержит ту
        же схему «как учится», что вы видите в Дизайнере экспериментов и в Мониторинге — она не
        абстрактная иллюстрация, а буквально то, что происходит в вашем запуске.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 2. DQN
// ---------------------------------------------------------------------------

const dqn: Lesson = {
  id: 'dqn',
  title: 'DQN — Deep Q-Network',
  tagline: 'Учим одну функцию Q(s,a) для всех действий и действуем жадно по ней',
  icon: Database,
  group: 'Value-based (дискретные действия)',
  badges: ['Value-based', 'Off-policy', 'Только дискретные действия'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="dqn"
        kind="gym"
        hyperparams={{ exploration_fraction: 0.2, buffer_size: 50_000, batch_size: 64, gamma: 0.99, target_update_interval: 1_000 }}
        show={{ network: false }}
      />

      <Section title="Идея">
        <p>
          Нейросеть Q_θ(s,a) принимает состояние и выдаёт по одному числу на каждое возможное
          действие — «насколько хорошо» его выбрать. Играть просто: выбрать действие с максимальным
          Q. Вся сложность — в том, как эту сеть обучить, потому что «правильного» Q никто не знает
          заранее, его приходится строить из собственного опыта агента (уравнение Беллмана из
          прошлого урока). Оригинальный рецепт —{' '}
          <PaperLink url="https://arxiv.org/abs/1312.5602">Mnih et al., 2013</PaperLink> (Atari по
          сырым пикселям), закреплённый в{' '}
          <PaperLink url="https://www.nature.com/articles/nature14236">Mnih et al., 2015</PaperLink>{' '}
          (Nature).
        </p>
      </Section>

      <Section title="Replay buffer + target-сеть">
        <p>
          Наивное обучение прямо на свежем опыте нестабильно: соседние шаги сильно скоррелированы, а
          цель обучения (Q_target) меняется на каждом шаге вместе с самой сетью — она «бежит за своим
          хвостом». DQN решает обе проблемы:
        </p>
        <FlowRow
          items={[
            { icon: Dices, title: 'ε-greedy шаг', detail: 'взаимодействие со средой' },
            { icon: Database, title: 'Replay buffer', detail: 'храним переходы (s,a,r,s\')' },
            { icon: Shuffle, title: 'Случайный батч', detail: 'разрывает корреляции по времени' },
            { icon: Brain, title: 'Обновление Q_θ', detail: 'градиент по MSE(Q, target)' },
            { icon: RefreshCw, title: 'Target-сеть', detail: 'копия Q_θ, обновляется редко' },
          ]}
        />
        <Formula caption="Q_target — отдельная, «замороженная» на target_update_interval шагов копия сети — без неё цель обучения гуляла бы вместе с самой сетью">
{`target = r + γ · max_a' Q_target(s', a')
loss   = MSE( Q_θ(s,a), target )`}
        </Formula>
      </Section>

      <Section title="Разведка: ε-greedy с затухающим ε">
        <p>
          В начале обучения Q ещё ничего не значит — есть смысл действовать случайно (ε=1). К концу
          exploration_fraction доли обучения ε линейно спадает до exploration_final_eps, и агент
          почти всегда действует жадно.
        </p>
        <EpsilonDecayChart />
      </Section>

      <Callout tone="warning" title="Слабое место: переоценка Q (overestimation bias)">
        Операция max в target'е систематически завышает оценку Q — из шума max выбирает случайно
        завышенные значения чаще, чем заниженные. Rainbow DQN (следующий урок) начинается именно с
        фикса этой проблемы — Double Q-learning.
      </Callout>

      <Callout tone="good" title="Когда использовать">
        Дискретные действия, среда дешёвая для симуляции, важна сэмпл-эффективность (off-policy,
        буфер можно переиспользовать). Для более сложных дискретных задач почти всегда лучше сразу
        взять Rainbow DQN — тот же DQN, но без известных слабых мест.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 3. Rainbow DQN
// ---------------------------------------------------------------------------

const rainbow: Lesson = {
  id: 'rainbow_dqn',
  title: 'Rainbow DQN',
  tagline: 'DQN + Double Q-learning + Dueling-сеть + Prioritized Replay + n-step возврат + опционально QR-DQN/NoisyNet/RND',
  icon: Zap,
  group: 'Value-based (дискретные действия)',
  badges: ['Value-based', 'Off-policy', 'Только дискретные действия'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="rainbow_dqn"
        kind="gym"
        hyperparams={{ exploration_fraction: 0.2, buffer_size: 50_000, n_step: 3, batch_size: 64, gamma: 0.99, target_update_interval: 1_000 }}
        show={{ network: false }}
      />

      <p className="text-[13px] text-muted-foreground">
        Rainbow (<PaperLink url="https://arxiv.org/abs/1710.02298">Hessel et al., 2018</PaperLink>)
        объединяет шесть отдельных улучшений DQN в одну сеть; из шести здесь — четыре ядровых
        всегда включены плюс два опциональных (QR-DQN, ниже, и NoisyNet/RND — переключатели
        разведки из отдельного урока), что в сумме даёт полный набор из статьи.
      </p>

      <Section title="1. Double Q-learning — лечим переоценку">
        <p>
          Вместо того чтобы брать max по той же сети, что и обучается, действие для target'а выбирает
          «онлайн»-сеть, а <i>оценивает</i> его — target-сеть. Если онлайн-сеть переоценила какое-то
          действие, это не обязательно совпадёт с переоценкой у отдельной target-сети — смещение
          компенсируется (<PaperLink url="https://arxiv.org/abs/1509.06461">van Hasselt et al., 2016</PaperLink>).
        </p>
        <Formula>{`target = r + γ · Q_target( s', argmax_a' Q_θ(s', a') )`}</Formula>
      </Section>

      <Section title="2. Dueling-архитектура — разделяем «где я» и «что делать»">
        <p>
          Сеть считает Q(s,a) в два потока
          (<PaperLink url="https://arxiv.org/abs/1511.06581">Wang et al., 2016</PaperLink>): V(s)
          («насколько в принципе хорошо это состояние», неважно что делать) и A(s,a) («насколько
          действие a лучше/хуже среднего в этом состоянии») — и складывает их обратно.
        </p>
        <FlowRow
          items={[
            { icon: Brain, title: 'Признаки', detail: 'общий ствол сети' },
            { icon: Split, title: 'V(s) / A(s,a)', detail: 'два независимых потока' },
            { icon: Combine, title: 'Q(s,a)', detail: 'V(s) + A(s,a) − mean(A)' },
          ]}
        />
        <Formula caption="Вычитание среднего A — просто для идентифицируемости (иначе V и A можно сдвинуть на константу в разные стороны без изменения Q)">
{`Q(s,a) = V(s) + ( A(s,a) − mean_a A(s,a) )`}
        </Formula>
      </Section>

      <Section title="3. Prioritized Experience Replay (PER) — сэмплируем не равномерно">
        <p>
          Переходы, на которых сеть сильно ошибается (большая |TD-error| δ), несут больше информации
          для обучения — их стоит сэмплировать чаще, а не наравне со всеми
          (<PaperLink url="https://arxiv.org/abs/1511.05952">Schaul et al., 2016</PaperLink>). Чтобы
          это смещение не испортило сходимость, добавляется importance-sampling коррекция веса
          градиента.
        </p>
        <Formula caption="α управляет силой приоритизации (α=0 — обычный uniform replay), β растёт до 1 по ходу обучения и убирает смещение от неравномерного сэмплирования">
{`priority(i) = ( |δᵢ| + ε )^α
P(i)        = priority(i) / Σⱼ priority(j)
IS-вес(i)   = ( 1 / (N · P(i)) )^β`}
        </Formula>
      </Section>

      <Section title="4. n-step возврат — смотрим на n шагов вперёд перед bootstrap'ом">
        <p>
          Обычный DQN использует награду за 1 шаг + оценку Q дальше. n-step возврат суммирует n
          настоящих наград перед тем, как «одолжить» оценку у сети — меньше зависимости от текущей
          (пока неточной) Q, но выше вариативность.
        </p>
        <Formula>{`G_t^(n) = r_t + γr_{t+1} + ... + γ^(n-1)r_{t+n-1} + γⁿ · max_a Q_target(s_{t+n}, a)`}</Formula>
      </Section>

      <Section title="5. Distributional RL (QR-DQN) — опционально: учим распределение, а не число">
        <p>
          Включается флагом distributional — вместо одного числа Q(s,a) сеть предсказывает
          num_quantiles квантилей <i>распределения</i> будущей отдачи для каждого действия
          (<PaperLink url="https://arxiv.org/abs/1710.10044">Dabney et al., 2017</PaperLink> —
          квантильная регрессия, а не исходный fixed-support C51 из статьи Rainbow: не нужен шаг
          проекции распределения на фиксированную решётку, при том же самом эффекте). Каждый
          предсказанный квантиль сравнивается с <i>каждым</i> целевым сэмплом (у них нет
          покомпонентного соответствия — это просто N несортированных сэмплов одного и того же
          распределения) через квантильный Huber-лосс, взвешенный тем, насколько τᵢ близко к «цель
          оказалась выше/ниже этого квантиля».
        </p>
        <Formula caption="τᵢ = (i + 0.5) / num_quantiles — фиксированный набор целевых квантильных уровней; TD-ошибка для PER берётся как разница средних (по всем квантилям) целевого и текущего распределений">
{`δᵢⱼ = target_quantile_j − current_quantile_i
huber(δ) = 0.5·δ²  при |δ|≤κ,  иначе κ·(|δ|−0.5κ)
loss = Σᵢ Eⱼ[ |τᵢ − 1[δᵢⱼ<0]| · huber(δᵢⱼ) ]`}
        </Formula>
        <p className="text-[12px]">
          Комбинируется с Double Q-learning/Dueling/PER/n-step ровно так же, как обычный DQN, и с
          NoisyNet/RND (отдельный урок «Exploration») — включить всё сразу и есть from-scratch
          эквивалент полного 6-компонентного рецепта Rainbow. Единственное сочетание, которого нет:
          distributional + память (LSTM/GRU) — рекуррентная Q-сеть в этом приложении не имеет
          квантильного варианта.
        </p>
      </Section>

      <Callout tone="info" title="Что не совмещается с памятью">
        При включённой памяти (LSTM/GRU) PER, n-step <i>и</i> distributional отключаются — совместить
        приоритизацию/n-step возврат/квантили с окнами эпизодов для BPTT корректно заметно сложнее
        (нужна пер-последовательностная приоритизация в духе R2D2), см. урок «Память». Рекуррентный
        режим сохраняет только Double Q-learning + Dueling.
      </Callout>

      <Callout tone="good" title="Когда использовать">
        В любой ситуации, где подошёл бы DQN — Rainbow DQN почти никогда не хуже и часто заметно
        лучше, особенно на более сложных дискретных средах (Atari и подобные). Начинать стоит с
        четырёх «ядровых» компонентов; distributional добавляет точность оценки риска (не только
        среднего исхода, но и формы распределения) за дополнительную вычислительную стоимость.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 4. A2C
// ---------------------------------------------------------------------------

const a2c: Lesson = {
  id: 'a2c',
  title: 'A2C — Advantage Actor-Critic',
  tagline: 'Простейший actor-critic: короткий rollout, одно обновление, дальше',
  icon: Scale,
  group: 'Policy-gradient (актор-критик)',
  badges: ['Policy-gradient', 'On-policy', 'Дискретные и continuous'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="a2c"
        kind="gym"
        hyperparams={{ n_steps: 5, gamma: 0.99, ent_coef: 0.01 }}
        show={{ network: false }}
      />

      <Section title="Идея: двигаем политику напрямую">
        <p>
          В отличие от DQN, здесь нет функции Q, из которой действие достаётся через max — сеть
          <b> сама выдаёт распределение действий</b> π(a|s) (actor) и, отдельной головой, оценку V(s)
          (critic). Обучение — policy gradient: увеличиваем вероятность действий, которые оказались
          <i> лучше ожидаемого</i>, уменьшаем — которые хуже. A2C — синхронный вариант{' '}
          <PaperLink url="https://arxiv.org/abs/1602.01783">A3C (Mnih et al., 2016)</PaperLink>{' '}
          — один rollout, одно синхронное обновление, без асинхронных воркеров оригинальной статьи.
        </p>
        <FlowRow
          items={[
            { icon: Brain, title: 'Общий ствол', detail: 'извлечение признаков' },
            { icon: Split, title: 'Actor π(a|s)', detail: 'куда действовать' },
            { icon: TrendingUp, title: 'Critic V(s)', detail: 'насколько хорошо здесь' },
          ]}
        />
      </Section>

      <Section title="Advantage — почему не просто награда">
        <p>
          Сырая награда — шумный сигнал одобрения/наказания действия. <b>Advantage</b> A(s,a) =
          «насколько это действие лучше среднего для этого состояния» — если оно положительное,
          действие было лучше, чем critic ожидал, увеличиваем его вероятность.
        </p>
        <Formula caption="δ — TD-error за один шаг; A2C использует его напрямую как оценку advantage (bootstrap на n_steps шаге)">
{`δ_t = r_t + γ · V(s_{t+1}) − V(s_t)
∇J(θ) ≈ E[ ∇log π_θ(a_t|s_t) · δ_t ]`}
        </Formula>
      </Section>

      <Section title="Entropy-бонус — не дать политике «зацементироваться»">
        <p>
          Без штрафа политика быстро становится почти детерминированной (одно действие с вероятностью
          ≈1) — и разведка останавливается. ent_coef добавляет к цели обучения бонус за энтропию
          распределения π(·|s), поощряя оставаться немного «неуверенным».
        </p>
      </Section>

      <Callout tone="good" title="Когда использовать">
        Быстрый, дёшевый бейзлайн для сравнения — очень короткий rollout (n_steps по умолчанию 5) и
        одно обновление на каждый rollout делают шаг обучения почти мгновенным. Для серьёзной задачи
        обычно выгоднее PPO — тот же actor-critic, но с более аккуратным (и переиспользуемым) шагом
        обновления.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 5. PPO
// ---------------------------------------------------------------------------

const ppo: Lesson = {
  id: 'ppo',
  title: 'PPO — Proximal Policy Optimization',
  tagline: 'Actor-critic, который не даёт себе сделать слишком большой шаг обновления',
  icon: Target,
  group: 'Policy-gradient (актор-критик)',
  badges: ['Policy-gradient', 'On-policy', 'Дискретные и continuous', 'Дефолт по умолчанию'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="ppo"
        kind="gym"
        hyperparams={{ n_steps: 2048, gamma: 0.99, batch_size: 64, ent_coef: 0.0 }}
        show={{ network: false }}
      />

      <Section title="Проблема, которую PPO решает">
        <p>
          Обычный policy gradient (как в A2C) — очень чувствителен к размеру шага обучения: слишком
          большой шаг может резко «сломать» уже неплохую политику, и откатиться назад с
          on-policy-данными невозможно — новые данные собираются уже сломанной политикой. PPO
          (<PaperLink url="https://arxiv.org/abs/1707.06347">Schulman et al., 2017</PaperLink>)
          позволяет несколько эпох обучения на одном и том же rollout'е, но <b>обрезает</b> (clip)
          вклад тех действий, вероятность которых изменилась слишком сильно относительно того, что
          было на момент сбора данных.
        </p>
      </Section>

      <Section title="Clipped surrogate objective">
        <p>
          r(θ) — во сколько раз новая политика вероятнее выбрать то же действие, чем старая на момент
          сбора данных. Если действие было хорошим (A&gt;0), рост r поощряется — но только до
          1+ε, дальше эффект «срезается»: дальше увеличивать r не выгодно.
        </p>
        <Formula>{`r(θ) = π_θ(a|s) / π_θ_old(a|s)
L_CLIP(θ) = E[ min( r(θ)·A, clip(r(θ), 1−ε, 1+ε)·A ) ]`}</Formula>
        <PPOClipChart />
        <p className="text-[12px]">
          Синяя линия (A&gt;0, хорошее действие) растёт вместе с r и плоско срезается после 1+ε —
          дальше «награждать» рост вероятности не имеет смысла. Красная (A&lt;0, плохое действие) —
          зеркально: не даёт лишнего бонуса за снижение вероятности ниже 1−ε, чтобы не «переубедить»
          политику слишком резко в противоположную сторону.
        </p>
      </Section>

      <Section title="GAE — компромисс между смещением и дисперсией">
        <p>
          Вместо одного δ (как в A2C), PPO усредняет TD-ошибки на нескольких горизонтах с
          экспоненциальным затуханием (γλ)ᵏ —{' '}
          <PaperLink url="https://arxiv.org/abs/1506.02438">Generalized Advantage Estimation, Schulman et al., 2016</PaperLink>.
          Параметр gae_lambda крутит этот компромисс: λ=0 — как в A2C (низкая дисперсия, больше
          смещения от неточной V), λ=1 — почти Monte-Carlo (без смещения, но шумно).
        </p>
        <Formula>{`δ_t     = r_t + γV(s_{t+1}) − V(s_t)
A_t^GAE = Σ_{k=0}^{∞} (γλ)^k · δ_{t+k}`}</Formula>
      </Section>

      <Section title="Несколько эпох на одном rollout'е — почему это безопасно">
        <p>
          Именно clip делает переиспользование данных (n_epochs проходов по одному и тому же
          rollout'у батчами batch_size) безопасным — без него повторные обновления на тех же данных
          быстро уводили бы политику слишком далеко от того, что данные на самом деле описывают.
        </p>
      </Section>

      <Callout tone="good" title="Когда использовать">
        Хороший выбор по умолчанию почти для любой Gym-задачи — дискретной или continuous. Более
        стабилен и сэмпл-эффективнее A2C при сопоставимой простоте настройки; в этом приложении это
        рекомендуемая точка отсчёта, если не уверены, с чего начать.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 6. SAC
// ---------------------------------------------------------------------------

const sac: Lesson = {
  id: 'sac',
  title: 'SAC — Soft Actor-Critic',
  tagline: 'Off-policy actor-critic для непрерывных действий, максимизирует награду и энтропию сразу',
  icon: Waves,
  group: 'Continuous control',
  badges: ['Actor-Critic', 'Off-policy', 'Только continuous'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="sac"
        kind="gym"
        hyperparams={{ buffer_size: 100_000, batch_size: 256, gamma: 0.99, tau: 0.005, ent_coef: 0.2 }}
        show={{ network: false }}
      />

      <Section title="Maximum entropy RL — разведка встроена в саму цель">
        <p>
          SAC (<PaperLink url="https://arxiv.org/abs/1801.01290">Haarnoja et al., 2018</PaperLink>)
          не максимизирует просто сумму наград — он максимизирует награду <i>плюс</i> энтропию
          политики, взвешенную температурой α. Это не «трюк для разведки» сверху, а часть самой
          целевой функции — политика по построению остаётся стохастичной, даже когда уже нашла
          хорошее решение (что вдобавок делает её устойчивее к небольшим ошибкам/шуму).
        </p>
        <Formula>{`J(π) = E[ Σₜ γᵗ ( r_t + α · H(π(·|s_t)) ) ]`}</Formula>
      </Section>

      <Section title="Twin critics — та же идея, что Double Q, для continuous">
        <p>
          Действий бесконечно много (continuous), поэтому явный max по действиям невозможен — вместо
          этого обучаются <b>два</b> независимых Q-критика, и в target'е берётся минимум из них.
          Как и Double Q-learning, это подавляет систематическую переоценку.
        </p>
        <FlowRow
          items={[
            { icon: Split, title: 'Q1(s,a), Q2(s,a)', detail: 'два независимых критика' },
            { icon: GitBranch, title: 'min(Q1,Q2)', detail: 'консервативная оценка' },
            { icon: Brain, title: 'Обновление actor', detail: 'максимизирует Q − α·log π' },
          ]}
        />
        <Formula caption="soft target — тот же max-Q target, что и в DQN, но со штрафом за низкую энтропию действия">
{`y = r + γ · ( min(Q1_target, Q2_target)(s',a') − α · log π(a'|s') )`}
        </Formula>
      </Section>

      <Section title="Мягкое (Polyak) обновление target-сетей">
        <p>
          Вместо резкой периодической синхронизации (как target_update_interval в DQN), target-сети
          в SAC плавно «ползут» за онлайн-сетями на каждом шаге — множитель τ обычно очень мал (0.005).
        </p>
        <Formula>{`θ_target ← τ · θ + (1 − τ) · θ_target`}</Formula>
      </Section>

      <Section title="Автонастройка температуры α">
        <p>
          ent_coef можно оставить фиксированным числом, но обычно продуктивнее позволить ему
          подстраиваться автоматически под целевую энтропию (второй SAC-статьи —{' '}
          <PaperLink url="https://arxiv.org/abs/1812.05905">Haarnoja et al., 2018 (v2)</PaperLink>) —
          на старте, когда политика ничего не знает, разведки нужно больше, чем под конец обучения.
        </p>
      </Section>

      <Callout tone="good" title="Когда использовать">
        Continuous-задачи (Pendulum, MuJoCo, BipedalWalker, ...) — прямая альтернатива PPO/A2C для
        непрерывных действий, но off-policy: обычно сэмпл-эффективнее, зато чуть требовательнее к
        точной настройке (размер буфера, batch_size).
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 6b. DDPG
// ---------------------------------------------------------------------------

const ddpg: Lesson = {
  id: 'ddpg',
  title: 'DDPG — Deep Deterministic Policy Gradient',
  tagline: 'Простейший off-policy actor-critic для continuous control: actor выдаёт одно действие, а не распределение',
  icon: Compass,
  group: 'Continuous control',
  badges: ['Actor-Critic (детерминир.)', 'Off-policy', 'Только continuous'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="ddpg"
        kind="gym"
        hyperparams={{ buffer_size: 100_000, batch_size: 256, gamma: 0.99, tau: 0.005, exploration_noise: 0.1 }}
        show={{ network: false }}
      />

      <Section title="Детерминированная политика — принципиальное отличие от PPO/SAC">
        <p>
          Всюду выше actor выдавал <b>распределение</b> — Categorical/Gaussian/Beta, из которого
          сэмплируется действие. DDPG проще: actor μ_θ(s) — обычная сеть, которая на входе получает
          состояние и на выходе выдаёт <i>одно конкретное</i> действие, без всякого распределения.
          Играть с такой политикой тривиально — просто взять её выход; сложность в том, как обучать
          сеть без распределения, которое можно было бы «подтолкнуть» градиентом по log-вероятности.
        </p>
      </Section>

      <Section title="Deterministic Policy Gradient: критик указывает направление">
        <p>
          Раз своей стохастичности нет, обучение идёт через критика: если Q(s,a) продифференцировать
          по действию a, получится направление, в котором a нужно менять, чтобы Q выросло. DDPG
          (<PaperLink url="https://arxiv.org/abs/1509.02971">Lillicrap et al., 2015</PaperLink>)
          толкает actor именно туда — максимизирует Q(s, μ_θ(s)) напрямую градиентом по θ через
          критика.
        </p>
        <Formula caption="Критик должен быть достаточно точным, чтобы этот градиент был осмысленным — поэтому critic и actor обновляются вместе на каждом шаге, а не actor «с нуля» против случайного критика">
{`actor_loss = − E[ Q(s, μ_θ(s)) ]
∇_θ actor_loss ∝ − ∇_a Q(s,a)|_{a=μ_θ(s)} · ∇_θ μ_θ(s)`}
        </Formula>
      </Section>

      <Section title="Разведка: шум добавляется, а не сэмплируется">
        <p>
          У детерминированной политики нет встроенной стохастичности — разведка добавляется вручную:
          к выходу actor'а прибавляется гауссов шум (масштаб — доля exploration_noise от диапазона
          действий, чтобы одно и то же значение параметра значило похожее для среды с [-1,1] и среды
          с [-1,0,0]..[1,1,1]), затем результат обрезается в границы действия.
        </p>
        <FlowRow
          items={[
            { icon: Brain, title: 'μ_θ(s)', detail: 'детерминированное действие' },
            { icon: Dices, title: '+ N(0, σ²)', detail: 'гауссов шум разведки' },
            { icon: Split, title: 'clip', detail: 'обрезка в границы action space' },
          ]}
        />
      </Section>

      <Section title="Replay buffer, target-сети, Polyak — как у SAC, но один критик">
        <p>
          Buffer/target-сети/мягкое обновление — те же механизмы, что и в SAC (см. прошлый урок), но
          критик здесь <b>один</b>, а не два — а значит, ничто не подавляет систематическое
          переоценивание Q, ровно та же проблема, с которой начинался урок про DQN.
        </p>
        <Formula>{`target = r + γ·(1−done)·Q_target(s', μ_target(s'))
θ_target ← τ·θ + (1−τ)·θ_target`}</Formula>
      </Section>

      <Callout tone="warning" title="Известная слабость: нестабильность">
        Один критик легко переоценивает Q на никогда не виденных парах (s,a), а actor затем
        сознательно едет именно туда, где критик (ошибочно) обещает много — самоусиливающаяся
        петля. TD3 (следующий урок) — тот же алгоритм с тремя конкретными фиксами именно этой
        проблемы.
      </Callout>

      <Callout tone="good" title="Когда использовать">
        В основном как учебный/сравнительный бейзлайн — самый простой continuous off-policy
        actor-critic в этом курсе. Для реальной задачи почти всегда лучше сразу взять TD3.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 6c. TD3
// ---------------------------------------------------------------------------

const td3: Lesson = {
  id: 'td3',
  title: 'TD3 — Twin Delayed DDPG',
  tagline: 'DDPG + три точечных фикса нестабильности: twin critics, delayed updates, target policy smoothing',
  icon: GitFork,
  group: 'Continuous control',
  badges: ['Actor-Critic (детерминир.)', 'Off-policy', 'Только continuous', 'Надёжный дефолт'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="td3"
        kind="gym"
        hyperparams={{
          buffer_size: 100_000, batch_size: 256, gamma: 0.99, tau: 0.005,
          policy_noise: 0.2, noise_clip: 0.5, policy_delay: 2,
        }}
        show={{ network: false }}
      />

      <Section title="Три конкретных фикса поверх DDPG">
        <p>
          TD3 (<PaperLink url="https://arxiv.org/abs/1802.09477">Fujimoto et al., 2018</PaperLink>) —
          не новая идея, а инженерный разбор того, что именно у DDPG ломается, и три точечных
          решения:
        </p>
        <FlowRow
          items={[
            { icon: Split, title: 'Twin critics', detail: 'Q1, Q2 — берём min' },
            { icon: RefreshCw, title: 'Delayed updates', detail: 'actor реже, чем critic' },
            { icon: Shuffle, title: 'Target smoothing', detail: 'шум в target-действие' },
          ]}
        />
      </Section>

      <Section title="1. Twin critics — та же идея, что Double Q, для детерминированной политики">
        <p>
          Два независимых критика Q1, Q2; в target'е берётся минимум — тот же принцип, что и у
          Double DQN/SAC, применённый здесь к детерминированной политике.
        </p>
        <Formula>{`target = r + γ·(1−done)·min( Q1_target, Q2_target )(s', ã')`}</Formula>
      </Section>

      <Section title="2. Target policy smoothing — не дать критику опереться на острый пик Q">
        <p>
          Перед тем как передать action от target actor'а в target-критики, к нему добавляется
          маленький обрезанный гауссов шум. Без этого критик мог бы научиться завышать Q ровно в
          одной узкой точке действия — актор нашёл бы эту точку и «эксплуатировал» бы артефакт
          критика, а не настоящую динамику среды.
        </p>
        <Formula caption="policy_noise — масштаб шума, noise_clip — граница обрезки самого шума (не итогового действия)">
{`ã' = clip( μ_target(s') + clip(ε, −noise_clip, noise_clip), low, high ),  ε ~ N(0, policy_noise²)`}
        </Formula>
      </Section>

      <Section title="3. Delayed policy updates — дать критику сначала устояться">
        <p>
          Actor (и оба target-сети) обновляются не на каждом шаге, а раз в policy_delay обновлений
          критика. Градиент через критика в actor-loss полезен только когда критик сам достаточно
          точен — обновлять actor чаще смысла не имеет, только вносит лишний шум.
        </p>
      </Section>

      <Callout tone="good" title="Когда использовать">
        Обычно надёжнее DDPG почти без дополнительной сложности настройки — стандартный выбор для
        continuous control с (почти) детерминированной динамикой (MuJoCo-локомоция, CarRacing). SAC
        остаётся альтернативой, когда стохастическая, энтропийная разведка ценнее, чем
        детерминированность.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 6d. Continuous control: gSDE и Beta-политика
// ---------------------------------------------------------------------------

const continuousTechniques: Lesson = {
  id: 'continuous_techniques',
  title: 'Continuous control: gSDE и Beta-политика',
  tagline: 'Два способа сделать непрерывное действие PPO/A2C менее «дёрганым» и никогда не выходящим за границы',
  icon: SlidersHorizontal,
  group: 'Continuous control',
  badges: ['PPO · A2C', 'gSDE', 'Beta-распределение'],
  content: (
    <>
      <Section title="Проблема обычной Gaussian-головы">
        <p>
          Стандартная continuous-голова PPO/A2C — диагональный Gaussian с независимым, обучаемым
          log_std. У неё два практических недостатка: (1) шум сэмплируется независимо на каждом
          шаге — исполняемая траектория «дёргается» кадр к кадру даже когда среднее действие уже
          хорошее; (2) распределение <i>неограничено</i>, поэтому сэмплированное действие всегда
          обрезается (clip) в границы action space — а сама обрезка — недифференцируемая операция,
          которую градиент политики не видит: сеть может продолжать толкать «сырое» среднее ещё
          дальше за границу (газ = 1.4), хотя это исполняется абсолютно так же, как газ = 1.0.
        </p>
      </Section>

      <Section title="gSDE — коррелированный во времени шум">
        <p>
          gSDE («generalized State-Dependent Exploration»,{' '}
          <PaperLink url="https://arxiv.org/abs/2005.05719">Raffin et al., 2021</PaperLink>) не
          сэмплирует шум заново каждый шаг, а строит его как линейную функцию признаков состояния через матрицу
          exploration_mat, которая перевыбирается редко — раз в sde_sample_freq шагов, а не на
          каждом. Шум остаётся тем же направлением на протяжении нескольких шагов подряд — рулевое
          колесо «плывёт» плавно, а не дёргается.
        </p>
        <Formula caption="exploration_mat перевыбирается раз в sde_sample_freq шагов; log_prob/entropy считаются через маргинальное (по exploration_mat) гауссово распределение, зависящее только от обучаемого log_std">
{`noise = features(s) @ exploration_mat   (не пересэмплируется каждый шаг)
action = mean(s) + noise`}
        </Formula>
      </Section>

      <Section title="Beta-политика — границы действия соблюдены по построению">
        <p>
          Beta-распределение (<PaperLink url="https://proceedings.mlr.press/v70/chou17a/chou17a.pdf">Chou et al., 2017</PaperLink>)
          имеет носитель ровно на [0,1] — аффинно растягивается на [low, high] конкретного action
          space и <b>никогда</b> не требует обрезки: любой сэмпл уже валидное действие. На
          CarRacing, где газ/тормоз — [0,1]-действие с жёсткой односторонней границей, это устраняет
          именно тот источник смещённого градиента, о котором шла речь выше
          (<PaperLink url="https://arxiv.org/abs/2111.02202">Petrazzini &amp; Antonelo, 2021</PaperLink>{' '}
          сообщают +63% успешных эпизодов относительно Gaussian на этой же задаче).
        </p>
        <Formula caption="α, β > 1 (softplus+1 в реализации) — не даёт плотности вырождаться в «ванну» с массой у самых краёв диапазона">
{`x ~ Beta(α, β),  x ∈ [0,1]
action = low + (high − low) · x`}
        </Formula>
      </Section>

      <FlowRow
        items={[
          { icon: Waves, title: 'Gaussian (дефолт)', detail: 'неограниченный носитель, нужен clip' },
          { icon: SlidersHorizontal, title: 'gSDE', detail: 'тот же Gaussian, но коррелированный шум' },
          { icon: Target, title: 'Beta', detail: 'носитель точно на границах, clip не нужен' },
        ]}
      />

      <Callout tone="info" title="Оба переключателя не для любой среды">
        gSDE/use_beta осмысленны только для continuous (Box) action space — на дискретных
        действиях включённый флаг просто ничего не делает. Если включены оба одновременно, Beta
        побеждает: gSDE — специфически гауссова схема разведки, несовместимая с Beta-головой.
      </Callout>

      <Callout tone="good" title="Когда что включать">
        Beta — для действий с жёсткими (особенно односторонними) границами, где резкая обрезка
        Gaussian реально мешает (CarRacing-подобные среды). gSDE — для плавного «рулевого» управления
        (робототехническая локомоция), где именно временная корреляция шума, а не сами границы,
        определяет качество разведки.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 7. AlphaZero
// ---------------------------------------------------------------------------

const alphazero: Lesson = {
  id: 'alphazero',
  title: 'AlphaZero',
  tagline: 'Планирование через MCTS + одна сеть на policy и value + обучение на собственных играх',
  icon: Swords,
  group: 'Планирование и self-play',
  badges: ['Planning', 'Self-play', 'Только настольные игры'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="alphazero-demo"
        kind="alphazero"
        hyperparams={{ games_per_iteration: 20, num_simulations: 25, eval_games: 10, win_rate_threshold: 0.55 }}
        show={{ network: false }}
      />

      <Section title="Одна сеть — два выхода">
        <p>
          Сеть fθ(s) = (p, v) смотрит на позицию и сразу выдаёт (1) p — распределение вероятностей по
          ходам («куда обычно играют в такой позиции») и (2) v — оценку позиции («кто, вероятно,
          выиграет»). Это не замена поиску — это то, что делает поиск (MCTS) практичным: без сети
          пришлось бы обходить дерево вариантов до самого конца партии на каждом ходу. Рецепт из{' '}
          <PaperLink url="https://arxiv.org/abs/1712.01815">Silver et al., 2017 — AlphaZero</PaperLink>{' '}
          (обобщение AlphaGo Zero на шахматы/сёги — без всякого knowledge-specific тюнинга под
          конкретную игру, в отличие от AlphaGo).
        </p>
      </Section>

      <Section title="MCTS, управляемый сетью (PUCT)">
        <p>
          На каждом реальном ходу выполняется num_simulations симуляций поиска по дереву. Каждая
          выбирает путь вглубь по формуле PUCT (введена в{' '}
          <PaperLink url="https://www.nature.com/articles/nature16961">Silver et al., 2016 — AlphaGo</PaperLink>) —
          балансу «что уже выглядит хорошо» (Q) и «что стоит попробовать» (бонус разведки, тем
          больше, чем реже ход посещался):
        </p>
        <MCTSTreeDiagram />
        <Formula caption="P(s,a) — вероятность из сети, N — счётчик посещений, c — константа разведки">
{`UCB(s,a) = Q(s,a) + c · P(s,a) · √N(s) / (1 + N(s,a))`}
        </Formula>
        <p>
          После num_simulations симуляций итоговое решение — не argmax по сети, а распределение по
          тому, <i>сколько раз</i> каждый ход был посещён поиском — то есть поиск сам «уточняет»
          политику сети перед тем, как сделать ход.
        </p>
      </Section>

      <Section title="Self-play — откуда берутся тренировочные данные">
        <p>
          Данных «правильных ходов от эксперта» нет — агент играет сам с собой, используя описанный
          выше MCTS-поиск на каждом ходу. Уточнённое поиском распределение ходов (π_MCTS) становится
          целью для policy-головы, а итог партии (кто выиграл) — целью для value-головы.
        </p>
        <Formula>{`loss(θ) = ( z − v )² − π_MCTS · log(p) + λ‖θ‖²`}</Formula>
      </Section>

      <Section title="Арена и gating — защита от регресса">
        <p>
          Новая сеть после обучения не заменяет «чемпиона» автоматически — они играют eval_games
          партий друг против друга, и замена происходит только если новая сеть выигрывает не хуже
          win_rate_threshold. Это единственный алгоритм в курсе с явной защитой от «а что если новая
          версия оказалась хуже старой».
        </p>
      </Section>

      <Callout tone="warning" title="Ограничение: нужен быстрый, детерминированный симулятор">
        AlphaZero предполагает, что можно «промотать» партию вперёд из любой позиции сколько угодно
        раз за симуляцию поиска — работает для настольных игр с полной информацией (Tic-Tac-Toe,
        Connect Four, Gomoku), но не подходит для сред с шумной/дорогой/невоспроизводимой динамикой.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 7b. World Models — общая идея
// ---------------------------------------------------------------------------

const worldModelsOverview: Lesson = {
  id: 'world_models_overview',
  title: 'World Models: общая идея',
  tagline: 'Учим модель динамики среды, чтобы «репетировать» внутри неё вместо того, чтобы каждый раз идти в настоящую среду',
  icon: Orbit,
  group: 'World Models',
  badges: ['Model-based RL', 'Dreamer', 'MBPO/PETS', 'World Models (Ha)', 'EfficientZero V2'],
  content: (
    <>
      <Section title="Model-free vs. model-based: чем world model отличается от всего курса до сих пор">
        <p>
          Каждый алгоритм этого курса до сих пор был <b>model-free</b>: чтобы узнать, что случится
          после действия a в состоянии s, единственный способ — реально его выполнить
          (env.step()) и посмотреть. <b>World model</b> — это отдельная нейросеть, которая учится
          <i> предсказывать</i> следующее наблюдение/награду/конец эпизода по (s, a), не выполняя
          действие в настоящей среде. Раз есть такая модель, её можно «прокручивать» сколько угодно
          раз почти бесплатно — это и называют <b>воображением</b> (imagination) или планированием.
        </p>
        <ImaginationDiagram />
      </Section>

      <Section title="Зачем: один настоящий шаг стоит дорого, один воображаемый — почти бесплатно">
        <p>
          env.step() может быть медленным (физический симулятор, рендер картинки) или просто
          ограниченным по количеству доступных попыток. Обучаемый forward-проход через маленькую
          сеть — секунды на тысячи шагов. Если модель динамики достаточно точна, замена части
          настоящих шагов воображаемыми — это способ выжать намного больше «опыта» из того же
          числа реальных взаимодействий со средой (сэмпл-эффективность).
        </p>
      </Section>

      <Section title="Три семейства world models в этом приложении — один и тот же конструктор «World Models», разные архитектуры">
        <p>
          На странице <b>World Models</b> (отдельный раздел навигации, аналог Network Builder) можно
          создать спецификацию одного из трёх типов и обучить её как самостоятельный артефакт —
          без какого-либо RL-агента вообще (get random-policy эксперимент, чисто «насколько хорошо
          модель предсказывает динамику») — или дать алгоритму сослаться на неё через
          world_model_id, так же, как алгоритмы уже ссылаются на сохранённые архитектуры сети
          (Network Builder).
        </p>
        <ul className="ml-4 list-disc space-y-1">
          <li>
            <b>RSSM</b> (Recurrent State-Space Model,{' '}
            <PaperLink url="https://arxiv.org/abs/1912.01603">Hafner et al., 2019 — Dreamer</PaperLink>)
            — детерминированное GRU-состояние h + стохастический гауссов латент z, вместе несущие
            всю историю наблюдений. Используется <b>Dreamer</b>-ом для обучения actor-critic
            целиком внутри воображаемых траекторий.
          </li>
          <li>
            <b>Ensemble</b> (ансамбль вероятностных моделей динамики,{' '}
            <PaperLink url="https://arxiv.org/abs/1805.12114">Chua et al., 2018 — PETS</PaperLink>)
            — несколько независимых сетей, предсказывающих распределение (Δнаблюдение, награда).
            Разногласие между членами ансамбля — сигнал «модель здесь не уверена». Используется{' '}
            <b>MBPO</b> (<PaperLink url="https://arxiv.org/abs/1906.08253">Janner et al., 2019</PaperLink>{' '}
            — дополняет replay buffer SAC воображаемыми переходами) и <b>PETS</b> (прямое
            планирование поиском без всякой политики).
          </li>
          <li>
            <b>VAE + MDN-RNN</b> (классические «World Models»,{' '}
            <PaperLink url="https://arxiv.org/abs/1803.10122">Ha &amp; Schmidhuber, 2018</PaperLink>)
            — VAE сжимает каждое наблюдение в компактный код z, MDN-RNN предсказывает распределение
            следующего z как смесь гауссиан. Используется <b>World Models (Ha)</b> — крошечный
            линейный контроллер обучается на замороженной паре (VAE, MDN-RNN) через Evolution
            Strategies.
          </li>
        </ul>
        <p>
          Четвёртый алгоритм этой группы, <b>EfficientZero V2</b> (
          <PaperLink url="https://arxiv.org/abs/2403.00564">Wang et al., ICML 2024</PaperLink>),
          не подключается к World Model Builder — у него нет отдельно переиспользуемого артефакта
          «модель мира»: representation/dynamics/prediction сети настолько тесно связаны с его
          собственными policy/value-головами и поиском, что переиспользовать их отдельно (как
          RSSM/ансамбль/VAE+MDN-RNN переиспользуются между несколькими алгоритмами выше) не имеет
          смысла — см. отдельный урок.
        </p>
      </Section>

      <Section title="Что общее у всех трёх — обучение на реальном опыте, всегда">
        <p>
          Модель динамики никогда не учится «из воздуха» — она всегда fit-ится на настоящих
          переходах (s, a, r, s', done), собранных из реальной среды (случайной или текущей
          политикой). Воображение работает только <i>после</i> того, как модель успела увидеть
          достаточно реальных данных — до этого её предсказания бессмысленны, а обучение на них
          вредно (см. предупреждения в уроках Dreamer/MBPO о том, почему real-experience-коллекция
          продолжается непрерывно, а не один раз в начале).
        </p>
      </Section>

      <Callout tone="info" title="Как это устроено в интерфейсе">
        Раздел <b>World Models</b> в навигации — создать спецификацию (тип + гиперпараметры
        архитектуры), обучить её как самостоятельный запуск (Мониторинг работает так же, как для
        любого другого запуска — те же loss-графики, GIF воображаемых-vs-реальных траекторий,
        визуализация латентного пространства) или подключить к Dreamer/MBPO/PETS/World Models (Ha) в
        Дизайнере экспериментов через выпадающий список — так же, как готовые архитектуры сети из
        Network Builder подключаются к любому алгоритму.
      </Callout>

      <Callout tone="tip" title="Дальше — по одному уроку на каждый из пяти алгоритмов">
        Следующие четыре урока разбирают Dreamer, MBPO+PETS, World Models (Ha) и EfficientZero V2
        детально — что именно внутри каждого world model, как он обучается и как именно его
        воображение (или поиск) превращается в поведение агента.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 7c. Dreamer (RSSM)
// ---------------------------------------------------------------------------

const dreamer: Lesson = {
  id: 'dreamer',
  title: 'Dreamer (RSSM)',
  tagline: 'Учим actor-critic целиком «во сне» — внутри воображаемых траекторий через выученную RSSM-модель',
  icon: Moon,
  group: 'World Models',
  badges: ['Model-based', 'World Model: RSSM', 'Дискретные и continuous'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="dreamer"
        kind="gym"
        hyperparams={{ batch_size: 32, seq_len: 50, imagination_horizon: 15, gamma: 0.99, collect_steps_per_iter: 100 }}
        show={{ network: false }}
      />

      <p className="text-[13px] text-muted-foreground">
        Dreamer (<PaperLink url="https://arxiv.org/abs/1912.01603">Hafner et al., 2019/2020</PaperLink>)
        — первый из трёх world-model-алгоритмов приложения, где сама RSSM обучается совместно с
        actor-critic на протяжении всего запуска, а не в отдельной первой фазе.
      </p>

      <Section title="RSSM: два состояния вместо одного">
        <p>
          RSSM несёт по времени пару (h, z) — детерминированное GRU-состояние h (помнит всё
          релевантное без потерь, никакого бутылочного горлышка) и стохастический гауссов латент z
          (моделирует настоящую неопределённость, которую один forward-проход не может разрешить —
          например, что за той дверью, которую агент ещё не открывал). z получают двумя разными
          способами:
        </p>
        <FlowRow
          items={[
            { icon: Eye, title: 'Posterior', detail: 'видит настоящее наблюдение — для обучения на реальных данных' },
            { icon: Moon, title: 'Prior', detail: 'предсказывает z по h без наблюдения — единственный способ во время воображения' },
          ]}
        />
      </Section>

      <Section title="Обучение модели на реальных последовательностях (ELBO-стиль)">
        <p>
          Модель фитится на окнах реального опыта (seq_len шагов подряд) четырьмя одновременными
          лоссами: реконструкция наблюдения из (h,z), KL между posterior и prior (заставляет prior
          учиться предсказывать то же самое, что posterior видит с доступом к настоящему кадру —
          это и есть то, что делает prior пригодным для воображения), предсказание награды и
          предсказание конца эпизода.
        </p>
        <Formula caption="free_nats — «бесплатный» минимум KL, ниже которого модель не штрафуется; без него самый простой способ занизить KL — обрушить posterior на prior, что уничтожает всю полезную информацию, которую z должен нести">
{`model_loss = recon_loss + kl_scale · max(KL(posterior ‖ prior), free_nats)
+ reward_loss + continue_loss`}
        </Formula>
      </Section>

      <Section title="Воображение: prior катает модель вперёд без единого реального шага">
        <p>
          Взяв любое реальное (h, z) как стартовую точку, imagination_horizon раз подряд вызывается
          только prior — actor выбирает действие по текущему признаку feat=(h,z), prior предсказывает
          следующее (h,z), reward-голова и continue-голова оценивают награду и вероятность
          продолжения эпизода. Ни одного env.step() за весь процесс.
        </p>
      </Section>

      <Section title="Actor-critic обучается только на воображаемых траекториях">
        <p>
          Critic оценивает V(feat) в конце горизонта (bootstrap), дальше возврат считается назад по
          времени вдоль воображаемой траектории — обычный дисконтированный возврат, а не
          промежуточный λ-возврат из статьи. Градиент в actor идёт целиком через reward/continue
          предсказания модели (путевой, differentiable rollout — поэтому и action, и prior должны
          быть сэмплированы дифференцируемо: Gumbel-softmax straight-through для дискретных
          действий, tanh-Gaussian для continuous), а не через critic напрямую.
        </p>
        <Formula caption="возврат для каждого шага воображаемой траектории — не забывая continue как «мягкий» множитель незавершённости эпизода">
{`G_t = reward_t + γ · continue_t · G_{t+1},   G_H = critic(feat_H)  (bootstrap)
actor_loss  = − mean(G) − entropy_coef · mean(entropy)
critic_loss = MSE( critic(feat), G.detach() )`}
        </Formula>
      </Section>

      <Callout tone="tip" title="Почему это ускоряет обучение">
        Один настоящий шаг collect_steps_per_iter → train_steps_per_iter обновлений
        actor-critic'а, каждое — на imagination_horizon воображаемых шагах. Реальных env.step()
        нужно на порядки меньше, чем PPO/SAC потратили бы на такое же количество обновлений политики
        — цена, которую платят взамен, это точность самой RSSM (см. предупреждение ниже).
      </Callout>

      <Callout tone="warning" title="Ограничение: воображение настолько хорошо, насколько хороша модель">
        Если RSSM ещё плохо предсказывает динамику (learning_starts не пройден, среда только
        начала собирать опыт), actor-critic обучается на систематически неверных траекториях — это
        не просто «шумно», а активно вредно. Поэтому реальный опыт продолжает собираться
        непрерывно (не одной фазой в начале, как в классических World Models) — модель дообучается
        параллельно с actor-critic-обновлениями на протяжении всего запуска.
      </Callout>

      <Callout tone="good" title="Параллельные среды ускоряют именно сбор реального опыта">
        training.num_envs {'>'} 1 запускает несколько сред параллельно (AsyncVectorEnv,
        подпроцесс на лейн) — (h, z) — обычный тензор (N, dim), поэтому продвинуть N лейнов на один
        реальный шаг вперёд — это тот же step_posterior с батчем N вместо 1. Это единственная часть
        Dreamer'а, которую не ускорить дополнительным GPU — обучение actor-critic на воображении
        уже и так batched.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 7d. MBPO и PETS (Ensemble Dynamics)
// ---------------------------------------------------------------------------

const mbpoPets: Lesson = {
  id: 'mbpo_pets',
  title: 'MBPO и PETS (Ensemble Dynamics)',
  tagline: 'Один и тот же ансамбль моделей динамики — по-разному использован: как источник данных для SAC или как модель для прямого поиска',
  icon: Crosshair,
  group: 'World Models',
  badges: ['Model-based', 'World Model: Ensemble', 'Только continuous'],
  content: (
    <>
      <p className="text-[13px] text-muted-foreground">
        Один и тот же ансамбль (<PaperLink url="https://arxiv.org/abs/1805.12114">Chua et al., 2018 — PETS</PaperLink>)
        используют два разных алгоритма приложения: MBPO (
        <PaperLink url="https://arxiv.org/abs/1906.08253">Janner et al., 2019</PaperLink>) как
        источник дополнительных данных для SAC, и PETS напрямую для планирования без всякой политики.
      </p>

      <Section title="Ансамбль вероятностных моделей — откуда берётся неопределённость">
        <p>
          Вместо одной сети динамики — ensemble_size независимых сетей, каждая предсказывает
          распределение (не точку!) «дельты» наблюдения и награды: Δобс ~ N(μ, σ²), r ~ N(μ_r,
          σ_r²). Каждый член обучается на своей bootstrap-подвыборке батча (с повторениями) — а не
          на одном и том же батче для всех — именно это заставляет ансамбль реально
          «расходиться» в областях, где данных мало (ровно тот сигнал неопределённости, на который
          опирается PETS' планирование).
        </p>
        <Formula caption="Гауссова NLL для каждого члена на своей bootstrap-подвыборке; total_loss усредняет обе части по всем членам">
{`obs_nll    = 0.5·( (Δобс_target − Δμ)² / exp(Δlogσ²) + Δlogσ² )
reward_nll = 0.5·( (r_target − μ_r)²  / exp(logσ²_r) + logσ²_r )`}
        </Formula>
        <p className="text-[12px]">
          В реализации все ensemble_size членов считаются одним батчевым matmul (torch.bmm) вместо
          Python-цикла «одна сеть — один forward» — тот же результат, но с O(1), а не O(ensemble_size)
          запусков ядра на слой, что особенно важно на GPU для маленького ансамбля из пары
          200-нейронных слоёв на каждого члена.
        </p>
      </Section>

      <Section title="MBPO — ансамбль как поставщик обучающих данных для SAC">
        <p>
          MBPO — это ровно тот же SAC (сеть/лоссы/target-сети из соответствующего урока), только
          каждое обновление критика/actor'а видит смесь настоящих и <i>воображаемых</i> переходов.
          Короткие (rollout_length шагов, обычно 1) воображаемые траектории «отрастают» от
          настоящих состояний из реального буфера, действие на каждом шаге выбирает текущая
          политика — а не случайная — потому что именно её и предстоит улучшать этими данными.
        </p>
        <ImaginationDiagram />
        <Formula caption="real_ratio контролирует долю настоящих переходов в каждом батче SAC — статья показывает, что «в основном реальные + немного модельных» превосходит оба крайних варианта (чистый real-data SAC и полностью Dyna-style модельные эпизоды)">
{`batch = real_ratio · real_buffer + (1 − real_ratio) · model_buffer`}
        </Formula>
        <Callout tone="info" title="Почему rollout_length короткий">
          Ошибка модели накапливается с каждым шагом воображения — MBPO's ключевая идея: если
          отращивать <i>короткие</i> траектории от настоящих (а не случайных или сильно устаревших)
          состояний, накопленная ошибка модели остаётся ограниченной, а не разгоняется на весь
          горизонт эпизода.
        </Callout>
      </Section>

      <Section title="PETS — тот же ансамбль, но без всякой политики: прямое планирование">
        <p>
          PETS не учит ни actor, ни critic вообще — каждое настоящее действие выбирается заново с
          нуля методом Cross-Entropy (CEM): сэмплируется num_candidates случайных
          последовательностей действий длиной cem_horizon, каждая прогоняется через ансамбль
          («trajectory sampling» — один и тот же случайно выбранный член ансамбля используется на
          всём горизонте одной траектории, иначе усреднение по шагам стёрло бы саму
          неопределённость, которую ансамбль должен нести), берутся num_elites с максимальной
          суммой предсказанной награды, по ним переоценивается (mean, std) для следующей итерации.
        </p>
        <Formula caption="Из финального (уточнённого за cem_iterations итераций) среднего берётся только самое первое действие — план целиком отбрасывается и строится заново на следующем настоящем шаге (стандартный MPC), т.к. ошибка модели растёт с горизонтом">
{`for iteration in 1..cem_iterations:
  candidates ~ N(mean, std²), clip к границам действия
  returns    = Σ_t predicted_reward( rollout через ансамбль )
  elites     = top-k(returns, num_elites)
  mean, std  = elites.mean(), elites.std()
action = clip(mean[0])`}
        </Formula>
      </Section>

      <Callout tone="tip" title="PETS — самый прямой способ увидеть, насколько хороша модель">
        У PETS нет никакого запасного навыка, накопленного отдельно от модели (в отличие от
        PPO/SAC, у которых свежий градиентный шаг может быть неудачным, но остальная сеть всё ещё
        помнит прошлое) — каждое действие целиком зависит от того, что ансамбль думает о будущем
        прямо сейчас. Кривая награды PETS — довольно честный прокси-показатель того, насколько
        улучшилась сама модель динамики за время обучения.
      </Callout>

      <Callout tone="warning" title="PETS не поддерживает training.num_envs {'>'} 1">
        Смысл алгоритма — по-настоящему онлайн-планирование от текущего состояния настоящей среды,
        один шаг (затем replan с нуля) за раз — у этого нет очевидного аналога «N параллельных
        лейнов», в отличие от сбора опыта с фиксированной политикой (MBPO/Dreamer это умеют).
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 7e. World Models (Ha & Schmidhuber): VAE + MDN-RNN
// ---------------------------------------------------------------------------

const worldModelsHa: Lesson = {
  id: 'world_models_ha',
  title: 'World Models (Ha & Schmidhuber): VAE + MDN-RNN',
  tagline: 'Оригинальный рецепт 2018 года: сжать наблюдение в код, предсказать распределение следующего кода, обучить крошечный контроллер сверху',
  icon: Aperture,
  group: 'World Models',
  badges: ['Model-based', 'World Model: VAE+MDN-RNN', 'Evolution Strategies'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="world_models_ha"
        kind="gym"
        hyperparams={{ world_model_phase_steps: 10_000, seq_len: 50, population_size: 16, sigma: 0.1 }}
        show={{ network: false }}
      />

      <p className="text-[13px] text-muted-foreground">
        Оригинальный «World Models» (<PaperLink url="https://arxiv.org/abs/1803.10122">Ha &amp;
        Schmidhuber, 2018</PaperLink>) — самый старый из трёх world-model-подходов приложения и
        единственный, где world model обучается один раз, а не совместно с политикой всё время.
      </p>

      <Section title="Две полностью раздельные сети: VAE и MDN-RNN">
        <p>
          VAE (<PaperLink url="https://arxiv.org/abs/1312.6114">Kingma &amp; Welling, 2013</PaperLink>)
          сжимает <i>одно</i> наблюдение за раз в компактный код z (без всякой рекуррентности —
          это целиком забота MDN-RNN). MDN-RNN — рекуррентная сеть, предсказывающая по (z_t, a_t) не
          одну точку z_{'{'}t+1{'}'}, а параметры <b>смеси гауссиан</b> — несколько возможных
          «продолжений» сразу, что важно там, где будущее реально неоднозначно (например, трасса
          CarRacing расходится на развилке — оба направления одинаково вероятны, пока агент не
          выбрал одно).
        </p>
        <FlowRow
          items={[
            { icon: Eye, title: 'Наблюдение oₜ', detail: 'сырое, большое' },
            { icon: Combine, title: 'VAE-энкодер', detail: 'сжатие' },
            { icon: Sparkles, title: 'zₜ', detail: 'компактный код' },
            { icon: History, title: 'MDN-RNN', detail: 'смесь гауссиан для z_{t+1}' },
          ]}
        />
        <Formula caption="K компонент смеси, каждая — своя (mean, log_std) над полным z и свой вес-логит; NLL — обычный логарифм смеси гауссиан">
{`p(z_{t+1} | z_t, a_t) = Σ_k softmax(logits)_k · N( z_{t+1}; mean_k, std_k² )
mdn_loss = − log p(z_{t+1}^{target} | z_t, a_t)`}
        </Formula>
      </Section>

      <Section title="Фаза 1 — собрать данные и обучить World Model">
        <p>
          Случайная политика (в этой реализации — намеренно: у этой фазы нет цели найти хорошее
          поведение, только увидеть разнообразную динамику) собирает world_model_phase_steps шагов
          опыта в SequenceReplayBuffer, а VAE + MDN-RNN обучаются одновременно на выборках окон из
          него — реконструкция + KL (VAE) и mixture NLL + предсказание награды/конца эпизода
          (MDN-RNN), одним общим лоссом.
        </p>
      </Section>

      <Section title="Фаза 2 — заморозить модель, обучить крошечный линейный контроллер">
        <p>
          После фазы 1 обе сети замораживаются (requires_grad_(False) — больше не меняются). Вся
          «умная» часть уже сделана моделью — контроллеру остаётся один линейный слой прямо от
          конкатенации [z, h] (латент VAE + скрытое состояние MDN-RNN) к действию. Он обучается
          <b> Evolution Strategies</b> (тот же рецепт mirrored sampling + centered ranks, что и в
          отдельном уроке про ES) напрямую на настоящую отдачу эпизода — без единого backward-прохода
          через среду.
        </p>
        <Callout tone="tip" title="Почему controller настолько маленький работает">
          Идея статьи именно в этом: если world model уже выучила хорошее представление о динамике,
          политике сверху почти нечего добавлять — линейный слой оказывается достаточным, потому что
          вся сложность задачи уже «переехала» в [z, h].
        </Callout>
      </Section>

      <Callout tone="info" title="Пропущенная (третья) фаза статьи">
        Оригинальная статья предлагает опциональную третью фазу — обучать контроллер целиком внутри
        полностью галлюцинированной M-model «мечты», а не в настоящей среде. Здесь это не
        реализовано (отдельная большая задача поверх уже большой фичи) — фаза 2 всегда оценивает
        контроллер на настоящих env-эпизодах.
      </Callout>

      <Callout tone="good" title="Параллельные среды ускоряют обе фазы по-разному">
        Фаза 1 (случайный сбор) — training.num_envs {'>'} 1 параллельных лейнов через тот же
        AsyncVectorEnv, что и везде. Фаза 2 не батчит саму популяцию по лейнам (в отличие от
        отдельного ES-алгоритма) — этот приём не переносится на картиночные наблюдения, а World
        Models изначально рассчитан именно на них — но повторы (episodes_per_eval {'>'} 1) <i>одного</i>{' '}
        кандидата теперь идут параллельно по num_envs лейнам вместо строго по очереди.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 7f. EfficientZero V2 (MuZero-family: планирование поиском по выученной модели)
// ---------------------------------------------------------------------------

const efficientzero: Lesson = {
  id: 'efficientzero',
  title: 'EfficientZero V2 (MuZero-family)',
  tagline: 'Не «воображать эпизод и учить actor-critic на нём» (Dreamer) и не «искать план методом CEM» (PETS) — а искать улучшенную политику маленьким деревом на каждом шаге',
  icon: GitBranch,
  group: 'World Models',
  badges: ['Model-based', 'MCTS/Gumbel search', 'Дискретные и continuous'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="efficientzero"
        kind="gym"
        hyperparams={{ num_sampled_actions: 8, num_simulations: 32, num_top_actions: 8, unroll_steps: 5 }}
        show={{ network: false }}
      />

      <p className="text-[13px] text-muted-foreground">
        EfficientZero V2 (<PaperLink url="https://arxiv.org/abs/2403.00564">Wang, Liu, Ye, You &amp; Gao,
        ICML 2024 Spotlight</PaperLink>) продолжает линию MuZero (
        <PaperLink url="https://arxiv.org/abs/1911.08265">Schrittwieser et al., 2020</PaperLink>) и
        EfficientZero (<PaperLink url="https://arxiv.org/abs/2111.00210">Ye et al., 2021</PaperLink>):
        вместо RSSM/ансамбля из предыдущих уроков учится компактная тройка сетей
        representation/dynamics/prediction, а действие каждый раз выбирает не сама политика
        напрямую, а маленький бюджетированный поиск по этой модели.
      </p>

      <Section title="Три сети вместо RSSM/ансамбля — и никакой реконструкции наблюдения">
        <FlowRow
          items={[
            { icon: Eye, title: 'h(obs) → s', detail: 'representation: реальное наблюдение → латент' },
            { icon: GitFork, title: 'g(s,a) → s\', reward', detail: 'dynamics: латентный переход + value prefix' },
            { icon: Target, title: 'f(s) → policy, value', detail: 'prediction: обе головы из одного латента' },
          ]}
        />
        <p>
          В отличие от Dreamer/World Models (Ha), латент s здесь не обязан быть достаточным, чтобы
          восстановить из него наблюдение обратно (никакого декодера/reconstruction loss вообще
          нет) — единственное требование к нему: быть достаточным, чтобы predict-нуть policy,
          value и следующий s. Это ровно то же самое «планирование в компактном пространстве
          задачи, а не в пространстве пикселей» решение, за которое и назван сам MuZero.
        </p>
      </Section>

      <Section title="Training unroll: как разворачивается предсказание вперёд">
        <p>
          Обучение берёт из буфера окно из <code>unroll_steps + 1</code> реальных шагов подряд: реальное
          <code> obs₀</code> и <code>unroll_steps</code> реальных действий <code>a₀, a₁, ...</code>, которые
          агент и правда выполнил. Представление <code>s₀ = h(obs₀)</code> строится один раз из настоящего
          наблюдения — а дальше каждый следующий <code>sₖ = g(sₖ₋₁, aₖ₋₁)</code> получается <i>чисто</i> из
          dynamics-сети по известному реальному действию, без единого повторного обращения к настоящему
          наблюдению. Policy/value читаются с каждого <code>sₖ</code>, reward — с каждого перехода.
        </p>
        <LatentChainDiagram />
      </Section>

      <Section title="Consistency loss: как латент не разваливается без реконструкции">
        <p>
          Без reconstruction loss нечем помешать representation-сети коллапсировать s в
          константу — модель будет «дешёво» минимизировать reward/value loss, если s ничего не
          несёт. Self-supervised consistency loss (SimSiam-стиль,{' '}
          <PaperLink url="https://arxiv.org/abs/2011.10566">Chen &amp; He, 2021</PaperLink>) решает
          это: предсказанный dynamics-ом следующий латент ŝ должен быть похож на настоящий
          h(next_obs) — но через асимметричную пару projector/predictor и stop-gradient на
          «настоящей» ветке, а не прямым MSE (симметричный MSE между двумя ветками одной и той же
          сети без этих приёмов тривиально коллапсирует в константу).
        </p>
        <Formula caption="stop-gradient на branch с настоящим следующим наблюдением — это единственное, что не даёт лоссу выродиться в тривиальный «одна и та же константа с обеих сторон»">
{`p_true = projector(h(next_obs)).detach()
p_pred = predictor(projector(ŝ))
consistency_loss = − cosine_similarity(p_true, p_pred)`}
        </Formula>
      </Section>

      <Section title="Value prefix: LSTM вместо предсказания «награда за один шаг»">
        <p>
          Reward-голова dynamics-сети — это не отдельное число на каждый шаг, а LSTM, несущая
          состояние через весь training-unroll и предсказывающая <i>накопленную</i> (не
          единичную) настоящую награду с начала unroll'а до текущего шага. Идея (унаследована от
          EfficientZero v1): ошибка предсказания одного шага накопится в любом случае за
          unroll_steps шагов вперёд — value prefix предсказывает эту накопленную сумму прямо, а не
          складывает unroll_steps отдельных, каждый со своей ошибкой, предсказаний.
        </p>
      </Section>

      <Section title="Категориальные value/reward-головы: почему не просто один скаляр + MSE">
        <p>
          И value-голова, и «value prefix» (reward-голова dynamics-сети) предсказывают не одно
          число, а <i>распределение</i> по фиксированному набору из <code>2 · value_support_size +
          1</code> «корзин» — ровно приём MuZero (Appendix F, «scaling and squashing»). Причина не
          декоративная: у скалярной MSE-головы loss растёт <i>квадратично</i> от масштаба цели —
          одна редкая большая награда (бонус, добыча, скачок score) в минибатче делает loss (и
          градиент) этого шага обучения огромным относительно всех остальных, обычных шагов.
          Категориальная голова вместо этого сначала сжимает весь диапазон через{' '}
          <code>signed_hyperbolic</code> (логарифмически по модулю), а затем учится cross-entropy
          по «two-hot» цели (вес размазан между двумя соседними корзинами) — loss любого одного
          шага ограничен сверху <code>log(число корзин)</code> независимо от того, насколько
          большой была именно эта цель.
        </p>
        <Formula caption="MuZero Appendix F: h — сжимающее преобразование, h⁻¹ — обратное; корзины равномерны в h-масштабе">
{`h(x)   = sign(x) · (sqrt(|x| + 1) − 1) + ε·x
h⁻¹(y) = sign(y) · (((sqrt(1 + 4ε(|y| + 1 + ε)) − 1) / 2ε)² − 1)

target = two_hot(h(x))                     # cross-entropy, не MSE(x̂, x)
x̂      = h⁻¹(Σ_bin  softmax(logits)_bin · bin_value)`}
        </Formula>
        <TwoHotBinsChart />
        <Callout tone="warning" title="Это не только про качество обучения — это про стабильность">
          Именно отсутствие этого приёма стояло за всплесками value_loss/reward_loss на всплесках
          награды в средах с «спайковой» наградой (например, NetHack: несколько шагов почти без
          награды, затем один большой скачок) — plain-скалярная MSE-голова превращает такой скачок
          в разрушительно большой градиент на этом шаге обучения. Категориальная голова устраняет
          это в первую очередь, а не просто «более точно оценивает большие значения».
        </Callout>
      </Section>

      <Section title="Gumbel search: настоящее дерево, только с бюджетированным отбором кандидатов вместо классического PUCT-MCTS">
        <p>
          Классический PUCT-MCTS (AlphaZero, оригинальный MuZero) тратит сотни симуляций на ход —
          нормально для настольной игры раз в несколько секунд, слишком дорого, чтобы гонять на{' '}
          <i>каждом</i> шаге среды. Gumbel search (
          <PaperLink url="https://arxiv.org/abs/2202.05489">Danihelka et al., 2022</PaperLink>,
          который переиспользует и сама статья EfficientZero V2 для дискретного случая) — это{' '}
          <i>настоящее</i> дерево с visit-count'ами и backup'ом значений на каждом узле, просто
          дешевле, чем PUCT: в корне сразу отбирается небольшое число K кандидатов
          Gumbel-Top-k trick'ом (логиты политики + шум Gumbel, топ-K после сортировки), а не все
          действия сразу, и это множество затем урезается турниром Sequential Halving — раунд за
          раундом отбрасывая половину кандидатов с худшей оценкой. Каждая из num_simulations
          симуляций спускается по дереву на одну реальную ступень глубже (корень → ещё не
          раскрытый узел), раскрывает его одним вызовом dynamics/prediction-сетей и разносит
          полученное значение обратно вверх по всему пройденному пути — ровно тот же механизм
          backup'а, что у PUCT-MCTS, просто без огромного бюджета симуляций на ход.
        </p>
        <GumbelHalvingDiagram />
        <Formula caption="v_mix/completed Q (Danihelka et al., 2022, Appendix D) — то, что даёт даже НЕ раскрытым узлам разумную оценку Q вместо того, чтобы их просто игнорировать">
{`# корень: Gumbel-Top-k из m кандидатов, round-robin по visit-count
# внутри дерева: argmax(improved_policy − visit_fraction), без Sequential Halving
completed_Q(a) = Q(a)                        если ребёнок a раскрыт
               = v_mix(node)                 если ещё не раскрыт  # не «игнорировать», а оценить через соседей
improved_policy = softmax(prior + sigma(completed_Q))            # sigma = (c_visit + max_visit) · c_scale
action = Sequential-Halving-победитель среди корневых кандидатов`}
        </Formula>
      </Section>

      <Section title="Continuous-действия — фирменное отличие V2 от исходного EfficientZero">
        <p>
          Gumbel-Top-k работает только для конечного набора действий — континуальное пространство
          нужно сначала как-то дискретизировать. V2 вместо этого сэмплирует K кандидатов-действий
          напрямую: часть — из текущей гауссовой политики (эксплуатация), часть — из искусственно
          расширенной версии той же политики (continuous_prior_scale — больше σ, шире покрытие,
          эксплорация). Тот же турнир Sequential Halving по симулированному Q выбирает победителя
          среди этих сэмплов — не нужно никакой дискретизации пространства действий вообще.
        </p>
      </Section>

      <Section title="Что на самом деле обучает policy/value-головы — не сыгранное действие, а результат поиска">
        <p>
          Ключевое отличие от model-free курса целиком: цель для policy-головы — не «то действие,
          которое агент сыграл», а «улучшенная политика» поиска — softmax(prior + sigma(completed
          Q)) по <i>всем</i> корневым кандидатам, ровно формула из статьи (Eq. 3) — то есть сама
          политика дистиллируется в сторону того, что поиск только что нашёл лучше неё самой.
          Value-цель для обучения — обычный n-step (td_steps) возврат на настоящих наградах с
          bootstrap текущей value-сетью, а не значение поиска напрямую (см. заметку ниже про
          Search-Based Value Estimation).
        </p>
      </Section>

      <Callout tone="info" title="Упрощения этой реализации относительно статьи">
        Дерево поиска и его формулы (Gumbel-Top-k, Sequential Halving, v_mix/completed Q, backup
        значений) реализованы без сокращений — это то же самое дерево, что в статье, сверено
        построчно с их же C++/Cython-путём поиска (`gumbel_cnode.cpp`), которым они реально
        обучают модели (их Python-путь `py_mcts.py` — отладочный, с собственным квирком: top-m
        там фактически не работает). Prioritized replay и reanalyze (см. ниже) — тоже перенесены,
        а не упрощены. Что осталось проще, чем у них: никакого отдельного async self-play/
        reanalyze/train на Ray — reanalyze у нас синхронный, встроенный в основной цикл (дороже по
        wall-clock, но без второго процесса); и никакого Dirichlet-шума в корневых приорах — сам
        Gumbel-Top-k уже даёт свежую случайность в том, какие действия попадут на рассмотрение.
      </Callout>

      <Callout tone="good" title="Prioritized Experience Replay + Reanalyze — перенесены из оригинала">
        Раньше буфер сэмплировал переходы равномерно, а value-цель была одним фиксированным
        n-step TD-таргетом. Теперь — как в `ez/agents/base.py` оригинала: (1) сэмплирование
        пропорционально |предсказание value − таргет|<sup>priority_alpha</sup> (Schaul et al.,
        2016) с importance-sampling коррекцией (priority_beta) лосса, чтобы это не смещало
        градиент; (2) reanalyze — каждые reanalyze_freq реальных шагов заново прогоняем
        search() текущей (уже более обученной) сетью по случайным сохранённым переходам и
        перезаписываем их policy_target/search_value — у оригинала это отдельные фоновые
        процессы, у нас — периодический проход внутри общего цикла; (3) value-таргет —
        max(TD-bootstrap, search_value) — их режим value_target='max': переход, который ещё не
        реанализировали, прозрачно откатывается на обычный TD-таргет.
      </Callout>

      <Callout tone="info" title="training.num_envs > 1 — это по-настоящему батчевый поиск">
        В отличие от многих model-based алгоритмов, где параллельные лейны означают просто «N
        независимых копий», здесь search() принимает сразу весь батч наблюдений и строит все N
        деревьев в лок-степе: каждая из num_simulations симуляций считает раскрытие узла для всех
        N лейнов ОДНИМ batched forward pass'ом через dynamics/prediction-сети, а не N отдельными.
      </Callout>

      <Callout tone="tip" title="consistency_loss_coef=5.0 по умолчанию — как в официальном atari.yaml">
        Официальный репозиторий EfficientZeroV2 весит consistency loss в 5 раз сильнее value loss
        (5.0 против 0.5 — для DMC/continuous 2.0) — это безопасно именно потому, что BatchNorm +
        асимметрия projector/predictor + stop-gradient на «истинной» ветке структурно не дают
        представлению схлопнуться независимо от веса лосса (в этом и смысл SimSiam/BYOL-трюка) —
        больший вес просто быстрее и точнее подгоняет представление, а не рискует коллапсом.
      </Callout>

      <Callout tone="good" title="train_freq/train_steps_per_iter не зависят от num_envs">
        num_timesteps растёт на num_envs за итерацию (все лейны шагают в лок-степе), поэтому наивная
        проверка «пересекли ли границу train_freq?» срабатывает не более раза за итерацию — даже
        если такой скачок пересёк сразу несколько границ. Без поправки на это рост num_envs тихо
        делил бы частоту обучения относительно собранного опыта на num_envs (в 24 → 48 лейнов —
        вдвое реже градиентных шагов на реальный шаг среды). Обучение считает, сколько границ
        train_freq реально пересечено за скачок, и делает train_steps_per_iter обновлений за
        каждую — соотношение «градиентных шагов на единицу опыта» остаётся постоянным при любом
        num_envs.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 7f2. UniZero — Transformer вместо рекуррентной динамики
// ---------------------------------------------------------------------------

const unizero: Lesson = {
  id: 'unizero',
  title: 'UniZero (MuZero + Transformer)',
  tagline: 'Тот же рецепт «планировать через выученную модель мира на каждом шаге», что у EfficientZero — но модель мира не рекуррентная, а causal Transformer над явной последовательностью токенов',
  icon: History,
  group: 'World Models',
  badges: ['Model-based', 'Transformer world model', 'MCTS/Gumbel search', 'Дискретные и continuous'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="unizero"
        kind="gym"
        hyperparams={{ embed_dim: 128, num_layers: 2, context_length: 6, num_simulations: 32, unroll_steps: 5 }}
        show={{ network: false }}
      />

      <p className="text-[13px] text-muted-foreground">
        UniZero (<PaperLink url="https://arxiv.org/abs/2406.10667">Pu, Zhao, Niu et al., ICLR 2025</PaperLink>,
        проект <PaperLink url="https://github.com/opendilab/LightZero">LightZero</PaperLink>) решает ту же
        задачу, что и предыдущий урок — планировать маленьким деревом через выученную модель мира на
        каждом реальном шаге — но меняет саму модель мира. У EfficientZero (и у самого MuZero) вся
        история сжимается в один вектор латентного состояния s, который LSTM/MLP-динамика шаг за шагом
        обновляет. UniZero вместо этого держит явную последовательность токенов — каждое прошлое
        наблюдение и действие остаётся своим собственным токеном, к которому Transformer может напрямую
        обратиться через attention на любом более позднем шаге, а не «наполовину забытым» внутри одного
        сжатого вектора.
      </p>

      <Section title="Токены вместо одного вектора состояния">
        <FlowRow
          items={[
            { icon: Eye, title: 'obsₜ → токен', detail: 'тот же encoder, что и везде в приложении' },
            { icon: Target, title: 'actionₜ → токен', detail: 'one-hot/raw вектор → embedding' },
            { icon: Layers, title: 'Causal Transformer', detail: 'внимание на все прошлые токены окна памяти' },
          ]}
        />
        <TokenAttentionDiagram />
        <p>
          Входная последовательность — <code>[obs₀, act₀, obs₁, act₁, ..., obsₜ]</code>: наблюдение и
          действие каждого прошлого шага (в пределах окна <code>context_length</code>) — отдельный токен.
          Головы читают конкретные позиции этой последовательности после Transformer'а: reward — с
          hidden state в позиции токена <i>действия</i> (модель предсказывает награду именно за это
          действие), policy/value — с позиции токена <i>наблюдения</i> (та же логика, что у EfficientZero:
          что делать/сколько это стоит, оценивается до выбора действия). И value, и reward — те же
          категориальные головы (signed-hyperbolic + two-hot, MuZero Appendix F), что и в EfficientZero —
          см. предыдущий урок за подробным разбором, почему это не «просто одна голова вместо другой».
        </p>
      </Section>

      <Section title="Полное использование траектории при обучении — не только первого наблюдения">
        <p>
          У EfficientZero обучающий unroll реально «видит» настоящее наблюдение только один раз (в самом
          начале), а дальше unroll идёт через собственные предсказания dynamics-сети — настоящие
          <code>next_obs</code> в этом unroll'е используются только как <i>цель</i> для consistency loss, а
          не как вход. У UniZero наоборот: вся последовательность из <code>unroll_steps + 1</code> настоящих
          наблюдений подаётся в Transformer <i>сразу</i>, за один forward pass, и каждое из них сразу даёт
          свой собственный reward/value/policy loss — не только первое. Ровно это в статье называется
          «полное использование траектории» и объясняется как одна из причин, почему UniZero сходится
          быстрее рекуррентной модели мира даже без каких-либо задач на длинную память.
        </p>
      </Section>

      <Section title="Consistency loss здесь играет другую роль — учит именно 'воображение'">
        <p>
          Раз обучение видит настоящие наблюдения на каждом шаге, для чего вообще нужен consistency loss?
          Ответ — для <i>поиска</i>: во время Gumbel-поиска (и во время предсказания на много шагов
          вперёд) настоящих будущих наблюдений ещё нет, и Transformer должен «представить», что увидел бы
          следующий obs-токен, чтобы продолжить последовательность и раскрыть узел дерева глубже. Голова
          <code>latent_head</code> учится предсказывать этот воображаемый следующий токен из hidden state
          позиции действия — а обучается она той же SimSiam-схемой (projector/predictor + BatchNorm +
          stop-gradient на «настоящей» ветке), что и consistency loss EfficientZero, просто целью здесь
          служит сырой (до-Transformer'ный) embedding настоящего следующего наблюдения — тот же тензор,
          что и так вычисляется как часть общего forward pass'а обучения, просто с <code>.detach()</code>.
        </p>
      </Section>

      <Section title="Поиск — тот же Gumbel search, что у EfficientZero, просто с другим 'шагом динамики'">
        <p>
          Дерево поиска (Gumbel-Top-k в корне, Sequential Halving, v_mix/completed Q, backup значений —
          см. предыдущий урок за формулами) здесь буквально то же самое, включая и то, как решается
          continuous-случай (сэмплирование K кандидатов из текущей гауссовой политики + расширенной
          версии). Меняется только то, что происходит при раскрытии одного узла: вместо одного вызова
          LSTM-динамики — два forward pass'а Transformer'а (действие → reward, воображаемое следующее
          наблюдение → policy/value), а «состояние» узла дерева — это не вектор фиксированного размера, а
          собственный кусок последовательности токенов этого узла (растёт на 2 токена на каждую ступень
          вглубь дерева).
        </p>
      </Section>

      <Callout tone="good" title="RoPE вместо обучаемых позиций — и поэтому персистентный KV-cache без потерь">
        Позиция токена здесь кодируется не обучаемым вектором (как в оригинале), а поворотом (RoPE, Su
        et al., 2021) query/key-векторов на угол, пропорциональный позиции, прямо перед скалярным
        произведением в attention — поэтому каждый вес внимания зависит только от <i>разницы</i> позиций
        двух токенов, а не от их абсолютных значений. Отсюда сразу два следствия. Первое: окно обучения,
        всегда нумерующее свои токены с нуля, и self-play, где у токена настоящая, всё растущая абсолютная
        позиция с начала эпизода, вычисляют математически одно и то же — никакой конвенции синхронизировать
        не нужно (в отличие от обучаемой таблицы embedding'ов, где значение для позиции 500 и позиции 0 —
        два ничем не связанных вектора). Второе: вытеснение самых старых записей из KV-cache становится
        обычным срезом тензора без каких-либо приближённых поправок позиций — в отличие от приёма оригинала
        (сдвиг + пересчёт позиционной «дельты»), нужного там именно потому, что его обучаемые embedding'и не
        инвариантны к сдвигу. Поэтому каждый лейн в <code>learn()</code> держит один KV-cache, который живёт
        весь эпизод, растёт на 2 токена за реальный шаг и подрезается обратно до <code>2×context_length</code>
        при переполнении — а не перестраивается с нуля каждый шаг, как было в промежуточной версии этого
        файла. Корень реального шага теперь стоит настоящий <code>O(1)</code>, а не <code>O(context_length)</code>.
      </Callout>

      <Callout tone="warning" title="Честная оговорка: вытеснение теряет позиционную неоднозначность, но не всё">
        RoPE снимает проблему позиций при вытеснении полностью — но не делает вытеснение стерильным во
        всех смыслах при <code>num_layers &gt; 1</code>. Собственное скрытое состояние выжившего токена
        (а значит, и его key/value на всех слоях выше первого) было посчитано <i>пока</i> вытеснённые токены
        ещё были доступны для внимания — поэтому немного их влияния остаётся зашитым в остаточном потоке
        (residual stream) даже после вытеснения. Это неизбежное свойство любого многослойного KV-cache со
        скользящим окном (в том числе в оригинальном UniZero) и не зависит от способа кодирования позиций —
        осознанный компромисс, а не незамеченная ошибка: единственная альтернатива — никогда не вытеснять,
        то есть платить <code>O(длина эпизода)</code> за каждый реальный шаг вместо <code>O(1)</code>.
      </Callout>

      <Callout tone="good" title="Prioritized Experience Replay + Reanalyze — перенесены как есть">
        Буфер, приоритизированное сэмплирование (Schaul et al., 2016) и периодический reanalyze (свежий
        search() текущей сетью поверх сохранённых переходов, value-таргет = max(TD-bootstrap,
        search_value)) — тот же самый механизм, что в EfficientZero, просто с дополнительным полем
        контекста (последние <code>context_length</code> реальных пар (observation, action) перед каждым
        сэмплированным переходом) — эпизоды и так хранятся целиком, так что это просто срез массива, а не
        отдельное хранилище.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 7g. Multi-agent RL: карта алгоритмов
// ---------------------------------------------------------------------------

const marlOverview: Lesson = {
  id: 'marl_overview',
  title: 'Multi-agent RL: карта алгоритмов',
  tagline: 'От полностью независимых политик до общего критика, видящего всех агентов сразу — и где на этой шкале стоят IPPO и QMIX',
  icon: Network,
  group: 'Multi-agent',
  badges: ['MARL', 'CTDE', 'Value decomposition', 'IPPO + QMIX реализованы'],
  content: (
    <>
      <Section title="Главный вопрос MARL: что расшарить между агентами">
        <p>
          Как только в среде больше одного обучаемого агента, у каждого алгоритма из первых уроков
          курса появляется развилка: обучать ли каждого агента полностью отдельно, или дать им
          как-то «подсматривать» друг за другом во время обучения. Большинство MARL-алгоритмов из
          литературы различаются именно ответом на этот вопрос — сама механика обучения политики
          (policy gradient, Q-learning, ...) внутри почти всегда одна из тех, что уже разобраны в
          курсе.
        </p>
        <FlowRow
          items={[
            { icon: Users, title: 'Independent Learning', detail: 'каждый агент — отдельная задача' },
            { icon: Network, title: 'CTDE', detail: 'общий критик на обучении, каждый действует сам' },
            { icon: Combine, title: 'Value decomposition', detail: 'общий Q раскладывается на агентов' },
          ]}
        />
      </Section>

      <Section title="Independent Learning — то, что реализовано в приложении (IPPO)">
        <p>
          Самый простой рецепт: каждый агент (или команда) — это полностью отдельная задача
          model-free RL, любой алгоритм курса без единой поправки на присутствие других агентов.
          Единственное взаимодействие между агентами — через саму динамику среды и правила
          награды. Удивительно конкурентоспособен на практике —{' '}
          <PaperLink url="https://arxiv.org/abs/2011.09533">de Witt et al., 2020</PaperLink>{' '}
          показывают, что независимый PPO не хуже намного более сложных специализированных
          MARL-алгоритмов на большинстве сценариев StarCraft Multi-Agent Challenge. В этом
          приложении это <b>IPPO</b> (через два урока) — policy-gradient точка на этой карте,
          реально выбираемая в Дизайнере экспериментов.
        </p>
      </Section>

      <Section title="CTDE — Centralized Training, Decentralized Execution">
        <p>
          Компромисс между «каждый сам за себя» и полностью централизованным контроллером: во время
          обучения критик (или что-то его заменяющее) видит <i>всё</i> — наблюдения и действия всех
          агентов сразу; во время исполнения каждый агент действует только по своему собственному,
          локальному наблюдению. Критик со «взглядом бога» решает главную проблему независимого
          обучения — что среда для одного агента кажется нестационарной, поскольку остальные
          агенты одновременно меняют свою политику.
        </p>
        <ul className="ml-4 list-disc space-y-1">
          <li>
            <b>MADDPG</b> (<PaperLink url="https://arxiv.org/abs/1706.02275">Lowe et al., 2017</PaperLink>)
            — DDPG-подобный actor на каждого агента, но критик Qᵢ(x, a₁, ..., a_N) каждого агента
            получает на вход наблюдения/действия <i>всех</i> агентов сразу, не только своего.
            Работает и в кооперативных, и в смешанных/конкурентных средах — критик каждого агента
            может учить свою собственную (не обязательно совпадающую с другими) цель.
          </li>
          <li>
            <b>COMA</b> (<PaperLink url="https://arxiv.org/abs/1705.08926">Foerster et al., 2018</PaperLink>)
            — общий централизованный критик оценивает совместное действие, а вклад <i>конкретного</i>{' '}
            агента в результат выделяется через <b>counterfactual baseline</b>: «что было бы, если бы
            этот агент выбрал другое действие, а остальные — нет» — явный ответ на «кто из команды
            заслужил похвалу/вину», без которого общая награда на команду — слишком грубый сигнал
            для каждого отдельного агента.
          </li>
          <li>
            <b>MAPPO</b> (<PaperLink url="https://arxiv.org/abs/2103.01955">Yu et al., 2021</PaperLink>)
            — тот же PPO, что и в этом курсе, просто с одним централизованным критиком (видит
            глобальное состояние) на всю команду и обычно общими весами актора между однотипными
            агентами. Статья по названию буквально об этом: «удивительная эффективность» почти
            неизменённого PPO на кооперативных MARL-бенчмарках, где ожидались специализированные
            методы.
          </li>
        </ul>
      </Section>

      <Section title="Value decomposition — для кооперативных команд с дискретными действиями">
        <p>
          Отдельное семейство под конкретный частный случай: команда полностью кооперативна (одна
          общая награда на всех) и действия дискретны. Идея — учить общий Q_tot всей команды, но
          <i> разложенный</i> так, что каждый агент может извлечь из него свой собственный
          Q_i(o_i, a_i) для decentralized-исполнения.
        </p>
        <Formula caption="VDN — самое простое разложение, какое можно придумать; QMIX обобщает его до произвольной монотонной (не обязательно линейной) функции смешивания, обученной отдельной сетью, которая зависит от глобального состояния">
{`VDN  (Sunehag et al., 2017):  Q_tot(o,a) = Σᵢ Qᵢ(oᵢ, aᵢ)
QMIX (Rashid et al., 2018):    Q_tot(o,a) = mixer( Q₁(o₁,a₁), ..., Q_N(o_N,a_N); s_global )
                                где ∂Q_tot/∂Qᵢ ≥ 0 для каждого i  (монотонность)`}
        </Formula>
        <p className="text-[12px]">
          <PaperLink url="https://arxiv.org/abs/1706.05296">VDN</PaperLink> и{' '}
          <PaperLink url="https://arxiv.org/abs/1803.11485">QMIX</PaperLink>. Монотонность в QMIX —
          не произвольное ограничение: она ровно то, что гарантирует «оптимальное совместное
          действие = каждый агент максимизирует свой собственный Qᵢ» (IGM, Individual-Global-Max) —
          без неё decentralized argmax по отдельным Qᵢ не обязательно совпадал бы с настоящим
          максимумом Q_tot. <b>QMIX</b> (через урок после IPPO) — второй реально выбираемый в
          приложении алгоритм на этой карте; VDN как отдельный алгоритм не реализован (QMIX
          строго обобщает его — не нужны оба).
        </p>
      </Section>

      <Callout tone="info" title="Что из этого реально доступно в приложении">
        Из всей этой карты в Дизайнере экспериментов реализованы <b>IPPO</b> (Independent Learning)
        и <b>QMIX</b> (Value decomposition) — оба выбираются как для многокомандных сцен Scene
        Builder, так и для готовых опубликованных бенчмарков категории «MARL» в галерее сред: MPE
        Simple Spread/Adversary/Tag и SISL Pursuit (PettingZoo), Butterfly Knights Archers Zombies
        (до 8 агентов), RWARE (Multi-Robot Warehouse, до 4 роботов), Level-Based Foraging (до
        4 агентов) и — через SMAClite — сами сценарии SMAC (2s3z/3s_vs_5z/MMM2, до 10 агентов).
        IPPO подходит для любых действий (discrete/continuous) и любых наград, но нужны
        2+ команды; QMIX — только для дискретного движения, но команда может быть и одна
        (RWARE/LBForaging/Simple Spread/Pursuit/SMAC — все однокомандные, кооперативные, и именно на
        них были представлены VDN/QMIX в оригинальных статьях, см. следующий урок). Для SMAC(lite)
        QMIX — единственный вариант из всех: у него ситуативное пространство действий (не любая
        атака доступна в любой момент), а маскирование по <code>get_avail_actions()</code> сделано
        только в qmix.py. MADDPG/COMA/VDN/MAPPO здесь — обзор ландшафта MARL для контекста, а не
        выбираемые в приложении алгоритмы.
      </Callout>

      <Callout tone="tip" title="А почему не Dota 2 / настоящий StarCraft II?">
        Dota 2 (OpenAI Five) требует лицензированного игрового клиента и, для результатов уровня
        оригинальной статьи, тысяч GPU-лет обучения — далеко за рамками десктоп-приложения. Настоящий{' '}
        <b>SMAC</b> (StarCraft Multi-Agent Challenge —{' '}
        <PaperLink url="https://arxiv.org/abs/1902.04043">Samvelyan et al., 2019</PaperLink>, QMIX'а
        собственный бенчмарк) в теории ближе — не нужно обучать десятки тысяч часов, — но требует сам
        игровой клиент StarCraft II (~30 ГБ, headless-сборка для Linux от Blizzard) и карты, что не
        устанавливается одним <code>pip install</code>. Поэтому в галерее сред — <b>SMAClite</b> (
        <PaperLink url="https://arxiv.org/abs/2305.05566">Michalski et al., 2023</PaperLink>) —
        почти чистый Python/Numpy-реимплемент того же SMAC (тот же API, та же награда — авторы
        проверяли transfer learning между версиями), без самой игры. MPE/SISL/Butterfly, RWARE,
        Level-Based Foraging и SMAClite — те же самые «классические» бенчмарки, на которых
        оценивались MADDPG/QMIX/VDN/MAPPO в их собственных статьях (и на которых до сих пор
        сравнивают новые MARL-алгоритмы — см.{' '}
        <PaperLink url="https://arxiv.org/abs/2006.07869">Papoudakis et al., 2021 (EPyMARL)</PaperLink>),
        устанавливаются чистым Python-пакетом (без бинарников движка) и проходят целый эпизод на
        CPU меньше чем за секунду (SMAClite чуть тяжелее остальных — до ~40 шагов/сек на MMM2/10
        агентов, из-за расчёта коллизий движения).
      </Callout>

      <Callout tone="tip" title="Дальше — как именно устроены IPPO и QMIX">
        Следующие два урока разбирают оба реализованных в приложении MARL-алгоритма — Independent
        PPO и QMIX по командам Scene Builder — на уровне того, что происходит в коде на каждом шаге
        rollout'а/обновления.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 7g. IPPO — Independent PPO (Multi-agent)
// ---------------------------------------------------------------------------

const ippo: Lesson = {
  id: 'ippo',
  title: 'IPPO — Independent PPO',
  tagline: 'По одной полностью независимой копии PPO на каждую команду в Scene Builder — простейший базовый рецепт MARL',
  icon: Users,
  group: 'Multi-agent',
  badges: ['Multi-agent', 'On-policy', 'Только Scene Builder'],
  content: (
    <>
      <Section title="Independent Learning — самый простой рецепт MARL">
        <p>
          IPPO не вводит ничего принципиально нового поверх PPO (см. соответствующий урок) — он
          запускает <b>по одной полностью отдельной</b> сети/оптимизатору/rollout-буферу PPO на
          каждую распознанную team в сцене Scene Builder (team_ids). Команды не делят ни критика, ни
          градиенты, ни явный канал общения — они взаимодействуют только через общую динамику среды
          и правила награды сцены (tag/team_shared_reward). Это стандартный MARL-бейзлайн
          «Independent Learning» (<PaperLink url="https://arxiv.org/abs/2011.09533">de Witt et al., 2020</PaperLink>)
          — и естественная первая MARL-реализация, раз обычный PPO уже умеет всё, что нужно одной
          команде (см. прошлый урок про карту MARL-алгоритмов — почему это не «слабая» точка на
          шкале, а вполне конкурентоспособный выбор).
        </p>
        <FlowRow
          items={[
            { icon: Users, title: 'Team A', detail: 'свой ActorCriticNet + PPO' },
            { icon: GitBranch, title: 'Общая среда', detail: 'динамика + правила награды' },
            { icon: Users, title: 'Team B', detail: 'свой ActorCriticNet + PPO' },
          ]}
        />
      </Section>

      <Section title="Один общий шаг среды, N независимых обновлений политики">
        <p>
          На каждом шаге среды каждая команда действует своей собственной политикой по своему
          «срезу» агентов сцены; все агенты всех команд одновременно шагают в среде одним
          env.step(). После n_steps шагов rollout'а каждая команда обучается отдельно — тот же
          clipped surrogate objective + GAE, что у обычного PPO, просто N раз, по одному на команду.
        </p>
      </Section>

      <Callout tone="warning" title="Ограничение применимости: только среды с 2+ командами">
        Требует, чтобы среда сама разделяла агентов на 2+ team (атрибут `team_ids`) — обычная
        одноагентная Gym-среда или однокомандная сцена/бенчмарк (например MPE Simple Spread, SISL
        Pursuit, RWARE, Level-Based Foraging, SMAC(lite) — все полностью кооперативные, одна
        команда) сюда не подходят, для них выбирается обычный PPO или, если движение дискретно,
        QMIX (следующий урок; в отличие от IPPO, ему одна команда вполне подходит — а для
        SMAC(lite), с его ситуативным пространством действий, QMIX и вовсе единственный вариант).
        Реализуют многокомандный
        `team_ids` многокомандные сцены Scene Builder (`SceneMultiAgentEnv`) и PettingZoo-бенчмарки
        с несколькими ролями (Simple Adversary, Simple Tag, Knights Archers Zombies;
        `PettingZooVectorEnv`).
      </Callout>

      <Callout tone="good" title="Когда использовать">
        Любая многокомандная сцена (конкуренция, погоня, командная координация) в Scene Builder или
        MARL-бенчмарк PettingZoo — Independent Learning удивительно конкурентоспособен даже против
        намного более сложных MARL-схем на многих задачах (см.{' '}
        <PaperLink url="https://arxiv.org/abs/2011.09533">de Witt et al., 2020</PaperLink>). Работает
        и для continuous-движения, и для любых наград (в отличие от QMIX, следующий урок), так что
        это разумный дефолт, если не уверены, какой из двух брать.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 7h. QMIX — Value decomposition (Multi-agent)
// ---------------------------------------------------------------------------

const qmix: Lesson = {
  id: 'qmix',
  title: 'QMIX — Monotonic Value Decomposition',
  tagline: 'Общий Q_tot команды из индивидуальных Qᵢ агентов через монотонную mixing-сеть — кооперация вместо независимости',
  icon: Combine,
  group: 'Multi-agent',
  badges: ['Multi-agent', 'Off-policy', 'Value decomposition', 'Только discrete + Scene Builder'],
  content: (
    <>
      <Section title="Идея: общая Q-сеть на команду + монотонный mixer">
        <p>
          QMIX (<PaperLink url="https://arxiv.org/abs/1803.11485">Rashid et al., 2018</PaperLink>) —
          второй MARL-алгоритм в приложении, альтернатива IPPO для тех же многокомандных сцен Scene
          Builder, но с принципиально другим устройством: вместо policy gradient на каждую команду
          отдельно, QMIX учит одну <b>общую Q-сеть</b> Qᵢ(oᵢ, a) на всех агентов команды (parameter
          sharing, как у IPPO внутри команды) и одну <b>mixing-сеть</b>, которая комбинирует
          выбранные Q-значения всех агентов команды в общее Q_tot — обучается это всё как обычный
          off-policy DQN (replay buffer + target-сеть), просто на командных, а не индивидуальных
          переходах.
        </p>
        <Formula caption="Игрушечный пример на команде из двух агентов — mixer монотонен по каждому входному Qᵢ">
{`Q₁(o₁, a₁), Q₂(o₂, a₂)  →  mixer(Q₁, Q₂; s_global)  →  Q_tot
                              (веса mixer'а неотрицательны ⇒ ∂Q_tot/∂Qᵢ ≥ 0)`}
        </Formula>
      </Section>

      <Section title="Mixing-сеть: гиперсеть с неотрицательными весами">
        <p>
          Mixer — не обычный MLP над Q-значениями, а <b>гиперсеть</b>: маленькая сеть, которая по
          глобальному состоянию s предсказывает <i>веса</i> двухслойного MLP-микшера (`w1`, `b1`,
          `w2`, `b2`), а не сами активации. `w1`/`w2` пропускаются через `abs()` — это и есть тот
          самый монотонности-по-построению трюк из предыдущего урока: раз веса не могут быть
          отрицательными, увеличение любого входного Qᵢ никогда не может уменьшить Q_tot.
        </p>
        <Callout tone="info" title="Глобальное состояние s — в этом приложении">
          В классическом QMIX (например, StarCraft Multi-Agent Challenge) s — отдельный сигнал,
          который среда явно предоставляет поверх локальных наблюдений. Сцены Scene Builder такого
          не дают, поэтому s здесь — просто конкатенация локальных наблюдений <i>всех</i> агентов
          команды (то, что команда и так целиком наблюдает) — валидное и распространённое
          упрощение, когда отдельного «state»-канала у среды нет.
        </Callout>
      </Section>

      <Section title="Обучение: TD-цель на команду, как в DQN">
        <p>
          Каждый реальный шаг среды даёт одну <b>командную</b> транзакцию на каждую команду:
          наблюдения/действия всех её агентов, плюс одна общая награда (среднее по наградам
          агентов команды — совпадает с team_shared_reward, если он включён в правилах сцены) и
          один флаг done (транзакция терминальна, если хоть один агент команды завершил эпизод в
          этот шаг). TD-цель считается через <i>target</i>-копии Q-сети и mixer'а — greedy-действие
          каждого агента по своей Q-сети, затем те же Q-значения через target mixer:
        </p>
        <Formula caption="Стандартная DQN-цель, просто на Q_tot команды вместо скалярного Q одного агента">
{`y = r_команда + γ · (1 - done) · Q_tot_target(s', argmax_a₁ Q₁'(o₁',a), ..., argmax_a_N Q_N'(o_N',a))
loss = MSE( Q_tot(s, a₁, ..., a_N),  y )`}
        </Formula>
        <p>
          Разведка — обычный ε-greedy на каждого агента отдельно (общее затухающее по всему
          прогону ε, как у DQN), не гиперсеть/mixer.
        </p>
      </Section>

      <Callout tone="warning" title="Ограничения: только дискретное движение, 2+ агента (не обязательно 2+ команды)">
        Требует среду с 2+ агентами (team_ids), но, в отличие от IPPO, необязательно 2+ *команд* —
        одна команда из 2+ агентов подходит тоже (см. следующий callout, почему это не бесполезный
        частный случай) — и, дополнительно, только discrete4/discrete8 движение (не continuous):
        value decomposition по конструкции раскладывает <i>дискретный</i> совместный Q по агентам,
        для непрерывных действий нужен был бы другой механизм (например MADDPG-подобный
        детерминированный actor, см. прошлый урок).
      </Callout>

      <Callout tone="info" title="Почему одна команда — это не «недо-IPPO», а собственный классический случай QMIX">
        IPPO на одной команде вырождается в бесполезно раздробленную копию обычного PPO — делить
        нечего, конкурента нет. QMIX на одной команде — совсем другое дело: это ровно тот сценарий,
        на котором были представлены и VDN, и сам QMIX (полностью кооперативная команда, общая
        награда, разложение общего Q_tot по агентам ради явного вознаграждения за координацию) —
        поэтому в приложении QMIX выбирается и для однокомандных, чисто кооперативных бенчмарков:
        MPE Simple Spread, SISL Pursuit, RWARE (Multi-Robot Warehouse), Level-Based Foraging и
        SMAC(lite) (2s3z/3s_vs_5z/MMM2 — сам оригинальный бенчмарк статьи QMIX) — а IPPO для них не
        показывается (2+ команды физически нет).
      </Callout>

      <Callout tone="info" title="SMAC(lite): маскирование действий — единственный случай, где QMIX не «любой из вариантов»">
        Во всех остальных built-in средах любое из discrete4/discrete8-действий всегда легально —
        случайное ε-исследование просто сэмплирует из полного пространства действий. В SMAC(lite)
        это не так: действие «атаковать юнита k» валидно только пока юнит k жив и в радиусе атаки,
        а у части ролей (например, Medivac в MMM2 — только лечит, не атакует) в принципе меньше
        действий, чем у остальных. Среда сообщает это через <code>get_avail_actions()</code>{' '}
        (булева маска на каждый шаг) — и только qmix.py её читает: ε-исследование сэмплирует
        только легальные действия, greedy-argmax маскирует нелегальные значением −∞ до максимума,
        а Double-DQN target маскирует argmax следующего состояния той же маской. Без этого — не
        баг, а просто нечестное обучение: адаптер среды (`TupleMarlVectorEnv._sanitize_actions`)
        подставляет любое нелегальное действие первым легальным вместо ошибки, так что запуск
        <b> dqn</b>/<b>ppo</b> на SMAC(lite) не крашится, но большую часть шагов агент получает не
        то действие, которое «выбрал» — именно поэтому в списке совместимых алгоритмов для
        <code> marlgym:smac_*</code> — только <b>qmix</b>.
      </Callout>

      <Callout tone="good" title="Когда предпочесть QMIX, а не IPPO">
        Когда движение дискретно и команда действительно кооперативна — особенно с 3+ агентами в
        одной команде и включённым team_shared_reward (шаблоны «Стая против жертв» и «Командная
        битва 3×3» в Scene Builder, или бенчмарки PettingZoo MPE Simple Adversary/Simple Tag/Knights
        Archers Zombies, RWARE, Level-Based Foraging, SMAC(lite)), где общий Q_tot явно поощряет координацию, а
        не просто параллельно обучает N одинаковых независимых policy. Для continuous-движения или
        когда команды соревнуются друг с другом, а не координируются внутри себя, IPPO — более
        общий выбор.
      </Callout>

      <Callout tone="warning" title="RWARE/LBForaging: reward_mean=0.0 на протяжении миллионов шагов — это норма">
        В отличие от MPE (плотная посреходовая награда), RWARE и, в меньшей степени, Level-Based
        Foraging — среды с крайне разреженной наградой (только за успешную доставку стеллажа /
        сбор еды нужного уровня). В литературе (EPyMARL, Papoudakis et al. 2021; расширенный
        бенчмарк AAMAS'25, arXiv:2502.04773) QMIX тренируют на LBF ~10 млн шагов, а на RWARE —
        <b> 30-40 млн</b>; независимые репродукции отдельно отмечают, что с дефолтными
        гиперпараметрами reward остаётся ровно 0.0 даже после 2-10 млн шагов, и первая ненулевая
        награда появляется лишь ближе к отметке ~5 млн (при удлинённом epsilon-anneal). Если запуск
        QMIX на marlgym:rware_* показывает <code>episode_reward_mean = 0</code> и стабильный (не
        растущий) loss — это ожидаемая форма кривой обучения на этой среде, а не расходимость
        алгоритма: сравнивайте с loss/Q_tot (должны быть стабильны), а не только с reward. Чтобы
        увидеть первый сигнал быстрее, увеличьте <code>buffer_size</code> (~100-200k),
        <code> batch_size</code> (~128-256) и <code>exploration_fraction</code> (~0.6-0.8) в
        конструкторе экспериментов, либо начните с более простого rware_tiny_2ag.
      </Callout>

      <Callout tone="warning" title="SMAC(lite): закладывайте время на прогон, не только шаги">
        В отличие от RWARE/LBForaging (проблема — миллионы шагов при разреженной награде),
        SMAC(lite) — плотная посреходовая награда (урон/убийства/победа), сходится в пределах
        рекомендованного бюджета шагов (2 млн на 2s3z, 5 млн на 3s_vs_5z, 10 млн на MMM2 — тиры
        Easy/Hard/Super Hard из оригинальной статьи SMAC). Узкое место здесь — throughput
        симуляции, не сходимость: numpy-реализация коллизий SMAClite даёт ~130-320 шагов/сек на
        2s3z/3s_vs_5z, но лишь ~40 на MMM2/10 агентах (больше юнитов → дороже расчёт избегания
        столкновений) — планируйте прогон по времени, а не только по числу шагов, особенно на MMM2.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 8. Память (LSTM/GRU)
// ---------------------------------------------------------------------------

const memory: Lesson = {
  id: 'memory',
  title: 'Память: LSTM и GRU',
  tagline: 'Когда одного наблюдения недостаточно, чтобы принять решение',
  icon: History,
  group: 'Продвинутое',
  badges: ['LSTM/GRU', 'PPO · A2C · DQN · Rainbow DQN'],
  content: (
    <>
      <Section title="Зачем: feed-forward сеть видит только «сейчас»">
        <p>
          Обычная (feed-forward) сеть в PPO/A2C/DQN/Rainbow DQN считает Q(s,a) или π(a|s) как функцию
          только текущего наблюдения — у неё физически нет доступа к тому, что было раньше. Если
          наблюдение <b>не</b> содержит всей нужной информации (частичная наблюдаемость — POMDP,
          подробнее в следующем уроке), такая сеть упирается в потолок качества, который не преодолеть
          никакой доучкой — ей просто не хватает входных данных.
        </p>
      </Section>

      <Section title="Куда встаёт рекуррентный блок">
        <p>
          Гиперпараметр «Память (рекуррентность)» вставляет LSTM или GRU между извлечением признаков
          и головами действия/ценности (или Q/advantage). Скрытое состояние hₜ переносится с шага на
          шаг внутри эпизода и сбрасывается на границе нового эпизода (episode_start=True).
        </p>
        <MemoryTimelineDiagram />
        <FlowRow
          items={[
            { icon: Eye, title: 'Наблюдение oₜ', detail: 'может быть неполным' },
            { icon: Brain, title: 'Feature extractor', detail: 'тот же, что и без памяти' },
            { icon: History, title: 'LSTM / GRU', detail: 'hₜ = f(hₜ₋₁, признаки)' },
            { icon: Target, title: 'Головы', detail: 'action / Q / value — видят hₜ, а не только oₜ' },
          ]}
        />
      </Section>

      <Section title="Как это обучается: truncated BPTT">
        <p>
          Градиент через рекуррентную сеть нужно распространять по времени (backpropagation through
          time) — но не через весь эпизод целиком, это дорого и нестабильно. Обучение режется на
          окна длиной memory_seq_len:
        </p>
        <ul className="ml-4 list-disc space-y-1">
          <li>
            <b>On-policy (PPO/A2C)</b> — весь rollout режется на чанки по memory_seq_len{' '}
            <i>в исходном порядке</i> (перемешивание сломало бы саму идею памяти), hidden state
            переносится и «отрезается» (detach) на границе чанков.
          </li>
          <li>
            <b>Off-policy (DQN/Rainbow DQN)</b> — реплей-буфер хранит целые эпизоды и сэмплирует окна
            фиксированной длины с нулевым начальным hidden state (см.{' '}
            <PaperLink url="https://arxiv.org/abs/1507.06527">Hausknecht &amp; Stone, 2015</PaperLink> — DRQN).
          </li>
        </ul>
      </Section>

      <Section title="Гиперпараметры">
        <ul className="ml-4 list-disc space-y-1">
          <li><b>memory_type</b> — Нет / LSTM / GRU. GRU проще (меньше параметров/чуть быстрее), LSTM обычно немного стабильнее на длинных зависимостях.</li>
          <li><b>memory_hidden_size</b> — «объём» памяти. Больше — можно запомнить больше, но больше параметров и риск переобучения на маленьких средах.</li>
          <li><b>memory_num_layers</b> — глубина рекуррентного блока (почти всегда 1 достаточно).</li>
          <li><b>memory_seq_len</b> — длина окна truncated-BPTT. Слишком маленькое — сеть не увидит зависимость, которая длиннее окна; слишком большое — дороже и шумнее обучение.</li>
        </ul>
      </Section>

      <Callout tone="warning" title="Границы применимости">
        SAC и AlphaZero памяти не поддерживают — continuous control обычно видит полное состояние, а
        настольные игры (AlphaZero) видят всю доску целиком, так что мотивация намного слабее. Своя
        (hand-designed) архитектура из конструктора сетей тоже игнорирует переключатель памяти — там
        пока нет LSTM/GRU как типа слоя.
      </Callout>

      <Callout tone="good" title="Как убедиться, что память реально помогает">
        Не верьте на слово — включите память на среде, где без неё физически невозможно (следующий
        урок), и сравните. На Memory Corridor обычный DQN даёт ~0.55 средней награды за эпизод (уровень
        случайного угадывания), DQN с memory_type=LSTM — 1.0.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 9. POMDP-среды
// ---------------------------------------------------------------------------

const pomdp: Lesson = {
  id: 'pomdp',
  title: 'POMDP-среды: где память реально нужна',
  tagline: 'Практика: 20 сред, специально устроенных так, чтобы feed-forward политика упиралась в потолок',
  icon: Eye,
  group: 'Продвинутое',
  badges: ['Practice', 'Tiger', 'Hallway', 'RockSample', 'LaserTag', 'Visual Memory Maze'],
  content: (
    <>
      <Section title="POMDP — Partially Observable MDP">
        <p>
          В обычном MDP (первый урок курса) текущее состояние содержит всё нужное для решения. В
          POMDP агент видит только <b>часть</b> состояния — недостающая часть иногда восстанавливается
          из истории наблюдений, а для этого и нужна память (предыдущий урок).
        </p>
        <PomdpCompareDiagram
          full={['позиция тележки x', 'скорость тележки ẋ', 'угол шеста θ', 'угловая скорость θ̇']}
          observed={['позиция тележки x', 'угол шеста θ']}
        />
        <p className="text-[12px]">PO-CartPole — то же самое, что CartPole, но скорости убраны из наблюдения.</p>
      </Section>

      <Section title="Шесть способов сделать среду частично наблюдаемой">
        <ul className="ml-4 list-disc space-y-1">
          <li>
            <b>Маскирование скорости</b> (PO-CartPole, PO-Pendulum, PO-Mountain Car, PO-Acrobot,
            PO-Lunar Lander) — убираем производные величины из наблюдения. Стандартный академический
            приём (RLlib's <code className="mx-1 font-mono text-[11px]">StatelessCartPole</code>,{' '}
            <PaperLink url="https://arxiv.org/abs/1512.04455">Heess et al., 2015</PaperLink>).
            PO-Lunar Lander маскирует сразу все три скорости — самый требовательный к памяти
            вариант из пятёрки.
          </li>
          <li>
            <b>Flickering-наблюдение</b> (Flickering CartPole, Flickering Pong) — с вероятностью 50%
            на каждом шаге всё наблюдение целиком гасится в ноль. Рецепт статьи{' '}
            <PaperLink url="https://arxiv.org/abs/1507.06527">Hausknecht &amp; Stone, 2015</PaperLink>,
            которая ввела DRQN — Flickering Pong воспроизводит их эксперимент буквально.
          </li>
          <li>
            <b>Memory Corridor / Memory Corridor (long)</b> — не обёртка вокруг существующей среды, а
            отдельная задача: один бит-подсказка виден только на первом шаге, дальше 7 (или 19 — в
            «long»-версии) «слепых» шагов, награда — только если финальное действие совпало с давно
            пропавшей подсказкой. «Long»-вариант показывает, что решает не любая память, а память
            достаточной ёмкости.
          </li>
          <li>
            <b>Repeat Previous (N-back)</b> — задача другого типа: каждый шаг показывается новый
            случайный символ, а верный ответ — символ, показанный ровно 3 шага назад. Память нужна
            не один раз в конце эпизода, а <i>непрерывно</i>, на каждом шаге — классический
            delayed-match-to-sample тест из литературы по памяти.
          </li>
          <li>
            <b>RockSample(7,8)</b> — классический масштабируемый бенчмарк из литературы по POMDP-
            <i>планированию</i> (<PaperLink url="https://scholar.google.com/scholar?q=Smith+Simmons+Heuristic+Search+Value+Iteration+for+POMDPs+2004">Smith &amp; Simmons, 2004</PaperLink>).
            Позиции 8 камней на сетке 7×7 известны, а
            их качество скрыто и проверяется только шумным сенсором, точность которого падает с
            расстоянием. Тут память нужна не для «вспомнить один сигнал», а чтобы копить несколько
            зашумлённых показаний в уверенное мнение о конкретном камне.
          </li>
          <li>
            <b>Visual Memory Maze</b> — картиночный аналог флагманского бенчмарка{' '}
            <PaperLink url="https://arxiv.org/abs/2306.13831">MiniGrid</PaperLink>'s
            <code className="mx-1 font-mono text-[11px]">MemoryEnv</code>, на котором в документации
            <code className="mx-1 font-mono text-[11px]">sb3-contrib</code>'s RecurrentPPO показывают,
            зачем вообще нужна память. Агент видит лишь окно 5×5 клеток вокруг себя, а не всю карту —
            здесь частичная наблюдаемость не искусственная маска, а настоящее ограниченное поле зрения.
          </li>
        </ul>
      </Section>

      <Section title="Ещё восемь — прямиком из классики POMDP-планирования и современных memory-бенчмарков">
        <p>
          Все среды выше решают одну и ту же задачу — «вспомнить факт, который был виден раньше» — просто
          на разных таймингах. Следующие восемь специально выбраны так, чтобы каждая требовала памяти
          по-своему <i>другому</i> поводу.
        </p>
        <ul className="ml-4 list-disc space-y-1">
          <li>
            <b>Tiger</b> — самый канонический toy-POMDP в литературе (
            <PaperLink url="https://scholar.google.com/scholar?q=Kaelbling+Littman+Cassandra+Planning+and+acting+in+partially+observable+stochastic+domains+1998">Kaelbling, Littman &amp; Cassandra, 1998</PaperLink>):
            тигр за одной из двух дверей, можно слушать (85% точности, но платно) или открыть дверь.
            Одного «слушания» недостаточно — нужно копить несколько независимых зашумлённых подсказок в
            уверенное мнение, прежде чем рисковать.
          </li>
          <li>
            <b>Heaven/Hell</b> — впервые в этом курсе память нужна не для того, чтобы <i>удержать</i>
            подсказку, а чтобы <i>решить пойти и посмотреть</i> её: указатель стоит в тупике сбоку от
            главного пути. Без похода к указателю агент обречён угадывать на развилке 50/50.
          </li>
          <li>
            <b>Hallway</b> — оригинальный бенчмарк на <i>state aliasing</i> (
            <PaperLink url="https://scholar.google.com/scholar?q=Littman+Cassandra+Kaelbling+Learning+policies+for+partially+observable+environments+scalable+methods+1995">Littman, Cassandra &amp; Kaelbling, 1995</PaperLink>):
            две разные комнаты дают идентичное показание сенсора, но требуют
            противоположных действий. Любая фиксированная реакция на одинаковое наблюдение гарантированно
            ошибается в одной из них — разрешить это может только память о пройденном пути.
          </li>
          <li>
            <b>Battleship</b> и <b>Minesweeper (POMDP)</b> — классические настольные/логические игры,
            переформулированные так, что наблюдение — это только результат самого последнего выстрела или
            открытой клетки, а не карта всего поля целиком. Без памяти о всех прошлых действиях агент не
            может избежать повторов, за которые здесь штрафуют.
          </li>
          <li>
            <b>Concentration</b> («найди пару») — один из диагностических тестов памяти в бенчмарке{' '}
            <PaperLink url="https://arxiv.org/abs/2303.01859">POPGym</PaperLink>: значение карты
            видно только в момент её переворота, всё остальное время — «закрыто, неизвестно».
          </li>
          <li>
            <b>LaserTag</b> — впервые скрытый факт не статичен, а <i>движется сам</i>: противник виден
            только в узкой прямой видимости. Нужна не разовая память, а постоянно обновляемое мнение о
            том, где цель находится сейчас.
          </li>
          <li>
            <b>Active T-Maze / Active T-Maze (long)</b> — активная версия Memory Corridor (
            <PaperLink url="https://arxiv.org/abs/2307.03864">Ni et al., 2023</PaperLink>):
            подсказка не показывается сама, её нужно запросить действием «посмотреть» в самом
            начале одностороннего коридора. Проверяет не удержание информации, а то, научился ли агент
            понимать, <i>когда стоит её запросить</i>.
          </li>
        </ul>
      </Section>

      <Section title="Попробуйте сами">
        <p>
          В Дизайнере экспериментов откройте среду <b>Memory Corridor</b> или <b>Repeat Previous
          (N-back)</b>, возьмите DQN, запустите два коротких эксперимента: memory_type = «Нет» и
          memory_type = «LSTM». Разница видна не по интуиции, а прямо в цифрах Мониторинга — на Memory
          Corridor награда за эпизод у обычной сети колеблется около 0.5 (случайное угадывание), у
          LSTM-версии быстро уходит к 1.0; на Repeat Previous точность держится около 0.25 (1 из 4
          символов вслепую) против ~0.96 с LSTM — эти числа получены прямо в этом репозитории.
        </p>
      </Section>

      <Callout tone="info" title="Все 20 сред — в галерее «Среды»">
        Категория «POMDP (нужна память)» на странице Среды: PO-CartPole, PO-Pendulum, PO-Mountain Car,
        PO-Acrobot, PO-Lunar Lander, Flickering CartPole, Flickering Pong, Memory Corridor, Memory
        Corridor (long), Repeat Previous (N-back), RockSample(7,8), Visual Memory Maze, Tiger,
        Heaven/Hell, Hallway, Battleship, Minesweeper (POMDP), Concentration, LaserTag, Active T-Maze и
        Active T-Maze (long). Совместимы с любым алгоритмом того же типа действия
        (дискретный/continuous) — эффект памяти нагляднее всего на DQN/Rainbow DQN/PPO/A2C с включённым
        memory_type.
      </Callout>
    </>
  ),
}

const exploration: Lesson = {
  id: 'exploration',
  title: 'Exploration: NoisyNet и RND',
  tagline: 'Два разных вопроса: как разнообразить действия и как награждать агента за настоящую новизну',
  icon: Sparkles,
  group: 'Продвинутое',
  badges: ['ε-greedy', 'NoisyNet', 'RND', 'Sparse rewards'],
  content: (
    <>
      <Section title="Action exploration и intrinsic motivation — не одно и то же">
        <p>
          <b>ε-greedy и NoisyNet</b> определяют, как агент отклоняется от текущего argmax Q.
          <b> RND</b> меняет обучающий сигнал: незнакомое состояние временно получает дополнительную
          intrinsic-награду. Поэтому в Дизайнере это два независимых переключателя: можно сочетать
          ε-greedy + RND или NoisyNet + RND.
        </p>
      </Section>

      <Section title="ε-greedy: простой и надёжный baseline">
        <p>
          С вероятностью ε выбирается случайное действие, иначе — argmax Q(s,a). Линейный schedule
          постепенно переводит агента от разведки к эксплуатации.
        </p>
        <Formula>{`ε(t) = ε₀ + min(1, t / T_decay) · (ε_min − ε₀)`}</Formula>
        <EpsilonDecayChart />
      </Section>

      <Section title="NoisyNet: обучаемый шум в параметрах">
        <p>
          Вместо независимой случайности на каждом шаге шум добавляется прямо в веса Q-головы
          (<PaperLink url="https://arxiv.org/abs/1706.10295">Fortunato et al., 2018</PaperLink>).
          Его масштаб обучается градиентом, поэтому действия меняются согласованно в зависимости
          от состояния. При оценке шум отключается и используются средние веса.
        </p>
        <Formula>{`W = μ_W + σ_W ⊙ ε_W
b = μ_b + σ_b ⊙ ε_b`}</Formula>
        <Callout tone="tip" title="Когда выбирать NoisyNet">
          Для DQN/Rainbow на дискретных действиях, когда ε-greedy слишком часто делает бессвязные
          случайные шаги. NoisyNet несовместим с произвольной NetworkSpec, потому что должен заменить
          конкретные Linear-слои Q-головы.
        </Callout>
      </Section>

      <Section title="RND: любопытство через ошибку предсказания">
        <p>
          Случайная target-сеть навсегда заморожена
          (<PaperLink url="https://arxiv.org/abs/1810.12894">Burda et al., 2018 — Random Network Distillation</PaperLink>).
          Predictor пытается повторить её выход. Для часто встречавшихся состояний ошибка
          становится маленькой; для новых — большой и превращается во временный бонус новизны.
        </p>
        <Formula>{`r_intrinsic(s') = || f_target(s') − f_predictor(s') ||²
r_total = r_extrinsic + β · normalize(r_intrinsic)`}</Formula>
        <AlgorithmDiagram
          algorithmId="dqn"
          kind="gym"
          hyperparams={{
            action_exploration: 1, noisy_sigma0: 0.5,
            intrinsic_exploration: 1, rnd_bonus_coef: 0.1,
            buffer_size: 50_000, batch_size: 64, gamma: 0.99,
          }}
          show={{ network: false }}
        />
        <Callout tone="warning" title="Смотрите на extrinsic reward отдельно">
          Большой total reward может состоять почти целиком из curiosity-бонуса. Монитор поэтому
          рисует extrinsic и intrinsic линии отдельно: реальный успех среды нельзя подменять
          внутренней наградой RND.
        </Callout>
      </Section>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 9b. Evolution Strategies (ES)
// ---------------------------------------------------------------------------

const es: Lesson = {
  id: 'es',
  title: 'Evolution Strategies (ES)',
  tagline: 'Полностью без backprop через среду: чёрный ящик, зашумлённые копии политики, оценивается только итоговая отдача',
  icon: Dna,
  group: 'Продвинутое',
  badges: ['Gradient-free', 'Чёрный ящик', 'Дискретные и continuous'],
  content: (
    <>
      <AlgorithmDiagram
        algorithmId="es"
        kind="gym"
        hyperparams={{ population_size: 32, sigma: 0.1, learning_rate: 0.02, episodes_per_eval: 1 }}
        show={{ network: false }}
      />

      <Section title="Принципиально другой подход: параметры сети — это то, что «мутирует»">
        <p>
          Каждый другой алгоритм в этом курсе дифференцируемо считает loss из отдельных переходов и
          делает backward(). ES — нет: он воспринимает <b>весь эпизод целиком</b> как чёрный ящик —
          функцию отдачи от <i>параметров</i> политики, и оценивает градиент по ним методом конечных
          разностей на популяции случайно зашумлённых копий
          (<PaperLink url="https://arxiv.org/abs/1703.03864">Salimans et al., 2017</PaperLink> —
          «OpenAI-ES»). Ни replay-буфера, ни функции ценности, ни единого backward-прохода через среду —
          а значит, разреженность/недифференцируемость/длина награды среды совершенно не важны,
          в отличие от всех остальных алгоритмов курса — цена — много полных прогонов эпизода на
          одно обновление.
        </p>
      </Section>

      <Section title="Mirrored sampling (antithetic variates)">
        <p>
          Каждый шум ε оценивается сразу в обе стороны — θ+σε и θ−σε — что снижает дисперсию оценки
          градиента бесплатно: для пары не нужно два независимых сэмпла шума, только один на пару
          лейнов.
        </p>
      </Section>

      <Section title="Fitness shaping через centered ranks">
        <p>
          Сырая отдача эпизода заменяется её <b>рангом</b> в популяции, линейно отмасштабированным в
          [-0.5, 0.5], прежде чем использоваться как вес в оценке градиента — делает обновление
          инвариантным к масштабу награды и устойчивым к одному эпизоду-выбросу, который иначе мог
          бы задавить всю остальную популяцию.
        </p>
        <Formula caption="half = population_size/2 пар (θ+σεᵢ, θ−σεᵢ); итоговый шаг передаётся в Adam как обычный градиент (с обратным знаком — Adam спускается, а нужно подниматься), что даёт адаптивный per-parameter шаг практически бесплатно">
{`rank(i) ∈ [-0.5, 0.5] — линейный ранг отдачи в популяции
grad = (1 / (half · σ)) · Σᵢ ( rank(i) − rank(half+i) ) · εᵢ
θ ← Adam-step(θ, −grad)`}
        </Formula>
      </Section>

      <Section title="Diagonal, а не полная (CMA-ES) ковариация">
        <p>
          Полная ковариационная матрица —{' '}
          <PaperLink url="https://scholar.google.com/scholar?q=Hansen+Ostermeier+Completely+Derandomized+Self-Adaptation+in+Evolution+Strategies+2001">CMA-ES, Hansen &amp; Ostermeier, 2001</PaperLink>{' '}
          — стоит O(d²) по памяти и вычислениям — непрактично для сети даже с несколькими тысячами
          параметров. ES здесь использует диагональную (separable) версию — масштабируется на сети
          любого размера за счёт того, что не моделирует корреляции между параметрами.
        </p>
      </Section>

      <Callout tone="tip" title="Параллелизм здесь другой, чем у всех остальных алгоритмов">
        У каждого другого алгоритма training.num_envs лейнов делят <i>одну и ту же</i> сеть —
        разные только наблюдения. У ES каждый лейн — это по-настоящему <i>другой</i> набор весов
        (свой кандидат популяции), поэтому параллелизм — по популяции, а не по времени: до
        num_envs кандидатов одновременно проходят свой полный эпизод, ограничивая время на группу
        самым медленным её членом, а не суммой всех.
      </Callout>

      <Callout tone="good" title="Когда использовать">
        Разреженная или явно недифференцируемая награда, или когда backprop через среду в принципе
        невозможен/не нужен — простота реализации оправдывает избыточные по сравнению с
        PPO/SAC требования к числу эпизодов на одно обновление.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 9c. Обёртки среды: frame stacking, frame skip, нормализация
// ---------------------------------------------------------------------------

const wrappers: Lesson = {
  id: 'wrappers',
  title: 'Обёртки среды: frame stacking, frame skip, нормализация',
  tagline: 'Небольшие, независимые от алгоритма преобразования наблюдения/награды — узлы графа в Дизайнере экспериментов',
  icon: Layers,
  group: 'Продвинутое',
  badges: ['Wrappers', 'Frame Stack', 'Frame Skip', 'Препроцессинг'],
  content: (
    <>
      <Section title="Обёртка (wrapper) — прозрачная прослойка вокруг среды">
        <p>
          Обёртка оборачивает среду и меняет то, что видит агент (наблюдение), что он получает в
          ответ (награда) или что он может отправить (действие) — сам алгоритм ничего об этом не
          знает, для него это просто «другая среда» с другой формой наблюдения/шкалой награды. В
          Дизайнере экспериментов обёртки — отдельные узлы графа между средой и алгоритмом,
          применяются в том порядке, в котором соединены.
        </p>
      </Section>

      <Section title="Frame Stack — компенсация «памяти одного кадра»">
        <p>
          Рецепт восходит к оригинальной Atari DQN-статье
          (<PaperLink url="https://www.nature.com/articles/nature14236">Mnih et al., 2015</PaperLink> —
          num_stack=4 кадра, frame_skip=4 там же): одно наблюдение часто не содержит скорости/направления движения — по одному кадру
          неподвижного маятника нельзя понять, в какую сторону он падает. Frame Stack склеивает
          последние num_stack наблюдений <b>по каналам</b> ((H,W,C) → (H,W,C·num_stack)), а не по
          новой оси — это принципиально: 4D-форма (num_stack, H, W, C) не проходит проверку «это
          картинка?» в этом приложении и незаметно превращается в вектор, теряя всю
          пространственную структуру для CNN. Для векторных (не картиночных) наблюдений это
          вырождается в «склеить последние N векторов признаков» — тоже валидная история, просто без
          CNN-семантики.
        </p>
        <FlowRow
          items={[
            { icon: Eye, title: 'oₜ₋₃ .. oₜ', detail: 'последние num_stack кадров' },
            { icon: Layers, title: 'Frame Stack', detail: 'склейка по оси каналов' },
            { icon: Brain, title: 'CNN/MLP', detail: 'видит движение, не только позицию' },
          ]}
        />
        <Callout tone="info" title="Дешёвая альтернатива памяти (LSTM/GRU)">
          Frame Stack и рекуррентная память (отдельный урок) решают похожую проблему по-разному:
          Frame Stack даёт сети фиксированное окно последних кадров без всякой рекуррентности —
          дешевле и проще, но не помогает, если нужная информация была видна раньше, чем
          num_stack шагов назад (см. Memory Corridor в уроке про POMDP-среды, где нужна именно
          настоящая память).
        </Callout>
      </Section>

      <Section title="Frame Skip — меньше шагов сети на то же «игровое время»">
        <p>
          Повторяет одно и то же действие skip раз подряд, суммирует награду, возвращает только
          последний кадр — вдвое (при skip=2) меньше forward/backward-проходов сети на то же
          количество кадров симуляции. Применяется <i>до</i> Resize/Grayscale в цепочке (ближе к
          самой среде), чтобы те работали один раз на пропущенный блок, а не на каждый сырой кадр.
        </p>
        <Callout tone="info" title="Не Atari-style max-pool двух кадров">
          В отличие от классической Atari DQN-обёртки (берёт попиксельный максимум двух последних
          кадров, чтобы убрать мерцающие спрайты), эта реализация просто берёт последний кадр —
          для «плавных» сред типа CarRacing max-pool двух кадров быстро движущейся машины создаёт
          двойные контуры ровно там, где для точного руления нужна чёткая граница трассы.
        </Callout>
      </Section>

      <Section title="Resize / Grayscale — меньше пикселей на вход CNN">
        <p>
          Уменьшение разрешения (обычно до 64×64) и перевод в оттенки серого — цвет редко несёт
          важную для управления информацию, а количество пикселей на входе CNN напрямую определяет
          стоимость каждого forward/backward-прохода. Рекомендуемая тройка для картиночных сред —
          Frame Skip → Resize → Grayscale, именно в этом порядке (см. CarRacing в галерее сред).
        </p>
      </Section>

      <Section title="Нормализация и обрезка — про масштаб, не про содержимое">
        <ul className="ml-4 list-disc space-y-1">
          <li><b>Normalize Observation</b> — приводит каждую координату наблюдения к скользящим mean=0/std=1; полезно, когда координаты наблюдения в средах измеряются в очень разных единицах/масштабах.</li>
          <li><b>Normalize Reward</b> — то же самое для награды со своим дисконтированным скользящим масштабом; помогает, когда сырая награда сильно различается по величине между шагами/эпизодами.</li>
          <li><b>Clip Reward</b> — жёсткая обрезка награды в [min, max] за шаг; проще, чем нормализация, и полностью убирает влияние редких «выбросов» награды на градиент.</li>
          <li><b>Clip Action</b> — обрезает continuous-действие в границы action space перед тем, как оно доходит до среды (страховка на случай, если сама политика этого не гарантирует).</li>
          <li><b>Time Limit</b> — жёсткий потолок числа шагов в эпизоде; без него эпизод в среде без естественного конца может длиться бесконечно.</li>
        </ul>
      </Section>

      <Callout tone="good" title="Где это настраивается">
        В Дизайнере экспериментов — узлы-обёртки между средой и алгоритмом на графе; каждый тип из
        каталога выше доступен как отдельный узел со своими параметрами. Порядок соединения узлов —
        это и есть порядок применения обёрток к сырой среде.
      </Callout>
    </>
  ),
}

// ---------------------------------------------------------------------------
// 10. Шпаргалка
// ---------------------------------------------------------------------------

interface CheatRow {
  name: string
  kind: string
  policy: string
  actions: string
  memory: string
  when: string
}

const CHEAT_ROWS: CheatRow[] = [
  { name: 'DQN', kind: 'Value-based', policy: 'Off-policy', actions: 'Дискретные', memory: 'да', when: 'Простой бейзлайн на дискретных действиях' },
  { name: 'Rainbow DQN', kind: 'Value-based', policy: 'Off-policy', actions: 'Дискретные', memory: 'да', when: 'То же, что DQN, но почти всегда лучше; +QR-DQN для полного Rainbow' },
  { name: 'A2C', kind: 'Actor-Critic', policy: 'On-policy', actions: 'Discrete + continuous', memory: 'да', when: 'Быстрый дёшевый бейзлайн' },
  { name: 'PPO', kind: 'Actor-Critic', policy: 'On-policy', actions: 'Discrete + continuous', memory: 'да', when: 'Рекомендуемый дефолт почти всегда' },
  { name: 'SAC', kind: 'Actor-Critic', policy: 'Off-policy', actions: 'Только continuous', memory: 'нет', when: 'Continuous-задачи, нужна сэмпл-эффективность' },
  { name: 'DDPG', kind: 'Actor-Critic (детерминир.)', policy: 'Off-policy', actions: 'Только continuous', memory: 'нет', when: 'Простой бейзлайн для сравнения с TD3/SAC' },
  { name: 'TD3', kind: 'Actor-Critic (детерминир.)', policy: 'Off-policy', actions: 'Только continuous', memory: 'нет', when: 'Надёжный дефолт для (почти) детерминированной динамики' },
  { name: 'Evolution Strategies', kind: 'Gradient-free (чёрный ящик)', policy: '—', actions: 'Discrete + continuous', memory: 'нет', when: 'Разреженная/недифференцируемая награда, нужна простота' },
  { name: 'AlphaZero', kind: 'Planning + self-play', policy: '—', actions: 'Дискретные (доска)', memory: 'нет', when: 'Настольные игры с полной информацией' },
  { name: 'Dreamer', kind: 'Model-based (RSSM)', policy: '—', actions: 'Discrete + continuous', memory: 'нет', when: 'Дорогой/медленный env.step() — учим actor-critic в воображении' },
  { name: 'MBPO', kind: 'Model-based (Ensemble) + SAC', policy: 'Off-policy', actions: 'Только continuous', memory: 'нет', when: 'SAC + короткие модельные rollouts для сэмпл-эффективности' },
  { name: 'PETS', kind: 'Model-based (Ensemble) + планирование', policy: '—', actions: 'Только continuous', memory: 'нет', when: 'Прямое CEM-планирование без обучаемой политики' },
  { name: 'World Models (Ha)', kind: 'Model-based (VAE+MDN-RNN) + ES', policy: '—', actions: 'Discrete + continuous', memory: 'нет', when: 'Классический рецепт: сжать → предсказать → крошечный ES-контроллер' },
  { name: 'EfficientZero V2', kind: 'Model-based (MuZero-family) + Gumbel search', policy: '—', actions: 'Discrete + continuous', memory: 'нет', when: 'Планирование бюджетированным поиском на каждом шаге, без готового World Model артефакта' },
  { name: 'IPPO', kind: 'Actor-Critic (multi-agent)', policy: 'On-policy', actions: 'Discrete + continuous', memory: 'да', when: 'Многокомандные сцены Scene Builder или бенчмарки PettingZoo' },
  { name: 'QMIX', kind: 'Value decomposition (multi-agent)', policy: 'Off-policy', actions: 'Только дискретные', memory: 'нет', when: 'Кооперация 2+ агентов (1 команда или больше), дискретное движение; единственный вариант для SMAC(lite)' },
]

const cheatsheet: Lesson = {
  id: 'cheatsheet',
  title: 'Шпаргалка: что выбрать',
  tagline: 'Один экран, который сравнивает все алгоритмы курса по ключевым осям',
  icon: ListChecks,
  group: 'Итог',
  badges: ['Сравнение', 'Итог курса'],
  content: (
    <>
      <Section title="Сравнительная таблица">
        <div className="overflow-x-auto rounded-lg border border-border/70">
          <table className="w-full min-w-[640px] text-left text-[12px]">
            <thead className="bg-muted/50 text-[11px] uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-3 py-2">Алгоритм</th>
                <th className="px-3 py-2">Тип</th>
                <th className="px-3 py-2">On/off-policy</th>
                <th className="px-3 py-2">Действия</th>
                <th className="px-3 py-2">Память</th>
                <th className="px-3 py-2">Когда брать</th>
              </tr>
            </thead>
            <tbody>
              {CHEAT_ROWS.map((row) => (
                <tr key={row.name} className="border-t border-border/50">
                  <td className="px-3 py-2 font-medium">{row.name}</td>
                  <td className="px-3 py-2 text-muted-foreground">{row.kind}</td>
                  <td className="px-3 py-2 text-muted-foreground">{row.policy}</td>
                  <td className="px-3 py-2 text-muted-foreground">{row.actions}</td>
                  <td className="px-3 py-2 text-muted-foreground">{row.memory}</td>
                  <td className="px-3 py-2 text-muted-foreground">{row.when}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Быстрое дерево решений">
        <ul className="ml-4 list-disc space-y-1">
          <li>Настольная игра, полная информация → <b>AlphaZero</b>.</li>
          <li>Continuous-действия, нужна сэмпл-эффективность и стабильная (не строго детерминированная) динамика → <b>SAC</b>.</li>
          <li>Continuous-действия, динамика близка к детерминированной (MuJoCo-locomotion, Car Racing) → <b>TD3</b> (DDPG — для сравнения/учебных целей).</li>
          <li>Continuous или дискретные, нужен простой стабильный дефолт → <b>PPO</b>.</li>
          <li>Дискретные действия, важен off-policy replay → <b>Rainbow DQN</b> (или DQN для простоты); включите <b>distributional (QR-DQN)</b> для полного набора из 6 ингредиентов Rainbow.</li>
          <li>Награда разреженная/недифференцируемая, или хочется вообще без backprop через среду → <b>Evolution Strategies</b>.</li>
          <li>Наблюдение может быть неполным (POMDP) → любой из вышеперечисленных Gym-алгоритмов (кроме ES) + <b>память (LSTM/GRU)</b>.</li>
          <li>env.step() дорогой/медленный, важна сэмпл-эффективность любой ценой → <b>World Models</b>: Dreamer (дискретные/continuous), MBPO или PETS (continuous), World Models (Ha) — см. отдельную группу уроков.</li>
          <li>Многокомандная сцена в Scene Builder или бенчмарк PettingZoo (категория «MARL» в галерее сред) → <b>IPPO</b> (policy gradient, discrete/continuous) или, для дискретного движения и кооперации внутри команды, <b>QMIX</b> (value decomposition).</li>
        </ul>
      </Section>

      <Callout tone="tip" title="Дальше — практика">
        Откройте Дизайнер экспериментов, соберите граф среда → алгоритм → training, взгляните на живую
        схему архитектуры сети (та же, что и в этом курсе) и запустите первый эксперимент. Курс никуда
        не убежит — к нему можно вернуться в любой момент через раздел «Учебный центр» в навигации.
      </Callout>
    </>
  ),
}

export const LESSONS: Lesson[] = [
  fundamentals, dqn, rainbow, a2c, ppo, sac, ddpg, td3, continuousTechniques, alphazero,
  worldModelsOverview, dreamer, mbpoPets, worldModelsHa, efficientzero, unizero, marlOverview, ippo, qmix,
  memory, pomdp, exploration, es, wrappers, cheatsheet,
]
