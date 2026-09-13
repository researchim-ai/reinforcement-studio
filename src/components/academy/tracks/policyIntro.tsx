import { TrendingUp } from 'lucide-react'
import type { Lesson } from '@/components/academy/lessonShared'
import { LessonLink } from '@/components/academy/lessonShared'
import { Callout, Formula, MiniTable, PaperLink, Section, WorkedExample } from '@/components/academy/visuals'

export const reinforce: Lesson = {
  id: 'reinforce',
  title: 'REINFORCE: градиент политики',
  tagline: 'Увеличиваем вероятность действий, после которых возврат был большим. Без Q, зато с дисперсией',
  icon: TrendingUp,
  group: 'Policy-gradient (актор-критик)',
  badges: ['REINFORCE', 'Score function', 'Baseline', 'On-policy'],
  content: (
    <>
      <Section title="Идея без критика">
        <p>
          Value-based методы строят Q и только потом берут argmax. Policy gradient учит π_θ напрямую:
          если после действия a возврат оказался большим, увеличить log π_θ(a|s); если маленьким —
          уменьшить. Теорема градиента политики:{' '}
          <PaperLink url="https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf">
            Sutton et al., 2000
          </PaperLink>
          . REINFORCE — Monte Carlo-версия:{' '}
          <PaperLink url="https://link.springer.com/article/10.1007/BF00992696">Williams, 1992</PaperLink>.
        </p>
        <Formula caption="Gₜ — полный возврат от шага t; без минуса baseline градиент очень шумный">
{`∇J(θ) ≈ E[ ∇ log π_θ(aₜ | sₜ) · Gₜ ]`}
        </Formula>
        <WorkedExample title="Посчитаем: два эпизода без baseline">
          <MiniTable
            headers={['эпизод', 'действие', 'G', 'что делает градиент']}
            rows={[
              ['1', 'налево', '1', 'слегка поднимает π(налево)'],
              ['2', 'направо', '5', 'сильнее поднимает π(направо)'],
            ]}
            caption="Оба G > 0, оба действия усиливаются. «Налево» не штрафуется — оно просто слабее джекпота. Вычтите V(s)≈3: тогда налево получит вес −2, направо +2. Это и есть смысл baseline / advantage."
          />
        </WorkedExample>
      </Section>

      <Section title="Почему это работает интуитивно">
        <p>
          ∇ log π — направление в пространстве параметров, которое увеличивает вероятность именно
          выбранного a. Умножение на G масштабирует шаг: редкий джекпот тянет сильнее, чем шаг с
          G≈0. Это credit assignment «в лоб»: весь будущий возврат вешается на каждое прошлое
          действие эпизода.
        </p>
      </Section>

      <Section title="Baseline — вычитаем «обычный» возврат">
        <p>
          Можно вычесть из G любую функцию, не зависящую от текущего действия — градиент в среднем
          тот же, дисперсия ниже. Естественный baseline — V(s): тогда вес становится advantage
          G − V(s). Как только V учится параллельно, получается actor-critic.{' '}
          <LessonLink id="a2c">A2C</LessonLink> идёт дальше: вместо полного G берёт TD/n-step, чтобы
          не ждать конца эпизода.
        </p>
        <Formula>{`∇J(θ) ≈ E[ ∇ log π_θ(aₜ | sₜ) · (Gₜ − b(sₜ)) ]`}</Formula>
      </Section>

      <Section title="Чего в приложении нет — и почему">
        <p>
          Отдельной кнопки «REINFORCE» в Дизайнере нет: на практике его едят дисперсия и необходимость
          закончить эпизод. PPO — тот же policy gradient с клипом, чтобы шаг θ не разрушил политику,
          и GAE вместо сырого G. Если поняли эту страницу, клип PPO перестаёт быть «магическим
          0.2» и становится предохранителем поверх REINFORCE.
        </p>
      </Section>

      <Callout tone="warning" title="Дисперсия — не мелочь">
        Два эпизода CartPole с одинаковым первым действием могут закончиться на шаге 12 и на шаге
        500. G отличается в десятки раз, градиент — тоже. Без baseline и без нормализации advantage
        (как в PPO) обучение дёргается. Не вините «маленькую сеть», пока не вычли среднее.
      </Callout>

      <Callout tone="tip" title="Дальше">
        Критик + короткий rollout + энтропия = рабочий алгоритм в Дизайнере.{' '}
        <LessonLink id="a2c">A2C</LessonLink>, затем стабильный дефолт{' '}
        <LessonLink id="ppo">PPO</LessonLink>.
      </Callout>
    </>
  ),
}

export const POLICY_INTRO_LESSONS: Lesson[] = [reinforce]
