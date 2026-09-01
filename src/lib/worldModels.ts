import type { WorldModelConfig, WorldModelType, WorldModelTypeInfo } from '@/api/types'

export const WORLD_MODEL_TYPE_LABELS: Record<WorldModelType, string> = {
  rssm: 'RSSM (Dreamer-style)',
  ensemble: 'Ensemble Dynamics (MBPO / PETS)',
  vae_mdnrnn: 'VAE + MDN-RNN (классические World Models)',
}

export const WORLD_MODEL_TYPE_DESCRIPTIONS: Record<WorldModelType, string> = {
  rssm:
    'Recurrent State-Space Model — детерминированный GRU-state + стохастический латент, обучается на реконструкцию, ' +
    'KL, награду и continue-флаг. Используется алгоритмом Dreamer для actor-critic обучения целиком внутри воображения.',
  ensemble:
    'Ансамбль вероятностных feed-forward моделей динамики, предсказывающих распределение (Δnaблюдение, награда). ' +
    'Используется MBPO (для дополнения реального replay buffer воображаемыми переходами) и PETS (для CEM-планирования).',
  vae_mdnrnn:
    'Классические "World Models" (Ha & Schmidhuber): VAE сжимает наблюдения в латентный z, MDN-RNN предсказывает ' +
    'распределение следующего z. Используется алгоритмом World Models (VAE + MDN-RNN) с ES-контроллером.',
}

// Which of the four World Model algorithms (rl_core/algorithms/native_runner.py
// ::WORLD_MODEL_TYPE_FOR_ALGO) expects each type — purely informational (the
// Designer's own `AlgorithmSpec.world_model_type` already enforces this),
// shown on the World Model Builder page so a freshly created spec's intended
// use is obvious before ever touching the Designer.
export const ALGORITHMS_FOR_TYPE: Record<WorldModelType, string[]> = {
  rssm: ['Dreamer'],
  ensemble: ['MBPO', 'PETS'],
  vae_mdnrnn: ['World Models (VAE + MDN-RNN)'],
}

export function defaultConfigForType(typeInfo: WorldModelTypeInfo[] | undefined, type: WorldModelType): WorldModelConfig {
  return typeInfo?.find((t) => t.id === type)?.default_config ?? {}
}
