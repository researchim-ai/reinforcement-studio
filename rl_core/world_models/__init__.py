"""World Models — learned dynamics models of an environment (encoder +
latent transition + decoder/reward/continue heads), plus RL algorithms that
train and/or plan against one instead of (or in addition to) a model-free
policy.

Three from-scratch families, picked per `WorldModelSpec.type` (see
`spec.py`):

- `"rssm"` (`rssm.py`) — Dreamer-style Recurrent State-Space Model:
  deterministic GRU state + a stochastic Gaussian latent, trained with a
  reconstruction + KL (ELBO-style) + reward + continue loss on real
  sequences, then *imagined* forward for actor-critic training
  (`rl_core/algorithms/native/dreamer.py`) without ever stepping the real
  env during that phase.
- `"ensemble"` (`ensemble.py`) — MBPO/PETS-style ensemble of probabilistic
  feed-forward dynamics models (predict `Δobs, reward` with a Gaussian
  NLL loss). Used two ways: short imagined rollouts augmenting an
  off-policy replay buffer (`rl_core/algorithms/native/mbpo.py`), or as the
  model a Cross-Entropy Method planner searches over directly, no learned
  policy at all (`rl_core/algorithms/native/pets.py`).
- `"vae_mdnrnn"` (`vae_mdnrnn.py`) — the original Ha & Schmidhuber (2018)
  "World Models": a VAE encodes single frames/observations, an MDN-RNN
  predicts the *distribution* of the next latent (a mixture of Gaussians,
  not a single point) conditioned on the action, and a small linear
  controller acting on `[z, h]` is optimized directly against real
  episode return via Evolution Strategies
  (`rl_core/algorithms/native/world_models_ha.py`).

Every family is environment-agnostic the same way the rest of this app's
native networks are (`rl_core/algorithms/native/networks.py`'s
`build_feature_extractor`/`is_image_space`): a vector `Box` observation
gets an MLP encoder/decoder, an image `Box` observation gets a small conv
encoder/decoder — callers never need to know or care which.

A `WorldModelSpec` can be trained two ways:
1. Standalone, with no RL agent at all — a `kind: "world_model"` run
   collects random-policy experience and fits the model purely for its own
   sake (see `trainer.py`, launched exactly like any other training run:
   `POST /training/start`, watched on the same Training Monitor page). This
   is the "just build/inspect a world model" flow from the World Model
   Builder page (`/world-models`).
2. Jointly with an agent, by one of the four algorithms above picking a
   saved spec via `algorithm.world_model_id` (resolved by `store.py`,
   exactly mirroring how `algorithm.network_spec_id` resolves a saved
   Network Builder architecture in `rl_core/netbuilder_store.py`) — or with
   no saved spec at all, in which case the algorithm just builds a fresh
   one from its own hyperparams and trains it online alongside the policy.
"""
