"""ResearchImZero - our own MuZero-family algorithm, seeded from the
best-measured synergy of `unizero.py` and `efficientzero.py` rather than
being a port of any one published paper. Empirical motivation (see the
chat history this file was written from): a from-scratch, exact port of
the reference UniZero (PUCT + Dirichlet noise, an EMA target network,
MSE-to-target-tokenizer consistency loss, AdamW, `learning_starts=0`,
PER/reanalyze off by default - all now living in `unizero.py`, kept
around unmodified as the "faithful to the paper" reference implementation
    10|for anyone who wants exact reference behavior) trained *dramatically*
worse in practice on this app's own hard-exploration image envs (NetHack)
than an earlier, less "correct" revision of the same file that borrowed
`efficientzero.py`'s own training recipe wholesale - same architecture,
same buffer, wildly different search/loss/optimizer/schedule choices.
This file keeps **UniZero's actual architectural contribution** (a causal
Transformer over an interleaved `[obs_0, act_0, obs_1, ...]` token
sequence, RoPE, the persistent incrementally-evicted per-lane KV-cache -
see `unizero.py`'s own module docstring for why each of these matters,
    20|all reused here verbatim, unmodified) and swaps back in
**`efficientzero.py`'s own proven training recipe** everywhere that
recipe, empirically, is what actually made the earlier good runs learn
fast: Gumbel search instead of PUCT, SimSiam consistency instead of an
MSE latent target, context-aware EMA TD targets convexly blended with
fresh reanalysis values, plain `Adam` instead of `AdamW`, PER + reanalyze +
`learning_starts=500` on by default instead of off. Nothing here is
claimed to be "more correct" relative to any paper - the entire point of
this file existing separately from `unizero.py` is to be the one we keep
tuning by what actually works, not what matches a reference implementation.

    30|**Search: Gumbel-Top-k + Sequential Halving** (Danihelka et al., 2022 -
same algorithm `efficientzero.py` uses, *not* classic PUCT), ported
verbatim (`_SearchNode`/`_MinMaxStats`/`_transformed_completed_qs`/
`_select_action`/`_sequential_halving`/`_HalvingSchedule`/
`_backpropagate` below are `efficientzero.py`'s own copies, byte-for-byte
except for one adaptation) with exactly one adaptation: a node's "state"
is its own incremental KV-cache slice (`cache`, extended two tokens per
tree edge - one action, one predicted-observation, see `_step_imagine`),
not a fixed-size latent + LSTM cell, and a node's reward is a plain
per-edge scalar (UniZero's reward head predicts one transition's real
    40|reward directly, no LSTM "value prefix" cumulative-sum trick to unwind)
rather than `efficientzero.py`'s own value-prefix delta. Continuous
actions get the same sampled-candidate treatment
(`_sample_continuous_candidates`, ported from `efficientzero.py` as-is)
run through the identical Gumbel machinery discrete actions use - no
separate continuous search algorithm. The policy target this produces
is, correspondingly, `efficientzero.py`'s own: the *improved policy*
(Gumbel's own completed-Q-value + prior blend, Appendix D of Danihelka et
al.) for discrete actions, and the visit-weighted-average sampled
candidate for continuous ones - not a plain visit-count distribution
    50|(`unizero.py`'s reference-faithful choice) or a genuine importance-
weighted density fit (`unizero.py`'s own Sampled-UniZero continuous
policy loss) - both swapped back for simplicity/proven-in-practice
reasons, not because either alternative is wrong.

**EMA bootstrap target + SimSiam latent target.** The n-step TD bootstrap
uses a slow-moving copy of tokenizer/action embedding/Transformer/heads
and receives the same real historical context as the online value head.
The latent-consistency branch deliberately remains SimSiam against the
stop-gradiented online tokenizer:
- **Consistency loss is SimSiam, not MSE-to-target.** `_Projector`/
  `_Predictor` (ported verbatim from `efficientzero.py`) add a
  `BatchNorm1d`-equipped projector head + an asymmetric predictor on the
  *predicted* branch only, trained via negative cosine similarity against
  the stop-gradiented *online* tokenizer's own embedding of the real next
  observation. Chen & He, 2021's own ablation (Table 2c) is why the
  `BatchNorm1d` specifically isn't decorative - stop-gradient *alone*
   70|  still collapses in practice; empirically forcing every batch's
  projected features to spread out (via BN) is what actually prevents the
  every-observation-embeds-to-the-same-point degenerate solution, not the
  stop-gradient asymmetry by itself.
- **Value target additionally blends in a periodically-refreshed search
  value** via bounded `reanalyze_value_mix`, not a one-sided `max` that
  systematically selects optimistic MCTS errors -
  `_reanalyze()` (also ported as-is, `reanalyze_batch_size=64`/
  `reanalyze_freq=200` on by default here, unlike `unizero.py`'s `0`)
  periodically refreshes `search_value` with the network's *current*-
   80|weights root value estimate for old (episode, timestep) samples, so a
  transition's value target isn't purely at the mercy of however stale
  the online net's own bootstrap was back when that transition first
  landed in the buffer.

**Prioritized replay + reanalyze on by default** (`priority_alpha=1.0`,
`reanalyze_batch_size=64` - `efficientzero.py`'s own defaults, both `0`/
off in `unizero.py` to match the reference UniZero's own policy-level
choices) - both concepts are orthogonal to "recurrent vs Transformer
dynamics", ported wholesale from `efficientzero.py`'s `_EfficientZeroBuffer`
   90|(this file's own `_ResearchImZeroBuffer` only differs from it by the
extra `context_obs`/`context_action`/`context_valid` slicing every
Transformer training window needs - see that class's own docstring).
Active per-lane trajectories are sampled as soon as their next
observations exist, so long episodes no longer starve training until
their first terminal transition.

**Closed-loop latent overshooting.** Alongside stable teacher forcing, a
small sub-batch is recursively rolled through `_step_imagine`: each next
step consumes the model's own predicted latent exactly as tree search
does. Multi-step reward/value/latent losses therefore train the world
model in its deployment mode instead of only on real next-observation
tokens.

**`learning_starts=500`, not `0`.** `train_freq=1` means one
gradient-update group (two updates by default) per vector iteration (`learn()`'s own
`boundaries_crossed` comment) - training from the very first few
transitions, before the buffer has any behavioral diversity, is a real,
observed failure mode: a from-scratch model + a tiny buffer + this much
training pressure per new transition collapses the policy's own entropy
  100|within the first few hundred steps, and - depending on the env's own
episode-termination condition - a collapsed-but-"safe" policy can then
stop producing terminal transitions at all, starving the buffer of
anything fresher and locking the collapse in. `500` (matching
`efficientzero.py`'s own default, and this file's own earlier - before it
briefly, mistakenly matched the reference's bare `0` policy-level
default - choice) gives the buffer a little real behavioral diversity to
train against before training starts at all.

**Lighter default hyperparameters** (`num_heads=4`, `batch_size=64`,
 110|`unroll_steps=5`, adaptive `num_simulations=8→32`, `gamma=0.99`,
`consistency_loss_coef=2.0`, `value_support_size=300`) - not this file's
earlier attempt at matching the reference UniZero's own *policy-level*
defaults (`num_heads=8`, `batch_size=256`, `unroll_steps=10`,
`num_simulations=50`, `gamma=0.997`, `consistency_loss_coef=10.0`,
`value_support_size=50`), which are individually defensible but,
combined, meaningfully increase both per-step compute (fewer real
env-steps per wall-clock second - directly hurts a hard-exploration env's
odds of encountering reward at all before something like the collapse
above locks in) and how large a single early, low-diversity training
 120|batch's influence on the network is. `Adam`, not `AdamW` - simpler,
matches both `efficientzero.py` and this file's own earlier working
recipe; no evidence weight decay specifically mattered either way here.

Everything else - `_Tokenizer`/`_ActionEmbed`/`_CausalTransformer`/RoPE/
`_TransformerCache`/incremental KV-cache eviction/`_step_imagine`/
`_replay_context_to_cache`/`_advance_lane_caches`/`_embed_root_batch`/
categorical value-reward heads/label smoothing - is `unizero.py`'s own
machinery, reused as-is; see that file's module docstring for the design
rationale behind each. `rotary_emb` (default on) toggles the same RoPE-
 130|vs-learned-absolute-embedding choice `unizero.py` has, for the same
reasons.

Remaining simplifications vs. either parent file: no multitask/LPIPS/
pixel-decoder/open-loop-consistency/async-Ray-workers machinery (same
as both parents); reanalyze runs synchronously in-loop, not in a separate
process (same trade-off both parents make).

**Performance, not architecture.** None of the below changes what gets
computed - same search algorithm, same losses, same defaults - only how
fast: `search()`'s per-simulation, per-lane Python loop used to call
`.item()`/`.detach().cpu().numpy()` on a fresh one-row CUDA tensor slice
*inside* that loop (`batch_size * num_simulations` separate GPU syncs
per real env step, the actual reason a from-scratch profiling pass found
this process CPU-bound at 100% while the GPU sat mostly idle even though
it "looked" GPU-heavy on paper); every such site now converts its whole-
batch tensor to numpy exactly once per simulation and only ever indexes
plain numpy rows inside the per-lane loop. All of `search()`'s own
forward passes plus the two persistent-cache helpers
(`_replay_context_to_cache`/`_advance_lane_caches`) now run under
`torch.inference_mode()` instead of `torch.no_grad()` - strictly stronger
(inference tensors can never leak into a later autograd graph at all,
closing off the exact class of bug behind this file's one past `CUDA out
of memory` incident) and slightly cheaper (no version-counter/view
bookkeeping `no_grad()` still pays for). `_train_step`'s own forward pass
runs under automatic mixed precision (`use_amp`, default on for CUDA,
always off on CPU) - bf16 where the GPU supports it (no loss scaling
needed, fp32's exponent range), fp16 with a `GradScaler` otherwise -
PyTorch's own standard recipe, roughly halving both step time and memory
on a supported GPU. `__init__` also flips on `cudnn.benchmark` and TF32
matmuls (`torch.set_float32_matmul_precision("high")`) once, process-wide,
whenever running on CUDA - free wins for the CNN `_Tokenizer` on image
envs and every Linear/attention matmul respectively, no-ops elsewhere.

None of this touches **environment stepping** itself - that's `num_envs`
(`rl_core/algorithms/vec_env.py`'s own job, shared by every native
algorithm): `num_envs > 1` already runs one `AsyncVectorEnv` subprocess
worker per lane, true multi-core env parallelism, with `search()` itself
already batching all `num_envs` lanes' trees into one shared GPU forward
pass per simulation - so the single highest-leverage speed knob outside
this file is simply raising `training.num_envs` in the experiment config
to however many CPU cores are free, not anything below.

**Schedules, not fixed constants** (all toggleable, all default to the
new behavior, all fall back to this file's original flat/static behavior
when turned off - added after the above shipped and got run for a while,
not part of the original UniZero/EfficientZero synergy above): a
constant learning rate, `priority_beta`, and search-exploration
temperature for an entire run turned out to individually leave
performance on the table once a run went on long enough to reach the
late-training regime where PER and bootstrapped targets are most exposed.
- **`lr_schedule`** (default on): cosine-decays `learning_rate` down to
  `learning_rate * lr_min_fraction` from `learning_starts` onward
  (`_update_learning_rate`) - damps the value-loss oscillation a
  constantly-high LR otherwise feeds once PER has sharpened in on a few
  high-priority transitions late in training.
- **`use_target_for_bootstrap`** (default on): a slow-moving EMA/Polyak
  copy of tokenizer+action embedding+transformer+heads (`target_update_theta` per step,
  `_update_target_network`) provides the n-step TD bootstrap value
  instead of the online net, with real context at every landing state.
  This fixes the classic moving-target and memoryless-bootstrap problems
  for just that one use (never
  for the consistency loss's own target - that's still always the online
  tokenizer, deliberately, per that section above). Paired with
  **`search_value_max_staleness`** and **`reanalyze_value_mix`**: only a
  fresh reanalyzed value participates in a bounded convex blend with TD.
- **`priority_beta_start` → `priority_beta`** (default `0.4` → `1.0`):
  the PER importance-sampling correction anneals up across training
  (`_update_priority_beta`) instead of applying full-strength IS
  correction from step one, when the value network the priority signal
  itself depends on is still close to random.
- **`priority_policy_weight`** (default `0.1`): replay priority also
  factors in a transition's own policy-target error
  (`first_step_policy_error`), not just its value error - a transition
  whose policy target the network gets badly wrong is just as worth
  replaying soon as one whose value estimate is off.
- **Targeted reanalyze**: `sample_for_reanalyze` picks episodes/
  transitions weighted by priority *and* how many generations since they
  were last refreshed (`_episode_last_reanalyzed_gen`/`reanalyzed_gen`),
  not uniformly by episode length regardless of either.
- **`temperature_anneal_start_scale`** (default `1.5`): both the root
  Gumbel noise scale and `policy_entropy_coef` get scaled up early in
  training and decay to their configured face value
  (`_current_exploration_scale`) - more exploration/entropy bonus while
  the policy is still close to random, tapering off as it isn't.
- **`intrinsic_exploration`** (default off): optional Random Network
  Distillation novelty bonus (`dqn.py`/`rainbow_dqn.py`'s own
  `RNDModule`, same hyperparameter names for consistency) added to the
  real reward before it ever reaches the buffer/TD targets - off by
  default since, unlike everything else in this section, it changes what
  reward is actually being optimized, not just how training gets there;
  worth turning on for the sparse/hard-exploration environments (the
  NetHack-family runs) this file was originally tuned against.
- **`adaptive_loss_weights`** (default off): Kendall et al. (2018)
  homoscedastic-uncertainty weighting - a learnable `log_var` per task
  loss (reward/value/policy only, clamped to `[-5, 5]`; see below for why
  `consistency` and the entropy bonus are both excluded) replaces this
  file's original static `*_loss_coef` ratios (`value_loss_coef=0.25` vs
  `policy_loss_coef=1.0` was one never-revisited guess) with weights that
  adapt to each task's own loss scale during training; the `*_loss_coef`
  values still matter as each task's prior multiplier, not as the final
  weight. `consistency_loss` (SimSiam) deliberately keeps its plain
  static `consistency_loss_coef` - it has no genuine noise floor the way
  reward/value/policy do (nothing stops the stop-grad predictor from
  driving it arbitrarily close to `0`), so folding it into this scheme
  let its own learned weight climb straight to the `[-5, 5]` clamp
  ceiling (~148x) and stay pinned there for the rest of training,
  ~200x `policy`'s concurrent weight - a real run measured exactly this
  and stalled well below what static coefficients alone used to reach.
- **`auto_scale_replay_ratio`** (default off): optional additional learner
  pressure, scaling updates per vector iteration by `num_envs // 4`.
  It is deliberately not combined with the number of aggregate timestep
  boundaries crossed: that old double scaling produced 144 optimizer
  steps per 24 collected transitions and dominated wall-clock training.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from rl_core.algorithms.base import CustomAlgorithm, TrainingCallback
from rl_core.algorithms.native.exploration import RNDModule, uses_rnd
from rl_core.algorithms.native.preprocessing import action_to_env, obs_batch_to_array, obs_flat_dim, obs_to_array
from rl_core.algorithms.vec_env import (
    action_space as venv_action_space,
    num_envs_of,
    obs_space as venv_obs_space,
    vec_reset,
    vec_step,
)
from rl_core.composite_netbuilder import (
    build_component_mlp,
    build_composite_encoder,
    dimensions_from_spec,
    validate_composite_spec,
)
from rl_core.world_models.nets import ObsEncoder

DEFAULT_HYPERPARAMS = {
    # `efficientzero.py`'s own default (plain `Adam`, not `AdamW` - see
    # module docstring's "no target network"/optimizer discussion).
    "learning_rate": 2e-4,
    # `1` (default): cosine-decay `learning_rate` down to
    # `learning_rate * lr_min_fraction` over the run, starting only once
    # `learning_starts` real steps have passed (before that, no gradient
    # steps happen at all) - flat `lr` for the *entire* run otherwise had
    # no mechanism to damp the late-training value-loss oscillation a
    # constantly-high LR feeds into once PER has sharpened in on a few
    # high-priority (and, with no target network, self-referential)
    # transitions. `0`: keep `learning_rate` flat, exactly as before this
    # option existed.
    "lr_schedule": 1,
    "lr_min_fraction": 0.05,
    "embed_dim": 128,
    "num_layers": 2,
    # Lighter than `unizero.py`'s reference-matching `8` - module
    # docstring's "Lighter default hyperparameters" section.
    "num_heads": 4,
    "dropout": 0.0,
    # Past real (obs, action) *transitions* kept before "now" in every
    # token sequence the Transformer ever sees (root of a real step, a
    # training window, a reanalyze pass) - `2 * context_length` tokens.
    "context_length": 6,
    "buffer_size": 2_000,
    "batch_size": 64,
    # Doubles as the training teacher-forcing window length *and* the max
    # search-tree depth budget (an imagined rollout can't run longer than
    # a real training unroll would need to correct it) - same dual role
    # `unroll_steps` plays in `efficientzero.py`.
    "unroll_steps": 5,
    "td_steps": 5,
    "num_sampled_actions": 8,
    "num_simulations": 32,
    "num_simulations_initial": 8,
    "search_ramp_steps": 100_000,
    "search_model_error_low": 0.02,
    "search_model_error_high": 0.30,
    "search_model_error_ema_decay": 0.99,
    # Gumbel-Top-k + Sequential Halving (`_HalvingSchedule`/
    # `_sequential_halving`) - `efficientzero.py`'s own defaults, not
    # PUCT's `pb_c_base`/`pb_c_init`/Dirichlet noise (`unizero.py`'s own
    # choice - Gumbel-Top-k's own per-simulation resampling is its own,
    # different, built-in exploration source at the root).
    "num_top_actions": 8,
    "c_visit": 50.0,
    "c_scale": 0.1,
    "policy_target_temperature": 1.0,
    # Both the root Gumbel noise scale (`search()`'s own exploration
    # source, module docstring's "Search" section) and `policy_entropy_
    # coef` get multiplied by a shared factor that starts at this value
    # (more exploration/entropy bonus early, when the policy is still
    # near-random and there's little to lose from extra noise) and
    # linearly decays to `1.0` (both knobs at their configured face
    # value) as training progresses - `_current_exploration_scale()`.
    # `1.0` here disables the schedule entirely (flat scale of `1.0` the
    # whole run, exactly as before this existed).
    "temperature_anneal_start_scale": 1.5,
    "temperature_anneal_steps": 100_000,
    "value_minmax_delta": 0.01,
    # `efficientzero.py`'s own, much wider default - not `unizero.py`'s
    # narrower `50` (module docstring's "Lighter default hyperparameters"
    # section: a wider support isn't actually *lighter* compute-wise, but
    # this specific value is what both this file's own earlier good runs
    # and `efficientzero.py` itself already use, so it's kept rather than
    # re-guessed).
    "value_support_size": 300,
    # No label smoothing - `efficientzero.py` has no equivalent knob, and
    # this file's own earlier good runs didn't use it either.
    "label_smoothing_eps": 0.0,
    "gamma": 0.99,
    "value_loss_coef": 0.25,
    "policy_loss_coef": 1.0,
    "reward_loss_coef": 1.0,
    # SimSiam consistency loss's own weight - `efficientzero.py`'s own
    # default (`2.0` here specifically matches this file's own earlier
    # good NetHack runs, between `efficientzero.py`'s policy-level `5.0`
    # and this file's UniZero-parent's reference-matching `10.0`).
    "consistency_loss_coef": 2.0,
    "path_consistency_coef": 0.0,
    # Autoregressive training in the exact predicted-latent mode used by
    # MCTS. Kept on a subset/horizon budget so it adds signal without
    # multiplying full teacher-forced training cost.
    "closed_loop_loss_coef": 0.5,
    "closed_loop_horizon": 3,
    "closed_loop_batch_fraction": 0.25,
    # Kendall et al. (2018) homoscedastic-uncertainty multi-task
    # weighting - `1` (default): each of the four task losses (reward,
    # value, policy; `consistency` is deliberately excluded, see
    # `__init__`'s own `self.loss_log_vars` comment) gets its own
    # learnable `log_var` alongside the network's own parameters
    # (`self.loss_log_vars`, optimized by the exact same
    # `self.optimizer`), and the loss actually backpropped is
    # `sum_i(coef_i * exp(-log_var_i) * loss_i + log_var_i)` instead of
    # `sum_i(coef_i * loss_i)` directly for those three - in theory, a
    # task whose loss is consistently *larger* than the others'
    # automatically gets down-weighted relative to one that's already
    # small, rather than every task's relative weight being whatever this
    # file's own `*_loss_coef` defaults happened to guess.
    #
    # Default is `0` (off), not `1`, on real evidence, not theory: on a
    # real NetHack run, `reward_loss` (mostly-zero reward, trivial to
    # predict most steps) sat an order of magnitude below `policy_loss`
    # (search-target distillation, inherently noisier) for the *entire*
    # run - measured `loss_weight_reward` tracking `1/reward_loss`
    # exactly (this scheme's own, correct equilibrium, not a bug) and
    # settling at `~70-90x` for the whole run, while `loss_weight_policy`
    # sat flat at `~0.9-1.1` throughout, `value` in between at `~15-20x`.
    # "Easy to minimize" isn't "unimportant" here - reward-prediction
    # being trivial doesn't mean it should get 75x more of the trunk's
    # gradient than the one task (policy) that actually determines
    # in-game behavior. Kendall et al.'s own multi-task setup (depth/
    # segmentation/instance losses, all sharing comparable pixel-level
    # aleatoric noise) doesn't have this problem; this file's task mix
    # does. `1`: opt-in for environments where reward/value/policy
    # happen to have more comparable achievable loss floors - the
    # `*_loss_coef` values above still matter as each task's *prior*
    # multiplier either way, not as the final weight.
    "adaptive_loss_weights": 0,
    # SimSiam projector/predictor hidden+output dim (`_Projector`/
    # `_Predictor`) - `efficientzero.py`'s own default.
    "proj_dim": 64,
    # Entropy bonus - generic exploration regularizer, kept from
    # `unizero.py` (`efficientzero.py` has no equivalent knob).
    "policy_entropy_coef": 5e-3,
    "continuous_prior_scale": 2.5,
    "train_freq": 1,
    "train_steps_per_iter": 2,
    "adaptive_train_steps": 0,
    "adaptive_train_steps_min": 2,
    "adaptive_train_steps_max": 4,
    # Opt-in extra learner pressure for large vector envs. Off by default:
    # `learn()` already notices boundaries crossed by the aggregate
    # timestep counter, and the historical fast/successful native runs
    # performed one optimizer update per vector iteration, not one per
    # transition. Enabling this scales that one update by `num_envs // 4`.
    "auto_scale_replay_ratio": 0,
    # `efficientzero.py`'s own default, and this file's own earlier
    # (pre-reference-matching) good-runs' choice - not `0` (module
    # docstring's own "learning_starts=500, not 0" section explains why
    # `0` is a real, observed failure mode combined with `train_freq=1`).
    "learning_starts": 500,
    "max_grad_norm": 5.0,
    # Prioritized replay (Schaul et al., 2016) + reanalyze, *on* by
    # default here - `efficientzero.py`'s own defaults (not `unizero.py`'s
    # reference-matching `0.0`/`0`, both off there).
    "priority_alpha": 1.0,
    # Final importance-sampling correction exponent (Schaul et al., 2016)
    # - linearly annealed *up* from `priority_beta_start` to this value
    # across training (module docstring's "Schedules, not fixed
    # constants" section), not applied at full strength from step one:
    # early on, `value_pred`/`value_target` are both still close to
    # random, so IS-correcting hard against a priority signal that
    # isn't trustworthy yet mostly just down-weights the few transitions
    # PER *did* get right for the wrong reasons. Set
    # `priority_beta_start` equal to this to disable annealing (flat
    # `priority_beta` the entire run, exactly as before this existed).
    "priority_beta": 1.0,
    "priority_beta_start": 0.4,
    # How much a transition's *policy*-target error (cross-entropy for
    # discrete, MSE for continuous - `first_step_policy_error` in
    # `_train_step`) adds to its replay priority, on top of the original
    # `|value_pred - value_target|` term - `0.0` recovers the old,
    # value-error-only priority exactly.
    "priority_policy_weight": 0.1,
    "learning_progress_priority_weight": 0.0,
    "learning_progress_priority_cap": 1.0,
    "min_priority": 1e-6,
    "replay_recent_fraction": 0.5,
    "replay_recent_window": 512,
    "replay_success_fraction": 0.0,
    "replay_success_top_quantile": 0.25,
    "adaptive_closed_loop": 0,
    "adaptive_closed_loop_max_horizon": 5,
    "adaptive_closed_loop_error_threshold": 0.08,
    "adaptive_closed_loop_min_grad_steps": 500,
    "adaptive_closed_loop_stable_steps": 50,
    "adaptive_closed_loop_ema_decay": 0.99,
    "episode_reward_rolling_window": 20,
    "reanalyze_freq": 200,
    "reanalyze_batch_size": 64,
    # `1` (default): a context-aware slow-moving EMA/Polyak copy provides
    # the n-step TD bootstrap and breaks the moving-target problem.
    # `0`: bootstrap with the online network; consistency always retains
    # its separate stop-gradient SimSiam target.
    "use_target_for_bootstrap": 1,
    "target_update_theta": 0.02,
    # How many `_reanalyze()` "generations" (module docstring's
    # "Schedules, not fixed constants" section) a transition's own
    # `search_value` may be behind the buffer's current one and still be
    # allowed to participate in the convex reanalysis/TD blend. Past
    # this, it is treated the same as never reanalyzed (`td_target` only).
    "search_value_max_staleness": 5,
    # Convex weight of a fresh reanalyzed root value in the TD target.
    # Unlike the old hard max, this cannot turn every positive search
    # error into systematic value overestimation.
    "reanalyze_value_mix": 0.25,
    # Automatic mixed precision for `_train_step`'s forward/backward pass
    # (module docstring's "Performance, not architecture" section) -
    # `1` (default): on, whenever running on CUDA (always off on CPU,
    # regardless of this flag - `__init__`'s own `self.use_amp`). Pure
    # speed/memory knob, changes no computed value beyond ordinary
    # floating-point rounding - set to `0` to debug a suspected precision
    # issue in isolation.
    "use_amp": 1,
    # Random Network Distillation (Burda et al., 2019) intrinsic reward -
    # `dqn.py`/`rainbow_dqn.py`'s own `RNDModule`, same
    # `intrinsic_exploration`/`rnd_*` knob names for consistency across
    # every native algorithm in this app. `0` (default, off): behaves
    # exactly as before this option existed - opt-in, not on by default,
    # since it changes what reward the agent actually optimizes (real +
    # novelty bonus, not just real) rather than being a pure training-
    # speed/stability knob. Worth turning on for sparse/hard-exploration
    # environments (the NetHack-family runs this file was originally
    # tuned against) - `episode_extrinsic_reward`/`episode_intrinsic_
    # reward` metrics let you see the two parts separately once enabled.
    "intrinsic_exploration": 0,
    "rnd_bonus_coef": 0.1,
    "rnd_learning_rate": 1e-4,
    "rnd_feature_dim": 128,
    "rnd_hidden_dim": 128,
    "rnd_bonus_clip": 5.0,
    # `1` (default): RoPE - `_CausalTransformer`'s own default, lossless
    # `O(1)`-per-real-step KV-cache eviction, reused from `unizero.py`
    # as-is (module docstring's "everything else" section). `0`: learned
    # absolute `nn.Embedding` instead - slower, exists as an alternative
    # for anyone who wants to compare.
    "rotary_emb": 1,
}


# ----------------------------------------------------------------------
# Tokenizer + embeddings - raw (pre-Transformer) per-token feature vectors.
# ----------------------------------------------------------------------
class _Tokenizer(nn.Module):
    """One observation -> one token embedding. Reuses the same image/vector
    auto-detecting feature extractor every world model family in this app
    shares (`rl_core/world_models/nets.py::ObsEncoder`)."""

    def __init__(
        self, observation_space: gym.Space, embed_dim: int, hidden_dim: int,
        encoder: nn.Module | None = None, component_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or ObsEncoder(observation_space)
        self.head = (
            build_component_mlp(self.encoder.out_dim, embed_dim, component_config)
            if component_config is not None
            else nn.Sequential(nn.Linear(self.encoder.out_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, embed_dim))
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.encoder(obs)))


class _ActionEmbed(nn.Module):
    """One action -> one token embedding. Discrete: a genuine
    `nn.Embedding` lookup table (`n_actions` rows), matching the
    reference's own `world_model.py::self.act_embedding_table` - *not*
    `Linear(one-hot) + tanh` (mathematically almost the same thing for a
    one-hot input - `Linear(one_hot)` already just selects one weight
    row/bias - but an `nn.Embedding` has no bias term added on top of the
    looked-up row and no extra `tanh` squashing it). `action` still
    arrives one-hot (`obs_to_array`'s convention every algorithm in this
    app shares, and `_ResearchImZeroBuffer`'s own storage format) purely so this
    class's *caller* doesn't need a separate discrete/continuous branch -
    `forward` itself converts it to an index via `argmax` (safe: actions
    are data, never something a loss backprops *through*). Continuous
    (`Box`): unchanged, `Linear(raw vector) + tanh` - the reference has no
    single equivalent here since its own continuous variant (Sampled
    UniZero) samples actions from a Gaussian and embeds *those* the same
    way `Box` already does in this file."""

    def __init__(self, action_dim: int, embed_dim: int, discrete: bool) -> None:
        super().__init__()
        self.discrete = discrete
        if discrete:
            self.embedding = nn.Embedding(max(1, action_dim), embed_dim)
        else:
            self.net = nn.Linear(action_dim, embed_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if self.discrete:
            return self.embedding(action.argmax(dim=-1))
        return torch.tanh(self.net(action))


# ----------------------------------------------------------------------
# Categorical value/reward support - ported verbatim from
# `efficientzero.py` (MuZero Appendix F "scaling and squashing"); see that
# file's module docstring for the full rationale.
# ----------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def _signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    z = torch.sqrt(1.0 + 4.0 * eps * (torch.abs(x) + 1.0 + eps)) - 1.0
    return torch.sign(x) * ((z / (2.0 * eps)) ** 2 - 1.0)


def _scalar_to_two_hot(x: torch.Tensor, support_size: int, label_smoothing_eps: float = 0.0) -> torch.Tensor:
    """MuZero Appendix F's linear-interpolation "soft one-hot", optionally
    blended with a uniform distribution first - reference's own
    `phi_transform` (`lzero/policy/scaling_transform.py`): `smooth_target
    = (1-eps)*two_hot + eps/num_bins`. `eps=0.0` (this function's own
    default, and every pre-existing caller before label smoothing was
    added) reproduces a plain two-hot exactly."""
    num_bins = 2 * support_size + 1
    x = _signed_hyperbolic(x).clamp(-support_size, support_size)
    lower = x.floor()
    upper = lower + 1
    lower_weight = (upper - x).clamp(0.0, 1.0)
    upper_weight = 1.0 - lower_weight
    lower_idx = (lower + support_size).long().clamp(0, num_bins - 1)
    upper_idx = (upper + support_size).long().clamp(0, num_bins - 1)
    out = torch.zeros(*x.shape, num_bins, device=x.device, dtype=torch.float32)
    out.scatter_add_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
    out.scatter_add_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
    if label_smoothing_eps > 0.0:
        out = (1.0 - label_smoothing_eps) * out + label_smoothing_eps / num_bins
    return out


def _logits_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    bins = torch.arange(-support_size, support_size + 1, device=logits.device, dtype=torch.float32)
    x = (probs * bins).sum(-1)
    return _signed_parabolic(x)


# ----------------------------------------------------------------------
# Rotary position embeddings (Su et al., 2021, arXiv:2104.09864) - see
# module docstring's "Positional encoding" section for why this (rather
# than a learned additive `nn.Embedding` table) is what makes the
# incremental KV-cache below both correct *and* fast. No table, no
# maximum-position bound: `_rope_cos_sin` computes each token's rotation
# straight from its own (arbitrarily large) position index.
# ----------------------------------------------------------------------
def _rope_cos_sin(positions: torch.Tensor, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """`positions`: `(...)` long/float, any shape. Returns `(cos, sin)`,
    each `(..., head_dim)`, ready to combine with a `(..., head_dim)`
    query/key tensor via `_apply_rope`'s rotate-half convention."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=positions.device) / half))
    freqs = positions.float().unsqueeze(-1) * inv_freq  # (..., half)
    emb = torch.cat([freqs, freqs], dim=-1)  # (..., head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x`: `(..., head_dim)`. `cos`/`sin`: broadcastable to `x`'s shape
    (e.g. `x` is `(B, H, L, D)` and `cos`/`sin` are `(B, 1, L, D)`)."""
    return x * cos + _rotate_half(x) * sin


# ----------------------------------------------------------------------
# Incremental KV-cache (see module docstring's "Persistent,
# incrementally-extended per-lane root cache" section for the overall
# design). Two call conventions exist side by side on every
# Transformer-touching class below:
#   - `forward`/`forward(...)` (no suffix): the *full-sequence* path, one
#     matmul over a whole (padded, right-padded/compact-positions) batch
#     of token sequences at once - used only for training's teacher-
#     forcing forward pass, which fundamentally needs every position's
#     hidden state from one pass for gradients to flow correctly, and
#     gains nothing from a cache since nothing is "reused" across calls.
#   - `forward_incremental_batch` (this section): appends *exactly one*
#     new token per lane to an existing (possibly `None`/empty) cache and
#     returns only that new token's hidden state - used for everything
#     that legitimately reuses previously-computed attention: every real
#     step's persistent per-lane cache (`NativeResearchImZero._advance_lane_caches`),
#     every reanalyze sample's one-off cold-start context replay
#     (`NativeResearchImZero._replay_context_to_cache`), and search-tree
#     expansion on top of either root (`NativeResearchImZero._step_imagine`).
# A token's position for RoPE purposes is a cache's `next_pos` (see
# `_TransformerCache`) - monotonic, never rewound by eviction, unlike
# `length` (the cache's current *physical* token count).
# ----------------------------------------------------------------------
def _cache_positions(
    caches: list["_TransformerCache | None"], device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns `(positions, embed_positions)`, both `(B,)` long, `0` for
    an empty/`None` lane. `positions` is each lane's raw, monotonic
    absolute token-since-episode-start counter (`next_pos`) - always fed
    to `_CausalTransformer.forward_incremental_batch` for RoPE rotation
    when it's on (translation-invariant, so its own unbounded growth
    never needs correcting - module docstring's "Positional encoding"
    section) and for that same call's own cache-bookkeeping (`next_pos`
    on the *returned* cache) regardless of RoPE. `embed_positions` is the
    same counter *rebased* by each cache's own `pos_origin` (bumped by
    `evict_front`, module docstring's "Positional encoding toggle"
    section) - only consulted when RoPE is off, to keep the learned
    additive embedding table's lookup index within the small range
    training's own full-sequence pass (`0..window_len-1`) actually
    trained on, rather than growing unboundedly with episode length."""
    positions = torch.as_tensor([0 if c is None else c.next_pos for c in caches], dtype=torch.long, device=device)
    embed_positions = torch.as_tensor(
        [0 if c is None else c.next_pos - c.pos_origin for c in caches], dtype=torch.long, device=device,
    )
    return positions, embed_positions
# ----------------------------------------------------------------------
class _LayerKV:
    """One Transformer layer's cached keys/values for one lane or search
    node. `k`/`v`: `(1, num_heads, length, head_dim)` - batch dim is
    always exactly 1 (one cache belongs to one lane/node; cross-lane
    batching for one incremental step happens by *padding* several
    `_LayerKV`s together inside `_CausalSelfAttention.forward_incremental_batch`,
    not by giving a `_LayerKV` itself a batch dimension > 1)."""

    __slots__ = ("k", "v")

    def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k = k
        self.v = v


class _TransformerCache:
    """One lane's or one search node's incremental KV-cache across every
    Transformer layer. Crucially, `forward_incremental_batch` never
    mutates a `_TransformerCache` it's given - `torch.cat`, which is how
    every layer extends `k`/`v`, always allocates a fresh output tensor
    and never writes back into its inputs - so it always *returns* a
    brand-new `_TransformerCache` rather than editing one in place. That
    means the exact same parent cache object can be hand to many
    `forward_incremental_batch` calls "in parallel" (e.g. every candidate
    action expanding the same search node within one simulation) with no
    aliasing risk and no explicit `.clone()` needed anywhere - the
    reference's "deep-clone KV on retrieve" requirement falls out for
    free from this class simply never being mutated to begin with.

    `length` (physical token count, shrinks on `evict_front`) and
    `next_pos` (the RoPE position the *next* appended token will get,
    monotonic, never shrinks) are deliberately two separate counters -
    see module docstring's "Positional encoding" section for why eviction
    only needs to touch the former. `pos_origin` is a third, related
    counter only consulted when RoPE is off (`rotary_emb=0`, module
    docstring's "Positional encoding toggle" section) - see
    `evict_front`'s own docstring."""

    __slots__ = ("layers", "length", "next_pos", "pos_origin")

    def __init__(self, num_layers: int) -> None:
        self.layers: list[_LayerKV | None] = [None] * num_layers
        self.length = 0
        self.next_pos = 0
        self.pos_origin = 0

    def evict_front(self, keep_last: int) -> None:
        """Drops every physically-cached token except the most recent
        `keep_last` - a plain tensor slice. Lossless *positionally* under
        RoPE (module docstring's "Positional encoding" section): a
        surviving cached key's rotation was baked in at write time and
        depends only on its own `next_pos` at that time, never on what
        else shares the cache, so dropping unrelated old entries perturbs
        nothing about the *relative offsets* any future query will see.
        Not lossless in an absolute sense for `num_layers > 1`, though: a
        surviving token's own hidden state (and thus its K/V at every
        layer above the first) was computed *before* eviction, while the
        now-dropped tokens were still attendable, so a little of their
        influence remains baked into the survivors via the residual
        stream - unavoidable for any multi-layer causal-attention KV-cache
        eviction scheme, this file's own or the reference's, and
        independent of positional encoding entirely (see module
        docstring's "Positional encoding" section, second bullet, for why
        that's an accepted trade rather than a bug). `next_pos` itself is
        untouched; eviction only shrinks physical storage, never rewinds
        the position counter new tokens keep counting up from.

        `pos_origin` *does* move here, by exactly `drop` - unconditionally,
        regardless of whether RoPE is even on (harmless dead bookkeeping
        in that case, never read). Only meaningful when RoPE is off
        (module docstring's "Positional encoding toggle" section): after
        this, `_cache_positions`' `embed_positions` output (`next_pos -
        pos_origin`) for any token appended *from now on* stays re-based
        to roughly the same small range it was in before this eviction,
        rather than drifting ever further from the small `0..window_len-1`
        range training's own full-sequence pass actually saw - exactly the
        reference's own "shift + re-based position-delta" approximation
        this file's own module docstring already flags as approximate
        (not exact) for the same reason `next_pos` itself is never
        rewound: tokens *already* cached before this call keep whatever
        (larger, pre-rebase) position their own embedding was originally
        looked up with baked in - there's no retroactive fix for those
        short of a full recompute."""
        if keep_last >= self.length:
            return
        drop = self.length - keep_last
        for layer in self.layers:
            if layer is not None:
                layer.k = layer.k[:, :, drop:, :]
                layer.v = layer.v[:, :, drop:, :]
        self.length = keep_last
        self.pos_origin += drop


class _CausalSelfAttention(nn.Module):
    """Separate `query`/`key`/`value` projections - not one fused `qkv`
    matmul - matching the reference's own
    `lzero/model/unizero_world_models/transformer.py::SelfAttention`
    (`self.key`/`self.query`/`self.value`, three independent `nn.Linear`s)
    rather than the fused-QKV convention this file used to take. Same
    FLOPs either way (one `3*embed_dim`-wide matmul vs three
    `embed_dim`-wide ones); the only real difference is three separate
    weight tensors to initialize/regularize instead of one shared one."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, rotary_emb: bool = True) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rotary_emb = rotary_emb
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.resid_drop = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attend_mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """`x`: `(B, L, E)`. `attend_mask`: `(B, L, L)` bool, `True` where
        query position `i` is allowed to attend to key position `j`
        (already combines causality with key-padding - see
        `_build_attend_mask`). `positions`: `(B, L)` long - each token's
        RoPE position (module docstring's "Positional encoding" section);
        rotates `q`/`k` only, never `v`, before the dot product - only
        when `self.rotary_emb` (module docstring's "Positional encoding
        toggle" section); when it's off, positional information already
        entered `x` additively upstream (`_CausalTransformer`'s own
        `pos_embed`), so there is nothing for attention itself to do with
        `positions` at all."""
        b, seq_len, embed_dim = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(self.query(x)), split_heads(self.key(x)), split_heads(self.value(x))
        if self.rotary_emb:
            cos, sin = _rope_cos_sin(positions, self.head_dim)  # (B, L, D)
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (B, 1, L, D) broadcasts over heads
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        mask = attend_mask.unsqueeze(1)  # (B, 1, L, L) broadcasts over heads
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, seq_len, embed_dim)
        return self.resid_drop(self.proj(out))

    def forward_incremental_batch(
        self, x_new: torch.Tensor, positions: torch.Tensor, layer_caches: list[_LayerKV | None],
    ) -> tuple[torch.Tensor, list[_LayerKV]]:
        """`x_new`: `(B, E)` - exactly one new token per lane (already
        LN'd by the caller, same as the full-sequence path's `ln1(x)`).
        `positions`: `(B,)` long - this new token's own RoPE position
        (each lane's cache's `next_pos`, module docstring's "Positional
        encoding" section); already-cached keys need no re-rotation -
        their rotation was baked in when *they* were the new token.
        `layer_caches[i]`: lane `i`'s existing `(1, H, L_i, D)` K/V for
        *this* layer, or `None` for an empty cache - `L_i` may differ
        across lanes (different lanes/nodes have different history
        depths), so every lane's K/V is padded to this call's own common
        max length for one batched attention call; the query is always
        the newest token for its lane, so it's unconditionally allowed to
        attend to every one of its own lane's real cached keys (no
        causal masking needed beyond padding - nothing is *later* than
        the newest token, so there's nothing to hide from it)."""
        b, embed_dim = x_new.shape

        def split_heads_1(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, self.num_heads, self.head_dim)

        q = split_heads_1(self.query(x_new))
        k_new = split_heads_1(self.key(x_new))
        v_new = split_heads_1(self.value(x_new))  # each (B, H, D)
        if self.rotary_emb:
            cos, sin = _rope_cos_sin(positions, self.head_dim)  # (B, D)
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (B, 1, D) broadcasts over heads
            q, k_new = _apply_rope(q, cos, sin), _apply_rope(k_new, cos, sin)
        lens = [0 if c is None else c.k.shape[2] for c in layer_caches]
        lmax = max(lens) if lens else 0
        k_all = torch.zeros(b, self.num_heads, lmax + 1, self.head_dim, device=x_new.device, dtype=x_new.dtype)
        v_all = torch.zeros_like(k_all)
        pad_mask = torch.zeros(b, lmax + 1, dtype=torch.bool, device=x_new.device)
        for i in range(b):
            n = lens[i]
            if n:
                k_all[i, :, :n] = layer_caches[i].k[0]
                v_all[i, :, :n] = layer_caches[i].v[0]
            k_all[i, :, n] = k_new[i]
            v_all[i, :, n] = v_new[i]
            pad_mask[i, : n + 1] = True
        mask = pad_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, Lmax+1) - one query per lane
        out = F.scaled_dot_product_attention(
            q.unsqueeze(2), k_all, v_all, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.squeeze(2).reshape(b, embed_dim)
        out = self.resid_drop(self.proj(out))
        new_layer_caches = [
            _LayerKV(k_all[i : i + 1, :, : lens[i] + 1].clone(), v_all[i : i + 1, :, : lens[i] + 1].clone())
            for i in range(b)
        ]
        return out, new_layer_caches


class _TransformerBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float, rotary_emb: bool = True,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = _CausalSelfAttention(embed_dim, num_heads, dropout, rotary_emb)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ffn_multiplier * embed_dim), nn.GELU(),
            nn.Linear(ffn_multiplier * embed_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attend_mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attend_mask, positions)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_incremental_batch(
        self, x: torch.Tensor, positions: torch.Tensor, layer_caches: list[_LayerKV | None],
    ) -> tuple[torch.Tensor, list[_LayerKV]]:
        """`x`: `(B, E)` - one new token per lane's residual stream."""
        attn_out, new_layer_caches = self.attn.forward_incremental_batch(self.ln1(x), positions, layer_caches)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_layer_caches


def _build_attend_mask(pad_mask: torch.Tensor) -> torch.Tensor:
    """`pad_mask`: `(B, L)` bool, `True` = real token, `False` = leading
    context padding (a token slot that predates this episode's start -
    never appended-to-then-later-invalid, since real tokens are only ever
    appended at the *end* of a node's sequence, so once a position is real
    it stays real for the rest of that sequence's life). Combines with a
    plain lower-triangular causal mask so no query ever attends to a pad
    position *or* a future position."""
    b, seq_len = pad_mask.shape
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=pad_mask.device))
    key_ok = pad_mask.unsqueeze(1).expand(b, seq_len, seq_len)
    mask = causal.unsqueeze(0) & key_ok
    # A pad *query* position (there's no such thing as a pad key that's
    # also never a query - every position is both) whose row has zero
    # allowed keys would make softmax divide 0/0 -> NaN, which then
    # poisons every *valid* query's output too (a masked key's attention
    # *weight* is exactly 0, but `0 * NaN` is still `NaN` in IEEE754 - the
    # classic masked-attention footgun). Force every position to always
    # be able to attend to itself so its own row is never all-`False`;
    # harmless for real (non-pad) positions since diagonal attention was
    # already implied by causality, and pad positions' own (now finite,
    # if meaningless) output is never read by anything - `key_ok` still
    # keeps every *other* query from ever attending to a pad key.
    eye = torch.eye(seq_len, dtype=torch.bool, device=pad_mask.device).unsqueeze(0)
    return mask | eye


class _CausalTransformer(nn.Module):
    def __init__(
        self, embed_dim: int, num_layers: int, num_heads: int, dropout: float,
        rotary_emb: bool = True, max_positions: int = 4096, ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        head_dim = embed_dim // num_heads
        if rotary_emb:
            assert head_dim % 2 == 0, "RoPE needs an even head_dim (embed_dim / num_heads)"
        self.embed_dim = embed_dim
        self.rotary_emb = rotary_emb
        self.max_positions = max_positions
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(embed_dim, num_heads, dropout, rotary_emb, ffn_multiplier)
                for _ in range(num_layers)
            ],
        )
        self.ln_out = nn.LayerNorm(embed_dim)
        # Learned absolute position embedding - only built/used when RoPE
        # is off (module docstring's "Positional encoding toggle"
        # section); added straight to the raw token embedding, once,
        # right here (not inside `_CausalSelfAttention` - unlike RoPE's
        # per-head QK rotation, an additive embedding has nothing
        # attention-specific about it), so every block downstream sees an
        # already-position-aware residual stream and needs no further
        # positional handling of its own either way.
        self.pos_embed = nn.Embedding(max_positions, embed_dim) if not rotary_emb else None

    def _add_pos_embed(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is None:
            return x
        return x + self.pos_embed(positions.clamp(0, self.max_positions - 1))

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """`tokens`: `(B, L, E)` raw embeddings. `pad_mask`: `(B, L)` bool,
        `True` = real. `positions`: `(B, L)` long, or `None` to default to
        plain `0..L-1` for every lane - safe under RoPE regardless of what
        absolute position the same tokens would carry anywhere else (see
        module docstring's "Positional encoding" section); also exactly
        the range training's own additive `pos_embed` (when RoPE's off)
        was itself trained on, so the same default doubles as the correct
        embedding-table index with no separate "rebased" value needed
        here (unlike the incremental path below - see
        `_cache_positions`). Returns post-Transformer hidden states, same
        shape as `tokens`."""
        b, seq_len, _ = tokens.shape
        if positions is None:
            positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(b, seq_len)
        x = self._add_pos_embed(tokens, positions)
        x = self.drop(x)
        attend_mask = _build_attend_mask(pad_mask)
        for block in self.blocks:
            x = block(x, attend_mask, positions)
        return self.ln_out(x)

    def forward_incremental_batch(
        self, token_embeds: torch.Tensor, positions: torch.Tensor, caches: list[_TransformerCache | None],
        embed_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[_TransformerCache]]:
        """`token_embeds`: `(B, E)` - one new raw token per lane.
        `positions`: `(B,)` long - this token's own RoPE position;
        normally just `caches[i].next_pos` (or `0` if `None`), but passed
        explicitly since `_replay_context_to_cache` sometimes needs to
        skip a lane for one round (see that method). Always used for
        `cache.next_pos` bookkeeping below regardless of `self.
        rotary_emb` - it's the one true monotonic absolute counter every
        caller (`_cache_positions`) derives everything else from.
        `embed_positions`: `(B,)` long, or `None` to default to
        `positions` - only consulted when RoPE is off, as the additive
        `pos_embed` table's lookup index; callers with an evicting,
        persistent cache (module docstring's "Positional encoding
        toggle" section) pass their own *rebased* value here
        (`_cache_positions`' own second return) instead of raw `positions`
        so that index stays small even arbitrarily late in a long
        episode. `caches[i]`: lane `i`'s existing cache, or `None`.
        Returns the new tokens' post-Transformer hidden states (`(B, E)`)
        and one fresh `_TransformerCache` per lane - the ones passed in
        are never mutated (see `_TransformerCache`'s own docstring)."""
        b = token_embeds.shape[0]
        num_layers = len(self.blocks)
        x = self._add_pos_embed(token_embeds, positions if embed_positions is None else embed_positions)
        x = self.drop(x)
        layer_caches_by_layer = [[c.layers[i] if c is not None else None for c in caches] for i in range(num_layers)]
        new_layer_caches_by_layer: list[list[_LayerKV]] = []
        for i, block in enumerate(self.blocks):
            x, new_layer_caches = block.forward_incremental_batch(x, positions, layer_caches_by_layer[i])
            new_layer_caches_by_layer.append(new_layer_caches)
        hidden = self.ln_out(x)
        # One device synchronization for the whole batch, not one
        # `positions[lane].item()` synchronization per lane. This method
        # is called 67 times per collect iteration at the default search
        # budget, so the previous scalar loop caused 1600+ avoidable
        # GPU->CPU sync points at `num_envs=24`.
        position_values = positions.detach().cpu().tolist()
        new_caches: list[_TransformerCache] = []
        for lane in range(b):
            cache = _TransformerCache(num_layers)
            parent = caches[lane]
            cache.length = (0 if parent is None else parent.length) + 1
            # `positions[lane]` (this new token's own RoPE position), not
            # `parent.next_pos`, is authoritative - a caller may pass a
            # position that doesn't equal the parent's own `next_pos`
            # (see `_replay_context_to_cache`'s per-round `active` lanes).
            cache.next_pos = int(position_values[lane]) + 1
            # `pos_origin` simply carries over unchanged - only
            # `evict_front` ever bumps it (module docstring's "Positional
            # encoding toggle" section); a plain incremental append never
            # evicts anything.
            cache.pos_origin = 0 if parent is None else parent.pos_origin
            for i in range(num_layers):
                cache.layers[i] = new_layer_caches_by_layer[i][lane]
            new_caches.append(cache)
        return hidden, new_caches


# ----------------------------------------------------------------------
# Heads - read off specific token positions' hidden states (see module
# docstring's "Head placement" bullet).
# ----------------------------------------------------------------------
class _Heads(nn.Module):
    def __init__(
        self, embed_dim: int, hidden_dim: int, action_space: gym.Space, support_size: int,
        component_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        self.support_size = support_size
        self.trunk = (
            build_component_mlp(embed_dim, hidden_dim, component_config, final_activation=nn.ELU())
            if component_config is not None
            else nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ELU())
        )
        self.value_head = nn.Linear(hidden_dim, 2 * support_size + 1)
        self.reward_head = nn.Linear(hidden_dim, 2 * support_size + 1)
        # Predicted *raw* next-token embedding - fed straight back in as
        # the imagined next "obs" token during search (see
        # `_step_imagine`); trained via the consistency loss in
        # `_train_step` to actually approximate what `_Tokenizer` would
        # have produced from the real next observation.
        self.latent_head = nn.Linear(hidden_dim, embed_dim)
        if self.discrete:
            self.action_dim = int(action_space.n)
            self.policy_head = nn.Linear(hidden_dim, self.action_dim)
        else:
            self.action_dim = int(np.prod(action_space.shape))
            self.mean_head = nn.Linear(hidden_dim, self.action_dim)
            self.log_std_head = nn.Linear(hidden_dim, self.action_dim)
            low = np.where(np.isfinite(action_space.low), action_space.low, -1.0).reshape(-1)
            high = np.where(np.isfinite(action_space.high), action_space.high, 1.0).reshape(-1)
            self.register_buffer("action_scale", torch.as_tensor((high - low) / 2.0, dtype=torch.float32))
            self.register_buffer("action_bias", torch.as_tensor((high + low) / 2.0, dtype=torch.float32))

    def value_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(h))

    def value(self, h: torch.Tensor) -> torch.Tensor:
        return _logits_to_scalar(self.value_logits(h), self.support_size)

    def reward_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.reward_head(self.trunk(h))

    def reward(self, h: torch.Tensor) -> torch.Tensor:
        return _logits_to_scalar(self.reward_logits(h), self.support_size)

    def latent(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.latent_head(self.trunk(h)))

    def policy_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.trunk(h))

    def policy_gaussian(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.trunk(h)
        mean = torch.tanh(self.mean_head(t)) * self.action_scale + self.action_bias
        log_std = self.log_std_head(t).clamp(-5.0, 1.0)
        return mean, log_std.exp()


class _Projector(nn.Module):
    """SimSiam-style projector `P1` (module docstring's "no target
    network" section) - `efficientzero.py`'s own class, ported verbatim.
    `BatchNorm1d` on the hidden layer is not decorative - Chen & He,
    2021's own ablation (Table 2c) shows stop-gradient *alone* still
    collapses in practice (every latent converging to ~the same vector,
    which trivially "matches" whatever the predictor outputs and costs
    nothing at all to reach); BN is what empirically keeps that from
    happening by forcing every batch's projected features to actually
    spread out (zero mean, unit variance *per feature, across the batch*)
    instead of letting the optimizer collapse them all together."""

    def __init__(
        self, embed_dim: int, proj_dim: int, component_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.net = (
            build_component_mlp(embed_dim, proj_dim, component_config)
            if component_config is not None
            else nn.Sequential(
                nn.Linear(embed_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ELU(), nn.Linear(proj_dim, proj_dim),
            )
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _Predictor(nn.Module):
    """SimSiam-style predictor `P2` - only applied on the *predicted*
    (dynamics) branch, never the stop-gradiented real-encoding branch
    (asymmetry), with the same anti-collapse `BatchNorm1d` as `_Projector`
    on its hidden layer - see that class's docstring. `efficientzero.py`'s
    own class, ported verbatim."""

    def __init__(self, proj_dim: int, component_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.net = (
            build_component_mlp(proj_dim, proj_dim, component_config)
            if component_config is not None
            else nn.Sequential(
                nn.Linear(proj_dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ELU(), nn.Linear(proj_dim, proj_dim),
            )
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ----------------------------------------------------------------------
# Gumbel-Top-k + Sequential Halving search tree (Danihelka et al., 2022) -
# `efficientzero.py`'s own search algorithm, ported verbatim except for
# one adaptation: a node's "state" is its own incremental KV-cache
# (`cache`, extended by two tokens - one action, one predicted-observation
# - per tree edge, see `_step_imagine`) instead of a fixed-size latent +
# LSTM cell, and a node's reward is a plain per-edge scalar (UniZero's
# reward head predicts one transition's real reward directly, no LSTM
# "value prefix" cumulative-sum trick to unwind) rather than
# `efficientzero.py`'s own value-prefix delta - see module docstring's
# "Search" section.
# ----------------------------------------------------------------------
class _MinMaxStats:
    """Running min/max of every backed-up value seen so far in one search
    tree, used to normalize Q-values into `[0, 1]` before they're compared
    against each other (Danihelka et al., 2022, Appendix A) - without this
    a tree whose rewards/values happen to live on a different scale than
    `c_visit`/`c_scale` were tuned for would search essentially randomly."""

    def __init__(self, delta: float) -> None:
        self.maximum = -float("inf")
        self.minimum = float("inf")
        self.delta = max(1e-6, delta)

    def update(self, value: float) -> None:
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            value = min(max(value, self.minimum), self.maximum)
            value = (value - self.minimum) / max(self.maximum - self.minimum, self.delta)
        return min(max(value, 0.0), 1.0)


class _SearchNode:
    """One node of the Gumbel search tree - either a root (one per lane
    being searched this real step) or an imagined node reached from its
    parent by taking one of the parent's candidate actions and stepping
    the dynamics (`_step_imagine`). Every candidate action slot gets its
    own (initially un-expanded) child node the instant its parent is
    expanded - "un-expanded child" and "child the tree hasn't
    visited/expanded yet" are the same thing here, exactly like the
    reference tree `efficientzero.py`'s own copy of this class docstring
    describes."""

    __slots__ = (
        "prior", "parent", "children", "visit_count", "value_sum", "reward_value",
        "cache", "candidate_action", "selected_children_idx",
    )

    def __init__(self, prior: float, parent: "_SearchNode | None" = None) -> None:
        self.prior = prior
        self.parent = parent
        self.children: list[_SearchNode] = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward_value = 0.0
        self.cache: _TransformerCache | None = None  # this node's own KV-cache (see `_step_imagine`)
        self.candidate_action: Any = None  # continuous only: sampled action leading to this node
        self.selected_children_idx: list[int] = []  # root only: Sequential Halving survivors

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def reward(self) -> float:
        return self.reward_value

    def expand(
        self, priors: np.ndarray, cache: _TransformerCache | None, reward_value: float,
        candidate_actions: list[Any] | None = None,
    ) -> None:
        self.cache = cache
        self.reward_value = reward_value
        self.children = [_SearchNode(float(p), self) for p in priors]
        if candidate_actions is not None:
            for child, action in zip(self.children, candidate_actions):
                child.candidate_action = action

    def children_visit_sum(self) -> int:
        return sum(c.visit_count for c in self.children)

    def q_of_child(self, idx: int, discount: float) -> float:
        child = self.children[idx]
        return child.reward() + discount * child.value()

    def v_mix(self, discount: float) -> float:
        """Appendix D of Danihelka et al., 2022: blend this node's own raw
        value estimate with however many of its children are already
        expanded, weighted by how much of the node's total prior
        probability mass those expanded children cover - gives
        `completed_qs` something better than "value at the root" to use
        as a stand-in Q for children the tree hasn't visited at all yet."""
        priors = np.array([c.prior for c in self.children], dtype=np.float64)
        pi = _softmax(priors)
        pi_sum = 0.0
        pi_q_sum = 0.0
        for i, child in enumerate(self.children):
            if child.expanded():
                pi_sum += pi[i]
                pi_q_sum += pi[i] * self.q_of_child(i, discount)
        if pi_sum < 1e-6:
            return self.value()
        visit_sum = self.children_visit_sum()
        return (self.value() + visit_sum * pi_q_sum / pi_sum) / (1.0 + visit_sum)

    def completed_qs(self, discount: float, minmax: _MinMaxStats) -> np.ndarray:
        v_mix = self.v_mix(discount)
        out = np.empty(len(self.children), dtype=np.float64)
        for i, child in enumerate(self.children):
            raw = self.q_of_child(i, discount) if child.expanded() else v_mix
            out[i] = minmax.normalize(raw)
        return out

    def improved_policy(self, transformed_completed_qs: np.ndarray) -> np.ndarray:
        priors = np.array([c.prior for c in self.children], dtype=np.float64)
        return _softmax(priors + transformed_completed_qs)


def _transformed_completed_qs(node: _SearchNode, minmax: _MinMaxStats, discount: float, c_visit: float, c_scale: float) -> np.ndarray:
    completed = node.completed_qs(discount, minmax)
    max_visit = max((c.visit_count for c in node.children), default=0)
    return (c_visit + max_visit) * c_scale * completed


def _select_action(node: _SearchNode, minmax: _MinMaxStats, discount: float, c_visit: float, c_scale: float) -> int:
    """Root: equal-visit round robin over the current Sequential-Halving
    survivors (`selected_children_idx`) - ties broken towards whichever
    survivor is earliest in Gumbel-Top-k rank, matching the reference
    `do_equal_visit`. Non-root: the paper's deterministic selection rule -
    argmax of (improved policy - visit fraction) over *all* children, no
    Sequential Halving below the root."""
    if node.parent is None:
        best_idx, best_visits = -1, float("inf")
        for idx in node.selected_children_idx:
            visits = node.children[idx].visit_count
            if visits < best_visits:
                best_visits, best_idx = visits, idx
        return best_idx
    transformed = _transformed_completed_qs(node, minmax, discount, c_visit, c_scale)
    improved = node.improved_policy(transformed)
    visits = np.array([c.visit_count for c in node.children], dtype=np.float64)
    denom = 1.0 + node.children_visit_sum()
    scores = improved - visits / denom
    return int(np.argmax(scores))


def _sequential_halving(
    root: _SearchNode, gumbel: np.ndarray, minmax: _MinMaxStats, keep: int, discount: float, c_visit: float, c_scale: float,
) -> None:
    """Cut the root's currently-surviving candidate set in half (by score =
    Gumbel noise + prior + transformed completed Q), same formula the
    reference `sequential_halving` uses past its first phase - the first
    phase (Gumbel-Top-k over *all* actions) is instead folded into where
    `selected_children_idx` gets initialized (simulation 0 of `search()`
    below), which is exactly equivalent given the reference's own
    equal-visit round robin always exhausts the top `num_top_actions`
    candidates before any lower-ranked one ever gets a look."""
    selected = root.selected_children_idx
    if len(selected) <= 1:
        return
    priors = np.array([c.prior for c in root.children], dtype=np.float64)
    transformed = _transformed_completed_qs(root, minmax, discount, c_visit, c_scale)
    scores = np.array([gumbel[i] + priors[i] + transformed[i] for i in selected])
    order = np.argsort(-scores)
    keep = max(1, min(keep, len(selected)))
    root.selected_children_idx = [selected[i] for i in order[:keep]]


def _backpropagate(path: list[_SearchNode], leaf_value: float, minmax: _MinMaxStats, discount: float) -> None:
    value = leaf_value
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        value = node.reward() + discount * value
        minmax.update(value)


class _HalvingSchedule:
    """Allocate complete equal-visit rounds before every halving.

    The reference floor formula is retained whenever it allocates at least
    one round.  Its zero-round corner case (for example ``n=32,m=16``)
    previously collapsed to one simulation and eliminated fifteen
    candidates before they were visited.  Here every non-final phase gets
    at least ``current_top`` simulations, while ``m`` is capped at half the
    total budget whenever that is feasible.  The final phase consumes the
    exact remainder, so the total remains exactly ``n``.
    """

    def __init__(self, num_simulations: int, num_top_actions: int) -> None:
        self.n = max(1, num_simulations)
        maximum_top = max(2, self.n // 2)
        self.m = max(2, min(num_top_actions, maximum_top))
        self.current_top = self.m
        self.next_cutoff = self._span(self.current_top, used=0)

    def _span(self, current_top: int, used: int) -> int:
        log2m = max(math.log2(self.m), 1e-9)
        if current_top > 2:
            reference_span = (
                math.floor(self.n / (log2m * current_top)) * current_top
            )
            span = max(current_top, reference_span)
        else:
            span = self.n - used
        return min(self.n, max(1, int(span)))

    def maybe_advance(self, simulation_idx: int) -> bool:
        ready = (simulation_idx + 1) >= self.next_cutoff
        if ready and self.current_top > 1:
            used = self.next_cutoff
            self.current_top = max(1, self.current_top // 2)
            self.next_cutoff = min(self.n, used + self._span(self.current_top, used))
        return ready


# ----------------------------------------------------------------------
# Replay buffer - PER + reanalyze support ported verbatim from
# `efficientzero.py`'s `_EfficientZeroBuffer` (see that class's own
# docstring), extended with `context_obs`/`context_action`/`context_valid`
# slicing so every sample also carries the *preceding* real context a
# Transformer window needs (episodes are stored as contiguous arrays, so
# this is a pure slice - no extra storage).
# ----------------------------------------------------------------------
class _ResearchImZeroBuffer:
    _NO_SEARCH_VALUE = -1e9
    _STATE_SCHEMA_VERSION = 1

    def __init__(
        self, capacity_episodes: int, obs_shape: tuple[int, ...], action_dim: int, policy_target_dim: int,
        context_length: int, num_lanes: int = 1, priority_alpha: float = 1.0, priority_beta: float = 1.0,
        min_priority: float = 1e-6, recent_fraction: float = 0.5, recent_window: int = 512,
        success_fraction: float = 0.0, success_top_quantile: float = 0.25,
    ) -> None:
        self.capacity = max(1, capacity_episodes)
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.policy_target_dim = policy_target_dim
        self.context_length = max(0, context_length)
        self.priority_alpha = max(0.0, priority_alpha)
        self.priority_beta = max(0.0, priority_beta)
        self.min_priority = max(1e-8, min_priority)
        self.recent_fraction = min(1.0, max(0.0, recent_fraction))
        self.recent_window = max(1, recent_window)
        self.success_fraction = min(1.0, max(0.0, success_fraction))
        if self.recent_fraction + self.success_fraction > 1.0 + 1e-12:
            raise ValueError("replay_recent_fraction + replay_success_fraction must be <= 1")
        self.success_top_quantile = min(1.0, max(1e-6, success_top_quantile))
        self.episodes: list[dict[str, np.ndarray]] = []
        self._episode_alpha_sum: list[float] = []
        self._episode_returns: list[float] = []
        # `sample_for_reanalyze`'s own bookkeeping (module docstring's
        # "Schedules, not fixed constants" section) - `_reanalyze_
        # generation` is a plain call counter (one "generation" per
        # `update_reanalyzed_targets` call, i.e. per real `_reanalyze()`
        # pass), `_episode_last_reanalyzed_gen[i]` the most recent
        # generation that touched *any* transition in episode `i` (`-1`:
        # never). Together they let `sample_for_reanalyze` prefer
        # episodes/transitions that are both high-priority *and* long
        # overdue for a refresh, instead of picking uniformly by episode
        # length regardless of either.
        self._episode_last_reanalyzed_gen: list[float] = []
        self._reanalyze_generation = 0
        self._max_priority = 1.0
        self._cur: list[dict[str, list[Any]]] = [self._new_episode() for _ in range(max(1, num_lanes))]

    @staticmethod
    def _new_episode() -> dict[str, list[Any]]:
        return {
            "obs": [], "action": [], "reward": [], "next_obs": [], "policy_target": [],
            "priority": [], "raw_learning_error": [], "search_value": [], "reanalyzed_gen": [],
            "policy_target_valid": [], "search_budget": [], "model_expansions": [],
            "behavior_policy": [], "audit_type": [],
            "stage_index": [], "stage_id": [], "nominal_stage_budget": [],
            "requested_budget": [], "predicted_stage_index": [],
            "predicted_stage_id": [], "predicted_requested_budget": [],
            "executed_search_expansions": [], "common_evaluator_expansions": [],
            "total_model_calls": [], "termination_reason": [],
        }

    def _packed_active(
        self,
        active: dict[str, list[Any]],
    ) -> dict[str, np.ndarray]:
        n = len(active["reward"])
        array_shapes = {
            "obs": self.obs_shape,
            "action": (self.action_dim,),
            "next_obs": self.obs_shape,
            "policy_target": (self.policy_target_dim,),
            "behavior_policy": (self.policy_target_dim,),
        }
        packed: dict[str, np.ndarray] = {}
        for key in self._new_episode():
            values = active[key]
            if key in array_shapes:
                packed[key] = (
                    np.stack(values).astype(np.float32, copy=True)
                    if n
                    else np.empty((0, *array_shapes[key]), dtype=np.float32)
                )
            else:
                packed[key] = np.asarray(values).copy()
        return packed

    def _validated_store_copy(
        self,
        store: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        required = set(self._new_episode())
        if not isinstance(store, dict) or not required.issubset(store):
            missing = sorted(required.difference(store if isinstance(store, dict) else {}))
            raise ValueError(f"Invalid ResearchImZero replay store; missing fields: {missing}")
        copied = {
            key: np.asarray(store[key]).copy()
            for key in required
        }
        n = len(copied["reward"])
        for key, value in copied.items():
            if value.ndim == 0 or value.shape[0] != n:
                raise ValueError(
                    f"Invalid ResearchImZero replay field {key!r}: expected length {n}",
                )
        expected_shapes = {
            "obs": (n, *self.obs_shape),
            "next_obs": (n, *self.obs_shape),
            "action": (n, self.action_dim),
            "policy_target": (n, self.policy_target_dim),
            "behavior_policy": (n, self.policy_target_dim),
        }
        for key, shape in expected_shapes.items():
            if copied[key].shape != shape:
                raise ValueError(
                    f"ResearchImZero replay {key!r} shape {copied[key].shape} "
                    f"does not match {shape}",
                )
        return copied

    def state_dict(self) -> dict[str, Any]:
        """Return an explicit CPU/numpy replay snapshot.

        Checkpoints intentionally include both finalized episodes and live
        lanes: dropping live lanes loses the newest transitions and their
        reanalysis/PER state on every resume.
        """
        return {
            "schema_version": self._STATE_SCHEMA_VERSION,
            "obs_shape": tuple(self.obs_shape),
            "action_dim": self.action_dim,
            "policy_target_dim": self.policy_target_dim,
            "num_lanes": len(self._cur),
            "episodes": [
                {key: np.asarray(value).copy() for key, value in episode.items()}
                for episode in self.episodes
            ],
            "active_lanes": [self._packed_active(active) for active in self._cur],
            "episode_alpha_sum": np.asarray(
                self._episode_alpha_sum, dtype=np.float64,
            ),
            "episode_returns": np.asarray(
                self._episode_returns, dtype=np.float64,
            ),
            "episode_last_reanalyzed_gen": np.asarray(
                self._episode_last_reanalyzed_gen, dtype=np.float64,
            ),
            "reanalyze_generation": int(self._reanalyze_generation),
            "max_priority": float(self._max_priority),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("schema_version") != self._STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported ResearchImZero replay checkpoint schema")
        if (
            tuple(state.get("obs_shape", ())) != tuple(self.obs_shape)
            or int(state.get("action_dim", -1)) != self.action_dim
            or int(state.get("policy_target_dim", -1)) != self.policy_target_dim
        ):
            raise ValueError("ResearchImZero replay dimensions do not match the environment")
        active_lanes = state.get("active_lanes", [])
        if int(state.get("num_lanes", -1)) != len(active_lanes):
            raise ValueError("ResearchImZero replay lane bookkeeping is inconsistent")

        saved_episodes = [
            self._validated_store_copy(episode)
            for episode in state.get("episodes", [])
        ]
        saved_last = np.asarray(
            state.get("episode_last_reanalyzed_gen", []),
            dtype=np.float64,
        )
        if len(saved_last) != len(saved_episodes):
            raise ValueError("ResearchImZero replay reanalysis bookkeeping is inconsistent")
        keep_from = max(0, len(saved_episodes) - self.capacity)
        self.episodes = saved_episodes[keep_from:]
        self._episode_last_reanalyzed_gen = saved_last[keep_from:].tolist()
        # Alpha sums and success returns are derived from the validated
        # transition arrays; recomputing prevents stale/corrupt sampling mass.
        self._episode_alpha_sum = [
            float(np.sum(episode["priority"].astype(np.float64) ** self.priority_alpha))
            for episode in self.episodes
        ]
        self._episode_returns = [
            float(episode["reward"].astype(np.float64).sum())
            for episode in self.episodes
        ]

        # A checkpoint may be inspected/resumed with fewer vector lanes.
        # Lane identity cannot be merged safely, so restore the matching
        # prefix and drop surplus live lanes rather than overflowing or
        # pretending their unfinished tails were terminal episodes.
        restored_active = [
            self._validated_store_copy(active)
            for active in active_lanes[: len(self._cur)]
        ]
        self._cur = [self._new_episode() for _ in self._cur]
        array_fields = {
            "obs", "action", "next_obs", "policy_target", "behavior_policy",
        }
        for lane, packed in enumerate(restored_active):
            for key, values in packed.items():
                self._cur[lane][key] = (
                    [row.copy() for row in values]
                    if key in array_fields
                    else values.tolist()
                )
        self._reanalyze_generation = max(
            0, int(state.get("reanalyze_generation", 0)),
        )
        saved_max_priority = float(state.get("max_priority", 1.0))
        if not np.isfinite(saved_max_priority):
            raise ValueError("ResearchImZero replay max priority must be finite")
        self._max_priority = max(self.min_priority, saved_max_priority)

    def _pad_one(self, obs_hist: list[np.ndarray], action_hist: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Right-pads (valid transitions packed at the *front*, in
        temporal order; padding, if any, trails at the end) rather than
        the more common "left-pad so the most recent is always at a fixed
        trailing slot" scheme - deliberately, so that a token's position
        (`_CausalTransformer`'s positional-embedding index) can always
        just be its plain array index with *no* separate position lookup,
        in both this class's full-sequence callers (`NativeResearchImZero.
        _embed_root_batch`/`_train_step`) and the incremental KV-cache
        path (`NativeResearchImZero._replay_context_to_cache`) - a
        short-history lane's real tokens get the exact same position
        numbers a long-history lane's corresponding real tokens would.
        Harmless either way under RoPE (module docstring's "Positional
        encoding" section - only *relative* offsets between tokens in one
        attention computation matter), but kept for the same reason it
        always was: one less thing to keep in sync between callers."""
        length = self.context_length
        ctx_obs = np.zeros((length, *self.obs_shape), dtype=np.float32)
        ctx_action = np.zeros((length, self.action_dim), dtype=np.float32)
        ctx_valid = np.zeros(length, dtype=bool)
        n = len(obs_hist)
        if n:
            ctx_obs[:n] = np.stack(obs_hist)
            ctx_action[:n] = np.stack(action_hist)
            ctx_valid[:n] = True
        return ctx_obs, ctx_action, ctx_valid

    def add(
        self, obs: np.ndarray, action_flat: np.ndarray, reward: float, next_obs: np.ndarray,
        policy_target: np.ndarray, done: bool, lane: int = 0,
        policy_target_valid: bool = True, search_budget: int = 0,
        model_expansions: int = 0, behavior_policy: np.ndarray | None = None,
        audit_type: str = "none",
        stage_index: int = -1, stage_id: str = "none",
        nominal_stage_budget: int | None = None,
        requested_budget: int | None = None,
        predicted_stage_index: int = -1, predicted_stage_id: str = "none",
        predicted_requested_budget: int | None = None,
        executed_search_expansions: int | None = None,
        common_evaluator_expansions: int = 0,
        total_model_calls: int | None = None,
        termination_reason: str = "budget_exhausted",
    ) -> None:
        cur = self._cur[lane]
        cur["obs"].append(np.asarray(obs, dtype=np.float32))
        cur["action"].append(np.asarray(action_flat, dtype=np.float32))
        cur["reward"].append(float(reward))
        cur["next_obs"].append(np.asarray(next_obs, dtype=np.float32))
        cur["policy_target"].append(np.asarray(policy_target, dtype=np.float32))
        cur["priority"].append(self._max_priority)
        cur["raw_learning_error"].append(float("nan"))
        cur["search_value"].append(self._NO_SEARCH_VALUE)
        cur["reanalyzed_gen"].append(-1)
        cur["policy_target_valid"].append(bool(policy_target_valid))
        cur["search_budget"].append(int(search_budget))
        cur["model_expansions"].append(int(model_expansions))
        cur["behavior_policy"].append(
            np.asarray(
                policy_target if behavior_policy is None else behavior_policy,
                dtype=np.float32,
            )
        )
        cur["audit_type"].append(str(audit_type))
        requested = search_budget if requested_budget is None else requested_budget
        cur["stage_index"].append(int(stage_index))
        cur["stage_id"].append(str(stage_id))
        cur["nominal_stage_budget"].append(
            int(requested if nominal_stage_budget is None else nominal_stage_budget)
        )
        cur["requested_budget"].append(int(requested))
        cur["predicted_stage_index"].append(int(predicted_stage_index))
        cur["predicted_stage_id"].append(str(predicted_stage_id))
        cur["predicted_requested_budget"].append(
            int(requested if predicted_requested_budget is None else predicted_requested_budget)
        )
        cur["executed_search_expansions"].append(
            int(model_expansions if executed_search_expansions is None else executed_search_expansions)
        )
        cur["common_evaluator_expansions"].append(int(common_evaluator_expansions))
        cur["total_model_calls"].append(
            int(model_expansions if total_model_calls is None else total_model_calls)
        )
        cur["termination_reason"].append(str(termination_reason))
        if done:
            self._flush_episode(lane)

    def _flush_episode(self, lane: int = 0) -> None:
        cur = self._cur[lane]
        if not cur["obs"]:
            return
        ep_len = len(cur["obs"])
        episode = {
            "obs": np.stack(cur["obs"]),
            "action": np.stack(cur["action"]),
            "reward": np.asarray(cur["reward"], dtype=np.float32),
            "next_obs": np.stack(cur["next_obs"]),
            "policy_target": np.stack(cur["policy_target"]),
            "priority": np.asarray(cur["priority"], dtype=np.float64),
            "raw_learning_error": np.asarray(cur["raw_learning_error"], dtype=np.float64),
            "search_value": np.asarray(cur["search_value"], dtype=np.float32),
            # `-1`: never reanalyzed - `sample_for_reanalyze`'s own
            # per-transition staleness tracking (see `__init__`'s own
            # comment on `_episode_last_reanalyzed_gen`).
            "reanalyzed_gen": np.asarray(cur["reanalyzed_gen"], dtype=np.int64),
            "policy_target_valid": np.asarray(cur["policy_target_valid"], dtype=bool),
            "search_budget": np.asarray(cur["search_budget"], dtype=np.int64),
            "model_expansions": np.asarray(cur["model_expansions"], dtype=np.int64),
            "behavior_policy": np.stack(cur["behavior_policy"]),
            "audit_type": np.asarray(cur["audit_type"], dtype=object),
            "stage_index": np.asarray(cur["stage_index"], dtype=np.int64),
            "stage_id": np.asarray(cur["stage_id"], dtype=object),
            "nominal_stage_budget": np.asarray(cur["nominal_stage_budget"], dtype=np.int64),
            "requested_budget": np.asarray(cur["requested_budget"], dtype=np.int64),
            "predicted_stage_index": np.asarray(cur["predicted_stage_index"], dtype=np.int64),
            "predicted_stage_id": np.asarray(cur["predicted_stage_id"], dtype=object),
            "predicted_requested_budget": np.asarray(cur["predicted_requested_budget"], dtype=np.int64),
            "executed_search_expansions": np.asarray(cur["executed_search_expansions"], dtype=np.int64),
            "common_evaluator_expansions": np.asarray(cur["common_evaluator_expansions"], dtype=np.int64),
            "total_model_calls": np.asarray(cur["total_model_calls"], dtype=np.int64),
            "termination_reason": np.asarray(cur["termination_reason"], dtype=object),
        }
        self.episodes.append(episode)
        self._episode_alpha_sum.append(float(np.sum(episode["priority"] ** self.priority_alpha)))
        self._episode_returns.append(float(episode["reward"].sum()))
        self._episode_last_reanalyzed_gen.append(-1.0)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)
            self._episode_alpha_sum.pop(0)
            self._episode_returns.pop(0)
            self._episode_last_reanalyzed_gen.pop(0)
        self._cur[lane] = self._new_episode()

    def __len__(self) -> int:
        return sum(len(ep["reward"]) for ep in self.episodes) + sum(len(cur["reward"]) for cur in self._cur)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_replay_transitions(self) -> int:
        """Every transition whose next observation is already available.

        N-step TD does not need to wait for an episode to end: the newest
        available transition in a live lane can bootstrap from its known
        ``next_obs``.  Exposing active lanes here removes the long-horizon
        starvation where NetHack collected tens of thousands of useful
        transitions before the first few ``done`` signals.
        """
        return len(self)

    @property
    def reanalyze_generation(self) -> int:
        """Current `_reanalyze_generation` counter - `_train_step`'s own
        freshness gate on `search_value` reads this to compute how many
        generations old a given sampled transition's own `search_value`
        is (see `DEFAULT_HYPERPARAMS["search_value_max_staleness"]`'s own
        docstring)."""
        return self._reanalyze_generation

    def _context_for(self, ep: dict[str, np.ndarray], start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = self.context_length
        lo = max(0, start - length)
        obs_hist = list(ep["obs"][lo:start])
        action_hist = list(ep["action"][lo:start])
        return self._pad_one(obs_hist, action_hist)

    def sample(self, batch_size: int, unroll_steps: int, td_steps: int, gamma: float) -> dict[str, np.ndarray]:
        length = self.context_length
        ctx_obs = np.zeros((batch_size, length, *self.obs_shape), dtype=np.float32)
        ctx_action = np.zeros((batch_size, length, self.action_dim), dtype=np.float32)
        ctx_valid = np.zeros((batch_size, length), dtype=bool)
        obs0 = np.zeros((batch_size, *self.obs_shape), dtype=np.float32)
        action = np.zeros((batch_size, unroll_steps, self.action_dim), dtype=np.float32)
        reward = np.zeros((batch_size, unroll_steps), dtype=np.float32)
        next_obs = np.zeros((batch_size, unroll_steps, *self.obs_shape), dtype=np.float32)
        policy_target = np.zeros((batch_size, unroll_steps, self.policy_target_dim), dtype=np.float32)
        policy_target_valid = np.ones((batch_size, unroll_steps), dtype=np.float32)
        search_budget = np.zeros((batch_size, unroll_steps), dtype=np.int64)
        model_expansions = np.zeros((batch_size, unroll_steps), dtype=np.int64)
        behavior_policy = np.zeros(
            (batch_size, unroll_steps, self.policy_target_dim),
            dtype=np.float32,
        )
        audit_type = np.full((batch_size, unroll_steps), "none", dtype=object)
        stage_index = np.full((batch_size, unroll_steps), -1, dtype=np.int64)
        stage_id = np.full((batch_size, unroll_steps), "none", dtype=object)
        nominal_stage_budget = np.zeros((batch_size, unroll_steps), dtype=np.int64)
        requested_budget = np.zeros((batch_size, unroll_steps), dtype=np.int64)
        predicted_stage_index = np.full((batch_size, unroll_steps), -1, dtype=np.int64)
        predicted_stage_id = np.full((batch_size, unroll_steps), "none", dtype=object)
        predicted_requested_budget = np.zeros((batch_size, unroll_steps), dtype=np.int64)
        executed_search_expansions = np.zeros((batch_size, unroll_steps), dtype=np.int64)
        common_evaluator_expansions = np.zeros((batch_size, unroll_steps), dtype=np.int64)
        total_model_calls = np.zeros((batch_size, unroll_steps), dtype=np.int64)
        termination_reason = np.full(
            (batch_size, unroll_steps), "budget_exhausted", dtype=object,
        )
        mask = np.zeros((batch_size, unroll_steps), dtype=np.float32)
        td_reward = np.zeros((batch_size, unroll_steps), dtype=np.float32)
        td_obs = np.zeros((batch_size, unroll_steps, *self.obs_shape), dtype=np.float32)
        bootstrap_steps = unroll_steps + td_steps
        bootstrap_action = np.zeros((batch_size, bootstrap_steps, self.action_dim), dtype=np.float32)
        bootstrap_next_obs = np.zeros(
            (batch_size, bootstrap_steps, *self.obs_shape), dtype=np.float32,
        )
        bootstrap_mask = np.zeros((batch_size, bootstrap_steps), dtype=bool)
        td_discount = np.zeros((batch_size, unroll_steps), dtype=np.float32)
        td_horizon = np.zeros((batch_size, unroll_steps), dtype=np.int64)
        td_bootstrap_mask = np.zeros((batch_size, unroll_steps), dtype=np.float32)
        search_value = np.full((batch_size, unroll_steps), self._NO_SEARCH_VALUE, dtype=np.float32)
        # `-1`: never reanalyzed - `_train_step`'s own freshness gate on
        # whether `search_value` is allowed to participate in
        # `value_target`'s `max()` (see `DEFAULT_HYPERPARAMS
        # ["search_value_max_staleness"]`'s own docstring).
        search_value_gen = np.full((batch_size, unroll_steps), -1, dtype=np.int64)
        episode_idx = np.zeros(batch_size, dtype=np.int64)
        timestep = np.zeros(batch_size, dtype=np.int64)
        sample_prob = np.zeros(batch_size, dtype=np.float64)
        sampled_active = np.zeros(batch_size, dtype=bool)
        sampled_recent = np.zeros(batch_size, dtype=bool)
        sampled_success = np.zeros(batch_size, dtype=bool)

        # Every source, including active lanes, owns durable per-transition
        # priorities. Sampling is an exact mixture of global PER and a
        # uniform recent tail from active lanes; ``sample_prob`` below is
        # the probability under that same mixture, so IS correction remains
        # valid instead of oscillating between two unaccounted samplers.
        sources: list[tuple[bool, int]] = [(False, i) for i in range(len(self.episodes))]
        source_mass = list(self._episode_alpha_sum)
        recent_ranges: dict[int, tuple[int, int]] = {}
        total_recent = 0
        for lane, cur in enumerate(self._cur):
            if cur["reward"]:
                sources.append((True, lane))
                active_priority = np.asarray(cur["priority"], dtype=np.float64)
                source_mass.append(float(np.sum(active_priority ** self.priority_alpha)))
                recent_start = max(0, len(cur["reward"]) - self.recent_window)
                recent_ranges[lane] = (recent_start, len(cur["reward"]))
                total_recent += len(cur["reward"]) - recent_start
        if not sources:
            raise RuntimeError("Cannot sample an empty ResearchImZero replay buffer")
        source_mass_arr = np.asarray(source_mass, dtype=np.float64)
        total_alpha_sum = float(source_mass_arr.sum())
        source_probs = (
            source_mass_arr / total_alpha_sum
            if total_alpha_sum > 0
            else np.full(len(sources), 1.0 / len(sources))
        )
        effective_recent_fraction = self.recent_fraction if total_recent > 0 else 0.0
        success_episode_indices: list[int] = []
        total_success_transitions = 0
        if self.success_fraction > 0.0 and self.episodes:
            cutoff = float(
                np.quantile(
                    np.asarray(self._episode_returns, dtype=np.float64),
                    1.0 - self.success_top_quantile,
                ),
            )
            success_episode_indices = [
                index for index, value in enumerate(self._episode_returns) if value >= cutoff
            ]
            total_success_transitions = sum(
                len(self.episodes[index]["reward"]) for index in success_episode_indices
            )
        effective_success_fraction = (
            self.success_fraction if total_success_transitions > 0 else 0.0
        )
        per_fraction = 1.0 - effective_recent_fraction - effective_success_fraction
        n_total_transitions = max(1, self.num_replay_transitions)

        for b in range(batch_size):
            mixture_draw = np.random.random()
            choose_recent = mixture_draw < effective_recent_fraction
            choose_success = (
                not choose_recent
                and mixture_draw < effective_recent_fraction + effective_success_fraction
            )
            if choose_recent:
                recent_offset = int(np.random.randint(total_recent))
                source_i = -1
                for candidate_i, (candidate_active, candidate_index) in enumerate(sources):
                    if not candidate_active:
                        continue
                    recent_start, recent_end = recent_ranges[candidate_index]
                    recent_len = recent_end - recent_start
                    if recent_offset < recent_len:
                        source_i = candidate_i
                        start = recent_start + recent_offset
                        break
                    recent_offset -= recent_len
                if source_i < 0:
                    raise RuntimeError("Recent replay source accounting became inconsistent")
            elif choose_success:
                success_offset = int(np.random.randint(total_success_transitions))
                source_index = -1
                for candidate_index in success_episode_indices:
                    candidate_len = len(self.episodes[candidate_index]["reward"])
                    if success_offset < candidate_len:
                        source_index = candidate_index
                        start = success_offset
                        break
                    success_offset -= candidate_len
                if source_index < 0:
                    raise RuntimeError("Success replay source accounting became inconsistent")
                source_i = source_index
            else:
                source_i = int(np.random.choice(len(sources), p=source_probs))
            is_active, source_index = sources[source_i]
            sampled_active[b] = is_active
            sampled_recent[b] = choose_recent
            sampled_success[b] = choose_success
            ep = self._cur[source_index] if is_active else self.episodes[source_index]
            t_len = len(ep["reward"])
            priority = np.asarray(ep["priority"], dtype=np.float64)
            local_weight = priority ** self.priority_alpha
            local_sum = float(local_weight.sum())
            if not choose_recent and not choose_success:
                if local_sum > 0:
                    local_probs = local_weight / local_sum
                    start = int(np.random.choice(t_len, p=local_probs))
                else:
                    start = int(np.random.randint(t_len))
            episode_idx[b] = -source_index - 1 if is_active else source_index
            timestep[b] = start
            per_probability = (
                source_probs[source_i] * float(local_weight[start] / local_sum)
                if local_sum > 0 else source_probs[source_i] / t_len
            )
            recent_probability = 0.0
            if is_active:
                recent_start, recent_end = recent_ranges[source_index]
                if recent_start <= start < recent_end:
                    recent_probability = 1.0 / total_recent
            success_probability = (
                1.0 / total_success_transitions
                if not is_active and source_index in success_episode_indices
                else 0.0
            )
            sample_prob[b] = max(
                per_fraction * per_probability
                + effective_recent_fraction * recent_probability
                + effective_success_fraction * success_probability,
                1e-12,
            )

            ctx_obs[b], ctx_action[b], ctx_valid[b] = self._context_for(ep, start)
            obs0[b] = ep["obs"][start]
            for j in range(bootstrap_steps):
                idx = start + j
                if idx >= t_len:
                    break
                bootstrap_action[b, j] = ep["action"][idx]
                bootstrap_next_obs[b, j] = ep["next_obs"][idx]
                bootstrap_mask[b, j] = True
            for k in range(unroll_steps):
                idx = start + k
                if idx >= t_len:
                    continue
                mask[b, k] = 1.0
                action[b, k] = ep["action"][idx]
                reward[b, k] = ep["reward"][idx]
                next_obs[b, k] = ep["next_obs"][idx]
                policy_target[b, k] = ep["policy_target"][idx]
                if "policy_target_valid" in ep:
                    policy_target_valid[b, k] = float(ep["policy_target_valid"][idx])
                if "search_budget" in ep:
                    search_budget[b, k] = int(ep["search_budget"][idx])
                if "model_expansions" in ep:
                    model_expansions[b, k] = int(ep["model_expansions"][idx])
                if "behavior_policy" in ep:
                    behavior_policy[b, k] = ep["behavior_policy"][idx]
                else:
                    behavior_policy[b, k] = ep["policy_target"][idx]
                if "audit_type" in ep:
                    audit_type[b, k] = str(ep["audit_type"][idx])
                requested_budget[b, k] = search_budget[b, k]
                nominal_stage_budget[b, k] = search_budget[b, k]
                predicted_requested_budget[b, k] = search_budget[b, k]
                executed_search_expansions[b, k] = model_expansions[b, k]
                total_model_calls[b, k] = model_expansions[b, k]
                if "stage_index" in ep:
                    stage_index[b, k] = int(ep["stage_index"][idx])
                    stage_id[b, k] = str(ep["stage_id"][idx])
                    nominal_stage_budget[b, k] = int(ep["nominal_stage_budget"][idx])
                    requested_budget[b, k] = int(ep["requested_budget"][idx])
                    predicted_stage_index[b, k] = int(ep["predicted_stage_index"][idx])
                    predicted_stage_id[b, k] = str(ep["predicted_stage_id"][idx])
                    predicted_requested_budget[b, k] = int(
                        ep["predicted_requested_budget"][idx]
                    )
                    executed_search_expansions[b, k] = int(
                        ep["executed_search_expansions"][idx]
                    )
                    common_evaluator_expansions[b, k] = int(
                        ep["common_evaluator_expansions"][idx]
                    )
                    total_model_calls[b, k] = int(ep["total_model_calls"][idx])
                    termination_reason[b, k] = str(ep["termination_reason"][idx])
                search_value[b, k] = ep["search_value"][idx]
                search_value_gen[b, k] = ep["reanalyzed_gen"][idx]

                n = min(td_steps, t_len - idx)
                td_sum, discount = 0.0, 1.0
                for j in range(n):
                    td_sum += discount * float(ep["reward"][idx + j])
                    discount *= gamma
                td_reward[b, k] = td_sum
                td_discount[b, k] = discount
                td_horizon[b, k] = n
                landing_idx = idx + n - 1
                td_obs[b, k] = ep["next_obs"][landing_idx]
                td_bootstrap_mask[b, k] = (
                    1.0 if is_active else (0.0 if landing_idx == t_len - 1 else 1.0)
                )

        is_weight = (1.0 / (n_total_transitions * sample_prob)) ** self.priority_beta
        is_weight = is_weight / max(float(is_weight.max()), 1e-12)

        return {
            "context_obs": ctx_obs, "context_action": ctx_action, "context_valid": ctx_valid,
            "obs0": obs0, "action": action, "reward": reward, "next_obs": next_obs,
            "policy_target": policy_target, "policy_target_valid": policy_target_valid,
            "search_budget": search_budget, "model_expansions": model_expansions,
            "behavior_policy": behavior_policy, "audit_type": audit_type,
            "stage_index": stage_index, "stage_id": stage_id,
            "nominal_stage_budget": nominal_stage_budget,
            "requested_budget": requested_budget,
            "predicted_stage_index": predicted_stage_index,
            "predicted_stage_id": predicted_stage_id,
            "predicted_requested_budget": predicted_requested_budget,
            "executed_search_expansions": executed_search_expansions,
            "common_evaluator_expansions": common_evaluator_expansions,
            "total_model_calls": total_model_calls,
            "termination_reason": termination_reason,
            "mask": mask, "td_reward": td_reward,
            "td_obs": td_obs,
            "bootstrap_action": bootstrap_action,
            "bootstrap_next_obs": bootstrap_next_obs,
            "bootstrap_mask": bootstrap_mask,
            "td_discount": td_discount, "td_horizon": td_horizon,
            "td_bootstrap_mask": td_bootstrap_mask,
            "search_value": search_value, "search_value_gen": search_value_gen,
            "episode_idx": episode_idx, "timestep": timestep,
            "is_weight": is_weight.astype(np.float32),
            "active_sample_fraction": np.asarray(sampled_active.mean(), dtype=np.float32),
            "recent_sample_fraction": np.asarray(sampled_recent.mean(), dtype=np.float32),
            "success_sample_fraction": np.asarray(sampled_success.mean(), dtype=np.float32),
            "success_pool_size": np.asarray(len(success_episode_indices), dtype=np.float32),
        }

    def update_priorities(self, episode_idx: np.ndarray, timestep: np.ndarray, values: np.ndarray) -> None:
        values = np.maximum(np.asarray(values, dtype=np.float64), self.min_priority)
        touched = set()
        for ep_i, t, v in zip(episode_idx.tolist(), timestep.tolist(), values.tolist()):
            if ep_i < 0:
                lane = -ep_i - 1
                if 0 <= lane < len(self._cur) and t < len(self._cur[lane]["priority"]):
                    self._cur[lane]["priority"][t] = v
                continue
            if ep_i >= len(self.episodes):
                continue
            self.episodes[ep_i]["priority"][t] = v
            touched.add(ep_i)
        self._max_priority = max(self._max_priority, float(values.max()) if len(values) else self._max_priority)
        for ep_i in touched:
            ep = self.episodes[ep_i]
            self._episode_alpha_sum[ep_i] = float(np.sum(ep["priority"] ** self.priority_alpha))

    def raw_learning_errors_for(
        self, episode_idx: np.ndarray, timestep: np.ndarray,
    ) -> np.ndarray:
        values = np.full(len(episode_idx), np.nan, dtype=np.float64)
        for i, (ep_i, t) in enumerate(zip(episode_idx.tolist(), timestep.tolist())):
            if ep_i < 0:
                lane = -ep_i - 1
                if 0 <= lane < len(self._cur):
                    raw = self._cur[lane].get("raw_learning_error", [])
                    if t < len(raw):
                        values[i] = float(raw[t])
            elif ep_i < len(self.episodes):
                raw = self.episodes[ep_i].get("raw_learning_error")
                if raw is not None and t < len(raw):
                    values[i] = float(raw[t])
        return values

    def update_raw_learning_errors(
        self,
        episode_idx: np.ndarray,
        timestep: np.ndarray,
        values: np.ndarray,
    ) -> None:
        for ep_i, t, value in zip(
            episode_idx.tolist(), timestep.tolist(), np.asarray(values).tolist(),
        ):
            if ep_i < 0:
                lane = -ep_i - 1
                if 0 <= lane < len(self._cur) and t < len(self._cur[lane]["reward"]):
                    raw = self._cur[lane].setdefault(
                        "raw_learning_error",
                        [float("nan")] * len(self._cur[lane]["reward"]),
                    )
                    raw[t] = float(value)
            elif ep_i < len(self.episodes) and t < len(self.episodes[ep_i]["reward"]):
                ep = self.episodes[ep_i]
                if "raw_learning_error" not in ep:
                    ep["raw_learning_error"] = np.full(
                        len(ep["reward"]), np.nan, dtype=np.float64,
                    )
                ep["raw_learning_error"][t] = float(value)

    def sample_for_reanalyze(
        self, n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sources: list[tuple[bool, int]] = [(False, i) for i in range(len(self.episodes))]
        sources.extend((True, lane) for lane, cur in enumerate(self._cur) if cur["reward"])
        n = min(n, self.num_replay_transitions)
        length = self.context_length
        if n <= 0 or not sources:
            empty_obs = np.zeros((0, *self.obs_shape), dtype=np.float32)
            return (
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), empty_obs,
                np.zeros((0, length, *self.obs_shape), dtype=np.float32), np.zeros((0, length, self.action_dim), dtype=np.float32),
                np.zeros((0, length), dtype=bool),
            )
        # One priority*staleness distribution over both finalized and live
        # transitions. Reanalyzing live targets is essential when early
        # low-budget MCTS policy labels would otherwise remain frozen until
        # a multi-thousand-step NetHack episode eventually terminates.
        gen = float(self._reanalyze_generation)
        transition_scores: list[np.ndarray] = []
        source_scores = np.zeros(len(sources), dtype=np.float64)
        for source_i, (is_active, source_index) in enumerate(sources):
            ep = self._cur[source_index] if is_active else self.episodes[source_index]
            priorities = np.asarray(ep["priority"], dtype=np.float64)
            rgen = np.asarray(ep["reanalyzed_gen"], dtype=np.float64)
            staleness = np.where(rgen < 0, gen + 1.0, np.maximum(1.0, gen - rgen + 1.0))
            scores = (priorities ** self.priority_alpha) * staleness
            transition_scores.append(scores)
            source_scores[source_i] = float(scores.sum())
        total_score = float(source_scores.sum())
        source_probs = (
            source_scores / total_score
            if total_score > 0
            else np.full(len(sources), 1.0 / len(sources))
        )
        source_choices = np.random.choice(len(sources), size=n, p=source_probs)
        episode_idx = np.zeros(n, dtype=np.int64)
        timestep = np.zeros(n, dtype=np.int64)
        obs_out = np.zeros((n, *self.obs_shape), dtype=np.float32)
        ctx_obs = np.zeros((n, length, *self.obs_shape), dtype=np.float32)
        ctx_action = np.zeros((n, length, self.action_dim), dtype=np.float32)
        ctx_valid = np.zeros((n, length), dtype=bool)
        for i, source_i in enumerate(source_choices.tolist()):
            is_active, source_index = sources[source_i]
            ep = self._cur[source_index] if is_active else self.episodes[source_index]
            t_len = len(ep["reward"])
            weight = transition_scores[source_i]
            weight_sum = float(weight.sum())
            t = int(np.random.choice(t_len, p=weight / weight_sum)) if weight_sum > 0 else int(np.random.randint(t_len))
            episode_idx[i] = -source_index - 1 if is_active else source_index
            timestep[i] = t
            obs_out[i] = ep["obs"][t]
            ctx_obs[i], ctx_action[i], ctx_valid[i] = self._context_for(ep, t)
        return episode_idx, timestep, obs_out, ctx_obs, ctx_action, ctx_valid

    def update_reanalyzed_targets(
        self,
        episode_idx: np.ndarray,
        timestep: np.ndarray,
        policy_targets: np.ndarray,
        search_values: np.ndarray,
        policy_target_valid: np.ndarray | None = None,
    ) -> None:
        self._reanalyze_generation += 1
        gen = self._reanalyze_generation
        for i, (ep_i, t) in enumerate(zip(episode_idx.tolist(), timestep.tolist())):
            if ep_i < 0:
                lane = -ep_i - 1
                if lane >= len(self._cur):
                    continue
                ep = self._cur[lane]
                if t >= len(ep["reward"]):
                    continue
                ep["policy_target"][t] = np.asarray(policy_targets[i], dtype=np.float32)
                ep["search_value"][t] = float(search_values[i])
                ep["reanalyzed_gen"][t] = gen
                if policy_target_valid is not None:
                    valid = ep.setdefault(
                        "policy_target_valid",
                        [True] * len(ep["reward"]),
                    )
                    valid[t] = bool(policy_target_valid[i])
                continue
            if ep_i >= len(self.episodes):
                continue
            ep = self.episodes[ep_i]
            if t >= len(ep["reward"]):
                continue
            ep["policy_target"][t] = policy_targets[i]
            ep["search_value"][t] = search_values[i]
            ep["reanalyzed_gen"][t] = gen
            if policy_target_valid is not None:
                if "policy_target_valid" not in ep:
                    ep["policy_target_valid"] = np.ones(len(ep["reward"]), dtype=bool)
                ep["policy_target_valid"][t] = bool(policy_target_valid[i])
            self._episode_last_reanalyzed_gen[ep_i] = float(gen)


class NativeResearchImZero(CustomAlgorithm):
    COMPOSITE_FAMILY = "researchimzero"

    def __init__(self, env: gym.Env, hyperparams: dict[str, Any], seed: int | None, device: str) -> None:
        super().__init__(env, hyperparams, seed, device)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self._obs_space = venv_obs_space(env)
        self._action_space = venv_action_space(env)
        self.discrete = isinstance(self._action_space, gym.spaces.Discrete)
        self.n_actions = int(self._action_space.n) if self.discrete else None
        self.action_dim = obs_flat_dim(self._action_space)

        network_spec = hyperparams.get("network_spec")
        if network_spec is not None:
            network_spec = validate_composite_spec(network_spec, self.COMPOSITE_FAMILY)
            self.hyperparams["network_spec"] = network_spec
        dimensions = dimensions_from_spec(
            network_spec,
            self.COMPOSITE_FAMILY,
            {
                "embed_dim": int(hyperparams.get("embed_dim", 128)),
                "num_layers": max(1, int(hyperparams.get("num_layers", 2))),
                "num_heads": max(1, int(hyperparams.get("num_heads", 4))),
                "ffn_multiplier": 4,
                "dropout": float(hyperparams.get("dropout", 0.0)),
                "rotary_emb": int(hyperparams.get("rotary_emb", 1)),
                "proj_dim": int(hyperparams.get("proj_dim", 64)),
            },
        )
        self.embed_dim = int(dimensions["embed_dim"])
        num_layers = int(dimensions["num_layers"])
        num_heads = int(dimensions["num_heads"])
        ffn_multiplier = int(dimensions["ffn_multiplier"])
        dropout = float(dimensions["dropout"])
        self.context_length = max(0, int(hyperparams.get("context_length", 6)))
        self.unroll_steps = max(1, int(hyperparams.get("unroll_steps", 5)))
        self.support_size = max(1, int(hyperparams.get("value_support_size", 300)))
        self.label_smoothing_eps = max(0.0, float(hyperparams.get("label_smoothing_eps", 0.0)))
        # Entropy *bonus* - a generic MuZero-family exploration
        # regularizer (kept from `unizero.py`, `efficientzero.py` has no
        # equivalent knob) - subtracted from the total loss, not added,
        # since higher policy entropy is the goal here.
        self.policy_entropy_coef = float(hyperparams.get("policy_entropy_coef", 5e-3))
        # `1` (default): RoPE. `0`: learned absolute embedding - module
        # docstring's "everything else" section (reused from `unizero.py`
        # as-is).
        self.rotary_emb = bool(int(dimensions["rotary_emb"]))

        components = network_spec["components"] if network_spec is not None else {}
        encoder = build_composite_encoder(self._obs_space, network_spec) if network_spec is not None else None
        self.tokenizer = _Tokenizer(
            self._obs_space, self.embed_dim, 2 * self.embed_dim, encoder, components.get("tokenizer"),
        ).to(device)
        self.action_embed = _ActionEmbed(self.action_dim, self.embed_dim, self.discrete).to(device)
        self.transformer = _CausalTransformer(
            self.embed_dim, num_layers, num_heads, dropout,
            rotary_emb=self.rotary_emb, ffn_multiplier=ffn_multiplier,
        ).to(device)
        self.heads = _Heads(
            self.embed_dim, 2 * self.embed_dim, self._action_space,
            self.support_size, components.get("heads"),
        ).to(device)
        # EMA/Polyak target copy, bootstrap-value-only (module docstring's
        # "Schedules, not fixed constants" section) - never trained
        # directly (no `requires_grad_(False)` needed since nothing ever
        # builds an optimizer over it; still explicitly `.eval()`'d so its
        # own `BatchNorm1d`-having submodules, if any get added later,
        # don't drift on whatever batch statistics a stray forward pass
        # happens to see). Deliberately *not* used for the consistency
        # loss's own target (that stays the online tokenizer, stop-
        # gradiented per-call, per `use_target_for_bootstrap`'s own
        # docstring) - only for `_train_step`'s n-step TD bootstrap.
        self.target_tokenizer = copy.deepcopy(self.tokenizer).eval()
        self.target_action_embed = copy.deepcopy(self.action_embed).eval()
        self.target_transformer = copy.deepcopy(self.transformer).eval()
        self.target_heads = copy.deepcopy(self.heads).eval()
        for module in (self.target_tokenizer, self.target_action_embed, self.target_transformer, self.target_heads):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        # SimSiam projector/predictor (module docstring's "no target
        # network" section) - `efficientzero.py`'s own consistency-loss
        # machinery, operating on `embed_dim`-sized token embeddings here
        # (the tokenizer's real-observation embedding vs. `heads.latent`'s
        # predicted-next-embedding, both already in that same space)
        # rather than `efficientzero.py`'s own fixed-size representation
        # latent - same role either way.
        proj_dim = int(dimensions["proj_dim"])
        self.projector = _Projector(self.embed_dim, proj_dim, components.get("projector")).to(device)
        self.predictor = _Predictor(proj_dim, components.get("predictor")).to(device)
        self.adaptive_loss_weights = bool(int(hyperparams.get("adaptive_loss_weights", 0)))
        # Order: [reward, value, policy] - deliberately *not* 4. Kendall
        # et al.'s `precision*loss + log_var` implicitly assumes every
        # task loss has a genuine, environment/label-driven noise floor
        # it asymptotes towards (their own multi-task setup: depth/
        # segmentation/instance losses never hit exactly `0`) - reward/
        # value/policy all qualify (n-step TD noise, env stochasticity,
        # search-target noise). `consistency_loss` (SimSiam, stop-grad)
        # doesn't: nothing stops the predictor from driving it arbitrarily
        # close to `0` once the projector/predictor pair aligns well, and
        # once a task's own loss keeps shrinking towards `0`, this scheme
        # has no floor on how high *that* task's own `precision` climbs
        # to compensate - a real run measured `loss_weight_consistency`
        # pinned at this file's own `exp(5)` clamp ceiling (~148x) for the
        # entire back half of training, ~200x `loss_weight_policy`'s
        # concurrent ~0.6-0.7, while `episode_reward_mean` stalled well
        # below runs that predate `adaptive_loss_weights` entirely (which
        # just used the static `consistency_loss_coef` below, unweighted-
        # further) - the trunk's gradient was overwhelmingly "get this
        # already-easy task a little easier still", not reward/value/
        # policy. `consistency_loss` keeps its own fixed `consistency_
        # loss_coef` for exactly this reason; only the three tasks with
        # an actual noise floor get adaptively weighted. `0.0` initial
        # `log_var` for those three means `exp(-log_var)=1`, i.e. training
        # starts out exactly equivalent to the static-`*_loss_coef`-only
        # behavior and only diverges from it as these get learned (see
        # `DEFAULT_HYPERPARAMS["adaptive_loss_weights"]`'s own docstring).
        self.loss_log_vars = nn.Parameter(torch.zeros(3, device=device))
        self._params = (
            list(self.tokenizer.parameters()) + list(self.action_embed.parameters())
            + list(self.transformer.parameters()) + list(self.heads.parameters())
            + list(self.projector.parameters()) + list(self.predictor.parameters())
            + ([self.loss_log_vars] if self.adaptive_loss_weights else [])
        )
        # Plain Adam, no weight decay - `efficientzero.py`'s own default;
        # no target network here at all (module docstring's "no target
        # network" section) so there's nothing an `AdamW`-style decay was
        # ever specifically compensating for either.
        self.base_learning_rate = float(hyperparams.get("learning_rate", 2e-4))
        self.optimizer = torch.optim.Adam(self._params, lr=self.base_learning_rate)
        self.lr_schedule = bool(int(hyperparams.get("lr_schedule", 1)))
        self.lr_min_fraction = max(0.0, min(1.0, float(hyperparams.get("lr_min_fraction", 0.05))))
        # Progress-fraction bookkeeping shared by every training-progress-
        # dependent schedule below (LR decay, priority-beta annealing,
        # search/entropy temperature annealing) - one `learn()` call sets
        # `_total_timesteps_estimate` once from its own `total_timesteps`
        # argument (`__init__` doesn't get one), `_num_timesteps` tracks
        # the real global step counter every schedule reads its own
        # progress fraction off of.
        self._num_timesteps = 0
        self._total_timesteps_estimate = 1
        self._grad_step_count = 0

        # Pure speed knobs - none of these change what's computed, only
        # how fast (module docstring's "Performance, not architecture"
        # section): TF32 matmuls + `cudnn.benchmark` are free wins on any
        # Ampere-or-newer GPU (no-ops elsewhere), automatic mixed
        # precision (`autocast` in `_train_step`, gated by `use_amp`)
        # roughly halves training-step wall time and memory on CUDA by
        # running most of the forward pass in bf16/fp16 while keeping the
        # optimizer's own master weights and reductions in fp32 - PyTorch's
        # own numerically-safe default recipe, not a precision hack this
        # file invented. All three are CPU-safe no-ops (`use_amp` forces
        # itself off outside CUDA below), so tests/CPU-only setups are
        # unaffected either way.
        self._is_cuda = str(device).startswith("cuda")
        if self._is_cuda:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        self.use_amp = bool(int(hyperparams.get("use_amp", 1))) and self._is_cuda
        self._amp_dtype = (
            torch.bfloat16 if (self._is_cuda and torch.cuda.is_bf16_supported()) else torch.float16
        )
        # bf16 has fp32's exponent range, so it never needs loss scaling
        # (`GradScaler` is a no-op-by-construction whenever `enabled=False`
        # below) - only the fp16 fallback (older GPUs without native bf16)
        # actually exercises it.
        self._grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self._amp_dtype is torch.float16)

        self.gamma = float(hyperparams.get("gamma", 0.99))
        self.batch_size = int(hyperparams.get("batch_size", 64))
        self.td_steps = max(1, int(hyperparams.get("td_steps", 5)))
        self.num_sampled_actions = max(2, int(hyperparams.get("num_sampled_actions", 8)))
        self.num_simulations = max(2, int(hyperparams.get("num_simulations", 32)))
        self.num_simulations_initial = max(
            2, min(self.num_simulations, int(hyperparams.get("num_simulations_initial", 8))),
        )
        self.search_ramp_steps = max(1, int(hyperparams.get("search_ramp_steps", 100_000)))
        self.search_model_error_low = max(
            0.0, float(hyperparams.get("search_model_error_low", 0.02)),
        )
        self.search_model_error_high = max(
            self.search_model_error_low + 1e-6,
            float(hyperparams.get("search_model_error_high", 0.30)),
        )
        self.search_model_error_ema_decay = min(
            0.9999, max(0.0, float(hyperparams.get("search_model_error_ema_decay", 0.99))),
        )
        self._model_error_ema = self.search_model_error_high
        self._last_search_num_simulations = self.num_simulations_initial
        self._last_search_base_simulations = self.num_simulations_initial
        # Gumbel-Top-k + Sequential Halving (module docstring's "Search"
        # section) - `efficientzero.py`'s own defaults, not PUCT's
        # `pb_c_base`/`pb_c_init`/Dirichlet noise.
        self.num_top_actions = max(2, int(hyperparams.get("num_top_actions", 8)))
        self.c_visit = float(hyperparams.get("c_visit", 50.0))
        self.c_scale = float(hyperparams.get("c_scale", 0.1))
        self.policy_target_temperature = max(1e-6, float(hyperparams.get("policy_target_temperature", 1.0)))
        self.temperature_anneal_start_scale = max(1.0, float(hyperparams.get("temperature_anneal_start_scale", 1.5)))
        self.temperature_anneal_steps = max(1, int(hyperparams.get("temperature_anneal_steps", 100_000)))
        self.value_minmax_delta = float(hyperparams.get("value_minmax_delta", 0.01))
        self.value_loss_coef = float(hyperparams.get("value_loss_coef", 0.25))
        self.policy_loss_coef = float(hyperparams.get("policy_loss_coef", 1.0))
        self.reward_loss_coef = float(hyperparams.get("reward_loss_coef", 1.0))
        self.consistency_loss_coef = float(hyperparams.get("consistency_loss_coef", 2.0))
        self.path_consistency_coef = max(
            0.0, float(hyperparams.get("path_consistency_coef", 0.0)),
        )
        self.closed_loop_loss_coef = max(0.0, float(hyperparams.get("closed_loop_loss_coef", 0.5)))
        self.closed_loop_horizon = max(1, min(
            self.unroll_steps, int(hyperparams.get("closed_loop_horizon", 3)),
        ))
        self.closed_loop_batch_fraction = min(
            1.0, max(0.0, float(hyperparams.get("closed_loop_batch_fraction", 0.25))),
        )
        self.adaptive_closed_loop = bool(int(hyperparams.get("adaptive_closed_loop", 0)))
        self.adaptive_closed_loop_max_horizon = max(
            self.closed_loop_horizon,
            min(self.unroll_steps, int(hyperparams.get("adaptive_closed_loop_max_horizon", 5))),
        )
        self.adaptive_closed_loop_error_threshold = max(
            0.0, float(hyperparams.get("adaptive_closed_loop_error_threshold", 0.08)),
        )
        self.adaptive_closed_loop_min_grad_steps = max(
            0, int(hyperparams.get("adaptive_closed_loop_min_grad_steps", 500)),
        )
        self.adaptive_closed_loop_stable_steps = max(
            1, int(hyperparams.get("adaptive_closed_loop_stable_steps", 50)),
        )
        self.adaptive_closed_loop_ema_decay = min(
            0.9999, max(0.0, float(hyperparams.get("adaptive_closed_loop_ema_decay", 0.99))),
        )
        self._adaptive_closed_loop_horizon = self.closed_loop_horizon
        self._adaptive_closed_loop_stable_count = 0
        self._closed_loop_error_ema = [float("nan")] * self.unroll_steps
        self.continuous_prior_scale = float(hyperparams.get("continuous_prior_scale", 2.5))
        self.train_freq = max(1, int(hyperparams.get("train_freq", 1)))
        _train_steps_per_iter_base = max(1, int(hyperparams.get("train_steps_per_iter", 2)))
        # `learn()` executes this many updates once a vector iteration
        # crosses a train-frequency boundary. Do not multiply by the
        # number of aggregate boundaries again there: with 24 envs that
        # old combination produced `(24 boundaries) * (24//4) = 144`
        # optimizer steps after every vector step, versus one in the
        # historical fast/successful EfficientZero runs. Besides reducing
        # throughput from ~100 to ~5 aggregate steps/s, it repeatedly fit
        # the first short completed episode before ongoing lanes became
        # replayable. Auto-scaling remains an explicit opt-in and means
        # exactly `num_envs//4` updates per vector iteration, not per
        # transition.
        self._replay_ratio_scale = (
            max(1, num_envs_of(env) // 4) if bool(int(hyperparams.get("auto_scale_replay_ratio", 0))) else 1
        )
        self.train_steps_per_iter = _train_steps_per_iter_base * self._replay_ratio_scale
        self.learning_starts = int(hyperparams.get("learning_starts", 500))
        self.max_grad_norm = float(hyperparams.get("max_grad_norm", 5.0))
        self.priority_alpha = max(0.0, float(hyperparams.get("priority_alpha", 1.0)))
        self.priority_beta = max(0.0, float(hyperparams.get("priority_beta", 1.0)))
        self.priority_beta_start = max(0.0, float(hyperparams.get("priority_beta_start", 0.4)))
        self.priority_policy_weight = max(0.0, float(hyperparams.get("priority_policy_weight", 0.1)))
        self.learning_progress_priority_weight = max(
            0.0, float(hyperparams.get("learning_progress_priority_weight", 0.0)),
        )
        self.learning_progress_priority_cap = max(
            0.0, float(hyperparams.get("learning_progress_priority_cap", 1.0)),
        )
        self.min_priority = max(1e-8, float(hyperparams.get("min_priority", 1e-6)))
        self.reanalyze_freq = max(1, int(hyperparams.get("reanalyze_freq", 200)))
        self.reanalyze_batch_size = max(0, int(hyperparams.get("reanalyze_batch_size", 64)))
        self.use_target_for_bootstrap = bool(int(hyperparams.get("use_target_for_bootstrap", 1)))
        self.target_update_theta = float(hyperparams.get("target_update_theta", 0.02))
        self.search_value_max_staleness = max(0, int(hyperparams.get("search_value_max_staleness", 5)))
        self.reanalyze_value_mix = min(1.0, max(0.0, float(hyperparams.get("reanalyze_value_mix", 0.25))))
        self.rnd_bonus_coef = float(hyperparams.get("rnd_bonus_coef", 0.1))
        self.rnd = (
            RNDModule(
                self._obs_space, device,
                feature_dim=int(hyperparams.get("rnd_feature_dim", 128)),
                hidden_dim=int(hyperparams.get("rnd_hidden_dim", 128)),
                learning_rate=float(hyperparams.get("rnd_learning_rate", 1e-4)),
                bonus_clip=float(hyperparams.get("rnd_bonus_clip", 5.0)),
            )
            if uses_rnd(hyperparams)
            else None
        )

        sample_obs_arr = obs_to_array(self._obs_space.sample(), self._obs_space)
        self._obs_shape = sample_obs_arr.shape
        # Discrete: one prior per real action (`n_actions`). Continuous:
        # the visit-weighted-average sampled candidate action
        # (`efficientzero.py`'s own, simpler choice - module docstring's
        # "Search" section - not `unizero.py`'s flattened
        # candidates-and-weights format, since there's no importance-
        # weighted-NLL policy loss here to consume that richer target).
        policy_target_dim = self.n_actions if self.discrete else self.action_dim
        self.buffer = _ResearchImZeroBuffer(
            capacity_episodes=int(hyperparams.get("buffer_size", 2_000)),
            obs_shape=self._obs_shape, action_dim=self.action_dim, policy_target_dim=policy_target_dim,
            context_length=self.context_length, num_lanes=num_envs_of(env),
            # Starts at `priority_beta_start`, annealed up to
            # `priority_beta` every `_train_step()` call
            # (`self.buffer.priority_beta` reassigned there each time,
            # see `_update_learning_rate`'s sibling schedule helpers).
            priority_alpha=self.priority_alpha, priority_beta=self.priority_beta_start, min_priority=self.min_priority,
            recent_fraction=float(hyperparams.get("replay_recent_fraction", 0.5)),
            recent_window=int(hyperparams.get("replay_recent_window", 512)),
            success_fraction=float(hyperparams.get("replay_success_fraction", 0.0)),
            success_top_quantile=float(hyperparams.get("replay_success_top_quantile", 0.25)),
        )
        self._last_metrics: dict[str, float] = {}
        self._episode_reward_rolling = deque(
            maxlen=max(1, int(hyperparams.get("episode_reward_rolling_window", 20))),
        )
        # `learn()`'s own persistent per-lane real-history cache (module
        # docstring's "Persistent, incrementally-extended per-lane root
        # cache" section) - `None` per lane means "no history yet this
        # episode" (the reference's own "first_step_flag").
        self._lane_cache: list[_TransformerCache | None] = [None] * num_envs_of(env)
        # `predict()`'s own analogous single-lane persistent cache,
        # separate from `self.buffer`'s (which only tracks lanes actually
        # being trained on) - reset on `episode_start=True` exactly like
        # every other memory-carrying algorithm in this app (see
        # `dreamer.py`'s `predict()`).
        self._eval_cache: _TransformerCache | None = None

    # ------------------------------------------------------------------
    # Token-sequence construction.
    # ------------------------------------------------------------------
    def _embed_root_batch(
        self, ctx_obs: np.ndarray, ctx_action: np.ndarray, ctx_valid: np.ndarray, current_obs: np.ndarray,
        tokenizer: nn.Module | None = None, action_embed: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """`ctx_*`: `(B, context_length, ...)`, right-padded (see
        `_ResearchImZeroBuffer._pad_one`). `current_obs`: `(B, *obs_shape)`.
        Returns `(tokens (B, Lmax, E), pad_mask (B, Lmax) bool, valid_len
        (B,) int)`, `Lmax = valid_len.max()` - the raw root token
        sequence the *training* teacher-forcing window starts from (used
        only there - every other caller needing a root state, real
        steps/search/reanalyze alike, uses the cache-based
        `_replay_context_to_cache`/`_advance_lane_caches`/`_step_imagine`
        path instead, module docstring's "Persistent, incrementally-
        extended per-lane root cache" section).
        `valid_len[i] = 2*n_i + 1` is lane `i`'s *own* real token count
        (`n_i` valid context transitions, plus the current-obs token);
        since lanes can have different amounts of real history, they
        occupy different-length prefixes of this shared, batch-padded
        tensor - `_train_step` needs `valid_len` to know where each
        lane's *own* next real token belongs (there's no single shared
        offset)."""
        b, length = ctx_obs.shape[0], self.context_length
        device = self.device
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        action_embed = self.action_embed if action_embed is None else action_embed
        n_valid = ctx_valid.sum(axis=1).astype(np.int64) if length > 0 else np.zeros(b, dtype=np.int64)
        valid_len = 2 * n_valid + 1
        lmax = int(valid_len.max()) if b > 0 else 1
        if length > 0:
            obs_flat = torch.as_tensor(ctx_obs.reshape(b * length, *self._obs_shape), dtype=torch.float32, device=device)
            obs_emb_all = tokenizer(obs_flat).view(b, length, self.embed_dim)
            act_flat = torch.as_tensor(ctx_action.reshape(b * length, self.action_dim), dtype=torch.float32, device=device)
            act_emb_all = action_embed(act_flat).view(b, length, self.embed_dim)
        cur_emb = tokenizer(torch.as_tensor(current_obs, dtype=torch.float32, device=device))
        tokens = torch.zeros(b, lmax, self.embed_dim, device=device)
        pad_mask = torch.zeros(b, lmax, dtype=torch.bool, device=device)
        for i in range(b):
            n = int(n_valid[i])
            if n > 0:
                tokens[i, 0 : 2 * n : 2] = obs_emb_all[i, :n]
                tokens[i, 1 : 2 * n : 2] = act_emb_all[i, :n]
            tokens[i, 2 * n] = cur_emb[i]
            pad_mask[i, : 2 * n + 1] = True
        return tokens, pad_mask, valid_len

    def _step_imagine(
        self, parent_caches: list[_TransformerCache | None], action: torch.Tensor,
    ) -> tuple[list[_TransformerCache], torch.Tensor, torch.Tensor]:
        """One simulated tree edge: two incremental steps (module
        docstring's "Persistent, incrementally-extended per-lane root
        cache" section) - append the chosen action's token, read the
        reward off its hidden state and predict the next *imagined*
        observation token from it, append that too, and read policy/value
        off its hidden state (module docstring's "Head placement" bullet).
        Each step attends only the one new query against its parent's
        already-computed keys/values. `parent_caches[i]`: lane `i`'s
        parent node's cache - shared freely across every child expanding
        the same parent this simulation, no explicit clone needed (see
        `_TransformerCache`'s own docstring). `search()` never inspects
        `parent_caches`/the returned child cache itself, just threads it
        through `_SearchNode.cache` and back into the next simulation's
        call here. Returns `(child_caches, reward_hidden, obs_hidden)` -
        hidden states, not final head outputs, so the caller picks
        discrete/continuous/value formatting itself."""
        positions1, embed_positions1 = _cache_positions(parent_caches, self.device)
        act_emb = self.action_embed(action)  # (B, E)
        h_act, caches1 = self.transformer.forward_incremental_batch(
            act_emb, positions1, parent_caches, embed_positions1,
        )
        z_pred = self.heads.latent(h_act)
        positions2, embed_positions2 = positions1 + 1, embed_positions1 + 1
        h_obs, caches2 = self.transformer.forward_incremental_batch(z_pred, positions2, caches1, embed_positions2)
        return caches2, h_act, h_obs

    def _adjust_search_predictions(
        self,
        action_hidden: torch.Tensor,
        next_hidden: torch.Tensor,
        reward: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional search-edge hook; strict identity by default."""
        del action_hidden, next_hidden
        return reward, next_value

    def _replay_context_to_cache(
        self, ctx_obs: np.ndarray, ctx_action: np.ndarray, ctx_valid: np.ndarray,
    ) -> list[_TransformerCache | None]:
        """Cold-start cache construction for a batch of *independent*
        historical (episode, timestep) samples with no live per-lane
        cache to reuse - `_reanalyze`'s samples, specifically (real env
        steps use `_advance_lane_caches`'s persistent cache instead - see
        module docstring's "Persistent, incrementally-extended per-lane
        root cache" section). Replays each lane's own valid (right-padded,
        temporal-order) context transitions one token at a time through
        the exact same incremental machinery every other caller uses.
        `ctx_*`: `(B, context_length, ...)`. Wrapped in `torch.no_grad()`
        for the same reason `_advance_lane_caches` is - this only ever
        feeds `_reanalyze()`'s own `search()` call (itself already
        `no_grad`), never `_train_step`'s teacher-forcing pass."""
        b, length = ctx_obs.shape[0], self.context_length
        caches: list[_TransformerCache | None] = [None] * b
        if length == 0:
            return caches
        n_valid = ctx_valid.sum(axis=1)
        max_n = int(n_valid.max()) if b > 0 else 0
        device = self.device
        with torch.inference_mode():
            for t in range(max_n):
                active = [i for i in range(b) if t < n_valid[i]]
                if not active:
                    continue
                sub_caches = [caches[i] for i in active]
                positions_obs, embed_positions_obs = _cache_positions(sub_caches, device)
                obs_emb = self.tokenizer(torch.as_tensor(ctx_obs[active, t], dtype=torch.float32, device=device))
                _, sub_caches = self.transformer.forward_incremental_batch(
                    obs_emb, positions_obs, sub_caches, embed_positions_obs,
                )
                positions_act, embed_positions_act = positions_obs + 1, embed_positions_obs + 1
                act_emb = self.action_embed(torch.as_tensor(ctx_action[active, t], dtype=torch.float32, device=device))
                _, sub_caches = self.transformer.forward_incremental_batch(
                    act_emb, positions_act, sub_caches, embed_positions_act,
                )
                for j, i in enumerate(active):
                    caches[i] = sub_caches[j]
            return caches

    def _advance_lane_caches(
        self, caches: list[_TransformerCache | None], obs_batch: np.ndarray, action_flat: np.ndarray,
    ) -> list[_TransformerCache]:
        """Extends each lane's *persistent* real-history cache by exactly
        one (obs, action) pair - the two tokens every real env step
        contributes - then trims it back down to `2 * context_length`
        tokens if it grew past that (module docstring's "Persistent,
        incrementally-extended per-lane root cache" section). Called from
        `learn()`'s collection loop and `predict()` after `search()` has
        already appended `obs_batch` on top of the *previous* call's
        returned caches to pick this step's action - appending that same
        `obs_batch` again here (rather than reusing `search()`'s own
        root-cache) keeps this method a self-contained, order-independent
        step so callers don't need to thread an extra "cache right after
        the root observation, before any action" value out of `search()`;
        the cost is one extra incremental Transformer call per lane per
        real step, negligible next to `search()`'s own `num_simulations`
        calls.

        Wrapped in `torch.no_grad()` deliberately, unlike `search()`'s own
        analogous root forward pass only by way of *also* needing it (that
        one already has its own `with torch.no_grad():`) - this method's
        *return value* is what actually becomes next real step's
        `self._lane_cache`/`self._eval_cache` (`learn()`/`predict()`'s own
        persistent-cache bookkeeping, module docstring's "Persistent,
        incrementally-extended per-lane root cache" section), which then
        gets threaded back in as *this same method's own* `caches` input
        on the *next* real step, and so on for the entire episode. Without
        `no_grad()` here, every real step's call would extend the autograd
        graph rooted at the *previous* step's (already graph-tracked)
        cached K/V tensors instead of starting fresh from detached leaves -
        an ever-growing graph, never freed (nothing ever calls `.backward()`
        on it or detaches it), for the entire length of one episode. On a
        long-horizon env (thousands of steps) that silently turns a model
        with a few hundred thousand parameters into a many-GiB `CUDA out of
        memory` well before the episode ends - `_train_step`/`_reanalyze`
        never call this method (they build their own short-lived,
        intentionally-graphed batches via `_embed_root_batch`), so nothing
        here ever legitimately needs gradients. `torch.inference_mode()`,
        not just `torch.no_grad()` - stronger than a speed optimization
        here specifically: an inference-mode tensor is *structurally*
        barred from ever being captured into a later autograd graph (no
        version counter at all), where a `no_grad()`-created tensor
        merely *starts out* not requiring grad but could still, in
        principle, end up used somewhere that does. Given this exact
        method already caused one real `CUDA out of memory` incident from
        a graph-leak in an earlier revision (see this docstring's own
        "without `no_grad()` here" paragraph above), the stronger
        guarantee is worth having on top of the speedup."""
        with torch.inference_mode():
            obs_emb = self.tokenizer(torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device))
            positions_obs, embed_positions_obs = _cache_positions(caches, self.device)
            _, caches1 = self.transformer.forward_incremental_batch(
                obs_emb, positions_obs, caches, embed_positions_obs,
            )
            act_emb = self.action_embed(torch.as_tensor(action_flat, dtype=torch.float32, device=self.device))
            positions_act, embed_positions_act = positions_obs + 1, embed_positions_obs + 1
            _, caches2 = self.transformer.forward_incremental_batch(
                act_emb, positions_act, caches1, embed_positions_act,
            )
            keep = 2 * self.context_length
            for cache in caches2:
                cache.evict_front(keep)
            return caches2

    def _append_actions_to_root_caches(
        self, root_caches: list[_TransformerCache], action_flat: np.ndarray,
    ) -> list[_TransformerCache]:
        """Persist search's already-computed root observation plus action.

        ``search`` has just appended the current real observation to every
        lane cache. Reusing that exact cache avoids tokenizing and attending
        over the same observation a second time in `_advance_lane_caches`.
        """
        with torch.inference_mode():
            positions, embed_positions = _cache_positions(root_caches, self.device)
            action_emb = self.action_embed(
                torch.as_tensor(action_flat, dtype=torch.float32, device=self.device),
            )
            _, caches = self.transformer.forward_incremental_batch(
                action_emb, positions, root_caches, embed_positions,
            )
            keep = 2 * self.context_length
            for cache in caches:
                cache.evict_front(keep)
            return caches

    # ------------------------------------------------------------------
    # Search - a genuine batched Gumbel search tree, see module docstring.
    # ------------------------------------------------------------------
    def _sample_continuous_candidates(self, mean_np: np.ndarray, std_np: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """K = `num_sampled_actions` candidate child actions for one node,
        sampled from *that node's own* predicted Gaussian policy - `k1`
        "on policy" (its own std) plus `k2` "wide" (inflated std,
        `continuous_prior_scale`x), the paper's `A_{S1}`/`A_{S2}` split
        (`efficientzero.py`'s own method, ported verbatim). Each
        candidate's prior is the (unnormalized) log-density of the
        on-policy Gaussian at that action - a consistent proposal-density
        prior for every slot regardless of which of the two samplers
        actually drew it, fed into the exact same Gumbel-Top-k/Sequential
        Halving/improved-policy formulas the discrete case uses.

        `mean_np`/`std_np`: already-`.cpu().numpy()`-converted `(action_dim,)`
        rows - *not* tensors any more (performance: `search()`'s own
        per-simulation, per-lane loop below used to call `.detach().cpu()`
        on a fresh one-row tensor slice per lane per simulation, which is
        one GPU sync/transfer each time; converting the *whole* batch
        tensor to numpy once per simulation and slicing plain numpy rows
        here instead turns `batch_size * num_simulations` GPU syncs into
        one)."""
        low, high = self._action_space.low, self._action_space.high
        k1 = max(1, self.num_sampled_actions // 2)
        k2 = max(1, self.num_sampled_actions - k1)
        cand_policy = np.random.normal(mean_np, std_np, size=(k1, self.action_dim))
        cand_wide = np.random.normal(mean_np, std_np * self.continuous_prior_scale, size=(k2, self.action_dim))
        candidates = np.clip(np.concatenate([cand_policy, cand_wide], axis=0), low, high).astype(np.float32)
        var = np.clip(std_np, 1e-6, None) ** 2
        log_probs = -0.5 * (((candidates - mean_np) ** 2) / var + np.log(2 * np.pi * var)).sum(axis=-1)
        return list(candidates), log_probs.astype(np.float64)

    def search(
        self, obs_batch: np.ndarray, root_caches: list[_TransformerCache | None], deterministic: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Batched Gumbel-Top-k + Sequential Halving search
        (`efficientzero.py`'s own search algorithm, see module docstring's
        "Search" section) - ALL B lanes' trees are searched together: one
        shared root tokenizer+Transformer pass, then `num_simulations`
        rounds where every lane descends its OWN tree by exactly one edge
        and all B lanes' chosen leaf expansions are computed in one
        batched `_step_imagine` call. Returns a length-B list of
        `{"env_action", "policy_target", "value_target"}`.

        `obs_batch`: `(B, *obs_shape)`, `root_caches[i]`: lane `i`'s
        existing real-history cache (or `None`) - `learn()`/`predict()`
        pass their own persistent per-lane cache directly, `_reanalyze()`
        passes a fresh cold-start replay from `_replay_context_to_cache`
        (module docstring's "everything else" section, `unizero.py`'s own
        machinery reused as-is here) - never mutated here (see
        `_TransformerCache`'s own docstring), so it's always safe for the
        caller to keep using its own copy afterward. `deterministic[i]`
        (defaults to all-`False`, i.e. "collecting") only affects whether
        the returned `env_action` is the argmax of the final policy target
        or sampled from it - unlike PUCT's root Dirichlet noise, Gumbel-
        Top-k's own per-simulation resampling at the root is its own,
        always-on exploration source, nothing to gate off for eval here
        (`efficientzero.py`'s own choice, see that file's module
        docstring's "No Dirichlet exploration noise" bullet)."""
        batch_size = obs_batch.shape[0]
        det = np.zeros(batch_size, dtype=bool) if deterministic is None else np.broadcast_to(deterministic, (batch_size,))
        # `torch.inference_mode()`, not `torch.no_grad()` - search never
        # needs autograd bookkeeping at all (not even a later `.backward()`
        # the way some `no_grad()` blocks elsewhere in this file still
        # get inspected under), so the (small but non-zero, and this is
        # by far the hottest loop in the whole algorithm - `num_simulations`
        # forward passes per real env step) extra version-counter/view-
        # tracking overhead `no_grad()` still pays is pure waste here.
        with torch.inference_mode():
            obs_emb = self.tokenizer(torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device))
            positions0, embed_positions0 = _cache_positions(root_caches, self.device)
            h_root, root_state_out = self.transformer.forward_incremental_batch(
                obs_emb, positions0, root_caches, embed_positions0,
            )
            root_value = self.heads.value(h_root)
            if self.discrete:
                root_logits = self.heads.policy_logits(h_root)
            else:
                root_mean, root_std = self.heads.policy_gaussian(h_root)
            # One GPU->CPU transfer for the *whole* root batch, not one per
            # lane (see `_sample_continuous_candidates`'s own docstring for
            # why this matters as much as it does) - every per-lane use
            # below is then plain numpy indexing, no further device sync.
            root_value_np = root_value.detach().cpu().numpy()
            if self.discrete:
                root_logits_np = root_logits.detach().cpu().numpy().astype(np.float64)
            else:
                root_mean_np = root_mean.detach().cpu().numpy()
                root_std_np = root_std.detach().cpu().numpy()

        n_slots = self.n_actions if self.discrete else self.num_sampled_actions
        base_num_simulations = self._current_num_simulations()
        current_num_simulations = self._effective_search_simulations(
            base_num_simulations,
            root_logits if self.discrete else None,
            None if self.discrete else root_std,
        )
        self._last_search_base_simulations = base_num_simulations
        self._last_search_num_simulations = current_num_simulations
        # Keep enough budget for an actual halving phase at the cheap
        # beginning of the schedule (8 simulations -> at most 4 roots).
        num_top_actions = max(
            2, min(self.num_top_actions, n_slots, max(2, current_num_simulations // 2)),
        )
        roots: list[_SearchNode] = []
        for i in range(batch_size):
            root = _SearchNode(prior=1.0)
            if self.discrete:
                # Raw logits, not softmax'd - `v_mix`/`improved_policy`
                # apply `_softmax` themselves where needed, and the
                # Gumbel-Top-k trick below specifically needs logits (its
                # `argmax(logit + Gumbel noise)` sampling identity only
                # holds in log-space).
                priors = root_logits_np[i]
                candidates = None
            else:
                candidates, priors = self._sample_continuous_candidates(root_mean_np[i], root_std_np[i])
            root.expand(priors, root_state_out[i], 0.0, candidate_actions=candidates)
            root.visit_count = 1
            root.value_sum = float(root_value_np[i])
            roots.append(root)

        minmax_list = [_MinMaxStats(self.value_minmax_delta) for _ in range(batch_size)]
        gumbel = np.random.gumbel(size=(batch_size, n_slots)) * self.policy_target_temperature * self._current_exploration_scale()
        schedule = _HalvingSchedule(current_num_simulations, num_top_actions)

        for sim_idx in range(current_num_simulations):
            leaf_nodes: list[_SearchNode] = []
            parent_caches: list[Any] = []
            chosen_actions: list[Any] = []
            search_paths: list[list[_SearchNode]] = []
            for lane in range(batch_size):
                root = roots[lane]
                if sim_idx == 0:
                    scores = gumbel[lane] + np.array([c.prior for c in root.children])
                    root.selected_children_idx = list(np.argsort(-scores)[:num_top_actions])
                node = root
                path = [node]
                while node.expanded():
                    idx = _select_action(node, minmax_list[lane], self.gamma, self.c_visit, self.c_scale)
                    node = node.children[idx]
                    path.append(node)
                parent = path[-2]
                leaf_nodes.append(node)
                parent_caches.append(parent.cache)
                chosen_actions.append(node.candidate_action if not self.discrete else parent.children.index(node))
                search_paths.append(path)

            if self.discrete:
                idx_t = torch.as_tensor(chosen_actions, dtype=torch.long, device=self.device)
                action_t = F.one_hot(idx_t, num_classes=self.n_actions).float()
            else:
                action_t = torch.as_tensor(np.stack(chosen_actions), dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                child_caches, h_act, h_obs = self._step_imagine(parent_caches, action_t)
                reward_scalar = self.heads.reward(h_act)
                next_value = self.heads.value(h_obs)
                reward_scalar, next_value = self._adjust_search_predictions(
                    h_act, h_obs, reward_scalar, next_value,
                )
                if self.discrete:
                    next_logits = self.heads.policy_logits(h_obs)
                else:
                    next_mean, next_std = self.heads.policy_gaussian(h_obs)
                # Again: one whole-batch GPU->CPU transfer per simulation,
                # not `batch_size` of them - this is the single biggest
                # search-time cost cut in this file (each `.item()`/
                # `.cpu()` call on a CUDA tensor is its own device sync;
                # `num_simulations * batch_size` of those per real env
                # step is exactly the "GPU busy but Python/CPU-bound"
                # profile a from-scratch profiling pass of this search
                # loop turned up).
                reward_np = reward_scalar.detach().cpu().numpy()
                value_np = next_value.detach().cpu().numpy()
                if self.discrete:
                    next_logits_np = next_logits.detach().cpu().numpy().astype(np.float64)
                else:
                    next_mean_np = next_mean.detach().cpu().numpy()
                    next_std_np = next_std.detach().cpu().numpy()

            for lane in range(batch_size):
                leaf = leaf_nodes[lane]
                rv = float(reward_np[lane])
                if self.discrete:
                    leaf.expand(next_logits_np[lane], child_caches[lane], rv)
                else:
                    candidates, priors = self._sample_continuous_candidates(next_mean_np[lane], next_std_np[lane])
                    leaf.expand(priors, child_caches[lane], rv, candidate_actions=candidates)
                _backpropagate(search_paths[lane], float(value_np[lane]), minmax_list[lane], self.gamma)

            if schedule.maybe_advance(sim_idx):
                for lane in range(batch_size):
                    _sequential_halving(
                        roots[lane], gumbel[lane], minmax_list[lane], schedule.current_top, self.gamma, self.c_visit, self.c_scale,
                    )

        results: list[dict[str, Any]] = []
        for lane in range(batch_size):
            root = roots[lane]
            transformed = _transformed_completed_qs(root, minmax_list[lane], self.gamma, self.c_visit, self.c_scale)
            improved = root.improved_policy(transformed)
            best_idx = int(root.selected_children_idx[0]) if root.selected_children_idx else int(np.argmax(improved))
            if self.discrete:
                policy_target = improved.astype(np.float32)
                if det[lane]:
                    env_action: Any = int(np.argmax(policy_target))
                else:
                    probs = policy_target / max(float(policy_target.sum()), 1e-8)
                    env_action = int(np.random.choice(len(probs), p=probs))
            else:
                candidates_arr = np.stack([c.candidate_action for c in root.children])
                # `efficientzero.py`'s own, simpler continuous policy
                # target: the visit-weighted (well, improved-policy-
                # weighted) *average* candidate action, not the root's
                # sampled candidates and their weights kept separate
                # (`unizero.py`'s own Sampled-UniZero choice) - consumed
                # by `_train_step`'s plain MSE continuous policy loss
                # below, not an importance-weighted NLL.
                policy_target = (improved[:, None] * candidates_arr).sum(axis=0).astype(np.float32)
                env_action = candidates_arr[best_idx].astype(np.float32)
            results.append({
                "env_action": env_action,
                "policy_target": policy_target,
                "value_target": root.value(),
                "root_cache": root.cache,
            })
        return results


    # ------------------------------------------------------------------
    # Reanalyze - ported wholesale from `efficientzero.py`'s `_reanalyze`.
    # ------------------------------------------------------------------
    def _reanalyze(self) -> dict[str, float] | None:
        if self.reanalyze_batch_size <= 0 or self.buffer.num_episodes == 0:
            return None
        episode_idx, timestep, obs_batch, ctx_obs, ctx_action, ctx_valid = self.buffer.sample_for_reanalyze(
            self.reanalyze_batch_size,
        )
        if obs_batch.shape[0] == 0:
            return None
        # No persistent cache exists for these arbitrary historical
        # samples - cold-start one by replaying each sample's own real
        # context (module docstring's "Persistent, incrementally-extended
        # per-lane root cache" section).
        root_caches = self._replay_context_to_cache(ctx_obs, ctx_action, ctx_valid)
        results = self.search(obs_batch, root_caches, deterministic=np.zeros(obs_batch.shape[0], dtype=bool))
        policy_targets = np.stack([r["policy_target"] for r in results])
        search_values = np.asarray([r["value_target"] for r in results], dtype=np.float32)
        policy_target_valid = np.asarray(
            [r.get("policy_target_valid", True) for r in results],
            dtype=bool,
        )
        self.buffer.update_reanalyzed_targets(
            episode_idx,
            timestep,
            policy_targets,
            search_values,
            policy_target_valid,
        )
        return {"reanalyze_mean_search_value": float(search_values.mean())}

    # ------------------------------------------------------------------
    # Training-progress-dependent schedules (module docstring's
    # "Schedules, not fixed constants" section) - LR decay, priority-beta
    # annealing, and search/entropy temperature annealing all read their
    # own progress off the *same* fraction here, rather than each
    # tracking it separately.
    # ------------------------------------------------------------------
    def _training_progress(self) -> float:
        """`0.0` right when `learning_starts` gradient steps begin, `1.0`
        once `self._num_timesteps` (set every real step in `learn()`)
        reaches the `total_timesteps` this call's `learn()` was given -
        every schedule below anneals linearly (or cosine, for LR) across
        this one shared `[0, 1]` fraction."""
        return min(1.0, max(0.0, self._num_timesteps / max(1, self._total_timesteps_estimate)))

    def _update_learning_rate(self) -> float:
        """Cosine-decays `self.optimizer`'s LR from `base_learning_rate`
        down to `base_learning_rate * lr_min_fraction` across training
        progress (module docstring's "Schedules, not fixed constants"
        section) - a no-op (flat `base_learning_rate`) whenever
        `lr_schedule` is off, for exact backwards compatibility with
        every run/test/saved config that predates this option. Returns
        the LR actually applied, purely for the returned metrics dict."""
        if not self.lr_schedule:
            return self.base_learning_rate
        progress = self._training_progress()
        floor = self.base_learning_rate * self.lr_min_fraction
        lr = floor + 0.5 * (self.base_learning_rate - floor) * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _update_priority_beta(self) -> float:
        """Linearly anneals `self.buffer.priority_beta` from
        `priority_beta_start` up to `priority_beta` across training
        progress (see `DEFAULT_HYPERPARAMS["priority_beta"]`'s own
        docstring for why) - a no-op whenever the two are equal (the
        pre-this-option flat-`priority_beta` behavior)."""
        progress = self._training_progress()
        beta = self.priority_beta_start + (self.priority_beta - self.priority_beta_start) * progress
        self.buffer.priority_beta = beta
        return beta

    def _update_target_network(self) -> None:
        """Polyak/EMA update of `target_tokenizer`/`target_transformer`/
        `target_heads` towards their online counterparts, one
        `target_update_theta`-sized step per `_train_step` call - a
        no-op (never even called) whenever `use_target_for_bootstrap` is
        off."""
        theta = self.target_update_theta
        with torch.no_grad():
            for target_module, online_module in (
                (self.target_tokenizer, self.tokenizer),
                (self.target_action_embed, self.action_embed),
                (self.target_transformer, self.transformer),
                (self.target_heads, self.heads),
            ):
                for target_param, online_param in zip(target_module.parameters(), online_module.parameters()):
                    target_param.mul_(1.0 - theta).add_(online_param, alpha=theta)
                for target_buf, online_buf in zip(target_module.buffers(), online_module.buffers()):
                    if torch.is_floating_point(target_buf):
                        target_buf.mul_(1.0 - theta).add_(online_buf, alpha=theta)
                    else:
                        target_buf.copy_(online_buf)

    def _current_exploration_scale(self) -> float:
        """`1.0` (no-op) once training progress reaches `1.0`, or always
        if `temperature_anneal_start_scale<=1.0`; otherwise linearly
        decays from `temperature_anneal_start_scale` down to `1.0` -
        shared by `search()`'s own root Gumbel noise scale and
        `_train_step`'s policy-entropy-bonus coefficient (see
        `DEFAULT_HYPERPARAMS["temperature_anneal_start_scale"]`'s own
        docstring)."""
        if self.temperature_anneal_start_scale <= 1.0:
            return 1.0
        progress = min(1.0, max(0.0, self._num_timesteps / self.temperature_anneal_steps))
        return 1.0 + (self.temperature_anneal_start_scale - 1.0) * (1.0 - progress)

    def _current_num_simulations(self) -> int:
        """Spend search compute only when the closed-loop model earns it."""
        step_cap = min(1.0, max(0.0, self._num_timesteps / self.search_ramp_steps))
        model_confidence = (
            self.search_model_error_high - self._model_error_ema
        ) / (self.search_model_error_high - self.search_model_error_low)
        progress = min(step_cap, min(1.0, max(0.0, model_confidence)))
        simulations = round(
            self.num_simulations_initial
            + progress * (self.num_simulations - self.num_simulations_initial),
        )
        return max(2, min(self.num_simulations, int(simulations)))

    def _effective_train_steps_per_iter(self) -> int:
        """Learner cadence hook; ResearchImZero keeps its configured count."""
        return self.train_steps_per_iter

    def _effective_search_simulations(
        self,
        base_simulations: int,
        policy_logits: torch.Tensor | None,
        policy_std: torch.Tensor | None,
    ) -> int:
        del policy_logits, policy_std
        return base_simulations

    def _effective_reanalyze_value_mix(
        self,
        action_hidden: torch.Tensor,
        next_hidden: torch.Tensor,
        base_mix: float,
    ) -> torch.Tensor:
        del action_hidden, next_hidden
        return torch.as_tensor(base_mix, dtype=torch.float32, device=self.device)

    def _effective_closed_loop_horizon(self) -> int:
        return (
            self._adaptive_closed_loop_horizon
            if self.adaptive_closed_loop
            else self.closed_loop_horizon
        )

    def _update_closed_loop_error_ema(self, errors: list[torch.Tensor]) -> None:
        if not self.adaptive_closed_loop:
            return
        decay = self.adaptive_closed_loop_ema_decay
        for depth, error in enumerate(errors):
            observed = float(error.detach().item())
            previous = self._closed_loop_error_ema[depth]
            self._closed_loop_error_ema[depth] = (
                observed
                if not np.isfinite(previous)
                else decay * previous + (1.0 - decay) * observed
            )
        required = min(3, self._adaptive_closed_loop_horizon)
        stable = (
            self._grad_step_count >= self.adaptive_closed_loop_min_grad_steps
            and all(
                np.isfinite(self._closed_loop_error_ema[depth])
                and self._closed_loop_error_ema[depth]
                < self.adaptive_closed_loop_error_threshold
                for depth in range(required)
            )
        )
        self._adaptive_closed_loop_stable_count = (
            self._adaptive_closed_loop_stable_count + 1 if stable else 0
        )
        if (
            self._adaptive_closed_loop_stable_count
            >= self.adaptive_closed_loop_stable_steps
            and self._adaptive_closed_loop_horizon
            < self.adaptive_closed_loop_max_horizon
        ):
            self._adaptive_closed_loop_horizon = self.adaptive_closed_loop_max_horizon
            self._adaptive_closed_loop_stable_count = 0

    def _rolling_reward_metrics(self) -> dict[str, float]:
        values = np.asarray(self._episode_reward_rolling, dtype=np.float64)
        return {
            "episode_reward_rolling_mean": float(values.mean()) if values.size else 0.0,
            "episode_reward_rolling_std": float(values.std()) if values.size else 0.0,
            "episode_reward_rolling_count": float(values.size),
        }

    def _compute_path_consistency_loss(
        self,
        value_logits: list[torch.Tensor],
        reward_predictions: list[torch.Tensor],
        mask: torch.Tensor,
        is_weight: torch.Tensor,
    ) -> torch.Tensor:
        if len(value_logits) < 2:
            return mask.new_zeros(())
        terms = mask.new_zeros(())
        denominator = mask.new_zeros(())
        for k in range(len(value_logits) - 1):
            valid_pair = mask[:, k] * mask[:, k + 1]
            target = (
                reward_predictions[k].detach()
                + self.gamma
                * _logits_to_scalar(value_logits[k + 1], self.support_size).detach()
            )
            target_two_hot = _scalar_to_two_hot(
                target, self.support_size, self.label_smoothing_eps,
            )
            cross_entropy = -(
                target_two_hot * F.log_softmax(value_logits[k], dim=-1)
            ).sum(-1)
            terms = terms + (cross_entropy * valid_pair * is_weight).sum()
            denominator = denominator + valid_pair.sum()
        return terms / denominator.clamp_min(1.0)

    def _apply_learning_progress_priority(
        self,
        current_priority: np.ndarray,
        episode_idx: np.ndarray,
        timestep: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        bonus = np.zeros_like(current_priority)
        if self.learning_progress_priority_weight <= 0.0:
            return current_priority, bonus
        previous_raw_error = self.buffer.raw_learning_errors_for(episode_idx, timestep)
        improvement = np.where(
            np.isfinite(previous_raw_error),
            np.maximum(previous_raw_error - current_priority, 0.0),
            0.0,
        )
        bonus = self.learning_progress_priority_weight * np.clip(
            improvement,
            0.0,
            self.learning_progress_priority_cap,
        )
        self.buffer.update_raw_learning_errors(episode_idx, timestep, current_priority)
        return current_priority + bonus, bonus

    def _after_core_train_step(
        self,
        batch: dict[str, Any],
        hidden: torch.Tensor,
        obs0_pos: torch.Tensor,
        value_targets: torch.Tensor,
    ) -> dict[str, float]:
        """Optional detached sidecar hook; a strict no-op for ResearchImZero."""
        del batch, hidden, obs0_pos, value_targets
        return {}

    # ------------------------------------------------------------------
    # Training - one forward pass over the full teacher-forced window
    # (context + real interleaved obs/action tokens), reading every step's
    # losses off that single pass (module docstring's "Full-trajectory
    # teacher forcing" bullet).
    # ------------------------------------------------------------------
    def _train_step(self) -> dict[str, float]:
        current_lr = self._update_learning_rate()
        current_priority_beta = self._update_priority_beta()
        self._grad_step_count += 1
        batch = self.buffer.sample(self.batch_size, self.unroll_steps, self.td_steps, self.gamma)
        device = self.device
        b, k_steps = self.batch_size, self.unroll_steps
        length = self.context_length

        obs0 = torch.as_tensor(batch["obs0"], dtype=torch.float32, device=device)
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)
        policy_target = torch.as_tensor(batch["policy_target"], dtype=torch.float32, device=device)
        policy_target_valid = torch.as_tensor(
            batch["policy_target_valid"], dtype=torch.float32, device=device,
        )
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=device)
        td_reward = torch.as_tensor(batch["td_reward"], dtype=torch.float32, device=device)
        td_discount = torch.as_tensor(batch["td_discount"], dtype=torch.float32, device=device)
        td_bootstrap_mask = torch.as_tensor(batch["td_bootstrap_mask"], dtype=torch.float32, device=device)
        # `_reanalyze()`'s periodically-refreshed MCTS root value, per
        # real transition (`_NO_SEARCH_VALUE` sentinel for never-
        # reanalyzed ones) - blended into the value target below via
        # `max()`, `efficientzero.py`'s own `value_target: 'max'` mode
        # (module docstring's "no target network" section).
        search_value = torch.as_tensor(batch["search_value"], dtype=torch.float32, device=device)
        # Freshness gate (module docstring's "Schedules, not fixed
        # constants" section): a transition whose `search_value` was
        # reanalyzed more than `search_value_max_staleness` generations
        # ago gets treated as if it had never been reanalyzed at all
        # (`_NO_SEARCH_VALUE`, always loses the `max()` below against
        # `td_target`) rather than an indefinitely-stale value winning
        # forever just because it happened to be large.
        search_value_gen = torch.as_tensor(batch["search_value_gen"], dtype=torch.long, device=device)
        staleness = self.buffer.reanalyze_generation - search_value_gen
        is_fresh = (search_value_gen >= 0) & (staleness <= self.search_value_max_staleness)
        search_value = torch.where(is_fresh, search_value, torch.full_like(search_value, self.buffer._NO_SEARCH_VALUE))
        is_weight = torch.as_tensor(batch["is_weight"], dtype=torch.float32, device=device)

        # Raw token embeddings for the whole window: context (compact,
        # `2*n_i` tokens for lane `i`'s own `n_i` valid context
        # transitions - see `_embed_root_batch`), then interleaved [obs0,
        # act0, next_obs0=obs1, act1, ..., obs_{K-1}, act_{K-1},
        # next_obs_{K-1}=obs_K] - `2*k_steps` more tokens. Lanes with
        # different amounts of real context (`valid_len`) occupy
        # different-length prefixes of this shared, batch-padded tensor -
        # each lane's *own* unroll tokens are placed right after its own
        # context ends, not at one fixed shared offset (see
        # `_embed_root_batch`'s docstring for why: with compact, no-gap
        # positions - the same convention the incremental KV-cache path
        # uses - a fixed shared offset would leave a position-numbering
        # gap for every lane whose own context is shorter than the
        # batch's longest).
        # Autocast wraps every forward call below (tokenizer, action
        # embed, transformer, heads, projector/predictor) - the whole
        # graph this step's `total_loss.backward()` runs through, not
        # just the online-net calls - `use_amp`/`self._amp_dtype`
        # (`__init__`'s own "Performance, not architecture" comment). A
        # pure no-op on CPU (`self.use_amp` is forced `False` there).
        with torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self.use_amp):
            root_tokens, root_pad, valid_len = self._embed_root_batch(
                batch["context_obs"], batch["context_action"], batch["context_valid"], batch["obs0"],
            )  # (B, Lmax0, E); `valid_len[i]` = lane `i`'s own real length (incl. obs0)
            act_embs = self.action_embed(action.view(b * k_steps, self.action_dim)).view(b, k_steps, self.embed_dim)
            next_obs_embs_grad = self.tokenizer(next_obs.view(b * k_steps, *self._obs_shape)).view(b, k_steps, self.embed_dim)

            lmax0 = root_tokens.shape[1]
            total_len = lmax0 + 2 * k_steps
            tokens = torch.zeros(b, total_len, self.embed_dim, device=device)
            pad_mask = torch.zeros(b, total_len, dtype=torch.bool, device=device)
            tokens[:, :lmax0] = root_tokens
            pad_mask[:, :lmax0] = root_pad
            for i in range(b):
                base_i = int(valid_len[i])
                end_i = base_i + 2 * k_steps
                tokens[i, base_i:end_i:2] = act_embs[i]
                tokens[i, base_i + 1 : end_i : 2] = next_obs_embs_grad[i]
                pad_mask[i, base_i:end_i] = True

            hidden = self.transformer(tokens, pad_mask)
            # Context-aware EMA bootstrap. Build one target-network
            # teacher-forced sequence per sampled start, extending its real
            # context through enough future transitions to cover every
            # k-step TD landing state. This is both cheaper than B*K
            # independent context replays and, crucially, does not train a
            # history-aware value head against a one-token/memoryless target.
            with torch.no_grad():
                bootstrap_tokenizer = self.target_tokenizer if self.use_target_for_bootstrap else self.tokenizer
                bootstrap_action_embed = (
                    self.target_action_embed if self.use_target_for_bootstrap else self.action_embed
                )
                bootstrap_transformer = self.target_transformer if self.use_target_for_bootstrap else self.transformer
                bootstrap_heads = self.target_heads if self.use_target_for_bootstrap else self.heads
                target_root, target_root_pad, target_valid_len = self._embed_root_batch(
                    batch["context_obs"], batch["context_action"], batch["context_valid"], batch["obs0"],
                    tokenizer=bootstrap_tokenizer, action_embed=bootstrap_action_embed,
                )
                bootstrap_steps = batch["bootstrap_action"].shape[1]
                target_actions = torch.as_tensor(
                    batch["bootstrap_action"], dtype=torch.float32, device=device,
                )
                target_next_obs = torch.as_tensor(
                    batch["bootstrap_next_obs"], dtype=torch.float32, device=device,
                )
                target_act_emb = bootstrap_action_embed(
                    target_actions.view(b * bootstrap_steps, self.action_dim),
                ).view(b, bootstrap_steps, self.embed_dim)
                target_obs_emb = bootstrap_tokenizer(
                    target_next_obs.view(b * bootstrap_steps, *self._obs_shape),
                ).view(b, bootstrap_steps, self.embed_dim)
                target_total_len = target_root.shape[1] + 2 * bootstrap_steps
                target_tokens = torch.zeros(b, target_total_len, self.embed_dim, device=device)
                target_pad = torch.zeros(b, target_total_len, dtype=torch.bool, device=device)
                target_tokens[:, : target_root.shape[1]] = target_root
                target_pad[:, : target_root.shape[1]] = target_root_pad
                target_steps_valid = batch["bootstrap_mask"].sum(axis=1).astype(np.int64)
                for i in range(b):
                    base_i = int(target_valid_len[i])
                    n_i = int(target_steps_valid[i])
                    target_tokens[i, base_i : base_i + 2 * n_i : 2] = target_act_emb[i, :n_i]
                    target_tokens[i, base_i + 1 : base_i + 2 * n_i : 2] = target_obs_emb[i, :n_i]
                    target_pad[i, base_i : base_i + 2 * n_i] = True
                target_hidden = bootstrap_transformer(target_tokens, target_pad)
                td_horizon = torch.as_tensor(batch["td_horizon"], dtype=torch.long, device=device)
                target_root_pos = torch.as_tensor(target_valid_len - 1, dtype=torch.long, device=device)
                target_landing_pos = target_root_pos[:, None] + 2 * (
                    torch.arange(k_steps, device=device)[None, :] + td_horizon
                )
                target_batch_idx = torch.arange(b, device=device)[:, None].expand(b, k_steps)
                bootstrap_values = bootstrap_heads.value(
                    target_hidden[target_batch_idx, target_landing_pos],
                )
                bootstrap_values = bootstrap_values * td_bootstrap_mask
            # `obs0_pos[i]` = lane `i`'s own obs0 token position (its context's
            # last real slot) - differs per lane now, so every hidden-state
            # read below gathers by index instead of slicing a shared offset.
            obs0_pos = torch.as_tensor(valid_len - 1, dtype=torch.long, device=device)
            batch_idx = torch.arange(b, device=device)

            reward_loss = torch.zeros((), device=device)
            value_loss = torch.zeros((), device=device)
            policy_loss = torch.zeros((), device=device)
            consistency_loss = torch.zeros((), device=device)
            policy_entropy = torch.zeros((), device=device)
            first_step_value_pred: torch.Tensor | None = None
            first_step_value_target: torch.Tensor | None = None
            first_step_policy_error: torch.Tensor | None = None
            value_targets: list[torch.Tensor] = []
            path_value_logits: list[torch.Tensor] = []
            path_reward_predictions: list[torch.Tensor] = []
            effective_sve_mixes: list[torch.Tensor] = []

            for k in range(k_steps):
                m = mask[:, k]
                wm = m * is_weight
                denom = m.sum().clamp_min(1.0)
                policy_mask = m * policy_target_valid[:, k]
                policy_weight = policy_mask * is_weight
                policy_denom = policy_mask.sum().clamp_min(1.0)
                h_obs_k = hidden[batch_idx, obs0_pos + 2 * k]  # obs_k's own hidden state
                h_act_k = hidden[batch_idx, obs0_pos + 2 * k + 1]  # act_k's own hidden state

                value_pred_logits = self.heads.value_logits(h_obs_k)
                # Bootstrap values above come from one context-aware EMA
                # sequence. Fresh reanalysis can contribute through a
                # bounded convex blend; stale or absent search values fall
                # back to pure n-step TD.
                with torch.no_grad():
                    td_target = td_reward[:, k] + td_discount[:, k] * bootstrap_values[:, k]
                    sve_next_hidden = (
                        hidden[batch_idx, obs0_pos + 2 * (k + 1)]
                        if k + 1 < k_steps
                        else h_obs_k
                    )
                    effective_mix = self._effective_reanalyze_value_mix(
                        h_act_k, sve_next_hidden, self.reanalyze_value_mix,
                    )
                    mixed_target = (
                        (1.0 - effective_mix) * td_target
                        + effective_mix * search_value[:, k]
                    )
                    value_target = torch.where(is_fresh[:, k], mixed_target, td_target)
                    effective_sve_mixes.append(torch.as_tensor(effective_mix).detach())
                value_targets.append(value_target.detach())
                value_two_hot = _scalar_to_two_hot(value_target, self.support_size, self.label_smoothing_eps)
                v_loss = -(value_two_hot * F.log_softmax(value_pred_logits, dim=-1)).sum(-1)
                value_loss = value_loss + (v_loss * wm).sum() / denom
                if k == 0:
                    with torch.no_grad():
                        first_step_value_pred = _logits_to_scalar(value_pred_logits, self.support_size).detach()
                        first_step_value_target = value_target.detach()

                reward_logits_k = self.heads.reward_logits(h_act_k)
                if self.path_consistency_coef > 0.0:
                    path_value_logits.append(value_pred_logits)
                    path_reward_predictions.append(
                        _logits_to_scalar(reward_logits_k, self.support_size),
                    )
                reward_two_hot = _scalar_to_two_hot(reward[:, k], self.support_size, self.label_smoothing_eps)
                r_loss = -(reward_two_hot * F.log_softmax(reward_logits_k, dim=-1)).sum(-1)
                reward_loss = reward_loss + (r_loss * wm).sum() / denom

                if self.discrete:
                    logp = F.log_softmax(self.heads.policy_logits(h_obs_k), dim=-1)
                    p_loss = -(policy_target[:, k] * logp).sum(-1)
                    entropy_k = -(logp.exp() * logp).sum(-1)
                else:
                    mean, std = self.heads.policy_gaussian(h_obs_k)
                    # `efficientzero.py`'s own, simpler continuous policy loss
                    # (module docstring's "Search" section): plain MSE toward
                    # `search()`'s visit-weighted-*average* candidate action
                    # (`policy_target[:, k]`, shape `(B, action_dim)`) - not
                    # `unizero.py`'s own importance-weighted NLL against the
                    # full sampled-candidate set.
                    p_loss = F.mse_loss(mean, policy_target[:, k], reduction="none").sum(-1)
                    # Differential entropy of an (assumed-diagonal) Gaussian,
                    # summed over action dims - closed form, no sampling
                    # needed: `0.5*log(2*pi*e*sigma^2)` per dim.
                    var = std.clamp_min(1e-6) ** 2
                    entropy_k = (0.5 * torch.log(2.0 * math.pi * math.e * var)).sum(-1)
                policy_loss = policy_loss + (p_loss * policy_weight).sum() / policy_denom
                policy_entropy = (
                    policy_entropy + (entropy_k * policy_weight).sum() / policy_denom
                )
                if k == 0:
                    # Priority (below, after the unroll loop) blends this
                    # in alongside the value error - a transition whose
                    # *policy* target the network gets badly wrong is just
                    # as informative to replay again soon as one whose
                    # value estimate is off, but the old
                    # `|value_pred - value_target|`-only priority never
                    # surfaced it (module docstring's "Schedules, not
                    # fixed constants" section).
                    first_step_policy_error = (
                        p_loss * policy_target_valid[:, k]
                    ).detach()

                # SimSiam consistency loss (module docstring's "no target
                # network" section, `efficientzero.py`'s own recipe): the
                # dynamics-predicted next-token embedding (`heads.latent
                # (h_act_k)`) should approximate the *real* next
                # observation's own token embedding, trained via negative
                # cosine similarity against a stop-gradiented copy of the
                # *online* tokenizer's own embedding of it (`next_obs_embs_
                # grad[:, k].detach()` - already computed above as part of
                # this window's own input tokens, no extra forward pass
                # needed) rather than a separate momentum target network.
                # `_Projector`applied to both branches (shared, SimSiam's own
                # asymmetric-predictor-only-on-one-side convention),
                # `_Predictor` only on the predicted branch - `BatchNorm1d`
                # inside both is what actually prevents the every-observation-
                # embeds-to-the-same-point collapse stop-gradient alone
                # doesn't (Chen & He, 2021, Table 2c - see `_Projector`'s own
                # docstring).
                true_next_emb = next_obs_embs_grad[:, k].detach()
                p_true = F.normalize(self.projector(true_next_emb), dim=-1).detach()
                z_pred = self.heads.latent(h_act_k)
                p_pred = F.normalize(self.predictor(self.projector(z_pred)), dim=-1)
                # `1 - cos_sim`, not the more common `-cos_sim` - both have
                # the *exact same gradient* (a constant `+1` shift), but
                # `-cos_sim` can go negative (cos_sim in `[-1, 1]`), which
                # silently breaks `adaptive_loss_weights`'s own Kendall-
                # style `precision * loss + log_var` term below: that
                # formula's stability derivation assumes `loss >= 0` (an
                # interior minimum only exists there) - handed a
                # *negative* loss, minimizing it has no interior minimum
                # at all, so `precision = exp(-log_var)` runs away to
                # `+inf` instead of converging (this file's own postmortem:
                # `loss_weight_consistency` measured climbing 2 -> 9896 in
                # under 7k steps on a real NetHack run, dragging
                # `total_loss` to `-13486` and swamping every other task's
                # gradient in the process - training never recovered).
                # `1 - cos_sim` (`[0, 2]`, floored at exactly `0`) is the
                # same fix in spirit as most public SimSiam
                # implementations' own `2 - 2*cos_sim` convention - not a
                # tuning choice, a correctness fix for `adaptive_loss_
                # weights` specifically.
                c_loss = 1.0 - (p_true * p_pred).sum(-1)
                consistency_loss = consistency_loss + (c_loss * wm).sum() / denom

            path_consistency_loss = torch.zeros((), device=device)
            if self.path_consistency_coef > 0.0 and k_steps > 1:
                path_consistency_loss = self._compute_path_consistency_loss(
                    path_value_logits, path_reward_predictions, mask, is_weight,
                )

            # Closed-loop latent overshooting. A small sampled sub-batch is
            # rolled forward recursively through `_step_imagine`, so step
            # k+1 consumes the model's own predicted observation token from
            # step k exactly as MCTS does. Teacher forcing above remains the
            # stable anchor; this auxiliary branch closes the train/search
            # exposure gap without trusting wholly imagined trajectories.
            closed_loop_loss = torch.zeros((), device=device)
            closed_loop_latent_errors: list[torch.Tensor] = []
            effective_closed_loop_horizon = self._effective_closed_loop_horizon()
            closed_n = min(
                b,
                max(2, int(round(b * self.closed_loop_batch_fraction))),
            ) if b >= 2 and self.closed_loop_batch_fraction > 0 and self.closed_loop_loss_coef > 0 else 0
            if closed_n:
                cl_obs_emb = self.tokenizer(obs0[:closed_n])
                cl_positions = torch.zeros(closed_n, dtype=torch.long, device=device)
                _, cl_caches = self.transformer.forward_incremental_batch(
                    cl_obs_emb, cl_positions, [None] * closed_n,
                )
                cl_terms = torch.zeros((), device=device)
                cl_denom = torch.zeros((), device=device)
                for k in range(effective_closed_loop_horizon):
                    cl_caches, cl_h_act, cl_h_obs = self._step_imagine(cl_caches, action[:closed_n, k])
                    cl_mask = mask[:closed_n, k]
                    cl_weight = cl_mask * is_weight[:closed_n]

                    cl_reward_target = _scalar_to_two_hot(
                        reward[:closed_n, k], self.support_size, self.label_smoothing_eps,
                    )
                    cl_reward_loss = -(
                        cl_reward_target * F.log_softmax(self.heads.reward_logits(cl_h_act), dim=-1)
                    ).sum(-1)
                    cl_pred_latent = self.heads.latent(cl_h_act)
                    cl_true = F.normalize(
                        self.projector(next_obs_embs_grad[:closed_n, k].detach()), dim=-1,
                    ).detach()
                    cl_pred = F.normalize(
                        self.predictor(self.projector(cl_pred_latent)), dim=-1,
                    )
                    cl_latent_loss = 1.0 - (cl_true * cl_pred).sum(-1)
                    closed_loop_latent_errors.append(
                        (cl_latent_loss * cl_mask).sum() / cl_mask.sum().clamp_min(1.0),
                    )
                    step_loss = self.reward_loss_coef * cl_reward_loss + self.consistency_loss_coef * cl_latent_loss

                    if k + 1 < k_steps:
                        cl_value_target = _scalar_to_two_hot(
                            value_targets[k + 1][:closed_n], self.support_size, self.label_smoothing_eps,
                        )
                        cl_value_loss = -(
                            cl_value_target * F.log_softmax(self.heads.value_logits(cl_h_obs), dim=-1)
                        ).sum(-1)
                        step_loss = step_loss + self.value_loss_coef * cl_value_loss

                    horizon_weight = 0.8 ** k
                    cl_terms = cl_terms + horizon_weight * (step_loss * cl_weight).sum()
                    cl_denom = cl_denom + horizon_weight * cl_mask.sum()
                closed_loop_loss = cl_terms / cl_denom.clamp_min(1.0)

            n = float(k_steps)
            reward_loss, value_loss, policy_loss, consistency_loss, policy_entropy = (
                reward_loss / n, value_loss / n, policy_loss / n, consistency_loss / n, policy_entropy / n,
            )
            if self.adaptive_loss_weights:
                # Kendall et al. (2018) - see `DEFAULT_HYPERPARAMS
                # ["adaptive_loss_weights"]`'s own docstring. `precision_i
                # = exp(-log_var_i)` down-weights a task whose loss the
                # network currently can't push down much further (a large,
                # stubborn loss keeps its own `log_var` high, shrinking its
                # `precision`), the `+ log_var_i` regularizer is what stops
                # every `log_var` from simply drifting to `+inf` to trivially
                # zero out every task's contribution.
                static_coefs = torch.stack(
                    [
                        torch.as_tensor(self.reward_loss_coef, device=device),
                        torch.as_tensor(self.value_loss_coef, device=device),
                        torch.as_tensor(self.policy_loss_coef, device=device),
                    ],
                )
                # `consistency_loss` is *not* in here - see `__init__`'s
                # own `self.loss_log_vars` comment for why (no genuine
                # noise floor -> unbounded weight growth even after the
                # `1 - cos_sim` sign fix, just capped at the clamp ceiling
                # instead of at `+inf`). It keeps its plain static
                # `consistency_loss_coef` term, added on unweighted-
                # further, same as the `else` branch right below.
                task_losses = torch.stack([reward_loss, value_loss, policy_loss])
                # Defense-in-depth, on top of the `1 - cos_sim` fix above:
                # clamps `precision` to `[~0.007, ~148]` (`log_var` to
                # `[-5, 5]`) no matter which task loss it's attached to -
                # even a strictly non-negative loss that keeps shrinking
                # towards (but never quite reaching) `0` late in training
                # would otherwise let its own `precision` climb without a
                # real ceiling (this file's own postmortem saw `loss_
                # weight_reward` climbing monotonically the entire run,
                # never plateauing, off an already-tiny `reward_loss`).
                # Bounded, not clamped to a single fixed value - each
                # task's weight still adapts freely *within* this range.
                log_var = self.loss_log_vars.clamp(-5.0, 5.0)
                precision = torch.exp(-log_var)
                total_loss = (
                    (static_coefs * precision * task_losses + log_var).sum()
                    + self.consistency_loss_coef * consistency_loss
                )
            else:
                total_loss = (
                    self.reward_loss_coef * reward_loss + self.value_loss_coef * value_loss
                    + self.policy_loss_coef * policy_loss + self.consistency_loss_coef * consistency_loss
                )
            total_loss = total_loss + self.closed_loop_loss_coef * closed_loop_loss
            total_loss = total_loss + self.path_consistency_coef * path_consistency_loss
            # Entropy *bonus* - reference's own `policy_entropy_weight`
            # (default `5e-3`): subtracted from the total loss (not
            # added) since higher policy entropy is the *goal* here, one
            # of MuZero-family exploration regularizers this file had no
            # equivalent of before. Not part of the adaptive-weighting
            # blend above - it's a regularizer, not a task with its own
            # loss to balance against the others.
            total_loss = total_loss - self.policy_entropy_coef * self._current_exploration_scale() * policy_entropy
        # `GradScaler` is `enabled=False` (pure passthrough) whenever
        # we're not doing fp16 autocast (`__init__`'s own
        # `self._grad_scaler`) - bf16/CPU/no-AMP all just call
        # `.backward()`/`.step()` exactly as before.
        self.optimizer.zero_grad()
        self._grad_scaler.scale(total_loss).backward()
        self._grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self._params, self.max_grad_norm)
        self._grad_scaler.step(self.optimizer)
        self._grad_scaler.update()
        if self.adaptive_loss_weights:
            # In-place, post-step - `total_loss`'s own `.clamp(-5.0, 5.0)`
            # (forward-pass-only) keeps a saturated `log_var` from making
            # things *worse* (zero gradient once clamped), but doesn't
            # stop Adam's own momentum from still carrying the raw
            # parameter further out on a run long enough for it to
            # matter - this keeps the actual stored parameter itself
            # inside the same bounds the loss was computed with, so the
            # reported `loss_weight_*` metrics below always match what
            # `_train_step` actually used.
            self.loss_log_vars.data.clamp_(-5.0, 5.0)
        if self.use_target_for_bootstrap:
            self._update_target_network()
        self._update_closed_loop_error_ema(closed_loop_latent_errors)
        if closed_n:
            observed_model_error = float(closed_loop_loss.detach().item())
            decay = self.search_model_error_ema_decay
            self._model_error_ema = (
                decay * self._model_error_ema + (1.0 - decay) * observed_model_error
            )

        assert first_step_value_pred is not None and first_step_value_target is not None
        assert first_step_policy_error is not None
        value_error = (first_step_value_pred - first_step_value_target).abs()
        # `priority_policy_weight` (default `0.1`) - deliberately modest:
        # `value_error` and a discrete cross-entropy/continuous MSE
        # `policy_error` live on unrelated, uncalibrated scales, so this
        # is a heuristic blend, not a principled combination - just
        # enough to stop pure value-error priority from being blind to
        # "value is fine, but the policy target here is way off" samples,
        # without letting one badly-fit transition's policy loss swamp
        # the value signal the rest of this file's PER machinery was
        # actually tuned around.
        combined_priority = value_error + self.priority_policy_weight * first_step_policy_error
        fresh_priority = combined_priority.cpu().numpy()
        fresh_priority, learning_progress_bonus = self._apply_learning_progress_priority(
            fresh_priority, batch["episode_idx"], batch["timestep"],
        )
        self.buffer.update_priorities(batch["episode_idx"], batch["timestep"], fresh_priority)

        if self.rnd:
            # `dqn.py`'s own convention: the RND predictor trains off
            # replayed `next_obs`, not on-the-fly during collection - it
            # only ever *reads* fresh observations there (`.bonus()`,
            # under `no_grad`) to compute the novelty bonus itself.
            self.rnd.update_predictor(
                batch["next_obs"].reshape(-1, *self._obs_shape), mask=batch["mask"].reshape(-1),
            )
        stacked_value_targets = torch.stack(value_targets, dim=1).detach()
        sidecar_metrics = self._after_core_train_step(
            batch, hidden.detach(), obs0_pos, stacked_value_targets,
        )

        return {
            "reward_loss": float(reward_loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "consistency_loss": float(consistency_loss.item()),
            "closed_loop_loss": float(closed_loop_loss.item()),
            "path_consistency_loss": float(path_consistency_loss.item()),
            **{
                f"closed_loop_latent_error_h{horizon + 1}": float(error.item())
                for horizon, error in enumerate(closed_loop_latent_errors)
            },
            "policy_entropy": float(policy_entropy.item()),
            "mean_priority": float(fresh_priority.mean()) if fresh_priority.size else 0.0,
            "learning_progress_priority_bonus": float(learning_progress_bonus.mean()),
            "mean_is_weight": float(is_weight.mean().item()),
            "active_sample_fraction": float(batch["active_sample_fraction"]),
            "recent_sample_fraction": float(batch["recent_sample_fraction"]),
            "success_sample_fraction": float(batch["success_sample_fraction"]),
            "success_pool_size": float(batch["success_pool_size"]),
            "effective_closed_loop_horizon": float(effective_closed_loop_horizon),
            **{
                f"closed_loop_latent_error_ema_h{depth + 1}": float(value)
                for depth, value in enumerate(self._closed_loop_error_ema)
                if np.isfinite(value)
            },
            "total_loss": float(total_loss.item()),
            "learning_rate": current_lr,
            "priority_beta": current_priority_beta,
            "search_num_simulations": float(self._last_search_num_simulations),
            "search_base_simulations": float(self._last_search_base_simulations),
            "effective_sve_mix_mean": float(
                torch.stack(
                    [
                        value.expand(b) if value.ndim == 0 else value
                        for value in effective_sve_mixes
                    ],
                ).mean().item()
            ),
            "search_model_error_ema": float(self._model_error_ema),
            **(
                {
                    "loss_weight_reward": float(torch.exp(-self.loss_log_vars[0]).item()),
                    "loss_weight_value": float(torch.exp(-self.loss_log_vars[1]).item()),
                    "loss_weight_policy": float(torch.exp(-self.loss_log_vars[2]).item()),
                    # No `loss_weight_consistency` here anymore - see
                    # `__init__`'s own `self.loss_log_vars` comment: it
                    # keeps a fixed `consistency_loss_coef` weight, not an
                    # adaptive one, so there's no per-step value to report.
                }
                if self.adaptive_loss_weights
                else {}
            ),
            **(
                {"rnd_bonus_mean": float(self.rnd.bonus_mean), "rnd_predictor_loss": float(self.rnd.last_loss)}
                if self.rnd
                else {}
            ),
            **sidecar_metrics,
        }

    # ------------------------------------------------------------------
    # CustomAlgorithm contract
    # ------------------------------------------------------------------
    def learn(self, total_timesteps: int, callback: TrainingCallback) -> None:
        n_envs = num_envs_of(self.env)
        obs_list = vec_reset(self.env, seed=self.seed)
        obs_arr = obs_batch_to_array(obs_list, self._obs_space)
        ep_reward = np.zeros(n_envs, dtype=np.float64)
        ep_extrinsic = np.zeros(n_envs, dtype=np.float64)
        ep_intrinsic = np.zeros(n_envs, dtype=np.float64)
        ep_length = np.zeros(n_envs, dtype=np.int64)
        num_timesteps = 0
        # `_training_progress()`'s own denominator - every schedule (LR
        # decay, priority-beta annealing, search/entropy temperature
        # annealing) reads its progress fraction off `self._num_timesteps
        # / self._total_timesteps_estimate`, kept in sync with the local
        # `num_timesteps` below every real step so `_train_step`/
        # `_reanalyze` (methods, no `total_timesteps` of their own) can
        # read it too. A `.learn()` call resumed via `resume_from` at,
        # say, step 50k of a *new* 100k-step run still anneals cleanly:
        # this is this call's own total, not the checkpoint's original one.
        self._total_timesteps_estimate = max(1, total_timesteps)

        while num_timesteps < total_timesteps:
            env_actions: list[Any] = []
            action_flats: list[np.ndarray] = []
            policy_targets: list[np.ndarray] = []
            policy_target_valids: list[bool] = []
            search_budgets: list[int] = []
            model_expansion_counts: list[int] = []
            behavior_policies: list[np.ndarray] = []
            audit_types: list[str] = []
            behavior_metadata: list[dict[str, Any]] = []
            searched_root_caches: list[_TransformerCache] | None = None
            if num_timesteps < self.learning_starts:
                for _ in range(n_envs):
                    raw_action = self._action_space.sample()
                    policy_target = (
                        np.full(self.n_actions, 1.0 / self.n_actions, dtype=np.float32)
                        if self.discrete
                        # No real search yet during random warmup - the
                        # sampled action itself is the fallback "target"
                        # (`efficientzero.py`'s own choice, matching this
                        # file's `policy_target_dim = action_dim`).
                        else np.asarray(raw_action, dtype=np.float32).reshape(-1)
                    )
                    env_action = action_to_env(raw_action, self._action_space)
                    env_actions.append(env_action)
                    action_flats.append(obs_to_array(env_action, self._action_space))
                    policy_targets.append(policy_target)
                    policy_target_valids.append(True)
                    search_budgets.append(0)
                    model_expansion_counts.append(0)
                    behavior_policies.append(policy_target)
                    audit_types.append("none")
                    behavior_metadata.append({})
            else:
                # `self._lane_cache[lane]` is each lane's own persistent
                # real-history cache, carried over from the *previous*
                # real step (module docstring's "Persistent,
                # incrementally-extended per-lane root cache" section) -
                # no replay from `self.buffer` needed, unlike the
                # (removed) cold-start-every-step approach this file used
                # to take.
                results = self.search(obs_arr, self._lane_cache, deterministic=None)
                root_cache_candidates = [result.get("root_cache") for result in results]
                if all(cache is not None for cache in root_cache_candidates):
                    searched_root_caches = root_cache_candidates
                for result in results:
                    env_action = action_to_env(result["env_action"], self._action_space)
                    env_actions.append(env_action)
                    action_flats.append(obs_to_array(env_action, self._action_space))
                    policy_targets.append(result["policy_target"])
                    policy_target_valids.append(bool(result.get("policy_target_valid", True)))
                    search_budgets.append(int(result.get("search_budget", self._last_search_num_simulations)))
                    model_expansion_counts.append(int(result.get("model_expansions", self._last_search_num_simulations)))
                    behavior_policies.append(
                        np.asarray(result.get("behavior_policy", result["policy_target"]), dtype=np.float32),
                    )
                    audit_types.append(str(result.get("audit_type", "none")))
                    behavior_metadata.append(
                        {
                            "stage_index": int(result.get("stage_index", -1)),
                            "stage_id": str(result.get("stage_id", "none")),
                            "nominal_stage_budget": int(
                                result.get("nominal_stage_budget", search_budgets[-1])
                            ),
                            "requested_budget": int(
                                result.get("requested_budget", search_budgets[-1])
                            ),
                            "predicted_stage_index": int(
                                result.get("predicted_stage_index", -1)
                            ),
                            "predicted_stage_id": str(
                                result.get("predicted_stage_id", "none")
                            ),
                            "predicted_requested_budget": int(
                                result.get(
                                    "predicted_requested_budget",
                                    search_budgets[-1],
                                )
                            ),
                            "executed_search_expansions": int(
                                result.get(
                                    "executed_search_expansions",
                                    search_budgets[-1],
                                )
                            ),
                            "common_evaluator_expansions": int(
                                result.get("common_evaluator_expansions", 0)
                            ),
                            "total_model_calls": int(
                                result.get("total_model_calls", model_expansion_counts[-1])
                            ),
                            "termination_reason": str(
                                result.get("termination_reason", "budget_exhausted")
                            ),
                        }
                    )

            next_obs_list, rewards, terminated, truncated, infos = vec_step(self.env, env_actions)
            dones = terminated | truncated
            next_obs_arr = obs_batch_to_array(next_obs_list, self._obs_space)
            # `vec_step` uses SAME_STEP autoreset: `next_obs_arr` is the
            # fresh reset observation for a done lane (the state the next
            # policy call must see), while `final_obs` is the true next
            # observation of the terminal transition. Keep these roles
            # separate so replay/consistency never learn a cross-episode
            # `terminal -> reset` transition and the persistent cache
            # starts the new episode from a genuinely empty history.
            transition_next_obs_arr = next_obs_arr.copy()
            for lane in np.flatnonzero(dones):
                final_obs = infos[int(lane)].get("final_obs")
                if final_obs is not None:
                    transition_next_obs_arr[int(lane)] = obs_to_array(final_obs, self._obs_space)
            # RND intrinsic reward (`intrinsic_exploration`, off by
            # default - see `DEFAULT_HYPERPARAMS["intrinsic_exploration"]`'s
            # own docstring): `dqn.py`'s own per-lane convention, one
            # `.bonus()` call per lane's own fresh `next_obs` (each is
            # genuinely a new observation needing its own running-stats
            # update, not a batch that happens to share one). Everything
            # downstream - buffer storage, TD targets, `ep_reward` -
            # trains on `total_reward`; `ep_extrinsic`/`ep_intrinsic`
            # only exist to report the two parts separately once an
            # episode ends.
            extrinsic_reward = rewards.astype(np.float64)
            intrinsic_reward = (
                np.array(
                    [self.rnd_bonus_coef * self.rnd.bonus(transition_next_obs_arr[lane]) for lane in range(n_envs)],
                )
                if self.rnd
                else np.zeros(n_envs)
            )
            total_reward = extrinsic_reward + intrinsic_reward
            for lane in range(n_envs):
                self.buffer.add(
                    obs_arr[lane], action_flats[lane], float(total_reward[lane]), transition_next_obs_arr[lane],
                    policy_targets[lane], bool(dones[lane]), lane=lane,
                    policy_target_valid=policy_target_valids[lane],
                    search_budget=search_budgets[lane],
                    model_expansions=model_expansion_counts[lane],
                    behavior_policy=behavior_policies[lane],
                    audit_type=audit_types[lane],
                    **behavior_metadata[lane],
                )
            if num_timesteps >= self.learning_starts:
                done_lanes = np.flatnonzero(dones)
                for lane in done_lanes:
                    self._lane_cache[int(lane)] = None
                active_lanes = np.flatnonzero(~dones)
                if active_lanes.size:
                    active_actions = np.stack([action_flats[int(lane)] for lane in active_lanes])
                    if searched_root_caches is not None:
                        active_roots = [searched_root_caches[int(lane)] for lane in active_lanes]
                        advanced = self._append_actions_to_root_caches(active_roots, active_actions)
                    else:
                        active_caches = [self._lane_cache[int(lane)] for lane in active_lanes]
                        advanced = self._advance_lane_caches(
                            active_caches, obs_arr[active_lanes], active_actions,
                        )
                    for lane, cache in zip(active_lanes, advanced):
                        self._lane_cache[int(lane)] = cache
            obs_arr = next_obs_arr
            ep_reward += total_reward
            ep_extrinsic += extrinsic_reward
            ep_intrinsic += intrinsic_reward
            ep_length += 1
            prev_num_timesteps = num_timesteps
            num_timesteps += n_envs
            self._num_timesteps = num_timesteps

            if (
                num_timesteps >= self.learning_starts
                and self.buffer.num_replay_transitions >= max(1, self.learning_starts)
            ):
                effective_prev = max(prev_num_timesteps, self.learning_starts)
                boundaries_crossed = num_timesteps // self.train_freq - effective_prev // self.train_freq
                if boundaries_crossed > 0:
                    # One update group per vector iteration, matching the
                    # native algorithms' historical semantics. The jump
                    # by `n_envs` only decides whether a boundary was
                    # crossed; multiplying by its count here and scaling
                    # again in `__init__` was the 144-updates/iteration
                    # regression diagnosed on the real NetHack runs.
                    effective_train_steps = self._effective_train_steps_per_iter()
                    for _ in range(effective_train_steps):
                        self._last_metrics = self._train_step()
                    self._last_metrics["effective_train_steps"] = float(effective_train_steps)

                reanalyze_boundaries = num_timesteps // self.reanalyze_freq - effective_prev // self.reanalyze_freq
                if reanalyze_boundaries > 0:
                    reanalyze_metrics = self._reanalyze()
                    if reanalyze_metrics:
                        self._last_metrics = {**self._last_metrics, **reanalyze_metrics}

            any_done = False
            for lane in range(n_envs):
                if not dones[lane]:
                    continue
                any_done = True
                self._episode_reward_rolling.append(float(ep_reward[lane]))
                lane_metrics = (
                    {
                        **self._last_metrics,
                        "episode_extrinsic_reward": float(ep_extrinsic[lane]),
                        "episode_intrinsic_reward": float(ep_intrinsic[lane]),
                    }
                    if self.rnd
                    else self._last_metrics
                )
                lane_metrics = {**lane_metrics, **self._rolling_reward_metrics()}
                keep_going = callback.on_step(
                    num_timesteps, float(ep_reward[lane]), int(ep_length[lane]), lane_metrics,
                )
                ep_reward[lane], ep_extrinsic[lane], ep_intrinsic[lane], ep_length[lane] = 0.0, 0.0, 0.0, 0
                if not keep_going:
                    return
            if not any_done:
                keep_going = callback.on_step(
                    num_timesteps,
                    metrics={**self._last_metrics, **self._rolling_reward_metrics()},
                )
                if not keep_going:
                    return

    def predict(self, obs: Any, deterministic: bool = True, episode_start: bool = False) -> tuple[Any, Any]:
        if episode_start:
            self._eval_cache = None
        obs_arr = obs_to_array(obs, self._obs_space)
        # Same persistent-cache strategy `learn()` uses for its own real
        # steps (module docstring's "Persistent, incrementally-extended
        # per-lane root cache" section) - just one lane.
        results = self.search(obs_arr[None], [self._eval_cache], deterministic=np.array([deterministic]))
        env_action = action_to_env(results[0]["env_action"], self._action_space)
        action_flat = obs_to_array(env_action, self._action_space)
        self._eval_cache = self._append_actions_to_root_caches(
            [results[0]["root_cache"]], action_flat[None],
        )[0]
        return env_action, None

    def save(self, path: Path) -> None:
        torch.save(
            {
                "tokenizer_state_dict": self.tokenizer.state_dict(),
                "action_embed_state_dict": self.action_embed.state_dict(),
                "transformer_state_dict": self.transformer.state_dict(),
                "heads_state_dict": self.heads.state_dict(),
                "projector_state_dict": self.projector.state_dict(),
                "predictor_state_dict": self.predictor.state_dict(),
                "target_tokenizer_state_dict": self.target_tokenizer.state_dict(),
                "target_action_embed_state_dict": self.target_action_embed.state_dict(),
                "target_transformer_state_dict": self.target_transformer.state_dict(),
                "target_heads_state_dict": self.target_heads.state_dict(),
                "loss_log_vars": self.loss_log_vars.detach().cpu(),
                "hyperparams": self.hyperparams,
                "replay_buffer_state": self.buffer.state_dict(),
                **({"rnd_state": self.rnd.checkpoint_state()} if self.rnd else {}),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, env: gym.Env, device: str = "cpu") -> "NativeResearchImZero":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        algo = cls(env, payload.get("hyperparams", {}), None, device)
        algo.tokenizer.load_state_dict(payload["tokenizer_state_dict"])
        algo.action_embed.load_state_dict(payload["action_embed_state_dict"])
        algo.transformer.load_state_dict(payload["transformer_state_dict"])
        algo.heads.load_state_dict(payload["heads_state_dict"])
        if "projector_state_dict" in payload:
            algo.projector.load_state_dict(payload["projector_state_dict"])
            algo.predictor.load_state_dict(payload["predictor_state_dict"])
        if "target_tokenizer_state_dict" in payload:
            algo.target_tokenizer.load_state_dict(payload["target_tokenizer_state_dict"])
            algo.target_action_embed.load_state_dict(
                payload.get("target_action_embed_state_dict", payload["action_embed_state_dict"]),
            )
            algo.target_transformer.load_state_dict(payload["target_transformer_state_dict"])
            algo.target_heads.load_state_dict(payload["target_heads_state_dict"])
        if algo.rnd and payload.get("rnd_state"):
            algo.rnd.load_checkpoint_state(payload["rnd_state"])
        if "replay_buffer_state" in payload:
            algo.buffer.load_state_dict(payload["replay_buffer_state"])
        if "loss_log_vars" in payload:
            saved = payload["loss_log_vars"]
            if saved.shape == algo.loss_log_vars.shape:
                with torch.no_grad():
                    algo.loss_log_vars.copy_(saved.to(device))
            # else: shape changed (old checkpoints saved 4 - [reward,
            # value, policy, consistency] - this file now only adaptively
            # weights 3, see `__init__`'s own comment) - starting the 3
            # kept entries back at `0.0` (this file's own "equivalent to
            # static coefs" starting point) rather than crashing on load.
        return algo
