"""Generic, JSON-spec-driven network builder behind the visual Network
Architecture Builder (`/network-builder`) — turns a small declarative spec
(a linear "trunk" of layers + one or more named "heads") into a real
`torch.nn.Module`, with shape inference along the way so the UI can show
per-layer output shapes and catch mistakes (e.g. a `Linear` right after a
`Conv2d` with no `Flatten` in between) before anything actually trains.

Kept deliberately generic/role-agnostic at the pure-shape level — the
family-specific adapters below (`SpecActorCriticNet`, `SpecQNetwork`,
`SpecAlphaZeroNet`) are what map "a head named 'value'" etc. onto what each
existing algorithm family (native PPO/A2C, native DQN, AlphaZero) actually
expects, so none of those algorithms' own learn()/predict() code needs to
change to accept a hand-designed net — see rl_core/algorithms/native/ppo.py,
a2c.py, dqn.py and rl_core/alphazero/base.py for the call sites.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkSpecError(ValueError):
    """Raised for an invalid/incomplete spec — always caught by the preview
    endpoint and the trainers below and reported as a clean message, never
    a raw traceback, since this is meant to be editable by someone who
    isn't necessarily reading Python stack traces."""


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}

LAYER_TYPES = ["linear", "conv2d", "maxpool2d", "flatten", "activation", "dropout", "batchnorm"]
FAMILIES = ["actor_critic", "q_network", "dueling_q", "alphazero"]

# Which head names each family requires, and what their default final-layer
# output size is (None = must come from the caller, e.g. env action count).
FAMILY_HEADS: dict[str, dict[str, int | None]] = {
    "actor_critic": {"action": None, "value": 1},
    "q_network": {"q": None},
    "dueling_q": {"advantage": None, "value": 1},
    "alphazero": {"policy": None, "value": 1},
}


def _build_layer(layer: dict[str, Any], shape: list[int]) -> tuple[nn.Module, list[int]]:
    """`shape` is the running shape *without* the batch dim — either
    `[C, H, W]` (conv territory) or `[N]` (flat/linear territory)."""
    kind = layer.get("type")
    if kind == "linear":
        if len(shape) != 1:
            raise NetworkSpecError(f"Linear ожидает плоский вход, а пришло {shape} — добавь Flatten перед этим слоем")
        out_features = int(layer.get("out_features") or 0)
        if out_features <= 0:
            raise NetworkSpecError("У Linear должно быть указано положительное out_features")
        return nn.Linear(shape[0], out_features), [out_features]
    if kind == "conv2d":
        if len(shape) != 3:
            raise NetworkSpecError(f"Conv2d ожидает вход [C,H,W], а пришло {shape}")
        c, h, w = shape
        out_channels = int(layer.get("out_channels") or 0)
        if out_channels <= 0:
            raise NetworkSpecError("У Conv2d должно быть указано положительное out_channels")
        kernel_size = int(layer.get("kernel_size", 3))
        stride = int(layer.get("stride", 1))
        padding = int(layer.get("padding", 0))
        if h + 2 * padding < kernel_size or w + 2 * padding < kernel_size:
            raise NetworkSpecError(f"Conv2d(k={kernel_size}) слишком большой для входа {h}×{w}")
        out_h = (h + 2 * padding - kernel_size) // stride + 1
        out_w = (w + 2 * padding - kernel_size) // stride + 1
        return nn.Conv2d(c, out_channels, kernel_size, stride=stride, padding=padding), [out_channels, out_h, out_w]
    if kind == "maxpool2d":
        if len(shape) != 3:
            raise NetworkSpecError(f"MaxPool2d ожидает вход [C,H,W], а пришло {shape}")
        c, h, w = shape
        kernel_size = int(layer.get("kernel_size", 2))
        stride = int(layer.get("stride", kernel_size))
        out_h = (h - kernel_size) // stride + 1
        out_w = (w - kernel_size) // stride + 1
        if out_h < 1 or out_w < 1:
            raise NetworkSpecError(f"MaxPool2d(k={kernel_size}) слишком большой для входа {h}×{w}")
        return nn.MaxPool2d(kernel_size, stride=stride), [c, out_h, out_w]
    if kind == "flatten":
        n = 1
        for d in shape:
            n *= d
        return nn.Flatten(), [n]
    if kind == "activation":
        fn = layer.get("fn", "relu")
        cls = _ACTIVATIONS.get(fn)
        if cls is None:
            raise NetworkSpecError(f"Неизвестная активация: {fn}")
        return cls(), shape
    if kind == "dropout":
        return nn.Dropout(float(layer.get("p", 0.1))), shape
    if kind == "batchnorm":
        if len(shape) == 1:
            return nn.BatchNorm1d(shape[0]), shape
        if len(shape) == 3:
            return nn.BatchNorm2d(shape[0]), shape
        raise NetworkSpecError(f"BatchNorm не поддерживает форму {shape}")
    raise NetworkSpecError(f"Неизвестный тип слоя: {kind!r}")


@dataclass
class BuiltHead:
    name: str
    module: nn.Module
    layer_shapes: list[list[int]] = field(default_factory=list)

    @property
    def out_shape(self) -> list[int]:
        return self.layer_shapes[-1] if self.layer_shapes else []


@dataclass
class BuiltNetwork:
    trunk: nn.Module
    trunk_shapes: list[list[int]]
    heads: dict[str, BuiltHead]
    input_shape: list[int]


def build_network(
    spec: dict[str, Any],
    input_shape: list[int],
    head_out_features: dict[str, int] | None = None,
) -> BuiltNetwork:
    """`head_out_features` fills in each head's final layer size when the
    spec left it blank (0/None/omitted) — e.g. `{"action": n_actions}` for
    an env-determined head, so the Builder UI never needs to ask the user
    "how many actions does your env have" by hand."""
    head_out_features = head_out_features or {}
    trunk_layers: list[nn.Module] = []
    trunk_shapes: list[list[int]] = []
    shape = list(input_shape)
    for layer in spec.get("trunk", []):
        module, shape = _build_layer(layer, shape)
        trunk_layers.append(module)
        trunk_shapes.append(list(shape))
    trunk = nn.Sequential(*trunk_layers)
    trunk_out_shape = shape

    heads: dict[str, BuiltHead] = {}
    for head_spec in spec.get("heads", []):
        name = head_spec.get("name")
        if not name:
            raise NetworkSpecError("У каждой головы должно быть имя")
        layers = [dict(l) for l in head_spec.get("layers", [])]
        if not layers:
            raise NetworkSpecError(f"У головы «{name}» нет слоёв")
        if layers[-1].get("out_features") in (None, 0) and name in head_out_features:
            layers[-1]["out_features"] = head_out_features[name]

        head_shape = list(trunk_out_shape)
        head_modules: list[nn.Module] = []
        layer_shapes: list[list[int]] = []
        for layer in layers:
            module, head_shape = _build_layer(layer, head_shape)
            head_modules.append(module)
            layer_shapes.append(list(head_shape))
        heads[name] = BuiltHead(name=name, module=nn.Sequential(*head_modules), layer_shapes=layer_shapes)

    if not heads:
        raise NetworkSpecError("Нужна хотя бы одна голова (например «action» или «value»)")

    return BuiltNetwork(trunk=trunk, trunk_shapes=trunk_shapes, heads=heads, input_shape=list(input_shape))


def require_heads(built: BuiltNetwork, family: str) -> None:
    required = FAMILY_HEADS.get(family, {})
    missing = [name for name in required if name not in built.heads]
    if missing:
        raise NetworkSpecError(
            f"Для типа сети «{family}» не хватает голов: {', '.join(missing)} "
            f"(нужны: {', '.join(required)})"
        )


class SpecNet(nn.Module):
    """Thin nn.Module wrapper around a `BuiltNetwork` — trunk, then every
    head applied to the trunk's output in parallel. `forward` returns a
    dict keyed by head name; family adapters below unwrap it into whatever
    plain-tensor/tuple shape their algorithm actually expects."""

    def __init__(self, built: BuiltNetwork) -> None:
        super().__init__()
        self.trunk = built.trunk
        self.head_names = list(built.heads.keys())
        for name, head in built.heads.items():
            self.add_module(f"head_{name}", head.module)
        self.input_shape = built.input_shape

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.trunk(x)
        return {name: getattr(self, f"head_{name}")(feat) for name in self.head_names}


def _obs_input_shape(observation_space: Any, is_image: bool) -> list[int]:
    if is_image:
        h, w, c = observation_space.shape
        return [c, h, w]
    from rl_core.algorithms.native.preprocessing import obs_flat_dim

    return [obs_flat_dim(observation_space)]


def _preprocess_obs(obs: torch.Tensor, is_image: bool) -> torch.Tensor:
    if is_image:
        return obs.permute(0, 3, 1, 2).float() / 255.0
    return obs.float()


class SpecActorCriticNet(nn.Module):
    """Drop-in replacement for `rl_core.algorithms.native.networks.ActorCriticNet`,
    built from a user-designed spec instead of the fixed 64x64 MLP — same
    public surface (`forward`/`distribution`/`deterministic_action`/
    `discrete`), so NativePPO/NativeA2C need zero changes to accept it.
    Requires an "action" head (discrete logits, or the Gaussian mean for
    continuous actions) and a "value" head (scalar)."""

    def __init__(self, observation_space: Any, action_space: Any, spec: dict[str, Any]) -> None:
        super().__init__()
        import gymnasium as gym
        from rl_core.algorithms.native.preprocessing import is_image_space

        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        action_dim = int(action_space.n) if self.discrete else int(np.prod(action_space.shape))
        self._is_image = is_image_space(observation_space)
        input_shape = _obs_input_shape(observation_space, self._is_image)

        built = build_network(spec, input_shape, head_out_features={"action": action_dim, "value": 1})
        require_heads(built, "actor_critic")
        self.net = SpecNet(built)
        if not self.discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(_preprocess_obs(obs, self._is_image))
        value = out["value"].squeeze(-1)
        return out["action"], value

    def distribution(self, obs: torch.Tensor):
        params, value = self.forward(obs)
        if self.discrete:
            return torch.distributions.Categorical(logits=params), value
        std = torch.exp(self.log_std).clamp(min=1e-6)
        return torch.distributions.Normal(params, std), value

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        params, _ = self.forward(obs)
        return torch.argmax(params, dim=-1) if self.discrete else params

    @staticmethod
    def log_prob(dist, action: torch.Tensor) -> torch.Tensor:
        lp = dist.log_prob(action)
        return lp if lp.dim() == 1 else lp.sum(dim=-1)

    @staticmethod
    def entropy(dist) -> torch.Tensor:
        ent = dist.entropy()
        return ent if ent.dim() == 1 else ent.sum(dim=-1)


class SpecQNetwork(nn.Module):
    """Drop-in replacement for `rl_core.algorithms.native.networks.QNetwork`.
    Requires a "q" head sized to the env's action count."""

    def __init__(self, observation_space: Any, n_actions: int, spec: dict[str, Any]) -> None:
        super().__init__()
        from rl_core.algorithms.native.preprocessing import is_image_space

        self._is_image = is_image_space(observation_space)
        input_shape = _obs_input_shape(observation_space, self._is_image)
        built = build_network(spec, input_shape, head_out_features={"q": n_actions})
        require_heads(built, "q_network")
        self.net = SpecNet(built)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(_preprocess_obs(obs, self._is_image))["q"]


class SpecDuelingQNetwork(nn.Module):
    """Drop-in replacement for
    `rl_core.algorithms.native.networks.DuelingQNetwork` (used by Rainbow
    DQN). Requires a "value" head (scalar) and an "advantage" head (sized to
    the env's action count); combined the same way as the fixed-architecture
    version: `Q = V + (A - mean(A))`."""

    def __init__(self, observation_space: Any, n_actions: int, spec: dict[str, Any]) -> None:
        super().__init__()
        from rl_core.algorithms.native.preprocessing import is_image_space

        self._is_image = is_image_space(observation_space)
        input_shape = _obs_input_shape(observation_space, self._is_image)
        built = build_network(spec, input_shape, head_out_features={"advantage": n_actions, "value": 1})
        require_heads(built, "dueling_q")
        self.net = SpecNet(built)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(_preprocess_obs(obs, self._is_image))
        value, advantage = out["value"], out["advantage"]
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


class SpecAlphaZeroNet(nn.Module):
    """Drop-in replacement for `rl_core.alphazero.network.AlphaZeroNet`.
    Input is the game's own `(3, rows, cols)` board encoding (already
    normalized, unlike Gym pixels) — no preprocessing needed. Requires a
    "policy" head (`action_size` logits) and a "value" head (1, tanh'd)."""

    def __init__(self, rows: int, cols: int, action_size: int, spec: dict[str, Any]) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.action_size = action_size
        built = build_network(spec, [3, rows, cols], head_out_features={"policy": action_size, "value": 1})
        require_heads(built, "alphazero")
        self.net = SpecNet(built)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out["policy"], torch.tanh(out["value"]).squeeze(-1)

    @torch.no_grad()
    def predict(self, encoded_state, device: str = "cpu") -> tuple[Any, float]:
        self.eval()
        x = torch.as_tensor(encoded_state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, float(value.item())


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _build_layers_seq(layers: list[dict[str, Any]], shape: list[int]) -> tuple[list[list[int]], str | None, int | None]:
    """Builds layers one at a time (unlike `build_network`, which aborts the
    whole spec on the first bad layer) so the Builder UI can still show
    every shape computed *before* the broken one, plus exactly which layer
    index failed, instead of losing all progress on one typo."""
    shapes: list[list[int]] = []
    for i, layer in enumerate(layers):
        try:
            _, shape = _build_layer(layer, shape)
        except NetworkSpecError as exc:
            return shapes, str(exc), i
        shapes.append(list(shape))
    return shapes, None, None


def preview_network(
    spec: dict[str, Any],
    input_shape: list[int],
    family: str,
    head_out_features: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Read-only "what would this build to" check for the Network Builder
    UI — walks the trunk then each head layer-by-layer (see
    `_build_layers_seq`) so a mistake in one layer/head doesn't hide the
    shapes of everything that built fine around it."""
    head_out_features = head_out_features or {}
    trunk_shapes, trunk_error, trunk_error_index = _build_layers_seq(spec.get("trunk", []), list(input_shape))
    trunk_out_shape = trunk_shapes[-1] if trunk_shapes else list(input_shape)

    heads: dict[str, Any] = {}
    overall_error = f"Ствол: {trunk_error}" if trunk_error else None
    if trunk_error is None:
        for head_spec in spec.get("heads", []):
            name = head_spec.get("name", "?")
            layers = [dict(l) for l in head_spec.get("layers", [])]
            if layers and layers[-1].get("out_features") in (None, 0) and name in head_out_features:
                layers[-1] = {**layers[-1], "out_features": head_out_features[name]}
            layer_shapes, head_error, head_error_index = _build_layers_seq(layers, list(trunk_out_shape))
            heads[name] = {
                "layer_shapes": layer_shapes,
                "out_shape": layer_shapes[-1] if layer_shapes else [],
                "error": head_error,
                "error_index": head_error_index,
            }
            if head_error and overall_error is None:
                overall_error = f"Голова «{name}»: {head_error}"

    missing = [name for name in FAMILY_HEADS.get(family, {}) if name not in heads]
    if missing and overall_error is None:
        overall_error = f"Не хватает голов: {', '.join(missing)}"

    total_params = None
    if overall_error is None:
        try:
            built = build_network(spec, input_shape, head_out_features=head_out_features)
            total_params = count_params(built.trunk) + sum(count_params(h.module) for h in built.heads.values())
        except NetworkSpecError:
            pass

    return {
        "ok": overall_error is None,
        "error": overall_error,
        "input_shape": list(input_shape),
        "trunk": [{"shape": s} for s in trunk_shapes],
        "trunk_error": trunk_error,
        "trunk_error_index": trunk_error_index,
        "heads": heads,
        "total_params": total_params,
    }
