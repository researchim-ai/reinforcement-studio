"""CRUD storage for user-designed network architectures (JSON specs
authored on the Network Builder page — `/network-builder`), plus the
lookup helper every runner uses to resolve an experiment's
`algorithm.network_spec_id` into the actual layer spec.

Unlike code plugins (rl_core/plugins/loader.py), these are pure data — no
code execution happens here at all; the real `nn.Module` only gets built at
train time by `rl_core/netbuilder.py`. That also means there's nothing to
"dry-run validate" in the sandboxed-code sense; the Builder UI instead gets
instant shape-inference feedback from `netbuilder.preview_network`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rl_core.paths import CUSTOM_NETWORKS_DIR

# Same family mapping as the frontend's `requiredFamilyFor`/`networkFamilyFor`
# (src/lib/networkBuilder.ts) — kept here too since `write_network_snapshot`
# needs it without importing anything from `rl_core.netbuilder` (which would
# pull in torch just to label a run's network.json).
_ALGO_FAMILY: dict[str, str] = {
    "dqn": "q_network",
    "rainbow_dqn": "dueling_q",
    "ppo": "actor_critic",
    "a2c": "actor_critic",
    "efficientzero": "efficientzero",
    "unizero": "unizero",
    "researchimzero": "researchimzero",
    "latentimzero": "latentimzero",
}


def _path_for(slug: str) -> Path:
    return CUSTOM_NETWORKS_DIR / f"{slug}.json"


def list_slugs() -> list[str]:
    if not CUSTOM_NETWORKS_DIR.exists():
        return []
    return sorted(p.stem for p in CUSTOM_NETWORKS_DIR.glob("*.json"))


def load(slug: str) -> dict[str, Any]:
    path = _path_for(slug)
    if not path.exists():
        raise FileNotFoundError(f"Архитектура сети «{slug}» не найдена")
    return json.loads(path.read_text())


def save(slug: str, doc: dict[str, Any]) -> None:
    _path_for(slug).write_text(json.dumps(doc, indent=2, ensure_ascii=False))


def delete(slug: str) -> None:
    path = _path_for(slug)
    if path.exists():
        path.unlink()


def meta(slug: str) -> dict[str, Any]:
    doc = load(slug)
    return {
        "id": slug,
        "slug": slug,
        "name": doc.get("name") or slug,
        "description": doc.get("description", ""),
        "family": doc.get("family", "actor_critic"),
        "format": doc.get("spec", {}).get("format", "trunk_heads_v1"),
    }


def list_meta() -> list[dict[str, Any]]:
    out = []
    for slug in list_slugs():
        try:
            out.append(meta(slug))
        except Exception as exc:  # noqa: BLE001 - a broken file shouldn't hide the rest of the list
            out.append({"id": slug, "slug": slug, "name": slug, "description": "", "family": "actor_critic",
                         "broken": True, "error": str(exc)})
    return out


def resolve_network_spec(config: dict[str, Any]) -> dict[str, Any] | None:
    """Every runner (native gym, custom gym via CustomAlgorithm, built-in +
    custom AlphaZero) calls this once at startup — looks up either
    `algorithm.network_spec` (an inline, unsaved spec — e.g. the
    Designer's "quick layer editor", which builds one on the fly without
    ever writing it to `CUSTOM_NETWORKS_DIR`) or `algorithm.network_spec_id`
    (set when a saved architecture from the Network Builder page is picked
    instead) and returns just the layer spec, or None if the experiment
    doesn't reference either at all. Inline takes priority, but the
    Designer only ever sends one of the two at a time."""
    algorithm_cfg = config.get("algorithm", {})
    inline_spec = algorithm_cfg.get("network_spec")
    if inline_spec:
        return inline_spec
    slug = algorithm_cfg.get("network_spec_id")
    if not slug:
        return None
    return load(slug)["spec"]


def family_for_algorithm(algo_id: str | None, kind: str) -> str | None:
    """Which `NetworkSpec` family (if any) this algorithm's architecture
    belongs to — `None` for algorithms that don't support a hand-designed
    net at all (SAC/DDPG/TD3/ES, custom Gym plugins) so callers know there's
    nothing meaningful to snapshot."""
    if kind == "alphazero":
        return "alphazero"
    return _ALGO_FAMILY.get((algo_id or "").lower())


def write_network_snapshot(run_dir: Path, config: dict[str, Any], network_spec: dict[str, Any] | None) -> None:
    """Writes `network.json` next to `config.json` at the start of every
    run — the *exact* architecture actually used, with metadata about where
    it came from. This matters because `config.json` alone doesn't always
    tell the full story: a `network_spec_id` is just a slug, and that saved
    file can be renamed/edited/deleted later; an inline `network_spec` (the
    Designer's quick layer editor) never touches disk anywhere else at all.
    Writing a fully-resolved copy here means every past run's network stays
    inspectable and reusable (re-`PUT`-able to `/networks/{slug}`, see
    `backend/routes/networks.py`) independent of what happens to the
    catalog afterwards.

    Always written, even when the algorithm used its default hardcoded
    architecture (`network_spec` is `None`) — `family` alone is still
    useful context, and a consistently-present file is simpler for the UI
    than having to special-case "old run, nothing to show" vs "new run,
    genuinely used the default"."""
    algorithm_cfg = config.get("algorithm", {})
    algo_id = algorithm_cfg.get("id")
    family = family_for_algorithm(algo_id, config.get("kind", "gym"))
    if algorithm_cfg.get("network_spec_id"):
        source = "catalog"
    elif algorithm_cfg.get("network_spec"):
        source = "inline"
    else:
        source = "default"
    doc = {
        "family": family,
        "format": (
            network_spec.get("format", "trunk_heads_v1")
            if network_spec is not None
            else (
                "composite_v1"
                if family in {"efficientzero", "unizero", "researchimzero", "latentimzero"}
                else "trunk_heads_v1"
            )
        ),
        "spec": network_spec,
        "source": source,
        "network_spec_id": algorithm_cfg.get("network_spec_id"),
        "algorithm_id": algo_id,
        "environment_id": config.get("environment", {}).get("id"),
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "network.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))


def read_network_snapshot(run_dir: Path) -> dict[str, Any] | None:
    """Counterpart to `write_network_snapshot` — `None` for runs that
    predate this feature (no `network.json` at all), never for runs that
    simply used the default architecture (those still get a file, just
    with `spec: None`)."""
    path = run_dir / "network.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
