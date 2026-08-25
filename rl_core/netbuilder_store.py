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
from pathlib import Path
from typing import Any

from rl_core.paths import CUSTOM_NETWORKS_DIR


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
    custom AlphaZero) calls this once at startup — looks up
    `algorithm.network_spec_id` (set by the Designer/AlgorithmNode when a
    saved architecture is picked instead of the algorithm's default net)
    and returns just the layer spec, or None if the experiment doesn't
    reference one at all."""
    slug = config.get("algorithm", {}).get("network_spec_id")
    if not slug:
        return None
    return load(slug)["spec"]
