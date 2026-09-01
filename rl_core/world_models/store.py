"""CRUD storage for user-designed World Model specs (the World Model
Builder page — `/world-models`) plus the lookup helper every algorithm
that can use one (`dreamer`/`mbpo`/`pets`/`world_models_ha`) calls to
resolve `algorithm.world_model_id` into an actual spec — mirrors
`rl_core/netbuilder_store.py`'s relationship to `network_spec_id` exactly.

Unlike Network Builder specs (pure data, an `nn.Module` only ever gets
built at train time), a World Model spec can additionally carry *trained
weights*: `<slug>.json` holds the spec itself (name/type/config/metadata),
and `<slug>/model.pt` — written either by a standalone `kind: "world_model"`
run being promoted here (`attach_checkpoint`) or in-place while training
one of the four world-model algorithms with `world_model_id` set — holds
whatever `spec.build_world_model(...)` produced, already trained.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rl_core.paths import CUSTOM_WORLD_MODELS_DIR


def _spec_path(slug: str) -> Path:
    return CUSTOM_WORLD_MODELS_DIR / f"{slug}.json"


def _dir_for(slug: str) -> Path:
    return CUSTOM_WORLD_MODELS_DIR / slug


def checkpoint_path(slug: str) -> Path:
    path = _dir_for(slug) / "model.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def has_checkpoint(slug: str) -> bool:
    return checkpoint_path(slug).exists()


def list_slugs() -> list[str]:
    if not CUSTOM_WORLD_MODELS_DIR.exists():
        return []
    return sorted(p.stem for p in CUSTOM_WORLD_MODELS_DIR.glob("*.json"))


def load(slug: str) -> dict[str, Any]:
    path = _spec_path(slug)
    if not path.exists():
        raise FileNotFoundError(f"World Model «{slug}» не найден")
    return json.loads(path.read_text())


def save(slug: str, doc: dict[str, Any]) -> None:
    _spec_path(slug).write_text(json.dumps(doc, indent=2, ensure_ascii=False))


def delete(slug: str) -> None:
    path = _spec_path(slug)
    if path.exists():
        path.unlink()
    ckpt_dir = _dir_for(slug)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def meta(slug: str) -> dict[str, Any]:
    doc = load(slug)
    return {
        "id": slug,
        "slug": slug,
        "name": doc.get("name") or slug,
        "description": doc.get("description", ""),
        "type": doc.get("type", "rssm"),
        "config": doc.get("config", {}),
        "environment_id": doc.get("environment_id"),
        "trained": has_checkpoint(slug),
        "trained_at": doc.get("trained_at"),
        "source_run_id": doc.get("source_run_id"),
    }


def list_meta() -> list[dict[str, Any]]:
    out = []
    for slug in list_slugs():
        try:
            out.append(meta(slug))
        except Exception as exc:  # noqa: BLE001 - a broken file shouldn't hide the rest of the list
            out.append({
                "id": slug, "slug": slug, "name": slug, "description": "", "type": "rssm", "config": {},
                "trained": False, "broken": True, "error": str(exc),
            })
    return out


def resolve_world_model_spec(config: dict[str, Any]) -> dict[str, Any] | None:
    """Every one of the four world-model algorithms calls this once at
    startup — looks up either `algorithm.world_model_spec` (an inline,
    unsaved spec — e.g. the Designer's quick type+config picker, never
    written to `CUSTOM_WORLD_MODELS_DIR`) or `algorithm.world_model_id`
    (a saved spec's slug — picks up its trained checkpoint too, if any)
    and returns `{"type", "config", "slug"?, "checkpoint_path"?}`, or
    `None` if the run doesn't reference either. Inline takes priority, but
    the Designer only ever sends one of the two at a time — same
    convention as `netbuilder_store.resolve_network_spec`."""
    algorithm_cfg = config.get("algorithm", {})
    inline = algorithm_cfg.get("world_model_spec")
    if inline:
        return {"type": inline["type"], "config": inline.get("config", {})}
    slug = algorithm_cfg.get("world_model_id")
    if not slug:
        return None
    doc = load(slug)
    resolved: dict[str, Any] = {"type": doc.get("type", "rssm"), "config": doc.get("config", {}), "slug": slug}
    if has_checkpoint(slug):
        resolved["checkpoint_path"] = str(checkpoint_path(slug))
    return resolved


def attach_checkpoint(slug: str, source_path: Path, source_run_id: str | None = None) -> None:
    """Copies a finished run's trained weights (`model.pt` — see
    `trainer.py` for standalone `kind: "world_model"` runs, or any of the
    four algorithms' own `save()` for a jointly-trained one) into this
    spec's own directory, so future `world_model_id` references pick up
    the trained weights instead of building a fresh, untrained model of
    the same shape. Mirrors `backend/routes/models.py::promote_run`'s
    "copy the checkpoint, remember where it came from" shape."""
    shutil.copy2(source_path, checkpoint_path(slug))
    doc = load(slug)
    doc["trained_at"] = datetime.now(timezone.utc).isoformat()
    doc["source_run_id"] = source_run_id
    save(slug, doc)


def write_world_model_snapshot(run_dir: Path, world_model_spec: dict[str, Any] | None) -> None:
    """Counterpart to `netbuilder_store.write_network_snapshot` — the
    *exact* spec a run actually used (resolved slug's config, or an inline
    one), written once at run start, independent of whatever happens to
    the saved catalog entry afterwards."""
    doc = {
        "type": world_model_spec.get("type") if world_model_spec else None,
        "config": world_model_spec.get("config") if world_model_spec else None,
        "world_model_id": world_model_spec.get("slug") if world_model_spec else None,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "world_model.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))


def read_world_model_snapshot(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "world_model.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
