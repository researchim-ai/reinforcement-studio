"""Storage, preview and legacy-backfill coverage for composite networks."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.routes import networks as network_routes
from backend.routes import training as training_routes
from rl_core import netbuilder_store
from rl_core.composite_netbuilder import default_composite_spec


def _config(family: str, *, network_spec=None, network_spec_id=None) -> dict:
    return {
        "kind": "gym",
        "environment": {"id": "CartPole-v1", "wrappers": []},
        "algorithm": {
            "id": family,
            "hyperparams": {
                "embed_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "proj_dim": 8,
            },
            "network_spec": network_spec,
            "network_spec_id": network_spec_id,
        },
        "training": {"total_timesteps": 10},
    }


def test_store_resolves_inline_before_catalog_and_snapshots_schema(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(netbuilder_store, "CUSTOM_NETWORKS_DIR", tmp_path / "catalog")
    netbuilder_store.CUSTOM_NETWORKS_DIR.mkdir()
    catalog_spec = default_composite_spec("unizero")
    inline_spec = default_composite_spec("researchimzero")
    netbuilder_store.save("catalog", {
        "name": "Catalog",
        "description": "",
        "family": "unizero",
        "spec": catalog_spec,
    })

    assert netbuilder_store.resolve_network_spec(_config(
        "researchimzero", network_spec=inline_spec, network_spec_id="catalog",
    )) == inline_spec
    assert netbuilder_store.resolve_network_spec(_config(
        "unizero", network_spec_id="catalog",
    )) == catalog_spec

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    netbuilder_store.write_network_snapshot(run_dir, _config("researchimzero", network_spec=inline_spec), inline_spec)
    snapshot = netbuilder_store.read_network_snapshot(run_dir)
    assert snapshot["format"] == "composite_v1"
    assert snapshot["family"] == "researchimzero"
    assert snapshot["source"] == "inline"
    assert snapshot["spec"] == inline_spec


def test_composite_preview_returns_real_components_and_fixed_outputs() -> None:
    request = network_routes.PreviewRequest(
        family="unizero",
        spec=default_composite_spec("unizero"),
        environment_id="CartPole-v1",
        hyperparams={"value_support_size": 10},
    )
    result = asyncio.run(network_routes.preview(request))
    assert result["ok"] is True
    assert {component["name"] for component in result["components"]} >= {
        "tokenizer", "action_embed", "transformer", "heads", "target_transformer",
    }
    assert result["fixed_outputs"]["policy"] == [2]
    assert result["fixed_outputs"]["value"] == [21]


def test_latentimzero_preview_exposes_research_core_and_probe_outputs() -> None:
    request = network_routes.PreviewRequest(
        family="latentimzero",
        spec=default_composite_spec("latentimzero"),
        environment_id="CartPole-v1",
        hyperparams={"value_support_size": 10},
    )
    result = asyncio.run(network_routes.preview(request))
    assert result["ok"] is True
    assert {component["name"] for component in result["components"]} >= {
        "tokenizer", "action_embed", "transformer", "heads", "uncertainty_probe",
    }
    assert result["fixed_outputs"]["policy"] == [2]
    assert result["fixed_outputs"]["value"] == [21]
    assert result["fixed_outputs"]["reward"] == [21]
    assert result["fixed_outputs"]["next_token"] == ["embed_dim"]
    assert result["fixed_outputs"]["uncertainty_reward_heads"] == ["uncertainty_members"]
    assert result["fixed_outputs"]["uncertainty_value_heads"] == ["uncertainty_members"]


def test_old_run_without_snapshots_gets_materialized_composite_spec(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "legacy"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(_config("researchimzero")))
    monkeypatch.setattr(training_routes, "RUNS_DIR", runs_dir)

    snapshot = asyncio.run(training_routes.get_run_network("legacy"))
    assert snapshot["family"] == "researchimzero"
    assert snapshot["format"] == "composite_v1"
    assert snapshot["source"] == "default"
    assert snapshot["spec"]["dimensions"]["embed_dim"] == 16
    assert (run_dir / "network.json").is_file()

    architecture = asyncio.run(training_routes.get_run_architecture("legacy"))
    assert architecture["components"]
    assert architecture["total_params"] > 0
    assert (run_dir / "architecture.json").is_file()


def test_start_preflight_rejects_encoder_incompatible_with_environment() -> None:
    spec = default_composite_spec("unizero")
    spec["encoder"] = {
        "kind": "image_cnn",
        "layers": [
            {"type": "conv2d", "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
            {"type": "flatten"},
        ],
    }
    request = training_routes.StartRunRequest(
        kind="gym",
        environment={"id": "CartPole-v1", "wrappers": []},
        algorithm={
            "id": "unizero",
            "hyperparams": {},
            "network_spec": spec,
            "network_spec_id": None,
        },
        training={"total_timesteps": 10},
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(training_routes.start_run(request))
    assert error.value.status_code == 400
    assert "image_cnn" in str(error.value.detail)
