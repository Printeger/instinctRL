import hashlib
import json
import os
from pathlib import Path

import pytest

from training_workloads.unified_h1.asset_closure import validate_asset_closure, validate_asset_environment


ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "training_workloads/unified_h1/candidate_config.json").read_text())
ASSET_ROOT = Path(CONFIG["execution"]["isaacsim_asset_root"])
MANIFEST = ASSET_ROOT / "ASSET_MANIFEST.json"
MANIFEST_SHA256 = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def test_exact_local_ground_closure_opens_without_network():
    result = validate_asset_closure(ASSET_ROOT, MANIFEST_SHA256)
    assert len(result["opened_files"]) == 4
    assert result["network_fallback"] is False
    assert result["material"]["path"].endswith("/OmniPBR.mdl")


def test_environment_binding_is_exact(monkeypatch):
    execution = {"isaacsim_asset_root_env":"ISAACSIM_ASSET_ROOT",
                 "isaacsim_asset_root":str(ASSET_ROOT),
                 "isaacsim_asset_manifest_sha256":MANIFEST_SHA256}
    monkeypatch.delenv("ISAACSIM_ASSET_ROOT", raising=False)
    with pytest.raises(PermissionError):
        validate_asset_environment(execution)
    monkeypatch.setenv("ISAACSIM_ASSET_ROOT", str(ASSET_ROOT))
    assert validate_asset_environment(execution)["network_fallback"] is False


def test_manifest_and_file_drift_fail_closed(tmp_path):
    copied = tmp_path / "5.1"
    copied.mkdir()
    source = json.loads(MANIFEST.read_text())
    source["asset_root"] = str(copied)
    for relative in source["files"]:
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ASSET_ROOT / relative).read_bytes())
    manifest = copied / "ASSET_MANIFEST.json"
    manifest.write_text(json.dumps(source, sort_keys=True, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert validate_asset_closure(copied, digest)["network_fallback"] is False
    target = copied / "Isaac/Environments/Grid/default_environment.usd"
    target.write_bytes(target.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="changed"):
        validate_asset_closure(copied, digest)
