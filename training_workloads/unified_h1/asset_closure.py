"""Pure-stdlib validation for the pinned Isaac Sim 5.1 ground asset closure."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


MANIFEST_NAME = "ASSET_MANIFEST.json"
REQUIRED_FILES = {
    "Isaac/Environments/Grid/default_environment.usd",
    "Isaac/Environments/Grid/Materials/Textures/WireframeBlur_blue.png",
    "Isaac/Environments/Grid/Materials/Textures/Wireframe_blue.png",
    "Isaac/Environments/Grid/Materials/Textures/WireframeBlur_basecolor.png",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, under: Path | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"asset closure entry is not a regular file: {path}")
    resolved = path.resolve()
    if under is not None and resolved != under and not resolved.is_relative_to(under):
        raise ValueError(f"asset closure entry escapes local root: {path}")
    return resolved


def validate_asset_closure(asset_root: Path, manifest_sha256: str) -> dict[str, Any]:
    """Open and hash every local dependency before AppLauncher can be imported."""
    if not asset_root.is_absolute() or asset_root.is_symlink() or asset_root.resolve() != asset_root:
        raise ValueError("Isaac asset root must be an exact absolute local directory")
    if not asset_root.is_dir() or "://" in str(asset_root):
        raise ValueError("Isaac asset root is unavailable or remote")
    manifest_path = _regular_file(asset_root / MANIFEST_NAME, under=asset_root)
    if sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("Isaac asset manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"schema_version", "created_at", "asset_root", "source_root", "files",
                "external_local_materials", "entrypoint", "remote_fallback_allowed"}
    if set(manifest) != required or manifest["schema_version"] != "isaac-sim-5.1-h1-ground-closure/1":
        raise ValueError("Isaac asset manifest schema mismatch")
    if manifest["asset_root"] != str(asset_root) or manifest["entrypoint"] not in REQUIRED_FILES:
        raise ValueError("Isaac asset manifest root or entrypoint mismatch")
    if manifest["remote_fallback_allowed"] is not False or set(manifest["files"]) != REQUIRED_FILES:
        raise ValueError("Isaac asset closure is incomplete or permits remote fallback")
    opened: dict[str, dict[str, Any]] = {}
    for relative, expected in manifest["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts or set(expected) != {"bytes", "sha256", "source_url"}:
            raise ValueError("invalid Isaac asset manifest entry")
        path = _regular_file(asset_root / relative, under=asset_root)
        raw = path.read_bytes()
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"Isaac asset closure changed: {relative}")
        if relative.endswith(".usd") and not raw.startswith(b"PXR-USDC"):
            raise ValueError("ground USD is not a readable crate")
        if relative.endswith(".png") and not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"ground texture is not a readable PNG: {relative}")
        opened[relative] = {"bytes": len(raw), "sha256": expected["sha256"]}
    materials = manifest["external_local_materials"]
    if set(materials) != {"OmniPBR.mdl"}:
        raise ValueError("ground material closure mismatch")
    material = materials["OmniPBR.mdl"]
    if set(material) != {"path", "bytes", "sha256", "resolution"}:
        raise ValueError("ground material binding schema mismatch")
    material_path = _regular_file(Path(material["path"]))
    raw = material_path.read_bytes()
    if len(raw) != material["bytes"] or hashlib.sha256(raw).hexdigest() != material["sha256"]:
        raise ValueError("local OmniPBR material changed")
    raw.decode("utf-8")
    return {"asset_root": str(asset_root), "manifest_sha256": manifest_sha256,
            "opened_files": opened, "material": {"path": str(material_path), "sha256": material["sha256"]},
            "network_fallback": False}


def validate_asset_environment(execution: Mapping[str, Any]) -> dict[str, Any]:
    variable = execution["isaacsim_asset_root_env"]
    observed = os.environ.get(variable)
    if observed != execution["isaacsim_asset_root"]:
        raise PermissionError(f"{variable} does not match the frozen local asset root")
    return validate_asset_closure(Path(observed), execution["isaacsim_asset_manifest_sha256"])


__all__ = ["MANIFEST_NAME", "REQUIRED_FILES", "sha256_file", "validate_asset_closure", "validate_asset_environment"]
