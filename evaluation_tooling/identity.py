"""Git-backed proof of the executing tooling and frozen product identities."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any

from .findings import Finding, FindingCollector


def _git(repository: Path, *arguments: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _package_files(package_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in package_root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_file() and not path.is_symlink():
            files[path.relative_to(package_root.parent).as_posix()] = path
    return files


def establish_identity(
    manifest: dict[str, Any],
    *,
    tooling_commit: str,
    product_commit: str,
) -> tuple[dict[str, Any], tuple[Finding, ...]]:
    """Prove declared pins from the loaded package bytes and actual Git state."""

    collector = FindingCollector()
    proof: dict[str, Any] = {
        "schema_version": "evaluation-identity-proof.v1",
        "tooling": {},
        "product": {},
    }
    package_root = Path(__file__).resolve().parent
    tooling_repository = Path(str(manifest.get("tooling_repository", ""))).resolve(strict=False)
    proof["tooling"]["package_root"] = str(package_root)
    proof["tooling"]["repository"] = str(tooling_repository)
    proof["tooling"]["commit"] = tooling_commit

    code, top_level, error = _git(tooling_repository, "rev-parse", "--show-toplevel")
    if code != 0:
        collector.add("TOOLING_REPOSITORY", "$.tooling_repository", error or "not a Git repository")
    else:
        resolved_top = Path(top_level).resolve()
        proof["tooling"]["git_toplevel"] = str(resolved_top)
        if resolved_top != tooling_repository:
            collector.add("TOOLING_REPOSITORY", "$.tooling_repository", "must name the exact Git top-level")
        if package_root != resolved_top / "evaluation_tooling":
            collector.add("TOOLING_ORIGIN", "evaluation_tooling.__file__", "loaded package is outside the declared repository")

    code, resolved_commit, error = _git(tooling_repository, "rev-parse", f"{tooling_commit}^{{commit}}")
    if code != 0 or resolved_commit != tooling_commit:
        collector.add("TOOLING_COMMIT", "$.tooling_commit", error or "commit does not resolve exactly")
    code, actual_tree, error = _git(tooling_repository, "rev-parse", f"{tooling_commit}^{{tree}}")
    if code != 0:
        collector.add("TOOLING_TREE", "$.tooling_tree", error or "tree does not resolve")
    else:
        proof["tooling"]["tree"] = actual_tree
        if actual_tree != manifest.get("tooling_tree"):
            collector.add("TOOLING_TREE", "$.tooling_tree", "declared tree differs from pinned commit tree")

    expected_paths: list[str] = []
    code, listing, error = _git(tooling_repository, "ls-tree", "-r", "--name-only", tooling_commit, "--", "evaluation_tooling")
    if code != 0:
        collector.add("TOOLING_FILES", "evaluation_tooling", error or "cannot enumerate pinned package")
    else:
        expected_paths = sorted(line for line in listing.splitlines() if line)
        actual_files = _package_files(package_root)
        for relative in sorted(set(expected_paths) - set(actual_files)):
            collector.add("TOOLING_FILE_MISSING", relative, "pinned package file is absent from loaded package")
        for relative in sorted(set(actual_files) - set(expected_paths)):
            collector.add("TOOLING_FILE_EXTRA", relative, "loaded package contains an unpinned file")
        file_rows: list[dict[str, Any]] = []
        for relative in sorted(set(expected_paths) & set(actual_files)):
            completed = subprocess.run(
                ["git", "show", f"{tooling_commit}:{relative}"],
                cwd=tooling_repository,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            actual_bytes = actual_files[relative].read_bytes()
            if completed.returncode != 0:
                collector.add("TOOLING_FILE", relative, completed.stderr.decode("utf-8", "replace").strip())
                continue
            expected_hash = _sha256(completed.stdout)
            actual_hash = _sha256(actual_bytes)
            file_rows.append({"path": relative, "bytes": len(actual_bytes), "sha256": actual_hash})
            if actual_hash != expected_hash:
                collector.add("TOOLING_FILE_HASH", relative, "loaded bytes differ from pinned Git blob")
        proof["tooling"]["files"] = file_rows
        proof["tooling"]["file_set_sha256"] = _sha256(
            "\n".join(f"{row['path']}\0{row['bytes']}\0{row['sha256']}" for row in file_rows).encode("utf-8")
        )

    product_repository = Path(str(manifest.get("product_repository", ""))).resolve(strict=False)
    proof["product"] = {
        "repository": str(product_repository),
        "commit": product_commit,
        "clean_required": manifest.get("product_clean_required"),
    }
    code, top_level, error = _git(product_repository, "rev-parse", "--show-toplevel")
    if code != 0:
        collector.add("PRODUCT_REPOSITORY", "$.product_repository", error or "not a Git repository")
    elif Path(top_level).resolve() != product_repository:
        collector.add("PRODUCT_REPOSITORY", "$.product_repository", "must name the exact Git top-level")
    code, head, error = _git(product_repository, "rev-parse", "HEAD")
    if code != 0:
        collector.add("PRODUCT_HEAD", "$.product_commit", error or "cannot resolve HEAD")
    else:
        proof["product"]["head"] = head
        if head != product_commit or head != manifest.get("product_commit"):
            collector.add("PRODUCT_HEAD", "$.product_commit", "actual product HEAD differs from the pin")
    code, tree, error = _git(product_repository, "rev-parse", "HEAD^{tree}")
    if code != 0:
        collector.add("PRODUCT_TREE", "$.product_tree", error or "cannot resolve HEAD tree")
    else:
        proof["product"]["tree"] = tree
        if tree != manifest.get("product_tree"):
            collector.add("PRODUCT_TREE", "$.product_tree", "actual product tree differs from the declaration")
    code, status, error = _git(product_repository, "status", "--porcelain=v1", "--untracked-files=all")
    if code != 0:
        collector.add("PRODUCT_STATUS", "$.product_clean_required", error or "cannot inspect product status")
    else:
        proof["product"]["clean"] = not bool(status)
        proof["product"]["status_sha256"] = _sha256((status + ("\n" if status else "")).encode("utf-8"))
        if manifest.get("product_clean_required") is True and status:
            collector.add("PRODUCT_DIRTY", "$.product_repository", "product repository is not clean")
    proof["status"] = "PASS" if collector.ok else "FAIL"
    return proof, collector.findings
