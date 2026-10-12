from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from evaluation_tooling.canonical import canonical_dumps, canonical_load
from evaluation_tooling.findings import RuntimeFailure
from evaluation_tooling.identity import establish_identity
import evaluation_tooling.identity as identity_module
from evaluation_tooling.runner import EvaluationRunner, load_submission, preflight
from evaluation_tooling.schemas import validate_manifest
from evaluation_tooling.sealer import seal_attempt
from evaluation_tooling.validation import validate_sealed
from tests.evaluation_tooling.helpers import (
    PRODUCT_COMMIT,
    PRODUCT_TREE,
    TOOLING_COMMIT,
    TOOLING_REPOSITORY,
    TOOLING_TREE,
    make_attempt,
    python_source_argv,
    terminal_file_row,
    write_canonical,
)


def command(root: Path, argv: list[str], *, label: str) -> dict[str, object]:
    return {
        "label": label,
        "argv": argv,
        "cwd": str(root),
        "environment": {},
        "expected_exit": 0,
        "timeout_seconds": 5,
        "semantic_result_path": None,
        "cleanup_argv": None,
    }


def rewrite_submission(attempt: Path, mutate_manifest=None, mutate_inputs=None) -> None:
    manifest = canonical_load(attempt / "ATTEMPT_MANIFEST.json")
    inputs = canonical_load(attempt / "INPUT_HASHES.json")
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    if mutate_inputs is not None:
        mutate_inputs(inputs)
    input_bytes = canonical_dumps(inputs)
    (attempt / "INPUT_HASHES.json").write_bytes(input_bytes)
    manifest["input_hashes_sha256"] = hashlib.sha256(input_bytes).hexdigest()
    write_canonical(attempt / "ATTEMPT_MANIFEST.json", manifest)


def pin_file(inputs: dict[str, object], path: Path, role: str) -> None:
    resolved = path.resolve()
    files = inputs["files"]
    assert isinstance(files, list)
    files.append({
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "role": role,
    })


def resign_packet(packet: Path, *, preserve_false_mtime: bool = False) -> None:
    terminal_path = packet / "TERMINAL_MANIFEST.json"
    terminal = canonical_load(terminal_path)
    rows = []
    for path in sorted(item for item in packet.rglob("*") if item.is_file() and item != terminal_path):
        rows.append(terminal_file_row(packet, path, terminal["schema_version"]))
    terminal["files"] = rows
    if not preserve_false_mtime:
        terminal["max_preseal_mtime_ns"] = max((packet / row["path"]).stat().st_mtime_ns for row in rows)
    terminal_path.write_bytes(canonical_dumps(terminal))


class HardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def runner(self, attempt: Path, **kwargs) -> EvaluationRunner:
        return EvaluationRunner(
            attempt,
            tooling_commit=TOOLING_COMMIT,
            product_commit=PRODUCT_COMMIT,
            **kwargs,
        )

    def positive_packet(self) -> Path:
        attempt = make_attempt(self.root)
        self.assertEqual(self.runner(attempt).run_all(), 0)
        self.assertEqual(
            seal_attempt(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)["status"],
            "PASS",
        )
        return attempt

    def clone_repository(self, name: str) -> Path:
        destination = self.root / name
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(TOOLING_REPOSITORY), str(destination)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return destination

    def identity_manifest(self, tooling_repository: Path) -> dict[str, object]:
        return {
            "tooling_repository": str(tooling_repository),
            "tooling_commit": TOOLING_COMMIT,
            "tooling_tree": TOOLING_TREE,
            "product_repository": str(TOOLING_REPOSITORY),
            "product_commit": PRODUCT_COMMIT,
            "product_tree": PRODUCT_TREE,
            "product_clean_required": False,
        }

    def test_fake_or_nonexistent_tooling_commit_rejected(self) -> None:
        attempt = make_attempt(self.root)
        fake = "f" * 40
        rewrite_submission(
            attempt,
            lambda value: value.update(tooling_commit=fake, tooling_tree="e" * 40),
            lambda value: value.update(tooling_commit=fake),
        )
        result = preflight(attempt, tooling_commit=fake, product_commit=PRODUCT_COMMIT)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("TOOLING_COMMIT", {item["code"] for item in result["findings"]})

    def test_changed_and_extra_loaded_package_bytes_rejected(self) -> None:
        clone = self.clone_repository("tooling_clone")
        manifest = self.identity_manifest(clone)
        package_file = clone / "evaluation_tooling" / "identity.py"
        with mock.patch.object(identity_module, "__file__", str(package_file)):
            proof, findings = establish_identity(manifest, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
            self.assertEqual((proof["status"], findings), ("PASS", ()))
            package_file.write_bytes(package_file.read_bytes() + b"\n# changed\n")
            _, findings = establish_identity(manifest, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
            self.assertIn("TOOLING_FILE_HASH", {finding.code for finding in findings})
            package_file.write_bytes(subprocess.run(
                ["git", "show", f"{TOOLING_COMMIT}:evaluation_tooling/identity.py"],
                cwd=clone, check=True, stdout=subprocess.PIPE,
            ).stdout)
            (clone / "evaluation_tooling" / "extra_module.py").write_text("value = 1\n")
            _, findings = establish_identity(manifest, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
            self.assertIn("TOOLING_FILE_EXTRA", {finding.code for finding in findings})

    def test_moved_branch_head_keeps_intact_pinned_commit_valid(self) -> None:
        clone = self.clone_repository("moved_head")
        marker = clone / "branch-marker.txt"
        marker.write_text("branch moved without package change\n")
        subprocess.run(["git", "add", "branch-marker.txt"], cwd=clone, check=True)
        environment = dict(os.environ)
        environment.update({
            "GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        })
        subprocess.run(
            ["git", "commit", "--quiet", "--no-gpg-sign", "-m", "fixture: move branch"],
            cwd=clone, env=environment, check=True,
        )
        manifest = self.identity_manifest(clone)
        with mock.patch.object(identity_module, "__file__", str(clone / "evaluation_tooling" / "identity.py")):
            proof, findings = establish_identity(manifest, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertEqual((proof["status"], findings), ("PASS", ()))

    def test_wrong_or_dirty_product_identity_rejected(self) -> None:
        product = self.clone_repository("product_clone")
        manifest = self.identity_manifest(TOOLING_REPOSITORY)
        manifest.update(product_repository=str(product), product_clean_required=True)
        (product / "dirty.txt").write_text("dirty\n")
        _, findings = establish_identity(manifest, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertIn("PRODUCT_DIRTY", {finding.code for finding in findings})
        (product / "dirty.txt").unlink()
        manifest["product_tree"] = "0" * 40
        _, findings = establish_identity(manifest, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertIn("PRODUCT_TREE", {finding.code for finding in findings})

    def test_runtime_artifact_exception_is_lifecycle_root_and_use_bounded(self) -> None:
        attempt = make_attempt(self.root)
        runner = self.runner(attempt)
        self.assertEqual(runner.prepare()["status"], "PASS")
        runtime_tmp = attempt / "results" / "runtime_support" / "tmp"
        runtime_cache = attempt / "results" / "runtime_support" / "cache"
        runtime_tmp.mkdir(parents=True)
        runtime_cache.mkdir(parents=True)
        generated_source = runtime_tmp / "generated.py"
        executable_lock = runtime_cache / "cache.lock"
        generated_source.write_text("inert = True\n")
        executable_lock.write_bytes(b"")
        executable_lock.chmod(0o770)

        _, _, before = load_submission(attempt, allow_tool_outputs=True)
        self.assertIn("ATTEMPT_LOCAL_EXECUTABLE", {finding.code for finding in before})

        state_path = attempt / "TOOL_STATE.json"
        state = canonical_load(state_path)
        state["runtime_started"] = True
        write_canonical(state_path, state)
        _, _, after = load_submission(attempt, allow_tool_outputs=True)
        after_locations = {finding.location for finding in after if finding.code == "ATTEMPT_LOCAL_EXECUTABLE"}
        self.assertNotIn(generated_source.relative_to(attempt).as_posix(), after_locations)
        self.assertNotIn(executable_lock.relative_to(attempt).as_posix(), after_locations)

        outside = attempt / "results" / "generated.py"
        outside.write_text("still_forbidden = True\n")
        link = runtime_tmp / "linked.py"
        link.symlink_to(generated_source)
        _, _, bounded = load_submission(attempt, allow_tool_outputs=True)
        codes_by_location = {(finding.code, finding.location) for finding in bounded}
        self.assertIn(("ATTEMPT_LOCAL_EXECUTABLE", "results/generated.py"), codes_by_location)
        self.assertIn(("ATTEMPT_SYMLINK", "results/runtime_support/tmp/linked.py"), codes_by_location)

        spec = canonical_load(attempt / "ATTEMPT_MANIFEST.json")["commands"][0]
        for field in ("argv", "cleanup_argv"):
            malicious = dict(spec)
            malicious[field] = [str(Path(sys.executable).resolve()), "-B", str(generated_source)]
            with self.assertRaises(RuntimeFailure) as raised:
                runner._assert_execution_confinement(malicious, field)
            self.assertIn("ATTEMPT_LOCAL_EXECUTABLE", str(raised.exception))

    def test_post_runtime_artifacts_are_terminally_inventoried(self) -> None:
        attempt = make_attempt(self.root)
        self.assertEqual(self.runner(attempt).run_all(), 0)
        runtime_tmp = attempt / "results" / "runtime_support" / "tmp"
        runtime_cache = attempt / "results" / "runtime_support" / "cache"
        runtime_tmp.mkdir(parents=True)
        runtime_cache.mkdir(parents=True)
        generated_source = runtime_tmp / "generated.py"
        executable_lock = runtime_cache / "cache.lock"
        generated_source.write_text("inert = True\n")
        executable_lock.write_bytes(b"")
        executable_lock.chmod(0o770)
        sealed = seal_attempt(
            attempt,
            tooling_commit=TOOLING_COMMIT,
            product_commit=PRODUCT_COMMIT,
        )
        self.assertEqual(sealed["status"], "PASS")
        terminal = canonical_load(attempt / "TERMINAL_MANIFEST.json")
        rows = {row["path"]: row for row in terminal["files"]}
        self.assertEqual(rows["results/runtime_support/tmp/generated.py"]["type"], "file")
        self.assertEqual(
            rows["results/runtime_support/tmp/generated.py"]["mode"],
            stat.S_IMODE(generated_source.stat().st_mode),
        )
        self.assertEqual(rows["results/runtime_support/cache/cache.lock"]["mode"], 0o770)
        self.assertEqual(validate_sealed(attempt)["status"], "PASS")

    def test_input_mutation_before_command_is_receipted_and_blocks_launch(self) -> None:
        marker = self.root / "not-created"
        attempt = make_attempt(self.root, commands=[command(self.root, ["/usr/bin/touch", str(marker)], label="blocked")])
        runner = self.runner(attempt)
        self.assertEqual(runner.prepare()["status"], "PASS")
        (self.root / "input.dat").write_bytes(b"mutated\n")
        receipt, exit_code = runner.run_command(0)
        self.assertEqual((receipt.receipt_status, exit_code), ("FAIL", 1))
        self.assertIn("FROZEN_IDENTITY_LOSS", receipt.primary_failure)
        self.assertFalse(marker.exists())
        self.assertEqual(canonical_load(attempt / "COMMAND_LEDGER.json")["unexpected_commands"], [1])

    def test_input_mutation_between_commands_stops_later_command(self) -> None:
        marker = self.root / "not-created"
        commands = [
            command(self.root, ["/usr/bin/true"], label="first"),
            command(self.root, ["/usr/bin/touch", str(marker)], label="second"),
        ]
        attempt = make_attempt(self.root, commands=commands)
        runner = self.runner(attempt)
        runner.prepare()
        self.assertEqual(runner.run_command(0)[1], 0)
        (self.root / "input.dat").write_bytes(b"between\n")
        self.assertEqual(runner.run_command(1)[1], 1)
        self.assertFalse(marker.exists())

    def test_command_mutating_input_fails_after_child_and_stops(self) -> None:
        marker = self.root / "not-created"
        source = self.root / "input.dat"
        commands = [
            command(self.root, ["/usr/bin/truncate", "-s", "0", str(source)], label="mutate"),
            command(self.root, ["/usr/bin/touch", str(marker)], label="must_not_run"),
        ]
        attempt = make_attempt(self.root, commands=commands)
        self.assertEqual(self.runner(attempt).run_all(), 1)
        self.assertFalse(marker.exists())
        receipt = canonical_load(attempt / "launcher_receipts" / "0001.json")
        self.assertIn("FROZEN_IDENTITY_LOSS", receipt["primary_failure"])

    def test_dynamic_compact_shell_stdin_and_relative_sources_rejected(self) -> None:
        attempt = make_attempt(self.root)
        manifest = canonical_load(attempt / "ATTEMPT_MANIFEST.json")
        cases = [
            ["/usr/bin/python3", "-cprint('x')"],
            ["/usr/bin/python3", "-Bc", "print('x')"],
            ["/usr/bin/sh", "-lc", "printf x"],
            ["/usr/bin/python3", "-"],
            ["/usr/bin/python3", "/dev/stdin"],
        ]
        for argv in cases:
            manifest["commands"][0]["argv"] = argv
            self.assertIn("DYNAMIC_CODE", {finding.code for finding in validate_manifest(manifest, attempt)})
        manifest["commands"][0]["argv"] = [str(Path(sys.executable).resolve()), "payload.py"]
        self.assertIn("ATTEMPT_LOCAL_EXECUTABLE", {finding.code for finding in validate_manifest(manifest, attempt)})

    def test_generated_unpinned_source_cannot_run(self) -> None:
        generated = self.root / "generated.py"
        commands = [
            command(self.root, ["/usr/bin/cp", str(self.root / "input.dat"), str(generated)], label="generate"),
            command(self.root, [str(Path(sys.executable).resolve()), str(generated)], label="execute_generated"),
        ]
        attempt = make_attempt(self.root, commands=commands)
        runner = self.runner(attempt)
        result = runner.prepare()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("UNPINNED_SOURCE", {item["code"] for item in result["findings"]})
        self.assertFalse(generated.exists())

    def test_unpinned_system_executable_is_blocked_before_launch(self) -> None:
        attempt = make_attempt(
            self.root,
            commands=[command(self.root, ["/usr/bin/true"], label="unpinned_system")],
            pin_command_files=False,
            adapt_native_commands=False,
        )
        self.assertEqual(self.runner(attempt).run_all(), 2)

    def test_exact_pinned_native_executable_rejects_regardless_of_bytes(self) -> None:
        executable = self.root / "copied-true"
        shutil.copy2("/usr/bin/true", executable)
        attempt = make_attempt(
            self.root,
            commands=[command(self.root, [str(executable)], label="pinned_copy")],
            adapt_native_commands=False,
        )
        result = preflight(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("LAUNCH_PROFILE", {item["code"] for item in result["findings"]})

    def test_direct_pinned_interpreter_and_absolute_source_pass(self) -> None:
        source = self.root / "absolute_payload.py"
        source.write_text("raise SystemExit(0)\n")
        python = str(Path(__import__("sys").executable).resolve())
        attempt = make_attempt(
            self.root,
            commands=[command(self.root, [python, str(source)], label="pinned_interpreter")],
        )
        self.assertEqual(self.runner(attempt).run_all(), 0)

    def test_env_relative_source_wrapper_cannot_execute(self) -> None:
        marker = self.root / "wrapper-marker"
        source = self.root / "relative_payload.py"
        source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n")
        attempt = make_attempt(
            self.root,
            commands=[command(
                self.root,
                ["/usr/bin/env", "python3", source.name],
                label="env_wrapper",
            )],
            pin_command_files=False,
            adapt_native_commands=False,
        )
        self.assertEqual(self.runner(attempt).run_all(), 2)
        self.assertFalse(marker.exists())

    def test_cleanup_uses_same_native_executable_gate(self) -> None:
        marker = self.root / "cleanup-marker"
        primary = python_source_argv(self.root, "cleanup_primary", "raise SystemExit(0)\n")
        spec = command(self.root, primary, label="cleanup_gate")
        spec["cleanup_argv"] = ["/usr/bin/touch", str(marker)]
        attempt = make_attempt(self.root, commands=[spec], adapt_native_commands=False)
        result = preflight(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("LAUNCH_PROFILE", {item["code"] for item in result["findings"]})
        self.assertFalse(marker.exists())

    def test_cleanup_confinement_does_not_overwrite_primary_failure(self) -> None:
        marker = self.root / "cleanup-marker"
        primary = python_source_argv(self.root, "primary_failure", "raise SystemExit(1)\n")
        cleanup = python_source_argv(self.root, "cleanup_failure", "raise SystemExit(1)\n")
        spec = command(self.root, primary, label="primary_failure")
        spec["cleanup_argv"] = cleanup
        attempt = make_attempt(self.root, commands=[spec], adapt_native_commands=False)
        runner = self.runner(attempt)
        self.assertEqual(runner.prepare()["status"], "PASS")
        receipt, exit_code = runner.run_command(0)
        self.assertEqual((receipt.primary_failure, receipt.cleanup_status, exit_code), ("RAW_EXIT:1", "FAIL", 1))
        self.assertFalse(marker.exists())

    def test_cleanup_source_pin_loss_blocks_cleanup_and_is_receipted(self) -> None:
        marker = self.root / "cleanup-marker"
        cleanup_source = self.root / "fixture_cleanup.py"
        cleanup_source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
        primary = python_source_argv(
            self.root,
            "mutate_cleanup",
            f"from pathlib import Path\nPath({str(cleanup_source)!r}).write_text('changed')\n",
        )
        spec = command(self.root, primary, label="cleanup_pin_loss")
        spec["cleanup_argv"] = [str(Path(sys.executable).resolve()), "-B", str(cleanup_source)]
        attempt = make_attempt(self.root, commands=[spec], adapt_native_commands=False)
        runner = self.runner(attempt)
        self.assertEqual(runner.prepare()["status"], "PASS")
        receipt, exit_code = runner.run_command(0)
        self.assertEqual((receipt.receipt_status, receipt.cleanup_status, exit_code), ("FAIL", "FAIL", 1))
        self.assertIn("FROZEN_IDENTITY_LOSS", receipt.primary_failure)
        self.assertFalse(marker.exists())

    def test_command_language_utilities_reject_for_primary_and_cleanup(self) -> None:
        cases = {
            "find": lambda root, marker: ["/usr/bin/find", str(root), "-maxdepth", "0", "-exec", "/usr/bin/touch", str(marker), ";"],
            "awk": lambda root, marker: ["/usr/bin/awk", f'BEGIN {{ system("/usr/bin/touch {marker}") }}'],
            "sed": lambda root, marker: ["/usr/bin/sed", "-n", f"e /usr/bin/touch {marker}"],
        }
        for utility, build in cases.items():
            for field in ("argv", "cleanup_argv"):
                with self.subTest(utility=utility, field=field), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                    root = Path(temporary)
                    marker = root / "marker"
                    native = build(root, marker)
                    if field == "argv":
                        spec = command(root, native, label=f"{utility}_primary")
                    else:
                        spec = command(root, python_source_argv(root, "primary", "raise SystemExit(0)\n"), label=f"{utility}_cleanup")
                        spec["cleanup_argv"] = native
                    attempt = make_attempt(root, commands=[spec], adapt_native_commands=False)
                    result = preflight(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
                    self.assertEqual(result["status"], "FAIL")
                    self.assertIn("LAUNCH_PROFILE", {item["code"] for item in result["findings"]})
                    self.assertFalse(marker.exists())

    def test_renamed_dispatcher_and_non_python_named_python3_reject(self) -> None:
        for original, renamed in (("/usr/bin/find", "renamed-dispatcher"), ("/usr/bin/true", "python3")):
            with self.subTest(renamed=renamed), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                root = Path(temporary)
                copied = root / renamed
                shutil.copy2(original, copied)
                marker = root / "marker"
                attempt = make_attempt(
                    root,
                    commands=[command(root, [str(copied), str(marker)], label="renamed")],
                    adapt_native_commands=False,
                )
                result = preflight(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
                self.assertEqual(result["status"], "FAIL")
                self.assertIn("LAUNCH_PROFILE", {item["code"] for item in result["findings"]})
                self.assertFalse(marker.exists())

    def test_wrong_and_symlinked_interpreters_reject(self) -> None:
        source = self.root / "payload.py"
        source.write_text("raise SystemExit(0)\n")
        wrong_path = self.root / "wrong-python"
        shutil.copy2(Path(sys.executable).resolve(), wrong_path)
        attempt = make_attempt(
            self.root,
            commands=[command(self.root, [str(wrong_path), str(source)], label="wrong_python")],
            adapt_native_commands=False,
        )
        self.assertEqual(preflight(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)["status"], "FAIL")

        link_root = self.root / "link"
        link_root.mkdir()
        linked_source = link_root / "payload.py"
        linked_source.write_text("raise SystemExit(0)\n")
        interpreter_link = link_root / "python"
        interpreter_link.symlink_to(Path(sys.executable).resolve())
        linked_attempt = make_attempt(
            link_root,
            commands=[command(link_root, [str(interpreter_link), str(linked_source)], label="linked_python")],
            adapt_native_commands=False,
        )
        rewrite_submission(linked_attempt, lambda value: value["commands"][0]["argv"].__setitem__(0, str(interpreter_link)))
        result = preflight(linked_attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("LAUNCH_PROFILE", {item["code"] for item in result["findings"]})

    def test_missing_unpinned_mutated_and_symlinked_sources_reject(self) -> None:
        missing = self.root / "missing.py"
        attempt = make_attempt(
            self.root,
            commands=[command(self.root, [str(Path(sys.executable).resolve()), str(missing)], label="missing_source")],
            adapt_native_commands=False,
        )
        result = self.runner(attempt).prepare()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("UNPINNED_SOURCE", {item["code"] for item in result["findings"]})

        for mode in ("mutated", "symlinked"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                root = Path(temporary)
                source = root / "source.py"
                source.write_text("raise SystemExit(0)\n")
                authored = source
                if mode == "symlinked":
                    authored = root / "linked.py"
                    authored.symlink_to(source)
                spec = command(root, [str(Path(sys.executable).resolve()), str(authored)], label=f"{mode}_source")
                attempt = make_attempt(root, commands=[spec], adapt_native_commands=False)
                if mode == "mutated":
                    runner = self.runner(attempt)
                    self.assertEqual(runner.prepare()["status"], "PASS")
                    source.write_text("raise SystemExit(1)\n")
                    self.assertIn("FROZEN_IDENTITY_LOSS", runner.run_command(0)[0].primary_failure)
                else:
                    rewrite_submission(attempt, lambda value: value["commands"][0]["argv"].__setitem__(1, str(authored)))
                    result = preflight(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
                    self.assertEqual(result["status"], "FAIL")
                    self.assertIn("SOURCE_SYMLINK", {item["code"] for item in result["findings"]})

    def test_all_persistence_boundaries_recover_same_draft(self) -> None:
        boundaries = [
            "before_directory", "after_directory", "after_stdout", "after_stderr",
            "after_cleanup_stdout", "after_cleanup_stderr", "after_launcher_receipt",
            "after_metadata", "after_aggregates", "after_runner_receipt",
        ]
        for boundary in boundaries:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                root = Path(temporary)
                spec = command(root, ["/usr/bin/true"], label="faulted")
                spec["cleanup_argv"] = ["/usr/bin/true"]
                attempt = make_attempt(root, commands=[spec])
                fired = False

                def inject(observed: str) -> None:
                    nonlocal fired
                    if observed == boundary and not fired:
                        fired = True
                        raise OSError(f"injected:{boundary}")

                runner = self.runner(attempt, fault_injector=inject)
                self.assertEqual(runner.prepare()["status"], "PASS")
                with self.assertRaisesRegex(OSError, f"injected:{boundary}"):
                    runner.run_command(0)
                resumed = self.runner(attempt)
                if boundary == "before_directory":
                    self.assertEqual(resumed.run_command(0)[1], 0)
                else:
                    with self.assertRaisesRegex(RuntimeFailure, "recovered interrupted command 1"):
                        resumed.run_command(0)
                    self.assertEqual(self.runner(attempt).run_command(0, repair_of=1)[1], 0)
                    self.assertTrue((attempt / "commands/0001_faulted/transaction_complete.json").is_file())
                sealed = seal_attempt(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)
                self.assertEqual(sealed["status"], "PASS")
                self.assertEqual(validate_sealed(attempt)["status"], "PASS")

    def test_terminal_receipts_are_exact_pass_and_fail(self) -> None:
        packet = self.positive_packet()
        receipt = canonical_load(sorted((packet / "runner_receipts").glob("*.json"))[-1])
        self.assertEqual((receipt["mode"], receipt["command_ids"], receipt["result"], receipt["failure"]), ("seal", [], "PASS", None))

        other_root = self.root / "failed"
        other_root.mkdir()
        attempt = make_attempt(other_root, required_results=["results/missing.json"])
        self.runner(attempt).run_all()
        self.assertEqual(seal_attempt(attempt, tooling_commit=TOOLING_COMMIT, product_commit=PRODUCT_COMMIT)["status"], "FAIL")
        receipt = canonical_load(sorted((attempt / "runner_receipts").glob("*.json"))[-1])
        self.assertEqual((receipt["mode"], receipt["result"], receipt["failure"]), ("seal", "FAIL", "PRESEAL_VALIDATION"))

    def test_sealed_inspection_is_interpreter_independent(self) -> None:
        packet = self.positive_packet()
        results = []
        for interpreter in (str(Path(sys.executable).resolve()), "/usr/bin/python3"):
            completed = subprocess.run(
                [interpreter, "-B", "-m", "evaluation_tooling", "inspect", str(packet)],
                cwd=TOOLING_REPOSITORY,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual((completed.returncode, completed.stderr), (0, b""), interpreter)
            results.append(json.loads(completed.stdout))
        self.assertEqual(results[0], results[1])
        self.assertEqual((results[0]["status"], results[0]["findings"]), ("PASS", []))

    def test_sealed_inspection_ignores_removed_live_sources(self) -> None:
        packet = self.positive_packet()
        before = validate_sealed(packet)
        for source in self.root.glob("fixture_*.py"):
            source.unlink()
        after = validate_sealed(packet)
        self.assertEqual(after, before)
        self.assertEqual((after["status"], after["findings"]), ("PASS", []))

    def test_malformed_launch_profile_proof_rejected_offline(self) -> None:
        packet = self.positive_packet()
        state_path = packet / "TOOL_STATE.json"
        state = canonical_load(state_path)
        state["identity_proof"]["launch_profile"]["status"] = "FAIL"
        state_path.write_bytes(canonical_dumps(state))
        resign_packet(packet)
        result = validate_sealed(packet)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("LAUNCH_PROOF_STATUS", {item["code"] for item in result["findings"]})

    def test_malformed_receipt_state_and_false_mtime_are_rejected(self) -> None:
        packet = self.positive_packet()
        for name in ("receipt", "state", "mtime"):
            mutated = self.root / name
            shutil.copytree(packet, mutated)
            if name == "receipt":
                receipt_path = sorted((mutated / "runner_receipts").glob("*.json"))[-1]
                receipt = canonical_load(receipt_path)
                receipt.update(invocation_id=999, mode="run", command_ids=[999])
                receipt_path.write_bytes(canonical_dumps(receipt))
                resign_packet(mutated)
            elif name == "state":
                state_path = mutated / "TOOL_STATE.json"
                state = canonical_load(state_path)
                state["lifecycle"] = "DRAFT"
                state["unknown"] = True
                state_path.write_bytes(canonical_dumps(state))
                resign_packet(mutated)
            else:
                terminal_path = mutated / "TERMINAL_MANIFEST.json"
                terminal = canonical_load(terminal_path)
                terminal["max_preseal_mtime_ns"] = 0
                terminal_path.write_bytes(canonical_dumps(terminal))
            result = validate_sealed(mutated)
            self.assertEqual(result["status"], "FAIL", name)

    def test_path_traversal_duplicate_ids_chronology_and_collect_all(self) -> None:
        packet = self.positive_packet()
        metadata_path = next((packet / "commands").glob("*/metadata.json"))
        metadata = canonical_load(metadata_path)
        metadata["stdout_path"] = "../outside.bin"
        metadata["command_id"] = 2
        metadata["ended_at"] = "2000-01-01T00:00:00+00:00"
        metadata_path.write_bytes(canonical_dumps(metadata))
        state_path = packet / "TOOL_STATE.json"
        state = canonical_load(state_path)
        state["unknown"] = True
        state_path.write_bytes(canonical_dumps(state))
        resign_packet(packet)
        result = validate_sealed(packet)
        codes = {item["code"] for item in result["findings"]}
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue({"PATH", "COMMAND_ID", "CHRONOLOGY", "UNKNOWN_FIELD"}.issubset(codes))
