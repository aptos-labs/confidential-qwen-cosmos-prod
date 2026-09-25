#!/usr/bin/env python3
"""Strict offline pre-spawn installer. No imports of vllm/torch, no patch fuzz."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parent
FILES = ("engine/messages.py", "engine/orchestrator.py", "entrypoints/omni_base.py")
HELPER = "input_meter.py"


class InstallError(RuntimeError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def syntax(data: bytes, name: str) -> None:
    try:
        compile(data, name, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise InstallError(f"invalid Python syntax: {name}") from exc


def regular_path(root: Path, name: str) -> Path:
    path = root / name
    for part in (root, *path.relative_to(root).parents):
        # Relative parents need resolving under root; root itself is absolute.
        candidate = part if part.is_absolute() else root / part
        if candidate.is_symlink() or not candidate.is_dir():
            raise InstallError(f"unsafe package directory: {name}")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise InstallError(f"not a regular package file: {name}")
    return path


@dataclass(frozen=True)
class Plan:
    root: Path
    before: dict[str, bytes | None]
    after: dict[str, bytes]
    already_installed: bool


def preflight(package_root: Path, artifact: Path = ARTIFACT) -> Plan:
    """Validate every hash, replacement and syntax before touching the package."""
    root = package_root.absolute()
    if root.is_symlink() or root.resolve() != root:
        raise InstallError("package root must be canonical, not a symlink")
    manifest = json.loads((artifact / "manifest.json").read_text())
    if manifest["format"] != 1 or set(manifest["files"]) != set(FILES):
        raise InstallError("unexpected manifest file set/format")
    helper = (artifact / HELPER).read_bytes()
    if digest(helper) != manifest["helper_sha256"]:
        raise InstallError("helper artifact hash mismatch")
    syntax(helper, HELPER)
    before: dict[str, bytes | None] = {}
    after: dict[str, bytes] = {HELPER: helper}
    original_states = []
    patched_states = []
    generated_patch = ""
    for name in FILES:
        path = regular_path(root, name)
        if not path.exists():
            raise InstallError(f"missing package file: {name}")
        current = path.read_bytes()
        before[name] = current
        spec = manifest["files"][name]
        is_original = digest(current) == spec["original_sha256"]
        is_patched = digest(current) == spec["patched_sha256"]
        if not is_original and not is_patched:
            raise InstallError(f"source drift: {name}")
        original_states.append(is_original)
        patched_states.append(is_patched)
        # Reverse exact replacements to validate the entire artifact even on
        # idempotent invocations, then reapply forward and verify both hashes.
        original = current.decode("utf-8")
        if is_patched:
            for edit in reversed(spec["replacements"]):
                if not edit["new"] or original.count(edit["new"]) != 1:
                    raise InstallError(f"non-unique reverse replacement: {name}")
                original = original.replace(edit["new"], edit["old"], 1)
        if digest(original.encode()) != spec["original_sha256"]:
            raise InstallError(f"original hash mismatch: {name}")
        patched = original
        for edit in spec["replacements"]:
            if not edit["old"] or not edit["new"] or patched.count(edit["old"]) != 1:
                raise InstallError(f"non-unique replacement: {name}")
            patched = patched.replace(edit["old"], edit["new"], 1)
        after[name] = patched.encode()
        if digest(after[name]) != spec["patched_sha256"]:
            raise InstallError(f"patched hash mismatch: {name}")
        syntax(original.encode(), name)
        syntax(after[name], name)
        generated_patch += "".join(
            difflib.unified_diff(
                original.splitlines(True),
                patched.splitlines(True),
                fromfile="a/vllm_omni/" + name,
                tofile="b/vllm_omni/" + name,
                n=0,
            )
        )
    if generated_patch.encode() != (artifact / "vllm-omni-v0.28.0.patch").read_bytes():
        raise InstallError("unified patch does not match manifest replacements")
    helper_path = regular_path(root, HELPER)
    before[HELPER] = helper_path.read_bytes() if helper_path.exists() else None
    already_installed = all(patched_states) and before[HELPER] == helper
    pristine = all(original_states) and before[HELPER] is None
    if not already_installed and not pristine:
        raise InstallError("mixed installation or unexpected pre-existing helper")
    return Plan(root, before, after, already_installed)


def assert_unchanged(plan: Plan) -> None:
    for name, expected in plan.before.items():
        path = regular_path(plan.root, name)
        current = path.read_bytes() if path.exists() else None
        if current != expected:
            raise InstallError(f"package changed after preflight: {name}")


def install(package_root: Path, *, check: bool = False, artifact: Path = ARTIFACT) -> str:
    plan = preflight(package_root, artifact)
    if plan.already_installed:
        return "already installed (all bytes verified)"
    if check:
        return "preflight passed (no writes)"

    # Stage ALL replacements on the destination filesystem before replacing
    # ANY original. Server/workers must be stopped; this is not hot-patching.
    staged: dict[str, Path] = {}
    replaced: list[str] = []
    try:
        for name, content in plan.after.items():
            target = plan.root / name
            fd, temporary = tempfile.mkstemp(prefix=".input-meter-", dir=target.parent)
            staged[name] = Path(temporary)
            with os.fdopen(fd, "wb") as stream:
                mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
                os.fchmod(stream.fileno(), mode)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if staged[name].read_bytes() != content:
                raise InstallError(f"staging verification failed: {name}")
        assert_unchanged(plan)
        for name, temporary in staged.items():
            os.replace(temporary, plan.root / name)
            replaced.append(name)
        verified = preflight(plan.root, artifact)
        if not verified.already_installed:
            raise InstallError("post-install verification failed")
    except BaseException:
        # Best-effort rollback for write errors/interrupts (not SIGKILL/power
        # failure). A subsequent mixed state is always refused, never repaired
        # permissively; restore the pristine image if rollback cannot finish.
        for name in reversed(replaced):
            previous = plan.before[name]
            target = plan.root / name
            if previous is None:
                target.unlink()
            else:
                target.write_bytes(previous)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return "installed (all patched bytes verified)"


def stage(package_root: Path, *, parent: Path = Path("/tmp"), artifact: Path = ARTIFACT) -> Path:
    """Patch a private import overlay without writing to the immutable image."""
    preflight(package_root, artifact)
    for path in package_root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise InstallError("unsupported package entry in staging source")
    overlay = Path(tempfile.mkdtemp(prefix="ccs-metered-", dir=parent)).resolve()
    try:
        target = overlay / "vllm_omni"
        shutil.copytree(
            package_root, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        for directory in (target, *(p for p in target.rglob("*") if p.is_dir())):
            directory.chmod(stat.S_IMODE(directory.stat().st_mode) | 0o700)
        install(target, artifact=artifact)
        return overlay
    except BaseException:
        shutil.rmtree(overlay)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate without writing")
    mode.add_argument("--stage", action="store_true", help="print a private patched PYTHONPATH")
    args = parser.parse_args()
    try:
        manifest = json.loads((ARTIFACT / "manifest.json").read_text())
        for package, version in manifest["versions"].items():
            if importlib.metadata.version(package) != version:
                raise InstallError(f"unsupported installed version: {package}")
        # Metadata lookup avoids importing vllm_omni and triggering runtime
        # initialization before the stable helper and all seams are installed.
        dist = importlib.metadata.distribution("vllm-omni")
        root = Path(dist.locate_file("vllm_omni")).absolute()
        print(stage(root) if args.stage else install(root, check=args.check))
    except (
        InstallError,
        OSError,
        ValueError,
        KeyError,
        importlib.metadata.PackageNotFoundError,
    ) as exc:
        parser.exit(1, f"input meter installation refused: {exc}\n")


if __name__ == "__main__":
    main()
