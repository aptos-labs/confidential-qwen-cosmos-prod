"""Show readable source changes relative to the immutable v0.0.5 bundle."""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA256 = "14cec7260731d54d538b75e2d06e5072029c4be3549f4c88dd962eccc9cf8111"


METER_FILES = {"install.py", "input_meter.py", "manifest.json", "vllm-omni-v0.28.0.patch"}


def source_path(name: str) -> Path:
    if re.fullmatch(r"paid_gateway/[A-Za-z_][A-Za-z0-9_]*\.py", name):
        return ROOT / "images/paid-gateway" / name
    if name in METER_FILES:
        return ROOT / "images/qwen-metered/model-metering" / name
    raise RuntimeError("unexpected embedded source path")


def unique_sources(pairs: list[tuple[str, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in pairs:
        if name in result:
            raise RuntimeError("duplicate source path")
        source_path(name)
        if not isinstance(value, str):
            raise RuntimeError("invalid source bundle")
        result[name] = value
    return result


def baseline_sources() -> dict[str, str]:
    raw = subprocess.check_output(["git", "show", "v0.0.5:tinfoil-config.yml"], cwd=ROOT)
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA256:
        raise RuntimeError("v0.0.5 runtime does not match its published digest")
    bundles = re.findall(r'base64\.b64decode\("""\s*([A-Za-z0-9+/=\s]+)"""\)\)', raw.decode())
    if len(bundles) != 2:
        raise RuntimeError("expected exactly two v0.0.5 source bundles")
    files: dict[str, str] = {}
    for bundle in bundles:
        data = json.loads(
            base64.b64decode("".join(bundle.split()), validate=True),
            object_pairs_hook=unique_sources,
        )
        if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
            raise RuntimeError("invalid source bundle")
        if files.keys() & data.keys():
            raise RuntimeError("duplicate source path")
        files.update(data)
    return files


def current_sources() -> dict[str, str]:
    paths = {
        "paid_gateway/" + path.name: path
        for path in (ROOT / "images/paid-gateway/paid_gateway").glob("*.py")
    }
    for name in METER_FILES:
        path = source_path(name)
        if path.exists() or path.is_symlink():
            paths[name] = path
    result = {}
    for name, path in paths.items():
        source_path(name)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("source must be a regular file")
        result[name] = path.read_bytes().decode("utf-8")
    return result


def main() -> None:
    baseline, current = baseline_sources(), current_sources()
    for name in sorted(baseline.keys() | current.keys()):
        old, new = baseline.get(name, ""), current.get(name, "")
        if name not in baseline:
            print(f"# Added source: {name}")
        if name not in current:
            print(f"# Deleted source: {name}")
        before = "v0.0.5/" + name if name in baseline else "/dev/null"
        after = source_path(name).relative_to(ROOT).as_posix() if name in current else "/dev/null"
        print(
            "".join(
                difflib.unified_diff(
                    old.splitlines(True), new.splitlines(True), fromfile=before, tofile=after
                )
            ),
            end="",
        )


if __name__ == "__main__":
    main()
