"""Offline script/packaging guards; actual image execution is a separate CI gate."""

import base64
import hashlib
import io
import json
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import smoke_image  # noqa: E402
import source_delta  # noqa: E402

DIGEST = "sha256:" + "a" * 64


def baseline_bytes(*values):
    return "\n".join(
        'json.loads(base64.b64decode("""'
        + base64.b64encode((v if isinstance(v, str) else json.dumps(v)).encode()).decode()
        + '"""))'
        for v in values
    ).encode()


def mock_baseline(monkeypatch, *values):
    raw = baseline_bytes(*values)
    monkeypatch.setattr(source_delta, "BASELINE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(source_delta, "read_baseline", lambda: raw)


def test_source_inventory_includes_added_deleted_changed_and_empty_files(tmp_path, monkeypatch):
    monkeypatch.setattr(source_delta, "ROOT", tmp_path)
    mock_baseline(
        monkeypatch,
        {"paid_gateway/old.py": "old\n", "paid_gateway/shared.py": "before\n"},
        {"install.py": "install\n"},
    )
    for name, text in {
        "paid_gateway/shared.py": "after\n",
        "paid_gateway/new.py": "raise AssertionError('data, not executable')\n",
        "paid_gateway/empty.py": "",
        "install.py": "install\n",
    }.items():
        path = source_delta.source_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    output = io.StringIO()
    with redirect_stdout(output):
        source_delta.main()
    report = output.getvalue()
    assert "# Added source: paid_gateway/new.py" in report
    assert "# Added source: paid_gateway/empty.py" in report
    assert "# Deleted source: paid_gateway/old.py" in report
    assert "-before" in report and "+after" in report
    assert "+raise AssertionError('data, not executable')" in report
    assert "install.py" not in report


@pytest.mark.parametrize(
    "name",
    [
        "../x.py",
        "/x.py",
        "paid_gateway/../x.py",
        "paid_gateway/sub/x.py",
        "paid_gateway/x.txt",
        "paid_gateway\\x.py",
        "paid_gateway//x.py",
        "unknown.py",
    ],
)
def test_unexpected_baseline_paths_are_rejected(monkeypatch, name):
    mock_baseline(monkeypatch, {name: "source"}, {})
    with pytest.raises(RuntimeError, match="unexpected embedded source path"):
        source_delta.baseline_sources()


@pytest.mark.parametrize(
    "values",
    [
        ({"paid_gateway/x.py": "one"}, {"paid_gateway/x.py": "two"}),
        ('{"paid_gateway/x.py":"one","paid_gateway/x.py":"two"}', {}),
    ],
)
def test_duplicate_baseline_paths_are_rejected(monkeypatch, values):
    mock_baseline(monkeypatch, *values)
    with pytest.raises(RuntimeError, match="duplicate source path"):
        source_delta.baseline_sources()


@pytest.mark.parametrize("values", [({},), ({}, {}, {}), ({"install.py": 1}, {}), ([], {})])
def test_invalid_bundle_count_or_type_is_rejected(monkeypatch, values):
    mock_baseline(monkeypatch, *values)
    with pytest.raises(RuntimeError):
        source_delta.baseline_sources()


def test_vendored_baseline_matches_its_published_digest():
    raw = source_delta.read_baseline()
    assert hashlib.sha256(raw).hexdigest() == source_delta.BASELINE_SHA256
    assert len(source_delta.baseline_sources()) > 0


def test_baseline_hash_cannot_be_substituted(monkeypatch):
    mock_baseline(monkeypatch, {}, {})
    monkeypatch.setattr(source_delta, "BASELINE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="published digest"):
        source_delta.baseline_sources()


def test_source_symlinks_are_not_followed(tmp_path, monkeypatch):
    monkeypatch.setattr(source_delta, "ROOT", tmp_path)
    target = tmp_path / "outside.txt"
    target.write_text("not runtime source", encoding="utf-8")
    source = source_delta.source_path("paid_gateway/link.py")
    source.parent.mkdir(parents=True)
    source.symlink_to(target)
    with pytest.raises(RuntimeError, match="regular file"):
        source_delta.current_sources()


@pytest.mark.parametrize("kind", ["gateway", "qwen"])
def test_release_reference_requires_matching_package_and_digest(kind):
    expected = smoke_image.PACKAGES[kind] + "@" + DIGEST
    assert smoke_image.image_ref_allowed(kind, expected, False)
    assert smoke_image.image_ref_allowed(kind, expected, True)
    other = "qwen" if kind == "gateway" else "gateway"
    assert not smoke_image.image_ref_allowed(
        kind, smoke_image.PACKAGES[other] + "@" + DIGEST, True
    )
    assert not smoke_image.image_ref_allowed(kind, smoke_image.PACKAGES[kind] + ":latest", False)


@pytest.mark.parametrize("image", ["gateway-test:smoke", "a", "local/gateway:v1.2", "image:TAG"])
def test_local_tags_require_explicit_local_mode(image):
    assert smoke_image.image_ref_allowed("gateway", image, True)
    assert not smoke_image.image_ref_allowed("gateway", image, False)


@pytest.mark.parametrize(
    "image",
    [
        "",
        "--privileged",
        "-v",
        "../image",
        "/image",
        "name with spaces",
        "image\n",
        "image;command",
        "$(command)",
        "repo//image",
        "UPPER:tag",
        "a" * 256,
        "repo@sha256:abc",
        "repo@sha256:" + "A" * 64,
    ],
)
def test_invalid_local_refs_never_reach_docker(monkeypatch, image):
    assert not smoke_image.image_ref_allowed("gateway", image, True)
    monkeypatch.setattr(sys, "argv", ["smoke_image.py", "gateway", "--local", "--", image])
    with mock.patch.object(smoke_image.subprocess, "check_output") as inspect:
        with pytest.raises(SystemExit) as error:
            smoke_image.main()
        assert error.value.code == 2
        inspect.assert_not_called()


@pytest.mark.parametrize("kind", ["gateway", "qwen"])
def test_smoke_container_is_readonly_tokenless_and_cleans_up(kind):
    image = smoke_image.PACKAGES[kind] + "@" + DIGEST
    with mock.patch.object(smoke_image.subprocess, "run") as run:
        smoke_image.run_image(kind, image, "installed" if kind == "qwen" else "valid")
    command = run.call_args_list[0].args[0]
    assert command[:3] == ["docker", "run", "--rm"]
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert "--privileged" not in command and "--gpus" not in command
    expected_python = "/app/.venv/bin/python3" if kind == "gateway" else "python3"
    assert command[command.index("--entrypoint") + 1] == expected_python
    assert command[-3] == image
    assert run.call_args_list[0].kwargs["check"] is True
    assert run.call_args_list[0].kwargs["timeout"] == 210
    assert run.call_args_list[1].args[0][:3] == ["docker", "rm", "--force"]
    assert run.call_args_list[1].args[0][3] == command[command.index("--name") + 1]


def test_smoke_failure_remains_failure_after_cleanup():
    failure = subprocess.CalledProcessError(9, ["docker", "run"])
    with mock.patch.object(smoke_image.subprocess, "run", side_effect=[failure, None]) as run:
        with pytest.raises(subprocess.CalledProcessError):
            smoke_image.run_image("gateway", "local:test", "valid")
        assert run.call_count == 2


def test_gateway_smoke_exercises_immutable_startup_and_argument_environment_guards(monkeypatch):
    image = smoke_image.PACKAGES["gateway"] + "@" + DIGEST
    info = [
        {
            "Architecture": "amd64",
            "Os": "linux",
            "Config": {
                "Entrypoint": ["/opt/ccs-gateway/entrypoint.sh"],
                "Cmd": [],
            },
        }
    ]
    monkeypatch.setattr(sys, "argv", ["smoke_image.py", "gateway", image])
    with (
        mock.patch.object(smoke_image.subprocess, "check_output", return_value=json.dumps(info)),
        mock.patch.object(smoke_image, "run_image") as run,
    ):
        smoke_image.main()
    assert [call.args[2] for call in run.call_args_list] == [
        "missing",
        "short",
        "whitespace",
        "hooks",
        "arguments",
        "overrides",
        "valid",
    ]
    assert "PYTHONHOME='/tmp/untrusted'" in smoke_image.GATEWAY
    assert "does not accept arguments" in smoke_image.GATEWAY
    compile(smoke_image.GATEWAY, "gateway-smoke", "exec")
    compile(smoke_image.QWEN, "qwen-smoke", "exec")


def test_gateway_build_and_entrypoint_use_application_interpreter():
    root = ROOT / "images/paid-gateway"
    entrypoint = (root / "entrypoint.sh").read_text(encoding="utf-8")
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "PATH=/app/.venv/bin:/usr/local/bin:/usr/bin:/bin" in entrypoint
    assert 'exec /app/.venv/bin/python3 -m paid_gateway "$@"' in entrypoint
    assert "unset PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP" in entrypoint
    assert "umask 077" in entrypoint
    assert "--system-site-packages" not in dockerfile
    assert "/app/.venv/bin/python3 -m pip install" in dockerfile
    assert "/app/.venv/bin/python3 -m pytest" in dockerfile
    assert dockerfile.rstrip().endswith("FROM runtime-base AS runtime")
    subprocess.run(["bash", "-n", str(root / "entrypoint.sh")], check=True)


def test_policy_job_executes_root_tests_without_registry_privileges():
    workflow = yaml.load(
        (ROOT / ".github/workflows/validate-runtime-images.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["policy"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job.get("permissions", workflow["permissions"]) == {"contents": "read"}
    serialized = json.dumps(job)
    assert "-m pytest -q tests" in serialized
    assert "pytest==9.1.1" in serialized and "PyYAML==6.0.3" in serialized
    assert "secrets." not in serialized and "docker" not in serialized


def test_all_workflow_shell_and_python_blocks_parse():
    shell_count = python_count = 0
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                if "run" not in step:
                    continue
                script = step["run"]
                subprocess.run(["bash", "-n"], input=script, text=True, check=True)
                shell_count += 1
                for block in re.findall(r"python3 - <<'PY'\n(.*?)\nPY(?:\n|$)", script, re.DOTALL):
                    compile(block, str(path), "exec")
                    python_count += 1
    assert shell_count >= 16
    assert python_count >= 7


@pytest.mark.parametrize(
    "context,directories",
    [
        ("paid-gateway", "/opt/ccs-gateway /opt/ccs-gateway/paid_gateway"),
        ("qwen-metered", "/opt/model-metering"),
    ],
)
def test_source_directories_remain_searchable_without_dac_capabilities(context, directories):
    dockerfile = (ROOT / "images" / context / "Dockerfile").read_text(encoding="utf-8")
    fix = "RUN chmod 0555 " + directories
    assert dockerfile.index("COPY --chmod=0444") < dockerfile.index(fix)
    assert dockerfile.index(fix) < dockerfile.index("FROM runtime-base AS test")
    program = smoke_image.GATEWAY if context == "paid-gateway" else smoke_image.QWEN
    assert ".stat().st_mode & 0o777 == 0o555" in program


def test_tinfoil_runtime_matches_signed_extra_large_2d_v011_shape():
    document = yaml.safe_load((ROOT / "tinfoil-config.yml").read_text(encoding="utf-8"))
    assert document["cpus"] == 32
    assert document["memory"] == 524288
    assert document["gpus"] == 8
    assert len(document["models"]) == 2
    assert len(document["containers"]) == 3


def test_gateway_has_explicit_egress_for_usage_reporter():
    config = yaml.safe_load((ROOT / "tinfoil-config.yml").read_text())
    assert config["networks"]["gateway-egress"] == {
        "egress": "allowlist",
        "allow": ["api.inference.aptoslabs.com"],
    }
    gateway = next(container for container in config["containers"] if container["name"] == "litellm")
    assert gateway["networks"] == ["models", "gateway-egress"]
    assert all(
        "gateway-egress" not in container.get("networks", [])
        for container in config["containers"]
        if container["name"] != "litellm"
    )
