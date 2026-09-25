"""Read-only image staging, without GPUs or modifications to installed packages."""

import stat

import install
import pytest


def snapshot(root):
    return {
        str(path.relative_to(root)): (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in root.rglob("*")
        if path.is_file()
    }


def test_stage_preserves_readonly_source_and_patches_private_copy(package, tmp_path):
    (package / "config.yaml").write_text("preserved: true\n")
    for path in package.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    package.chmod(0o555)
    before = snapshot(package)
    overlay = install.stage(package, parent=tmp_path)
    assert overlay.parent == tmp_path.resolve()
    assert stat.S_IMODE(overlay.stat().st_mode) == 0o700
    assert install.preflight(overlay / "vllm_omni").already_installed
    assert snapshot(package) == before
    assert (overlay / "vllm_omni/config.yaml").read_text() == "preserved: true\n"


def test_stage_never_reuses_an_existing_overlay(package, tmp_path):
    first = install.stage(package, parent=tmp_path)
    (first / "vllm_omni/input_meter.py").write_text("tampered\n")
    second = install.stage(package, parent=tmp_path)
    assert first != second
    assert install.preflight(second / "vllm_omni").already_installed


def test_stage_discards_bytecode(package, tmp_path):
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "input_meter.cpython-312.pyc").write_bytes(b"stale")
    (package / "module.pyc").write_bytes(b"stale")
    overlay = install.stage(package, parent=tmp_path)
    assert not list(overlay.rglob("*.pyc"))
    assert not list(overlay.rglob("__pycache__"))


def test_stage_rejects_drift_before_allocating_overlay(package, tmp_path):
    (package / install.FILES[0]).write_text("drift")
    with pytest.raises(install.InstallError, match="source drift"):
        install.stage(package, parent=tmp_path)
    assert not list(tmp_path.glob("ccs-metered-*"))


def test_stage_rejects_symlinks_outside_the_patched_files(package, tmp_path):
    (package / "unexpected.py").symlink_to(package / install.FILES[0])
    with pytest.raises(install.InstallError, match="unsupported package entry"):
        install.stage(package, parent=tmp_path)
    assert not list(tmp_path.glob("ccs-metered-*"))


def test_failed_staging_removes_only_its_own_overlay(package, tmp_path, monkeypatch):
    sentinel = tmp_path / "ccs-metered-preserve"
    sentinel.mkdir()
    (sentinel / "keep").write_text("untouched")
    before = snapshot(package)

    def fail(*args, **kwargs):
        raise install.InstallError("synthetic failure")

    monkeypatch.setattr(install, "install", fail)
    with pytest.raises(install.InstallError, match="synthetic failure"):
        install.stage(package, parent=tmp_path)
    assert list(tmp_path.glob("ccs-metered-*")) == [sentinel]
    assert (sentinel / "keep").read_text() == "untouched"
    assert snapshot(package) == before
