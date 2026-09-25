import ast
import asyncio
import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

import input_meter as meter
import install
import msgspec
import pytest
import torch

ARTIFACT = Path(__file__).resolve().parents[1]


def unavailable(value, reason):
    assert value.to_dict() == {
        "schema": "qwen3-omni-input/v1",
        "status": "unavailable",
        "reason": reason,
    }


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_modalities(modality):
    result = meter.count_input_positions([object()] * 10, [meter.FeaturePosition(modality, 2, 4)])
    assert result.total_input == 10 and result.text_input == 6
    assert getattr(result, modality + "_input") == 4


def test_interleaving_and_text():
    # Shared video/audio ranges are valid when actual embedding masks are disjoint.
    result = meter.count_input_positions(
        [object()] * 10,
        [
            meter.FeaturePosition("video", 2, 6, (True, False, True, False, False, False)),
            meter.FeaturePosition("audio", 2, 6, (False, True, False, True, False, True)),
        ],
    )
    assert result == meter.InputMeter(None, 10, 5, 0, 2, 3)
    assert meter.count_input_positions([object()] * 8, []) == meter.InputMeter(None, 8, 8, 0, 0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.total_input = 999


@pytest.mark.parametrize(
    "feature,reason",
    [
        (meter.FeaturePosition("wat", 0, 1), "unknown_modality"),
        (meter.FeaturePosition("image", -1, 1), "invalid_range"),
        (meter.FeaturePosition("image", True, 1), "invalid_range"),
        (meter.FeaturePosition("image", 0, 0), "invalid_range"),
        (meter.FeaturePosition("image", 9, 2), "invalid_range"),
        (meter.FeaturePosition("image", 0, 2, (True,)), "invalid_mask"),
        (meter.FeaturePosition("image", 0, 1, (1,)), "invalid_mask"),
        (meter.FeaturePosition("image", 0, 1, (False,)), "invalid_mask"),
        ({"modality": "image", "offset": 0, "length": 1}, "invalid_features"),
    ],
)
def test_malformed(feature, reason):
    unavailable(meter.count_input_positions([None] * 10, [feature]), reason)


def test_limits_overlap_and_spoof():
    feature = meter.FeaturePosition("audio", 0, 2)
    unavailable(
        meter.count_input_positions([None] * 10, [feature, feature]), "overlapping_embeddings"
    )
    unavailable(meter.count_input_positions([None], None), "invalid_features")
    unavailable(meter.count_input_positions([], []), "invalid_prompt")
    unavailable(
        meter.count_input_positions([None] * (meter.MAX_INPUT_TOKENS + 1), []), "limit_exceeded"
    )
    unavailable(
        meter.count_input_positions([None], [feature] * (meter.MAX_FEATURES + 1)), "limit_exceeded"
    )
    spoof = {meter.NAMESPACE: {"status": "available", "total_input": 123}, "timing": 1}
    protected = meter.protect_metrics(spoof, spoof[meter.NAMESPACE])
    assert protected["timing"] == 1
    assert protected[meter.NAMESPACE] == meter.InputMeter().to_dict()
    assert spoof[meter.NAMESPACE]["total_input"] == 123


class NeverRead:
    def __repr__(self):
        raise AssertionError("payload/token value read")

    def __iter__(self):
        raise AssertionError("payload iterated")


@pytest.mark.parametrize("omni", [False, True])
def test_adapter(runtime, omni):
    cls = runtime.OmniRequest if omni else runtime.Request
    request = cls([NeverRead()] * 8)
    if omni:
        request.additional_information = NeverRead()
    assert meter.capture_input_meter(request, runtime.config).text_input == 8
    request.mm_features = [
        runtime.Feature(
            "video", runtime.Range(1, 4, torch.tensor([True, False, True, False])), NeverRead()
        )
    ]
    snapshot = meter.capture_input_meter(request, runtime.config)
    assert snapshot == meter.InputMeter(None, 8, 6, 0, 2, 0)
    request.prompt_token_ids.clear()
    assert snapshot.total_input == 8


@pytest.mark.parametrize(
    "key,value",
    [
        ("resumable", True),
        ("prompt_embeds", object()),
        ("pooling_params", object()),
        ("sampling_params", None),
        ("prompt_is_token_ids", [True]),
        ("encoder_inputs", {}),
    ],
)
def test_adapter_rejects_paths(runtime, key, value):
    request = runtime.Request([None] * 8)
    setattr(request, key, value)
    unavailable(meter.capture_input_meter(request, runtime.config), "unsupported_path")


def test_adapter_type_and_model(runtime):
    unavailable(
        meter.capture_input_meter({"prompt_token_ids": [1]}, runtime.config), "unsupported_path"
    )
    request = runtime.Request([None])
    runtime.config.model_config.model_stage = "talker"
    unavailable(meter.capture_input_meter(request, runtime.config), "unsupported_model")
    runtime.config.model_config.model_stage = "thinker"
    runtime.config.model_config.architectures = ["Other"]
    unavailable(meter.capture_input_meter(request, runtime.config), "unsupported_model")


@pytest.mark.parametrize(
    "mask", [[True], torch.tensor([1]), torch.tensor([[True]]), torch.tensor([True, False])]
)
def test_adapter_masks(runtime, mask):
    request = runtime.Request([None] * 8, [runtime.Feature("image", runtime.Range(0, 1, mask))])
    unavailable(meter.capture_input_meter(request, runtime.config), "invalid_mask")


def test_installer_and_exact_patch(package):
    before = {p: p.read_bytes() for p in package.rglob("*.py")}
    assert "no writes" in install.install(package, check=True)
    assert before == {p: p.read_bytes() for p in package.rglob("*.py")}
    subprocess.run(
        ["git", "apply", "--check", str(ARTIFACT / "vllm-omni-v0.28.0.patch")],
        cwd=package.parent,
        check=True,
    )
    assert install.install(package).startswith("installed")
    assert install.install(package).startswith("already installed")
    (package / install.HELPER).write_text("# drift\n")
    with pytest.raises(install.InstallError):
        install.install(package)


@pytest.mark.parametrize("target", [*install.FILES, install.HELPER])
def test_drift_no_partial_writes(package, target):
    path = package / target
    path.write_bytes((path.read_bytes() if path.exists() else b"") + b"\n# drift\n")
    before = {p: p.read_bytes() for p in package.rglob("*") if p.is_file()}
    with pytest.raises(install.InstallError):
        install.install(package)
    assert before == {p: p.read_bytes() for p in package.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["hash", "replacement", "syntax", "helper"])
def test_artifact_preflight(package, tmp_path, damage):
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact, ignore=shutil.ignore_patterns("__pycache__", "tests"))
    manifest = json.loads((artifact / "manifest.json").read_text())
    spec = manifest["files"][install.FILES[-1]]
    if damage == "hash":
        spec["patched_sha256"] = "0" * 64
    elif damage == "replacement":
        spec["replacements"][-1]["old"] = "not present"
    elif damage == "syntax":
        helper = b"invalid syntax !\n"
        (artifact / install.HELPER).write_bytes(helper)
        manifest["helper_sha256"] = install.digest(helper)
    else:
        (artifact / install.HELPER).write_text("# changed\n")
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    before = {p: p.read_bytes() for p in package.rglob("*.py")}
    with pytest.raises(install.InstallError):
        install.install(package, artifact=artifact)
    assert before == {p: p.read_bytes() for p in package.rglob("*.py")}


def nodes(path, name):
    return next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


def harness(patched):
    module = ModuleType("meter_message_harness")
    sys.modules[module.__name__] = module
    module.__dict__.update(
        msgspec=msgspec,
        Literal=Literal,
        InputMeter=meter.InputMeter,
        OmniRequestOutput=Any,
        StageRequestMetrics=Any,
    )
    tree = ast.Module(
        body=[
            nodes(patched / "engine/messages.py", name)
            for name in ("EngineQueueMessage", "OutputMessage")
        ],
        type_ignores=[],
    )
    exec(compile(tree, "message_fixture", "exec"), module.__dict__)
    return module.OutputMessage


@pytest.mark.parametrize("codec", [msgspec.json, msgspec.msgpack])
def test_message_roundtrip(patched, codec):
    cls = harness(patched)
    snapshot = meter.InputMeter(None, 8, 4, 1, 2, 1)
    message = cls(
        request_id="synthetic",
        stage_id=2,
        engine_outputs=None,
        finished=True,
        input_meter=snapshot,
    )
    restored = codec.decode(codec.encode(message), type=cls)
    assert type(restored.input_meter) is meter.InputMeter
    assert restored.input_meter == snapshot
    default = cls(request_id="synthetic", stage_id=2, engine_outputs=None, finished=True)
    assert codec.decode(codec.encode(default), type=cls).input_meter is None


@pytest.mark.parametrize("stage_id,kind", [(0, "text"), (2, "audio")])
def test_final_stage_propagation_fixture(patched, stage_id, kind):
    route = nodes(patched / "engine/orchestrator.py", "_route_output")
    # Execute the real routing prefix through the frontend put, excluding
    # downstream forwarding. No duplicated implementation of the meter seam.
    stop = next(
        i
        for i, n in enumerate(route.body)
        if isinstance(n, ast.If) and "self._pd_pair" in ast.unparse(n.test)
    )
    route.body = route.body[:stop]
    cls = harness(patched)
    scope = {
        "Any": Any,
        "OrchestratorRequestState": Any,
        "OutputMessage": cls,
        "StageMetricsMessage": SimpleNamespace,
    }
    exec(compile(ast.Module(body=[route], type_ignores=[]), "route_fixture", "exec"), scope)
    snapshot = meter.InputMeter(None, 8, 4, 1, 2, 1)
    state = SimpleNamespace(
        input_meter=snapshot,
        stage_submit_ts={},
        streaming=SimpleNamespace(enabled=False),
        final_stage_id=stage_id,
        final_output_stage_ids={stage_id},
        finished_final_output_stage_ids=set(),
    )
    queue = asyncio.Queue()
    self = SimpleNamespace(
        _cfg_tracker=SimpleNamespace(is_companion=lambda _: False),
        stage_pools={stage_id: SimpleNamespace(final_output=True)},
        output_async_queue=queue,
        _is_duplex_session_request=lambda _: False,
    )
    output = SimpleNamespace(request_id="synthetic", finished=True, final_output_type=kind)
    asyncio.run(scope["_route_output"](self, stage_id, 0, output, state, None))
    result = queue.get_nowait()
    assert result.input_meter is snapshot
    method = nodes(patched / "entrypoints/omni_base.py", "_process_single_result")
    assert isinstance(method.body[-2], ast.Assign)
    assert (
        ast.unparse(method.body[-2].value)
        == "protect_metrics(response_metrics, result.input_meter)"
    )
    scope = {
        "response_metrics": {meter.NAMESPACE: {"total_input": 999}},
        "result": result,
        "protect_metrics": meter.protect_metrics,
    }
    exec(
        compile(ast.Module(body=[method.body[-2]], type_ignores=[]), "metrics_fixture", "exec"),
        scope,
    )
    assert scope["response_metrics"][meter.NAMESPACE] == snapshot.to_dict()
    add = nodes(patched / "engine/orchestrator.py", "_handle_add_request")
    captures = [
        n
        for n in ast.walk(add)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "capture_input_meter"
    ]
    assert len(captures) == 1
    assert (
        ast.unparse(captures[0])
        == "capture_input_meter(prompt, self.stage_pools[stage_id].stage_vllm_config)"
    )
    abort = nodes(patched / "engine/orchestrator.py", "_abort_request_ids")
    assert "input_meter=req_state.input_meter" in ast.unparse(abort)
