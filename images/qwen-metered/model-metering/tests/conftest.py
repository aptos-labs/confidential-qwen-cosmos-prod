"""CPU-only fixtures. Runtime doubles validate the adapter, not vLLM itself."""

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ARTIFACT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTIFACT))
import install  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--upstream", help="read-only vllm-omni checkout at the pinned commit")


@pytest.fixture
def runtime(monkeypatch):
    @dataclass
    class EngineCoreRequest:
        prompt_token_ids: object
        mm_features: object = None
        resumable: bool = False
        prompt_embeds: object = None
        prompt_is_token_ids: object = None
        sampling_params: object = "sampling"
        pooling_params: object = None

    @dataclass
    class OmniEngineCoreRequest(EngineCoreRequest):
        additional_information: object = None
        model_intermediate_buffer: object = None

    @dataclass
    class PlaceholderRange:
        offset: int
        length: int
        is_embed: object = None

    @dataclass
    class MultiModalFeatureSpec:
        modality: str
        mm_position: object
        data: object = None
        identifier: object = None

    for name, attrs in {
        "vllm": {},
        "vllm.v1": {},
        "vllm.multimodal": {},
        "vllm_omni": {},
        "vllm.v1.engine": {"EngineCoreRequest": EngineCoreRequest},
        "vllm_omni.engine": {"OmniEngineCoreRequest": OmniEngineCoreRequest},
        "vllm.multimodal.inputs": {
            "MultiModalFeatureSpec": MultiModalFeatureSpec,
            "PlaceholderRange": PlaceholderRange,
        },
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["Qwen3OmniMoeForConditionalGeneration"],
            model_stage="thinker",
            is_encoder_decoder=False,
            session_mode="turn",
        )
    )
    return SimpleNamespace(
        Request=EngineCoreRequest,
        OmniRequest=OmniEngineCoreRequest,
        Feature=MultiModalFeatureSpec,
        Range=PlaceholderRange,
        config=config,
    )


@pytest.fixture(scope="session")
def upstream(pytestconfig):
    value = pytestconfig.getoption("--upstream")
    if not value:
        pytest.skip("pass --upstream to validate the pinned three-file patch/AST fixtures")
    root = Path(value).resolve()
    # Exact source hashes are stronger than trusting the checkout's tag label.
    for name in install.FILES:
        assert (root / "vllm_omni" / name).is_file()
    return root


@pytest.fixture
def package(tmp_path, upstream):
    root = tmp_path / "vllm_omni"
    for name in install.FILES:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream / "vllm_omni" / name, destination)
    return root


@pytest.fixture
def patched(package):
    install.install(package)
    return package
