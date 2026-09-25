# SPDX-License-Identifier: Apache-2.0
"""Dependency-free Qwen3-Omni input-position meter; no token values are read.

Only the typed runtime adapter may supply positions. This is NOT an API for
accepting caller-provided counts or multimodal metadata.
"""

from dataclasses import dataclass

NAMESPACE = "model_meter_v1"
MAX_INPUT_TOKENS = 131_072
MAX_FEATURES = 4_096
MAX_POSITION_WORK = 1_048_576
MODALITIES = ("image", "video", "audio")
REASONS = frozenset(
    {
        "missing_meter",
        "unsupported_model",
        "unsupported_path",
        "invalid_prompt",
        "invalid_features",
        "unknown_modality",
        "invalid_range",
        "invalid_mask",
        "overlapping_embeddings",
        "inconsistent_totals",
        "limit_exceeded",
    }
)


class MeterUnavailableError(ValueError):
    """Contains a fixed reason code, never request content."""


@dataclass(frozen=True)
class FeaturePosition:
    modality: str
    offset: int
    length: int
    is_embed: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class InputMeter:
    """Immutable request snapshot. Unavailable means no numeric counters at all."""

    reason: str | None = "missing_meter"
    total_input: int | None = None
    text_input: int | None = None
    image_input: int | None = None
    video_input: int | None = None
    audio_input: int | None = None

    def __post_init__(self):
        counts = (
            self.total_input,
            self.text_input,
            self.image_input,
            self.video_input,
            self.audio_input,
        )
        if self.reason is not None:
            if self.reason not in REASONS or any(value is not None for value in counts):
                raise MeterUnavailableError("inconsistent_totals")
        elif any(
            type(value) is not int or value < 0 or value > MAX_INPUT_TOKENS for value in counts
        ) or self.total_input != sum(counts[1:]):
            raise MeterUnavailableError("inconsistent_totals")

    def to_dict(self) -> dict:
        result = {
            "schema": "qwen3-omni-input/v1",
            "status": "unavailable" if self.reason else "available",
        }
        if self.reason is not None:
            result["reason"] = self.reason
        else:
            for name in ("total_input", "text_input", "image_input", "video_input", "audio_input"):
                result[name] = getattr(self, name)
        return result


def authoritative_meter(value: object) -> InputMeter:
    # Never parse a dict supplied as metrics, extra_body or additional_information.
    return value if type(value) is InputMeter else InputMeter()


def protect_metrics(metrics: object, meter: object) -> dict:
    result = dict(metrics) if isinstance(metrics, dict) else {}
    result[NAMESPACE] = authoritative_meter(meter).to_dict()
    return result


def validate_range(offset: object, length: object, total: int) -> None:
    if (
        type(offset) is not int
        or type(length) is not int
        or offset < 0
        or length <= 0
        or offset + length > total
    ):
        raise MeterUnavailableError("invalid_range")


def count_input_positions(prompt_token_ids: object, features: object) -> InputMeter:
    """Count normalized *processed* embedding positions; None features is not valid.

    Adapter maps the dependency's explicit mm_features=None (text-only) to ().
    IDs are used only for len(): sentinels, delimiters and non-embedding positions
    are text. Cache state and media payloads are deliberately not arguments.
    """
    try:
        if type(prompt_token_ids) not in (list, tuple) or not prompt_token_ids:
            raise MeterUnavailableError("invalid_prompt")
        total = len(prompt_token_ids)
        if total > MAX_INPUT_TOKENS:
            raise MeterUnavailableError("limit_exceeded")
        if type(features) not in (list, tuple):
            raise MeterUnavailableError("invalid_features")
        if len(features) > MAX_FEATURES:
            raise MeterUnavailableError("limit_exceeded")
        occupied = bytearray(total)
        counts = dict.fromkeys(MODALITIES, 0)
        work = 0
        for feature in features:
            if type(feature) is not FeaturePosition:
                raise MeterUnavailableError("invalid_features")
            if type(feature.modality) is not str or feature.modality not in MODALITIES:
                raise MeterUnavailableError("unknown_modality")
            validate_range(feature.offset, feature.length, total)
            work += feature.length
            if work > MAX_POSITION_WORK:
                raise MeterUnavailableError("limit_exceeded")
            mask = feature.is_embed
            if mask is not None and (
                type(mask) is not tuple
                or len(mask) != feature.length
                or any(type(bit) is not bool for bit in mask)
            ):
                raise MeterUnavailableError("invalid_mask")
            feature_count = 0
            for relative in range(feature.length):
                if mask is not None and not mask[relative]:
                    continue
                position = feature.offset + relative
                if occupied[position]:
                    raise MeterUnavailableError("overlapping_embeddings")
                occupied[position] = 1
                feature_count += 1
            # Empty embeddings for a declared feature are outside the validated scope.
            if not feature_count:
                raise MeterUnavailableError("invalid_mask")
            counts[feature.modality] += feature_count
        multimodal = sum(counts.values())
        if multimodal != sum(occupied) or multimodal > total:
            raise MeterUnavailableError("inconsistent_totals")
        return InputMeter(
            None, total, total - multimodal, counts["image"], counts["video"], counts["audio"]
        )
    except MeterUnavailableError as exc:
        return InputMeter(str(exc))


def capture_input_meter(request: object, stage_vllm_config: object) -> InputMeter:
    """Snapshot only a typed, processed, ordinary Qwen3-Omni thinker request.

    Imports are deferred to keep the counter/message value dependency-free, not
    to provide a fallback: missing/incompatible runtime dependencies must fail.
    Caller metrics, additional_information, feature data and token values are
    never inspected. Only CPU boolean position masks are materialized.
    """
    import torch
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm.v1.engine import EngineCoreRequest
    from vllm_omni.engine import OmniEngineCoreRequest

    if type(request) not in (EngineCoreRequest, OmniEngineCoreRequest):
        return InputMeter("unsupported_path")
    model_config = getattr(stage_vllm_config, "model_config", None)
    architectures = getattr(model_config, "architectures", None)
    if (
        type(architectures) not in (list, tuple)
        or len(architectures) != 1
        or architectures[0]
        not in (
            "Qwen3OmniMoeForConditionalGeneration",
            "Qwen3OmniMoeThinkerForConditionalGeneration",
        )
        or getattr(model_config, "model_stage", None) != "thinker"
    ):
        return InputMeter("unsupported_model")
    if (
        getattr(model_config, "is_encoder_decoder", None) is not False
        or getattr(model_config, "session_mode", None) != "turn"
        or request.resumable is not False
        or request.prompt_embeds is not None
        or request.prompt_is_token_ids is not None
        or request.pooling_params is not None
        or request.sampling_params is None
        or getattr(request, "model_intermediate_buffer", None) is not None
        or any(
            getattr(request, name, None) is not None
            for name in (
                "encoder_input",
                "encoder_inputs",
                "encoder_prompt",
                "encoder_prompt_token_ids",
            )
        )
    ):
        return InputMeter("unsupported_path")

    # None means text-only ONLY after the runtime request/path is validated.
    features = request.mm_features
    features = () if features is None else features
    try:
        tokens = request.prompt_token_ids
        if type(tokens) not in (list, tuple) or not tokens:
            raise MeterUnavailableError("invalid_prompt")
        total = len(tokens)
        if total > MAX_INPUT_TOKENS:
            raise MeterUnavailableError("limit_exceeded")
        if type(features) not in (list, tuple):
            raise MeterUnavailableError("invalid_features")
        if len(features) > MAX_FEATURES:
            raise MeterUnavailableError("limit_exceeded")
        normalized = []
        work = 0
        for feature in features:
            if type(feature) is not MultiModalFeatureSpec:
                raise MeterUnavailableError("invalid_features")
            modality = feature.modality
            if type(modality) is not str or modality not in MODALITIES:
                raise MeterUnavailableError("unknown_modality")
            position = feature.mm_position
            if type(position) is not PlaceholderRange:
                raise MeterUnavailableError("invalid_range")
            validate_range(position.offset, position.length, total)
            work += position.length
            if work > MAX_POSITION_WORK:
                raise MeterUnavailableError("limit_exceeded")
            mask = position.is_embed
            if mask is not None:
                if (
                    type(mask) is not torch.Tensor
                    or mask.dtype != torch.bool
                    or mask.device.type != "cpu"
                    or mask.layout != torch.strided
                    or mask.ndim != 1
                    or mask.shape[0] != position.length
                ):
                    raise MeterUnavailableError("invalid_mask")
                mask = tuple(mask.tolist())
            normalized.append(FeaturePosition(modality, position.offset, position.length, mask))
        return count_input_positions(tokens, normalized)
    except MeterUnavailableError as exc:
        return InputMeter(str(exc))
