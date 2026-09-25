"""Strict request policy and cumulative meters for the pinned Qwen deployment."""

from __future__ import annotations

import base64
import copy
import math
from fractions import Fraction
from typing import Any

from .client import GatewayError, integer
from .media import wav_frames

QWEN = "qwen3-omni"
VIDEO = "cosmos3-super"
INPUT_LIMIT = 131072
AUDIO_RATE = 24000
BODY_LIMIT = 64 * 1024 * 1024
OUTPUT_LIMIT = 128 * 1024 * 1024


def fail_request() -> None:
    raise GatewayError("invalid_request", 400)


def inline_media(url: Any, kind: str) -> None:
    prefixes = {
        "image": (
            "data:image/png;base64,",
            "data:image/jpeg;base64,",
            "data:image/webp;base64,",
            "data:image/gif;base64,",
        ),
        "audio": ("data:audio/wav;base64,", "data:audio/mpeg;base64,", "data:audio/mp3;base64,"),
        "video": ("data:video/mp4;base64,",),
    }
    if not isinstance(url, str) or not any(url.startswith(p) for p in prefixes[kind]):
        fail_request()
    try:
        raw = base64.b64decode(url.split(",", 1)[1], validate=True)
    except (ValueError, TypeError):
        fail_request()
    if not raw or len(raw) > BODY_LIMIT:
        fail_request()


def request_string(value: Any, *, empty: bool = False) -> None:
    if not isinstance(value, str) or (not empty and not value):
        fail_request()


def request_schema(value: Any, depth: int = 0) -> None:
    """Bound nested schemas and never permit a resolver to leave this document."""
    if depth > 32:
        fail_request()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"$ref", "$dynamicRef", "$recursiveRef", "$id"} and (
                not isinstance(item, str) or not item.startswith("#")
            ):
                fail_request()
            request_schema(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            request_schema(item, depth + 1)
    elif type(value) is float and not math.isfinite(value):
        fail_request()


def request_function(value: Any, *, definition: bool) -> str:
    allowed = (
        {"name", "description", "parameters", "strict"} if definition else {"name", "arguments"}
    )
    if not isinstance(value, dict) or set(value) - allowed:
        fail_request()
    request_string(value.get("name"))
    if definition:
        if "description" in value:
            request_string(value["description"], empty=True)
        if "strict" in value and type(value["strict"]) is not bool:
            fail_request()
        if "parameters" in value:
            if not isinstance(value["parameters"], dict):
                fail_request()
            request_schema(value["parameters"])
    else:
        request_string(value.get("arguments"), empty=True)
    return value["name"]


def request_options(value: dict[str, Any]) -> None:
    for name in ("stream", "parallel_tool_calls", "logprobs"):
        if name in value and type(value[name]) is not bool:
            fail_request()
    ranges = {
        "temperature": (0, 2),
        "top_p": (0, 1),
        "frequency_penalty": (-2, 2),
        "presence_penalty": (-2, 2),
        "repetition_penalty": (0, 2),
    }
    for name, (low, high) in ranges.items():
        if name in value:
            number = value[name]
            if type(number) not in (int, float) or not low <= number <= high:
                fail_request()
            if name in {"top_p", "repetition_penalty"} and number == 0:
                fail_request()
    for name, low, high in (
        ("top_k", -1, 2**31 - 1),
        ("seed", -(2**63), 2**63 - 1),
        ("top_logprobs", 0, 20),
    ):
        if name in value and (type(value[name]) is not int or not low <= value[name] <= high):
            fail_request()
    if "stop" in value:
        stop = value["stop"]
        if isinstance(stop, str):
            request_string(stop)
        elif not isinstance(stop, list) or not 1 <= len(stop) <= 4:
            fail_request()
        else:
            for item in stop:
                request_string(item)
    names = set()
    if "tools" in value:
        tools = value["tools"]
        if not isinstance(tools, list) or not 1 <= len(tools) <= 128:
            fail_request()
        for tool in tools:
            if (
                not isinstance(tool, dict)
                or set(tool) != {"type", "function"}
                or tool["type"] != "function"
            ):
                fail_request()
            name = request_function(tool["function"], definition=True)
            if name in names:
                fail_request()
            names.add(name)
    if "tool_choice" in value:
        choice = value["tool_choice"]
        if isinstance(choice, str):
            if choice not in {"none", "auto", "required"} or (choice != "none" and not names):
                fail_request()
        else:
            if (
                not isinstance(choice, dict)
                or set(choice) != {"type", "function"}
                or choice["type"] != "function"
                or not isinstance(choice["function"], dict)
                or set(choice["function"]) != {"name"}
            ):
                fail_request()
            request_string(choice["function"]["name"])
            if choice["function"]["name"] not in names:
                fail_request()
    if "response_format" in value:
        form = value["response_format"]
        if not isinstance(form, dict) or not isinstance(form.get("type"), str):
            fail_request()
        if form["type"] in {"text", "json_object"}:
            if set(form) != {"type"}:
                fail_request()
        elif form["type"] == "json_schema":
            schema = form.get("json_schema")
            if (
                set(form) != {"type", "json_schema"}
                or not isinstance(schema, dict)
                or set(schema) - {"name", "description", "strict", "schema"}
                or not isinstance(schema.get("schema"), dict)
            ):
                fail_request()
            request_string(schema.get("name"))
            if "description" in schema:
                request_string(schema["description"], empty=True)
            if "strict" in schema and type(schema["strict"]) is not bool:
                fail_request()
            request_schema(schema["schema"])
        else:
            fail_request()


def normalize_chat(value: Any) -> tuple[dict[str, Any], list[str], set[str]]:
    allowed = {
        "model",
        "messages",
        "modalities",
        "stream",
        "stream_options",
        "audio",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "logprobs",
        "top_logprobs",
    }
    if not isinstance(value, dict) or set(value) - allowed or value.get("model") != QWEN:
        fail_request()
    request_options(value)
    value = copy.deepcopy(value)
    if type(value.get("stream", False)) is not bool:
        fail_request()
    outputs = value.get("modalities", ["text"])
    if (
        not isinstance(outputs, list)
        or not outputs
        or any(not isinstance(x, str) or x not in {"text", "audio"} for x in outputs)
        or len(set(outputs)) != len(outputs)
    ):
        fail_request()
    value["modalities"] = outputs
    if "max_tokens" in value and "max_completion_tokens" in value:
        fail_request()
    tokens = value.pop("max_completion_tokens", value.get("max_tokens", 1024))
    if type(tokens) is not int or not 1 <= tokens <= 4096:
        fail_request()
    value["max_tokens"] = tokens
    audio = value.get("audio")
    if "audio" in value and (
        not isinstance(audio, dict)
        or set(audio) - {"format", "voice"}
        or audio.get("format", "wav") != "wav"
    ):
        fail_request()
    if audio is not None and "voice" in audio:
        request_string(audio["voice"])
    messages = value.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 256:
        fail_request()
    inputs = {"text"}
    media_count = 0
    for message in messages:
        if not isinstance(message, dict) or set(message) - {
            "role",
            "content",
            "name",
            "tool_calls",
            "tool_call_id",
            "function_call",
        }:
            fail_request()
        if not isinstance(message.get("role"), str) or message["role"] not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
            "function",
        }:
            fail_request()
        for name in ("name", "tool_call_id"):
            if name in message:
                request_string(message[name])
        if "function_call" in message:
            request_function(message["function_call"], definition=False)
        if "tool_calls" in message:
            calls = message["tool_calls"]
            if not isinstance(calls, list) or not 1 <= len(calls) <= 128:
                fail_request()
            for call in calls:
                if (
                    not isinstance(call, dict)
                    or set(call) != {"id", "type", "function"}
                    or call["type"] != "function"
                ):
                    fail_request()
                request_string(call["id"])
                request_function(call["function"], definition=False)
        content = message.get("content")
        if content is None and (message.get("tool_calls") or message.get("function_call")):
            continue
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            fail_request()
        for item in content:
            if not isinstance(item, dict):
                fail_request()
            kind = item.get("type")
            request_string(kind)
            if kind == "text":
                if set(item) != {"type", "text"} or not isinstance(item["text"], str):
                    fail_request()
                continue
            media_count += 1
            if media_count > 32:
                fail_request()
            if kind in {"image_url", "video_url", "audio_url"}:
                label = kind.split("_")[0]
                obj = item.get(kind)
                if (
                    set(item) != {"type", kind}
                    or not isinstance(obj, dict)
                    or set(obj) - {"url", "detail"}
                ):
                    fail_request()
                if "detail" in obj and (
                    label != "image"
                    or not isinstance(obj["detail"], str)
                    or obj["detail"] not in {"auto", "low", "high"}
                ):
                    fail_request()
                inline_media(obj.get("url"), label)
                inputs.add(label)
                if label == "video":
                    inputs.add("audio")
            elif kind == "input_audio":
                obj = item.get("input_audio")
                if (
                    set(item) != {"type", "input_audio"}
                    or not isinstance(obj, dict)
                    or set(obj) != {"data", "format"}
                ):
                    fail_request()
                if not isinstance(obj["format"], str) or obj["format"] not in {"wav", "mp3"}:
                    fail_request()
                request_string(obj["data"])
                inline_media(f"data:audio/{obj['format']};base64,{obj['data']}", "audio")
                inputs.add("audio")
            else:
                fail_request()
    value["return_token_ids"] = True
    value["return_stage_metrics"] = True
    options = value.get("stream_options")
    if "stream_options" in value and (
        not isinstance(options, dict) or set(options) - {"include_usage", "continuous_usage_stats"}
    ):
        fail_request()
    if options is not None and any(type(v) is not bool for v in options.values()):
        fail_request()
    if value.get("stream", False):
        value["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
    else:
        if options is not None:
            fail_request()
        value.pop("stream_options", None)
    return value, sorted(inputs), set(outputs)


def cleaned(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: cleaned(v)
            for k, v in value.items()
            if k
            not in {
                "metrics",
                "token_ids",
                "prompt_token_ids",
                "prompt_text",
                "prompt_logprobs",
                "kv_transfer_params",
                "ec_transfer_params",
                "routed_experts",
            }
        }
    if isinstance(value, list):
        return [cleaned(v) for v in value]
    return value


class UsageOnlyMeter:
    """Hard gateway ceilings when billing admission is absent; no billing records."""

    def __init__(self, outputs: set[str]) -> None:
        self.outputs = outputs
        self.inputs: dict[str, int] | None = None
        self.text_tokens = 0
        self.frames = 0
        self.format: tuple[int, int, int] | None = None
        self.audio_seen = False

    def input_snapshot(self, response: dict[str, Any]) -> None:
        metrics = response.get("metrics")
        if not isinstance(metrics, dict):
            raise GatewayError("meter_unavailable")
        data = metrics.get("model_meter_v1")
        names = {"total_input", "text_input", "image_input", "video_input", "audio_input"}
        if (
            not isinstance(data, dict)
            or set(data) != names | {"schema", "status"}
            or data.get("schema") != "qwen3-omni-input/v1"
            or data.get("status") != "available"
        ):
            raise GatewayError("meter_unavailable")
        counts = {name: integer(data[name], 0, INPUT_LIMIT) for name in names}
        if counts["total_input"] != sum(counts[n] for n in names - {"total_input"}):
            raise GatewayError("meter_unavailable")
        if self.inputs is not None and self.inputs != counts:
            raise GatewayError("meter_unavailable")
        self.inputs = counts

    def audio(self, encoded: Any) -> None:
        if not isinstance(encoded, str):
            raise GatewayError("meter_unavailable")
        frames, rate, channels, width = wav_frames(encoded)
        shape = rate, channels, width
        if shape != (AUDIO_RATE, 1, 2) or (self.format is not None and self.format != shape):
            raise GatewayError("meter_unavailable")
        self.format = shape
        self.frames += frames
        self.audio_seen = True
        if self.frames > AUDIO_RATE * 120:
            raise GatewayError("output_limit")

    def text(self, choice: dict[str, Any], visible: bool) -> None:
        ids = choice.get("token_ids")
        if ids is None:
            if visible:
                raise GatewayError("meter_unavailable")
            return
        if (
            not isinstance(ids, list)
            or (visible and not ids)
            or any(type(v) is not int or v < 0 for v in ids)
        ):
            raise GatewayError("meter_unavailable")
        self.text_tokens += len(ids)
        if self.text_tokens > 4096:
            raise GatewayError("output_limit")

    def meter(self) -> dict[str, Any]:
        raise GatewayError("billing_unavailable")

    def ensure_complete(self) -> None:
        if self.inputs is None or ("audio" in self.outputs and not self.audio_seen):
            raise GatewayError("meter_unavailable")


class QwenMeter:
    def __init__(self, maximum: dict[str, Any], outputs: set[str]) -> None:
        try:
            if (
                maximum["model"] != QWEN
                or maximum["meter"]["kind"] != "tokens_and_generated_audio"
            ):
                raise ValueError()
            limits = maximum["meter"]["usage"]
            self.limits = limits["tokens"]
            self.audio_limit = limits["generated_audio"]
            for name in (
                "total_input",
                "text_input",
                "image_input",
                "video_input",
                "audio_input",
                "text_output",
            ):
                integer(self.limits[name])
            if "audio" in outputs:
                if (
                    self.audio_limit["kind"] != "samples"
                    or self.audio_limit["duration"]["sample_rate"] != AUDIO_RATE
                ):
                    raise ValueError()
                integer(self.audio_limit["duration"]["sample_count"], 1)
        except (KeyError, TypeError, ValueError) as exc:
            raise GatewayError("billing_unavailable") from exc
        self.outputs = outputs
        self.inputs: dict[str, int] | None = None
        self.text_tokens = 0
        self.frames = 0
        self.format: tuple[int, int, int] | None = None
        self.audio_seen = False

    def input_snapshot(self, response: dict[str, Any]) -> None:
        metrics = response.get("metrics")
        if not isinstance(metrics, dict):
            raise GatewayError("meter_unavailable")
        data = metrics.get("model_meter_v1")
        names = {"total_input", "text_input", "image_input", "video_input", "audio_input"}
        if (
            not isinstance(data, dict)
            or set(data) != names | {"schema", "status"}
            or data.get("schema") != "qwen3-omni-input/v1"
            or data.get("status") != "available"
        ):
            raise GatewayError("meter_unavailable")
        counts = {name: integer(data[name], 0, INPUT_LIMIT) for name in names}
        if counts["total_input"] != sum(counts[n] for n in names - {"total_input"}):
            raise GatewayError("meter_unavailable")
        if self.inputs is not None and self.inputs != counts:
            raise GatewayError("meter_unavailable")
        if any(counts[n] > self.limits[n] for n in names):
            raise GatewayError("output_limit")
        self.inputs = counts

    def audio(self, encoded: Any) -> None:
        if not isinstance(encoded, str):
            raise GatewayError("meter_unavailable")
        frames, rate, channels, width = wav_frames(encoded)
        shape = rate, channels, width
        if shape != (AUDIO_RATE, 1, 2) or (self.format is not None and self.format != shape):
            raise GatewayError("meter_unavailable")
        self.format = shape
        self.frames += frames
        self.audio_seen = True
        if self.frames > min(self.audio_limit["duration"]["sample_count"], AUDIO_RATE * 120):
            raise GatewayError("output_limit")

    def text(self, choice: dict[str, Any], visible: bool) -> None:
        ids = choice.get("token_ids")
        if ids is None:
            if visible:
                raise GatewayError("meter_unavailable")
            return
        if (
            not isinstance(ids, list)
            or (visible and not ids)
            or any(type(v) is not int or v < 0 for v in ids)
        ):
            raise GatewayError("meter_unavailable")
        self.text_tokens += len(ids)
        if self.text_tokens > min(self.limits["text_output"], 4096):
            raise GatewayError("output_limit")

    def meter(self) -> dict[str, Any]:
        if self.inputs is None:
            raise GatewayError("meter_unavailable")
        tokens = {
            **self.inputs,
            "total_output": self.text_tokens,
            "text_output": self.text_tokens,
            "audio_output": 0,
        }
        audio: dict[str, Any] = {"kind": "absent"}
        if self.frames:
            audio = {
                "kind": "samples",
                "duration": {"sample_count": self.frames, "sample_rate": AUDIO_RATE},
            }
        return {
            "model": QWEN,
            "meter": {
                "kind": "tokens_and_generated_audio",
                "usage": {"tokens": tokens, "generated_audio": audio},
            },
        }

    def ensure_complete(self) -> None:
        if self.inputs is None or ("audio" in self.outputs and not self.audio_seen):
            raise GatewayError("meter_unavailable")


def video_meter(duration: Fraction, maximum: dict[str, Any]) -> dict[str, Any]:
    try:
        if maximum["model"] != VIDEO or maximum["meter"]["kind"] != "generated_video_seconds":
            raise ValueError()
        cap = maximum["meter"]["usage"]
        limit = Fraction(integer(cap["numerator"], 1), integer(cap["denominator"], 1))
        if duration <= 0 or duration > min(limit, 15):
            raise GatewayError("output_limit")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise GatewayError("billing_unavailable") from exc
    return {
        "model": VIDEO,
        "meter": {
            "kind": "generated_video_seconds",
            "usage": {"numerator": duration.numerator, "denominator": duration.denominator},
        },
    }
