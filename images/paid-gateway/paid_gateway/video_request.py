"""Validate bounded native Cosmos3-Super form fields without rewriting multipart bytes.

The gateway forwards the caller's exact ``multipart/form-data`` body to vLLM-Omni's
``/v1/videos/sync`` endpoint for ``cosmos3-super``; this module only admits or
rejects it. Accepted text fields are ``model``, ``prompt``, ``negative_prompt``,
``size`` (required, exactly ``832x480`` or ``1280x720``), ``num_frames`` (required,
1..189), ``fps`` (24 only), ``num_inference_steps``, ``guidance_scale``,
``flow_shift``, ``seed``, ``max_sequence_length``, ``generate_sound`` (``true`` or
``false``), ``sound_duration`` (seconds, only with ``generate_sound=true``, >0 and at most
the video duration ``num_frames / 24``) and ``extra_params``.
``extra_params`` is a strict JSON object limited to the two template booleans and
``guardrails``, which may only restate ``true``: guardrails are never disabled per
request. At most one PNG/JPEG ``input_reference`` file enables image-to-video.
Every other field, file part, or value fails closed.
"""

from __future__ import annotations

import math
import re
from email import policy
from email.parser import BytesHeaderParser
from typing import Any

from .client import GatewayError, strict_json
from .protocol import VIDEO

SIZES = frozenset({"832x480", "1280x720"})
MAX_FRAMES = 189
FPS = 24
EXTRA_PARAMS = frozenset({"use_resolution_template", "use_duration_template", "guardrails"})
FIELDS = frozenset(
    {
        "model",
        "prompt",
        "negative_prompt",
        "size",
        "num_frames",
        "fps",
        "num_inference_steps",
        "guidance_scale",
        "flow_shift",
        "seed",
        "max_sequence_length",
        "generate_sound",
        "sound_duration",
        "extra_params",
    }
)
SOUND_FLAGS = frozenset({"true", "false"})
# Plain ASCII decimals only: no whitespace, '+', '_', or non-ASCII digits that
# Python's int()/float() would silently accept but the model server might not.
_WHOLE = re.compile(r"-?[0-9]{1,20}")
_DECIMAL = re.compile(r"-?[0-9]{1,20}(?:\.[0-9]{1,20})?(?:[eE]-?[0-9]{1,3})?")


def invalid() -> None:
    raise GatewayError("invalid_request", 400)


def number(value: Any, low: float, high: float, *, whole: bool = False) -> int | float:
    try:
        if not isinstance(value, str) or not (_WHOLE if whole else _DECIMAL).fullmatch(value):
            invalid()
        parsed = int(value) if whole else float(value)
        if not math.isfinite(parsed) or not low <= parsed <= high:
            invalid()
        return parsed
    except (TypeError, ValueError, OverflowError):
        invalid()
    raise AssertionError("unreachable")


def validate_video(body: bytes, content_type: str) -> list[str]:
    try:
        header = BytesHeaderParser(policy=policy.HTTP).parsebytes(
            b"Content-Type: " + content_type.encode("ascii") + b"\r\n\r\n"
        )
        boundary = header.get_boundary()
        if (
            header.get_content_type() != "multipart/form-data"
            or not boundary
            or not 1 <= len(boundary) <= 70
            or not boundary.isascii()
        ):
            invalid()
        marker = b"--" + boundary.encode()
        if not body.startswith(marker + b"\r\n") or body.count(b"\r\n" + marker) > 128:
            invalid()
        sections = body[len(marker) + 2 :].split(b"\r\n" + marker)
        if sections[-1] not in {b"--", b"--\r\n"}:
            invalid()
        fields: dict[str, str] = {}
        files: list[str] = []
        for index, section in enumerate(sections[:-1]):
            if index:
                if not section.startswith(b"\r\n"):
                    invalid()
                section = section[2:]
            headers, separator, data = section.partition(b"\r\n\r\n")
            if not separator or len(headers) > 8192 or headers.count(b"\r\n") > 30:
                invalid()
            part = BytesHeaderParser(policy=policy.HTTP).parsebytes(headers + b"\r\n\r\n")
            if part.defects or any(
                k.lower() not in {"content-disposition", "content-type"} for k in part
            ):
                invalid()
            if (
                len(part.get_all("content-disposition", [])) != 1
                or len(part.get_all("content-type", [])) > 1
            ):
                invalid()
            if part.get_content_disposition() != "form-data":
                invalid()
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            if not isinstance(name, str) or len(name) > 128:
                invalid()
            if filename is not None:
                # Cosmos3-Super conditions on at most one reference image.
                if name != "input_reference" or files or not data:
                    invalid()
                if (
                    len(filename) > 128
                    or any(c in filename for c in ("/", "\\", "\x00", "\r", "\n"))
                    or ".." in filename
                ):
                    invalid()
                media_type = part.get_content_type()
                if media_type == "image/png":
                    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                        invalid()
                elif media_type == "image/jpeg":
                    if not data.startswith(b"\xff\xd8\xff"):
                        invalid()
                else:
                    invalid()
                files.append(name)
                continue
            if name in fields or len(data) > 65536:
                invalid()
            fields[name] = data.decode("utf-8")
        if set(fields) - FIELDS or fields.get("model") != VIDEO:
            invalid()
        if not fields.get("prompt", "").strip():
            invalid()
        if fields.get("size") not in SIZES:
            invalid()
        if "num_frames" not in fields:
            invalid()
        frames = number(fields["num_frames"], 1, MAX_FRAMES, whole=True)
        if "fps" in fields:
            number(fields["fps"], FPS, FPS, whole=True)
        if "num_inference_steps" in fields:
            number(fields["num_inference_steps"], 1, 50, whole=True)
        for key in ("guidance_scale", "flow_shift"):
            if key in fields:
                number(fields[key], 0, 32)
        if "seed" in fields:
            number(fields["seed"], -(2**63), 2**63 - 1, whole=True)
        if "max_sequence_length" in fields:
            number(fields["max_sequence_length"], 1, 4096, whole=True)
        if "generate_sound" in fields and fields["generate_sound"] not in SOUND_FLAGS:
            invalid()
        if "sound_duration" in fields:
            # Audio is muxed into the video MP4, so it may not outlast the video.
            if fields.get("generate_sound") != "true":
                invalid()
            if not 0 < number(fields["sound_duration"], 0, frames / FPS):
                invalid()
        if "extra_params" in fields:
            extra = strict_json(fields["extra_params"])
            if not isinstance(extra, dict) or set(extra) - EXTRA_PARAMS:
                invalid()
            for key in ("use_resolution_template", "use_duration_template"):
                if key in extra and type(extra[key]) is not bool:
                    invalid()
            # Guardrails are always on; a request may only restate that, never disable it.
            if "guardrails" in extra and extra["guardrails"] is not True:
                invalid()
    except (UnicodeError, TypeError, ValueError, IndexError) as exc:
        raise GatewayError("invalid_request", 400) from exc
    return ["text", "image"] if files else ["text"]
