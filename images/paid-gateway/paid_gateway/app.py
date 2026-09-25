"""Fail-closed paid ASGI boundary, inside the Model CVM's existing gateway."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Final

import httpx
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .client import (
    _BILLING_ROUTE_ABSENT,
    BillingClient,
    GatewayError,
    production_billing,
    strict_json,
)
from .media import MeterError, mp4_duration
from .protocol import (
    BODY_LIMIT,
    INPUT_LIMIT,
    OUTPUT_LIMIT,
    QWEN,
    VIDEO,
    QwenMeter,
    UsageOnlyMeter,
    cleaned,
    normalize_chat,
    video_meter,
)
from .video_request import validate_video

LOG = logging.getLogger(__name__)
CHAT_PATH = "/v1/chat/completions"
VIDEO_PATH = "/v1/videos/sync"
MAX_EVENT = 4 * 1024 * 1024
GUARDRAIL_ERROR_LIMIT = 64 * 1024
CONTENT_POLICY_MESSAGE = "Request was blocked by the model's safety guardrails."
PRODUCTION_MODELS: Final = frozenset({QWEN, VIDEO})


def encode(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def upstream_json(raw: bytes | str) -> dict[str, Any]:
    """Validate container types before touching model-controlled nested values."""
    try:
        value = strict_json(raw)
    except GatewayError as exc:
        raise GatewayError("upstream_failed", 502) from exc
    if not isinstance(value, dict) or value.get("error"):
        raise GatewayError("upstream_failed", 502)
    if "metrics" in value and not isinstance(value["metrics"], dict):
        raise GatewayError("meter_unavailable")
    if value.get("modality") is not None and not isinstance(value["modality"], str):
        raise GatewayError("upstream_failed", 502)
    choices = value.get("choices", [])
    if not isinstance(choices, list):
        raise GatewayError("upstream_failed", 502)
    for choice in choices:
        if not isinstance(choice, dict):
            raise GatewayError("upstream_failed", 502)
        reason = choice.get("finish_reason")
        if reason is not None and not isinstance(reason, str):
            raise GatewayError("upstream_failed", 502)
        for key in ("message", "delta"):
            if key not in choice:
                continue
            message = choice[key]
            if not isinstance(message, dict):
                raise GatewayError("upstream_failed", 502)
            for name in ("content", "reasoning", "reasoning_content", "refusal"):
                if message.get(name) is not None and not isinstance(message[name], str):
                    raise GatewayError("upstream_failed", 502)
            if message.get("audio") is not None and (
                not isinstance(message["audio"], dict)
                or any(
                    message.get(name)
                    for name in (
                        "content",
                        "reasoning",
                        "reasoning_content",
                        "tool_calls",
                        "function_call",
                        "refusal",
                    )
                )
            ):
                raise GatewayError("meter_unavailable")
            if value.get("modality") == "audio" and any(
                message.get(name)
                for name in (
                    "reasoning",
                    "reasoning_content",
                    "tool_calls",
                    "function_call",
                    "refusal",
                )
            ):
                raise GatewayError("meter_unavailable")
    return value


def visible_text(message: dict[str, Any]) -> bool:
    return any(
        message.get(name)
        for name in (
            "content",
            "reasoning",
            "reasoning_content",
            "tool_calls",
            "function_call",
            "refusal",
        )
    )


class Session:
    def __init__(self, client: BillingClient, key: str) -> None:
        self.client, self.key = client, key
        self.id = "bill_" + uuid.uuid4().hex
        self.sequence = 0
        self.attempted = False
        self.finished = False
        self.last: dict[str, Any] | None = None
        self.billing_disabled = False
        self.deadline = asyncio.get_running_loop().time() + 30

    async def admit(
        self, model: str, inputs: list[str], outputs: list[str]
    ) -> dict[str, Any] | None:
        self.attempted = True
        result = await self.client.call(
            "admit",
            {
                "request_id": self.id,
                "api_key": self.key,
                "model": model,
                "input_modalities": inputs,
                "output_modalities": outputs,
            },
        )
        if result is _BILLING_ROUTE_ABSENT:
            self.billing_disabled = True
            return None
        try:
            remaining = (
                datetime.fromisoformat(result["expires_at"]) - datetime.now(UTC)
            ).total_seconds()
            if (
                result["request_id"] != self.id
                or result["model"] != model
                or not 1 < remaining <= 86400
            ):
                raise ValueError()
            self.deadline = asyncio.get_running_loop().time() + remaining - 0.5
            if not isinstance(result["maximum"], dict):
                raise ValueError()
            return result["maximum"]
        except (KeyError, TypeError, ValueError) as exc:
            raise GatewayError("billing_unavailable") from exc

    async def checkpoint(self, meter: dict[str, Any]) -> None:
        if self.billing_disabled or meter == self.last:
            return
        sequence = self.sequence + 1
        result = await self.client.call(
            "checkpoint",
            {"request_id": self.id, "api_key": self.key, "sequence": sequence, "meter": meter},
        )
        if (
            result.get("request_id") != self.id
            or type(result.get("sequence")) is not int
            or result["sequence"] != sequence
        ):
            raise GatewayError("billing_unavailable")
        self.sequence, self.last = sequence, meter

    async def finish(self, reason: str) -> None:
        if self.finished or not self.attempted or self.billing_disabled:
            return
        self.finished = True

        async def finalize() -> None:
            try:
                async with asyncio.timeout(4):
                    await self.client.call(
                        "settle", {"request_id": self.id, "api_key": self.key, "reason": reason}
                    )
            except (GatewayError, TimeoutError):
                # Durable checkpoints and the CCS lease sweeper recover this state.
                LOG.warning(
                    "billing settlement deferred",
                    extra={"event": "ccs.gateway_settlement", "outcome": "deferred_to_expiry"},
                )

        task = asyncio.create_task(finalize())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # finalize has its own deadline; consume its result if the request disappeared.
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            raise


class Writer:
    def __init__(self, scope: Scope, receive: Receive, send: Send, request_id: str) -> None:
        self.scope, self.receive, self.send = scope, receive, send
        self.request_id, self.started = request_id, False

    async def event(self, data: bytes) -> None:
        if not self.started:
            self.started = True
            await self.send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/event-stream"),
                        (b"cache-control", b"no-cache"),
                        (b"x-request-id", self.request_id.encode()),
                    ],
                }
            )
        await self.send({"type": "http.response.body", "body": data, "more_body": True})

    async def end(self) -> None:
        await self.event(b"data: [DONE]\n\n")
        await self.send({"type": "http.response.body", "body": b"", "more_body": False})

    async def full(self, data: bytes, media_type: str) -> None:
        self.started = True
        await Response(data, media_type=media_type, headers={"X-Request-ID": self.request_id})(
            self.scope, self.receive, self.send
        )

    async def error(self, error: GatewayError) -> None:
        if error.code == "content_policy_violation":
            value = {
                "error": {
                    "type": "invalid_request_error",
                    "code": error.code,
                    "message": CONTENT_POLICY_MESSAGE,
                }
            }
        else:
            value = {
                "error": {"type": "inference_error", "code": error.code, "message": error.code}
            }
        if self.started:
            await self.event(b"data: " + encode(value) + b"\n\n")
            await self.end()
        else:
            await JSONResponse(value, status_code=error.status)(
                self.scope, self.receive, self.send
            )


async def events(response: httpx.Response) -> AsyncIterator[str]:
    buffer = bytearray()
    parts: list[str] = []
    size, total = 0, 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > OUTPUT_LIMIT:
            raise GatewayError("output_limit")
        buffer.extend(chunk)
        while b"\n" in buffer:
            line, _, rest = buffer.partition(b"\n")
            buffer = bytearray(rest)
            if len(line) > MAX_EVENT:
                raise GatewayError("output_limit")
            line = line.rstrip(b"\r")
            if not line:
                if parts:
                    yield "\n".join(parts)
                    parts, size = [], 0
            elif line.startswith(b"data:"):
                size += len(line)
                if size > MAX_EVENT:
                    raise GatewayError("output_limit")
                parts.append(line[5:].lstrip(b" ").decode("utf-8"))
        if len(buffer) > MAX_EVENT:
            raise GatewayError("output_limit")
    if buffer or parts:
        raise GatewayError("upstream_failed")


class PaidGateway:
    def __init__(
        self,
        app: ASGIApp,
        billing: BillingClient | None = None,
        upstream: httpx.AsyncClient | None = None,
    ) -> None:
        self.app = app
        self.allowed_models = PRODUCTION_MODELS
        if billing is None:
            billing = production_billing()
        self.billing = billing
        self.upstream = upstream or httpx.AsyncClient(
            timeout=httpx.Timeout(1800, connect=10), trust_env=False
        )
        self.closed = False
        self.limits = {QWEN: asyncio.Semaphore(4), VIDEO: asyncio.Semaphore(1)}
        self.targets = {
            QWEN: "http://vllm-omni-qwen:8000/v1/chat/completions",
            VIDEO: "http://vllm-omni-cosmos:8001/v1/videos/sync",
        }
        self.observability_targets = (
            "http://vllm-omni-qwen:8000",
            "http://vllm-omni-cosmos:8001",
        )

    async def close(self) -> None:
        """Close both clients once, including when one client's cleanup fails."""
        if self.closed:
            return
        self.closed = True
        try:
            await self.billing.close()
        finally:
            await self.upstream.aclose()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":

            async def lifespan_send(message: Message) -> None:
                if message["type"] in {
                    "lifespan.startup.failed",
                    "lifespan.shutdown.failed",
                    "lifespan.shutdown.complete",
                }:
                    # Uvicorn may stop the loop as soon as it sees completion.
                    # Cleanup must finish BEFORE that notification, not after.
                    await self.close()
                await send(message)

            try:
                await self.app(scope, receive, lifespan_send)
            finally:
                # Also handle a crash/cancellation without a lifespan message.
                await self.close()
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if (
            path
            in {
                "/v1/models",
                "/health/liveliness",
                "/health/readiness",
                "/healthz",
            }
            and scope["method"] == "GET"
        ):
            await self.app(scope, receive, send)
            return
        if path == "/health" and scope["method"] == "GET":
            await self.health(scope, receive, send)
            return
        if path == "/metrics" and scope["method"] == "GET":
            await self.metrics(scope, receive, send)
            return
        if path not in {CHAT_PATH, VIDEO_PATH}:
            await JSONResponse({"error": {"code": "unsupported_endpoint"}}, status_code=404)(
                scope, receive, send
            )
            return
        if scope["method"] != "POST":
            await JSONResponse(
                {"error": {"code": "method_not_allowed"}},
                status_code=405,
                headers={"Allow": "POST"},
            )(scope, receive, send)
            return
        request = Request(scope, receive)
        model = QWEN if path == CHAT_PATH else VIDEO
        if model not in self.allowed_models:
            await JSONResponse({"error": {"code": "model_not_enabled"}}, status_code=403)(
                scope, receive, send
            )
            return
        semaphore = self.limits[model]
        if semaphore.locked():
            await JSONResponse(
                {"error": {"code": "busy"}}, status_code=429, headers={"Retry-After": "1"}
            )(scope, receive, send)
            return
        async with semaphore:
            try:
                values = request.headers.getlist("authorization")
                if len(values) != 1 or not values[0].startswith("Bearer "):
                    raise GatewayError("unauthorized", 401)
                key = values[0][7:]
                if (
                    not 1 <= len(key) <= 512
                    or not key.isascii()
                    or any(not 33 <= ord(c) <= 126 for c in key)
                ):
                    raise GatewayError("unauthorized", 401)
                body = bytearray()
                async for piece in request.stream():
                    if len(body) + len(piece) > BODY_LIMIT:
                        raise GatewayError("invalid_request", 413)
                    body.extend(piece)
                session = Session(self.billing, key)
                writer = Writer(scope, receive, send, session.id)
                operation = asyncio.create_task(
                    self.process(request, bytes(body), model, session, writer)
                )

                async def disconnected() -> None:
                    while True:
                        if (await receive())["type"] == "http.disconnect":
                            return

                watcher = asyncio.create_task(disconnected())
                try:
                    done, _ = await asyncio.wait(
                        {operation, watcher}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if watcher in done and not operation.done():
                        operation.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await operation
                finally:
                    watcher.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await watcher
                    if not operation.done():
                        operation.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await operation
            except GatewayError as error:
                await JSONResponse({"error": {"code": error.code}}, status_code=error.status)(
                    scope, receive, send
                )
            except ClientDisconnect:
                return

    async def _backend_get(self, suffix: str) -> list[httpx.Response | Exception]:
        async def fetch(base: str) -> httpx.Response | Exception:
            try:
                return await self.upstream.get(base + suffix)
            except httpx.HTTPError as exc:
                return exc

        return list(await asyncio.gather(*(fetch(base) for base in self.observability_targets)))

    async def health(self, scope: Scope, receive: Receive, send: Send) -> None:
        results = await self._backend_get("/health")
        backends = [
            {
                "endpoint": endpoint,
                "status": result.status_code if isinstance(result, httpx.Response) else None,
            }
            for endpoint, result in zip(self.observability_targets, results, strict=True)
        ]
        healthy = all(item["status"] == 200 for item in backends)
        await JSONResponse(
            {"status": "ok" if healthy else "unhealthy", "backends": backends},
            status_code=200 if healthy else 503,
        )(scope, receive, send)

    async def metrics(self, scope: Scope, receive: Receive, send: Send) -> None:
        results = await self._backend_get("/metrics")
        if any(
            not isinstance(result, httpx.Response) or result.status_code != 200
            for result in results
        ):
            await Response("backend_metrics_unavailable", status_code=503)(scope, receive, send)
            return
        body = b"\n".join(
            result.content for result in results if isinstance(result, httpx.Response)
        )
        await Response(
            body,
            media_type="text/plain",
            headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
        )(scope, receive, send)

    async def process(
        self, request: Request, body: bytes, model: str, session: Session, writer: Writer
    ) -> None:
        try:
            if len(request.headers.getlist("content-type")) != 1:
                raise GatewayError("invalid_request", 400)
            if model == QWEN:
                if request.headers["content-type"].split(";")[0].strip() != "application/json":
                    raise GatewayError("invalid_request", 400)
                original = strict_json(body)
                try:
                    payload, inputs, outputs = normalize_chat(original)
                except (TypeError, ValueError, KeyError) as exc:
                    raise GatewayError("invalid_request", 400) from exc
                maximum = await session.admit(model, inputs, sorted(outputs))
                if maximum is None:
                    meter = UsageOnlyMeter(outputs)
                else:
                    meter = QwenMeter(maximum, outputs)
                    if any(meter.limits[f"{kind}_input"] < INPUT_LIMIT for kind in inputs):
                        raise GatewayError("billing_unavailable")
                    if "text" in outputs and payload["max_tokens"] > meter.limits["text_output"]:
                        raise GatewayError("invalid_request", 400)
                include_usage = bool((original.get("stream_options") or {}).get("include_usage"))
                async with asyncio.timeout_at(session.deadline):
                    async with self.upstream.stream(
                        "POST",
                        self.targets[model],
                        content=encode(payload),
                        headers={
                            "Content-Type": "application/json",
                            "Accept-Encoding": "identity",
                        },
                    ) as response:
                        if response.status_code != 200 or response.headers.get(
                            "content-encoding", "identity"
                        ) not in {"", "identity"}:
                            raise GatewayError("upstream_failed", 502)
                        if payload.get("stream", False):
                            if not response.headers.get("content-type", "").startswith(
                                "text/event-stream"
                            ):
                                raise GatewayError("upstream_failed", 502)
                            await self.stream_chat(response, session, writer, meter, include_usage)
                        else:
                            data = await self.collect(response)
                            await self.buffered_chat(data, session, writer, meter)
            else:
                video_inputs = validate_video(body, request.headers["content-type"])
                maximum = await session.admit(model, video_inputs, ["video"])
                async with asyncio.timeout_at(session.deadline):
                    async with self.upstream.stream(
                        "POST",
                        self.targets[model],
                        content=body,
                        headers={
                            "Content-Type": request.headers["content-type"],
                            "Accept-Encoding": "identity",
                        },
                    ) as response:
                        if response.status_code == 400 and await self.guardrail_blocked(response):
                            raise GatewayError("content_policy_violation", 400)
                        if (
                            response.status_code != 200
                            or response.headers.get("content-encoding", "identity")
                            not in {"", "identity"}
                            or not response.headers.get("content-type", "").startswith("video/mp4")
                        ):
                            raise GatewayError("upstream_failed", 502)
                        data = await self.collect(response)
                    duration = mp4_duration(data)
                    if not session.billing_disabled:
                        await session.checkpoint(video_meter(duration, maximum))
                    await session.finish("completed")
                    await writer.full(data, "video/mp4")
        except (
            GatewayError,
            MeterError,
            httpx.HTTPError,
            TimeoutError,
            UnicodeError,
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
            RecursionError,
        ) as exc:
            error = exc if isinstance(exc, GatewayError) else GatewayError("upstream_failed", 502)
            LOG.warning(
                "paid generation stopped",
                extra={"event": "ccs.gateway_request", "outcome": error.code},
            )
            await session.finish("cancelled")
            await writer.error(error)
        finally:
            await session.finish("cancelled")

    async def guardrail_blocked(self, response: httpx.Response) -> bool:
        """True only for a bounded, well-formed Cosmos guardrail rejection body."""
        if response.headers.get("content-encoding", "identity") not in {"", "identity"}:
            return False
        data = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=16384):
            if len(data) + len(chunk) > GUARDRAIL_ERROR_LIMIT:
                return False
            data.extend(chunk)
        try:
            value = strict_json(bytes(data))
        except GatewayError:
            return False
        error = value.get("error") if isinstance(value, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        return isinstance(message, str) and "guardrail" in message.lower()

    async def collect(self, response: httpx.Response) -> bytes:
        data = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=16384):
            if len(data) + len(chunk) > OUTPUT_LIMIT:
                raise GatewayError("output_limit")
            data.extend(chunk)
        return bytes(data)

    async def buffered_chat(
        self, raw: bytes, session: Session, writer: Writer, meter: QwenMeter
    ) -> None:
        value = upstream_json(raw)
        if not isinstance(value, dict) or value.get("error"):
            raise GatewayError("upstream_failed", 502)
        meter.input_snapshot(value)
        kept = []
        for choice in value.get("choices", []):
            if not isinstance(choice, dict) or choice.get("finish_reason") not in {
                "stop",
                "length",
                "tool_calls",
                "function_call",
            }:
                raise GatewayError("upstream_failed", 502)
            message = choice.get("message") or {}
            delta = choice.get("delta") or {}
            if visible_text(delta) or delta.get("audio") is not None:
                raise GatewayError("meter_unavailable")
            audio = message.get("audio")
            if audio is not None:
                if "audio" not in meter.outputs:
                    continue
                meter.audio(audio.get("data"))
            else:
                if "text" not in meter.outputs:
                    continue
                meter.text(
                    choice,
                    visible_text(message),
                )
            kept.append(choice)
        if not kept:
            raise GatewayError("meter_unavailable")
        meter.ensure_complete()
        if not session.billing_disabled:
            await session.checkpoint(meter.meter())
        await session.finish("completed")
        value["choices"] = kept
        value = cleaned(value)
        if not session.billing_disabled:
            value["billable_usage"] = meter.meter()
        await writer.full(encode(value), "application/json")

    async def stream_chat(
        self,
        response: httpx.Response,
        session: Session,
        writer: Writer,
        meter: QwenMeter,
        include_usage: bool,
    ) -> None:
        pending: list[bytes] = []
        pending_bytes = 0
        sent_frames, sent_tokens = 0, 0
        last_flush = time.monotonic()
        done = False
        async for data in events(response):
            if data == "[DONE]":
                done = True
                break
            event = upstream_json(data)
            if not isinstance(event, dict) or event.get("error"):
                raise GatewayError("upstream_failed", 502)
            if "model_meter_v1" in (event.get("metrics") or {}):
                meter.input_snapshot(event)
            modality = event.get("modality")
            if modality is None and meter.outputs == {"text"}:
                modality = "text"
            if modality is not None and modality not in meter.outputs:
                continue
            choices = event.get("choices", [])
            terminal = False
            changed = False
            for choice in choices:
                reason = choice.get("finish_reason")
                if reason is not None and reason not in {
                    "stop",
                    "length",
                    "tool_calls",
                    "function_call",
                }:
                    raise GatewayError("upstream_failed", 502)
                delta = choice.get("delta") or {}
                message = choice.get("message") or {}
                if (
                    visible_text(message)
                    or message.get("audio") is not None
                    or delta.get("audio") is not None
                ):
                    raise GatewayError("meter_unavailable")
                terminal |= reason is not None
                if modality == "audio" and delta.get("content"):
                    meter.audio(delta["content"])
                    changed = True
                elif modality == "text":
                    previous = meter.text_tokens
                    meter.text(
                        choice,
                        visible_text(delta),
                    )
                    changed |= meter.text_tokens != previous
                elif visible_text(delta):
                    raise GatewayError("meter_unavailable")
            if changed and meter.inputs is None:
                raise GatewayError("meter_unavailable")
            event = cleaned(event)
            if not include_usage:
                event.pop("usage", None)
                if not choices:
                    continue
            output = b"data: " + encode(event) + b"\n\n"
            pending.append(output)
            pending_bytes += len(output)
            if pending_bytes > MAX_EVENT:
                raise GatewayError("output_limit")
            should_flush = (
                terminal
                or meter.frames - sent_frames >= 24000
                or meter.text_tokens - sent_tokens >= 8
                or time.monotonic() - last_flush >= 0.1
            )
            if (
                not session.billing_disabled
                and should_flush
                and meter.inputs is not None
                and (meter.frames or meter.text_tokens)
            ):
                await session.checkpoint(meter.meter())
                await writer.event(b"".join(pending))
                pending, pending_bytes = [], 0
                sent_frames, sent_tokens = meter.frames, meter.text_tokens
                last_flush = time.monotonic()
        if not done:
            raise GatewayError("upstream_failed", 502)
        meter.ensure_complete()
        if not session.billing_disabled:
            await session.checkpoint(meter.meter())
        await session.finish("completed")
        if pending:
            await writer.event(b"".join(pending))
        await writer.end()
