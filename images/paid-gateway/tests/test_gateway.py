"""Paid gateway tests use synthetic media and in-memory transports only."""

import asyncio
import base64
import copy
import hashlib
import hmac
import io
import json
import sys
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paid_gateway.app import PaidGateway
from paid_gateway.client import _BILLING_ROUTE_ABSENT, BillingClient, GatewayError
from paid_gateway.protocol import QWEN, VIDEO, normalize_chat
from paid_gateway.video_request import validate_video

KEY = "synthetic-internal-key"
SECRET = b"01234567890123456789012345678901"


def wav(frames=12000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\x00\x00" * frames)
    return base64.b64encode(buf.getvalue()).decode()


def metadata():
    return {
        "model_meter_v1": {
            "schema": "qwen3-omni-input/v1",
            "status": "available",
            "total_input": 4,
            "text_input": 4,
            "image_input": 0,
            "video_input": 0,
            "audio_input": 0,
        }
    }


def maximum(outputs):
    return {
        "model": QWEN,
        "meter": {
            "kind": "tokens_and_generated_audio",
            "usage": {
                "tokens": {
                    "total_input": 131072,
                    "text_input": 131072,
                    "image_input": 0,
                    "video_input": 0,
                    "audio_input": 0,
                    "text_output": 4096 if "text" in outputs else 0,
                    "audio_output": 0,
                    "total_output": 4096 if "text" in outputs else 0,
                },
                "generated_audio": {
                    "kind": "samples",
                    "duration": {"sample_count": 2880000, "sample_rate": 24000},
                }
                if "audio" in outputs
                else {"kind": "absent"},
            },
        },
    }


class FakeBilling:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    async def call(self, operation, value):
        self.calls.append((operation, copy.deepcopy(value)))
        if self.fail == "absent" and operation == "admit":
            return _BILLING_ROUTE_ABSENT
        if operation == self.fail:
            raise GatewayError("billing_rejected", 403)
        if operation == "admit":
            return {
                "request_id": value["request_id"],
                "model": value["model"],
                "maximum": maximum(value["output_modalities"]),
                "expires_at": (datetime.now(UTC) + timedelta(minutes=3)).isoformat(),
            }
        return {"request_id": value["request_id"], "sequence": value.get("sequence")}

    async def close(self):
        pass


async def downstream(scope, receive, send):
    await send({"type": "http.response.start", "status": 204, "headers": []})
    await send({"type": "http.response.body", "body": b"", "more_body": False})


def payload(stream=False):
    return {
        "model": QWEN,
        "messages": [{"role": "user", "content": "hello"}],
        "modalities": ["audio"],
        "max_tokens": 64,
        "stream": stream,
    }


def output(meter=True):
    result = {
        "model": QWEN,
        "choices": [
            {
                "message": {"content": "hidden-thinker-output"},
                "token_ids": [1, 2, 3],
                "finish_reason": "stop",
            },
            {
                "message": {"audio": {"data": wav()}},
                "token_ids": [100] * 20,
                "finish_reason": "stop",
            },
        ],
    }
    if meter:
        result["metrics"] = metadata()
    return result


async def invoke(body, *, response=None, fail=None, path="/v1/chat/completions"):
    billing = FakeBilling(fail)
    sent = []

    async def model(request):
        sent.append(request)
        assert billing.calls and billing.calls[0][0] == "admit"
        parsed = json.loads(request.content)
        assert parsed["return_token_ids"] is True and parsed["return_stage_metrics"] is True
        if not body.get("stream"):
            assert "stream_options" not in parsed
        return response or httpx.Response(200, json=output())

    gateway = PaidGateway(
        downstream, billing, httpx.AsyncClient(transport=httpx.MockTransport(model))
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(gateway), base_url="https://gateway.test"
        ) as client:
            result = await client.post(path, json=body, headers={"Authorization": "Bearer " + KEY})
        return result, billing, sent
    finally:
        await gateway.upstream.aclose()


def test_buffered_audio_is_acked_before_delivery_and_hidden_text_is_not_billed():
    async def case():
        result, billing, sent = await invoke(payload())
        assert result.status_code == 200 and len(sent) == 1
        assert "hidden-thinker-output" not in result.text
        assert "token_ids" not in result.text and "metrics" not in result.text
        assert [op for op, _ in billing.calls] == ["admit", "checkpoint", "settle"]
        checkpoint = billing.calls[1][1]["meter"]
        assert checkpoint == result.json()["billable_usage"]
        assert checkpoint["meter"]["usage"]["tokens"]["text_output"] == 0
        assert checkpoint["meter"]["usage"]["generated_audio"]["duration"]["sample_count"] == 12000
        assert billing.calls[-1][1]["reason"] == "completed"
        assert KEY not in result.text

    asyncio.run(case())


def test_usage_only_buffered_audio_skips_billing_records():
    async def case():

        result, billing, sent = await invoke(payload(), fail="absent")
        assert result.status_code == 200 and len(sent) == 1
        assert "hidden-thinker-output" not in result.text
        assert "token_ids" not in result.text and "metrics" not in result.text
        assert "billable_usage" not in result.json()
        assert [op for op, _ in billing.calls] == ["admit"]
        assert KEY not in result.text

    asyncio.run(case())


@pytest.mark.parametrize("failure", ["admit", "checkpoint"])
def test_billing_failure_never_forwards_unconfirmed_audio(failure):
    async def case():
        result, billing, sent = await invoke(payload(), fail=failure)
        assert result.status_code == 403
        assert wav() not in result.text
        assert len(sent) == (failure != "admit")
        assert billing.calls[-1][0] == "settle"
        assert billing.calls[-1][1]["reason"] == "cancelled"

    asyncio.run(case())


def test_missing_meter_is_not_treated_as_zero_usage():
    async def case():
        result, billing, _ = await invoke(
            payload(), response=httpx.Response(200, json=output(False))
        )
        assert result.status_code == 503
        assert [op for op, _ in billing.calls] == ["admit", "settle"]

    asyncio.run(case())


def test_streaming_audio_keeps_final_chunk_and_excludes_codec_tokens():
    async def case():
        records = [
            {
                "model": QWEN,
                "modality": "text",
                "metrics": metadata(),
                "choices": [
                    {
                        "delta": {"content": "hidden-thinker-output"},
                        "token_ids": [1, 2],
                        "finish_reason": "stop",
                    }
                ],
            }
        ]
        records.extend(
            {
                "model": QWEN,
                "modality": "audio",
                "metrics": metadata(),
                "choices": [
                    {
                        "delta": {"content": wav()},
                        "token_ids": [1] * 50,
                        "finish_reason": finish,
                    }
                ],
            }
            for finish in (None, "stop")
        )
        data = (
            b"".join(b"data: " + json.dumps(x).encode() + b"\n\n" for x in records)
            + b"data: [DONE]\n\n"
        )
        result, billing, _ = await invoke(
            payload(True),
            response=httpx.Response(
                200, content=data, headers={"Content-Type": "text/event-stream"}
            ),
        )
        assert result.status_code == 200 and result.text.count(wav()) == 2
        assert result.text.endswith("data: [DONE]\n\n")
        assert "hidden-thinker-output" not in result.text and "token_ids" not in result.text
        checkpoint = [value for op, value in billing.calls if op == "checkpoint"][-1]
        assert (
            checkpoint["meter"]["meter"]["usage"]["generated_audio"]["duration"]["sample_count"]
            == 24000
        )
        assert checkpoint["meter"]["meter"]["usage"]["tokens"]["text_output"] == 0

    asyncio.run(case())


def test_usage_only_streaming_skips_billing_checkpoints():
    async def case():
        response = sse(
            [
                {
                    "model": QWEN,
                    "modality": "audio",
                    "metrics": metadata(),
                    "choices": [
                        {
                            "delta": {"content": wav()},
                            "token_ids": [1] * 50,
                            "finish_reason": "stop",
                        }
                    ],
                }
            ]
        )
        result, billing, _ = await invoke(payload(True), response=response, fail="absent")
        assert result.status_code == 200 and wav() in result.text
        assert [op for op, _ in billing.calls] == ["admit"]
        assert "billable_usage" not in result.text

    asyncio.run(case())


@pytest.mark.parametrize(
    "path", ["/v1/responses", "/v1/audio/speech", "/chat/completions", "/v1/videos"]
)
def test_alternative_endpoints_cannot_bypass_paid_admission(path):
    async def case():
        result, billing, sent = await invoke(payload(), path=path)
        assert result.status_code == 404 and not billing.calls and not sent

    asyncio.run(case())


def test_disconnect_cancels_upstream_and_closes_reservation():
    async def case():
        started = asyncio.Event()
        aborted = asyncio.Event()
        billing = FakeBilling()

        async def model(request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                aborted.set()

        gateway = PaidGateway(
            downstream, billing, httpx.AsyncClient(transport=httpx.MockTransport(model))
        )
        body = json.dumps(payload()).encode()
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            await started.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            raise AssertionError("disconnected request must not receive a response")

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [
                (b"authorization", b"Bearer " + KEY.encode()),
                (b"content-type", b"application/json"),
            ],
        }
        await asyncio.wait_for(gateway(scope, receive, send), 5)
        assert aborted.is_set()
        assert [op for op, _ in billing.calls] == ["admit", "settle"]
        assert billing.calls[-1][1]["reason"] == "cancelled"
        await gateway.upstream.aclose()

    asyncio.run(case())


def test_signed_retries_keep_the_same_request_and_checkpoint_payload():
    async def case():
        bodies = []

        async def endpoint(request):
            bodies.append(request.content)
            h = request.headers
            message = "\n".join(
                [
                    "CCS-BILLING-V1",
                    "POST",
                    request.url.path,
                    h["x-tinfoil-reporter-id"],
                    h["x-tinfoil-timestamp"],
                    h["x-tinfoil-nonce"],
                    hashlib.sha256(request.content).hexdigest(),
                ]
            )
            assert hmac.compare_digest(
                h["x-tinfoil-signature"],
                hmac.new(SECRET, message.encode(), hashlib.sha256).hexdigest(),
            )
            if len(bodies) == 1:
                raise httpx.ReadTimeout("synthetic lost acknowledgement")
            value = json.loads(request.content)
            return httpx.Response(
                200, json={"request_id": value["request_id"], "sequence": value["sequence"]}
            )

        client = BillingClient(
            "https://billing.test",
            "model-router",
            SECRET,
            httpx.AsyncClient(transport=httpx.MockTransport(endpoint)),
        )
        result = await client.call(
            "checkpoint", {"request_id": "fixture", "api_key": KEY, "sequence": 1, "meter": {}}
        )
        assert result["sequence"] == 1 and len(bodies) == 2 and bodies[0] == bodies[1]
        await client.close()

    asyncio.run(case())


@pytest.mark.parametrize(
    "url", ["https://public.example/image.png", "http://169.254.169.254/", "file:///etc/passwd"]
)
def test_initial_release_rejects_remote_media(url):
    value = payload()
    value["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": url}}]
    with pytest.raises(GatewayError):
        normalize_chat(value)


def form(fields, files=()):
    marker = b"--fixture-boundary"
    parts = []
    for name, value in fields:
        parts.append(
            marker
            + b'\r\nContent-Disposition: form-data; name="'
            + name.encode()
            + b'"\r\n\r\n'
            + value.encode()
            + b"\r\n"
        )
    for name, value, *rest in files:
        media_type, filename = rest if rest else ("image/png", "frame.png")
        parts.append(
            marker
            + b'\r\nContent-Disposition: form-data; name="'
            + name.encode()
            + b'"; filename="'
            + filename.encode()
            + b'"\r\nContent-Type: '
            + media_type.encode()
            + b"\r\n\r\n"
            + value
            + b"\r\n"
        )
    return b"".join(parts) + marker + b"--\r\n"


VIDEO_TYPE = "multipart/form-data; boundary=fixture-boundary"
PNG = b"\x89PNG\r\n\x1a\nfixture"
JPEG = b"\xff\xd8\xff\xe0fixture"


def cosmos_fields(**overrides):
    fields = {
        "model": "cosmos3-super",
        "prompt": "a robot arm stacks blocks",
        "size": "832x480",
        "num_frames": "189",
    }
    fields.update(overrides)
    return [(name, value) for name, value in fields.items() if value is not None]


@pytest.mark.parametrize("size", ["832x480", "1280x720"])
def test_cosmos_text_to_video_is_admitted(size):
    fields = cosmos_fields(
        size=size,
        negative_prompt="blurry",
        fps="24",
        num_inference_steps="35",
        guidance_scale="7.5",
        flow_shift="5.0",
        seed="-42",
        max_sequence_length="512",
        extra_params=json.dumps(
            {"use_resolution_template": True, "use_duration_template": False, "guardrails": True}
        ),
    )
    assert validate_video(form(fields), VIDEO_TYPE) == ["text"]
    assert validate_video(form(cosmos_fields(size=size, num_frames="1")), VIDEO_TYPE) == ["text"]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"generate_sound": "true", "sound_duration": "7.875"}, id="full-189"),
        pytest.param({"generate_sound": "true"}, id="sound-default-duration"),
        pytest.param({"generate_sound": "false"}, id="sound-off"),
        pytest.param(
            {"num_frames": "48", "generate_sound": "true", "sound_duration": "2"},
            id="two-seconds",
        ),
        pytest.param(
            {"num_frames": "48", "generate_sound": "true", "sound_duration": "0.5"},
            id="shorter-than-video",
        ),
    ],
)
def test_cosmos_video_with_sound_is_admitted(overrides):
    assert validate_video(form(cosmos_fields(**overrides)), VIDEO_TYPE) == ["text"]
    body = form(
        cosmos_fields(**overrides), [("input_reference", JPEG, "image/jpeg", "frame.jpg")]
    )
    assert validate_video(body, VIDEO_TYPE) == ["text", "image"]


@pytest.mark.parametrize(
    "image,media_type,filename",
    [(PNG, "image/png", "frame.png"), (JPEG, "image/jpeg", "frame.jpg")],
)
def test_cosmos_image_to_video_accepts_one_reference(image, media_type, filename):
    body = form(cosmos_fields(), [("input_reference", image, media_type, filename)])
    assert validate_video(body, VIDEO_TYPE) == ["text", "image"]


@pytest.mark.parametrize(
    "fields,files",
    [
        pytest.param(cosmos_fields(size=None), [], id="missing-size"),
        pytest.param(cosmos_fields(size="1920x1080"), [], id="size-1920x1080"),
        pytest.param(cosmos_fields(size="640x480"), [], id="size-640x480"),
        pytest.param(cosmos_fields(num_frames="190"), [], id="num-frames-190"),
        pytest.param(cosmos_fields(num_frames="0"), [], id="num-frames-0"),
        pytest.param(cosmos_fields(num_frames=None), [], id="missing-num-frames"),
        pytest.param(cosmos_fields(num_frames=" 189"), [], id="num-frames-whitespace"),
        pytest.param(
            cosmos_fields(num_frames="\u0661\u0668\u0669"), [], id="num-frames-nonascii"
        ),
        pytest.param(cosmos_fields(fps="30"), [], id="fps-30"),
        pytest.param(cosmos_fields(num_inference_steps="51"), [], id="steps-51"),
        pytest.param(cosmos_fields(guidance_scale="nan"), [], id="guidance-nan"),
        pytest.param(cosmos_fields(seed=str(2**63)), [], id="seed-overflow"),
        pytest.param(cosmos_fields(max_sequence_length="4097"), [], id="max-seq-4097"),
        pytest.param(cosmos_fields(prompt=" "), [], id="blank-prompt"),
        pytest.param(cosmos_fields(prompt=None), [], id="missing-prompt"),
        pytest.param(
            cosmos_fields(extra_params='{"guardrails":false}'), [], id="guardrails-false"
        ),
        pytest.param(
            cosmos_fields(extra_params='{"guardrails":"false"}'), [], id="guardrails-string"
        ),
        pytest.param(cosmos_fields(extra_params='{"guardrails":0}'), [], id="guardrails-zero"),
        pytest.param(
            cosmos_fields(extra_params='{"guardrails":null}'), [], id="guardrails-null"
        ),
        pytest.param(
            cosmos_fields(extra_params='{"guardrails":true,"guardrails":false}'),
            [],
            id="guardrails-duplicate",
        ),
        pytest.param(
            cosmos_fields(extra_params='{"use_resolution_template":1}'),
            [],
            id="template-not-bool",
        ),
        pytest.param(cosmos_fields(extra_params='{"task":"i2v"}'), [], id="unknown-extra-key"),
        pytest.param(cosmos_fields(extra_params="[]"), [], id="extra-not-object"),
        pytest.param(
            cosmos_fields(),
            [("input_reference", PNG), ("input_reference", PNG)],
            id="two-files",
        ),
        pytest.param(cosmos_fields(), [("input_references", PNG)], id="input-references"),
        pytest.param(cosmos_fields(), [("input_reference", b"")], id="empty-file"),
        pytest.param(
            cosmos_fields(), [("input_reference", b"GIF89a", "image/gif", "a.gif")], id="gif"
        ),
        pytest.param(
            cosmos_fields(), [("input_reference", b"not an image")], id="bad-png-magic"
        ),
        pytest.param(
            cosmos_fields(),
            [("input_reference", PNG, "image/jpeg", "frame.jpg")],
            id="png-bytes-as-jpeg",
        ),
        pytest.param(
            cosmos_fields(), [("input_reference", PNG, "image/png", "../x.png")], id="traversal"
        ),
        pytest.param(cosmos_fields(generate_sound="True"), [], id="sound-flag-capitalized"),
        pytest.param(cosmos_fields(generate_sound="1"), [], id="sound-flag-one"),
        pytest.param(cosmos_fields(generate_sound=""), [], id="sound-flag-empty"),
        pytest.param(cosmos_fields(sound_duration="7.875"), [], id="duration-without-sound"),
        pytest.param(
            cosmos_fields(generate_sound="false", sound_duration="2"),
            [],
            id="duration-with-sound-off",
        ),
        pytest.param(
            cosmos_fields(generate_sound="true", sound_duration="7.9"),
            [],
            id="duration-longer-than-video",
        ),
        pytest.param(
            cosmos_fields(num_frames="48", generate_sound="true", sound_duration="2.5"),
            [],
            id="duration-longer-than-short-video",
        ),
        pytest.param(
            cosmos_fields(generate_sound="true", sound_duration="0"), [], id="duration-zero"
        ),
        pytest.param(
            cosmos_fields(generate_sound="true", sound_duration="-1"), [], id="duration-negative"
        ),
        pytest.param(
            cosmos_fields(generate_sound="true", sound_duration="nan"), [], id="duration-nan"
        ),
        pytest.param(
            cosmos_fields(generate_sound="true", sound_duration=" 2"), [], id="duration-space"
        ),
        pytest.param(cosmos_fields(seconds="4"), [], id="unknown-field-seconds"),
        pytest.param(cosmos_fields(width="832"), [], id="unknown-field-width"),
        pytest.param(cosmos_fields(model="qwen3-omni"), [], id="wrong-model"),
        pytest.param(cosmos_fields(model=None), [], id="missing-model"),
    ],
)
def test_cosmos_video_rejections(fields, files):
    with pytest.raises(GatewayError) as error:
        validate_video(form(fields, files), VIDEO_TYPE)
    assert (error.value.code, error.value.status) == ("invalid_request", 400)


def test_video_validates_uploads_without_rewriting_body():
    fields = cosmos_fields(size="1280x720", fps="24", extra_params='{"guardrails":true}')
    body = form(fields, [("input_reference", PNG)])
    original = bytes(body)
    assert validate_video(body, VIDEO_TYPE) == ["text", "image"]
    assert body == original
    for extra in [
        ("priority", "1"),
        ("num_inference_steps", "1000000"),
        ("aspect_ratio", "16:9"),
        ("image_reference", '"http://internal/"'),
        ("size", "832x480"),
    ]:
        with pytest.raises(GatewayError):
            validate_video(form([*fields, extra]), VIDEO_TYPE)


# Regression coverage exercises the real ASGI boundary, not only meter helpers.
def sse(records, *, done=True):
    data = b"".join(b"data: " + json.dumps(record).encode() + b"\n\n" for record in records)
    return httpx.Response(
        200,
        content=data + (b"data: [DONE]\n\n" if done else b""),
        headers={"Content-Type": "text/event-stream"},
    )


def audio_record(frames=24000):
    return {
        "metrics": metadata(),
        "modality": "audio",
        "choices": [
            {"delta": {"content": wav(frames)}, "token_ids": [999], "finish_reason": None}
        ],
    }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ids", [[1, 2, 3], None, [], [True], [-1], "secret-invalid-ids"])
def test_text_only_token_ids_are_authoritative(stream, ids):
    async def case():
        body = payload(stream)
        body["modalities"] = ["text"]
        choice = {
            "delta" if stream else "message": {"content": "visible-text"},
            "token_ids": ids,
            "finish_reason": "stop",
        }
        record = {"metrics": metadata(), "choices": [choice]}
        response = sse([record]) if stream else httpx.Response(200, json=record)
        result, billing, _ = await invoke(body, response=response)
        if ids == [1, 2, 3]:
            assert result.status_code == 200 and "visible-text" in result.text
            meter = next(v["meter"] for op, v in billing.calls if op == "checkpoint")
            assert meter["meter"]["usage"]["tokens"]["text_output"] == 3
            assert meter["meter"]["usage"]["generated_audio"] == {"kind": "absent"}
            assert "token_ids" not in result.text
        else:
            assert result.status_code == 503 and "visible-text" not in result.text
            assert [op for op, _ in billing.calls] == ["admit", "settle"]

    asyncio.run(case())


@pytest.mark.parametrize("fault", ["missing_done", "regression", "rate_change"])
def test_stream_fault_after_confirmed_audio_only_settles_confirmed_frames(fault):
    async def case():
        second = audio_record(12000)
        if fault == "regression":
            second["metrics"]["model_meter_v1"].update(total_input=3, text_input=3)
        elif fault == "rate_change":
            raw = bytearray(base64.b64decode(wav(12000)))
            raw[24:28] = (16000).to_bytes(4, "little")
            raw[28:32] = (32000).to_bytes(4, "little")
            second["choices"][0]["delta"]["content"] = base64.b64encode(raw).decode()
        result, billing, _ = await invoke(
            payload(True), response=sse([audio_record(), second], done=fault != "missing_done")
        )
        assert result.status_code == 200 and "inference_error" in result.text
        assert result.text.count(wav(24000)) == 1
        assert second["choices"][0]["delta"]["content"] not in result.text
        checkpoints = [v for op, v in billing.calls if op == "checkpoint"]
        assert len(checkpoints) == 1
        assert (
            checkpoints[0]["meter"]["meter"]["usage"]["generated_audio"]["duration"][
                "sample_count"
            ]
            == 24000
        )
        assert billing.calls[-1][1]["reason"] == "cancelled"

    asyncio.run(case())


@pytest.mark.parametrize("waiting_checkpoint", [False, True])
def test_disconnect_after_ack_closes_upstream_and_discards_unconfirmed_audio(waiting_checkpoint):
    async def case():
        disconnect = asyncio.Event()
        closed = asyncio.Event()
        confirmed = []
        sent = []

        class Billing(FakeBilling):
            async def call(self, operation, value):
                if operation == "checkpoint" and confirmed:
                    disconnect.set()
                    await asyncio.Event().wait()
                result = await super().call(operation, value)
                if operation == "checkpoint":
                    confirmed.append(copy.deepcopy(value))
                return result

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield sse([audio_record()], done=False).content
                assert confirmed, "upstream must not be advanced past an unacknowledged batch"
                yield sse(
                    [audio_record(24001 if waiting_checkpoint else 12000)], done=False
                ).content
                disconnect.set()
                await asyncio.Event().wait()

            async def aclose(self):
                closed.set()

        billing = Billing()
        upstream = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, stream=Stream(), headers={"content-type": "text/event-stream"}
                )
            )
        )
        gateway = PaidGateway(downstream, billing, upstream)
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {
                    "type": "http.request",
                    "body": json.dumps(payload(True)).encode(),
                    "more_body": False,
                }
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            assert confirmed, "no response bytes before checkpoint acknowledgement"
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [
                (b"authorization", b"Bearer " + KEY.encode()),
                (b"content-type", b"application/json"),
            ],
        }
        try:
            await asyncio.wait_for(gateway(scope, receive, send), 5)
            assert closed.is_set() and len(confirmed) == 1
            assert (
                confirmed[0]["meter"]["meter"]["usage"]["generated_audio"]["duration"][
                    "sample_count"
                ]
                == 24000
            )
            wire = b"".join(m.get("body", b"") for m in sent)
            assert wav(24000).encode() in wire
            assert wav(24001 if waiting_checkpoint else 12000).encode() not in wire
            assert billing.calls[-1][1]["reason"] == "cancelled"
        finally:
            await upstream.aclose()

    asyncio.run(case())


@pytest.mark.parametrize(
    "patch",
    [
        {"modalities": [[]]},
        {"audio": {"voice": []}},
        {"audio": None},
        {"temperature": "1"},
        {"temperature": True},
        {"top_p": []},
        {"seed": 1.5},
        {"stop": [1]},
        {"logprobs": 1},
        {"top_logprobs": True},
        {"parallel_tool_calls": "yes"},
        {"tools": [{}]},
        {"tools": [{"type": "function", "function": {"name": "f", "parameters": []}}]},
        {"tool_choice": {"type": "function", "function": {"name": []}}},
        {"response_format": {"type": []}},
        {"response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": []}}},
        {"messages": [{"role": [], "content": "hello"}]},
        {"messages": [{"role": "assistant", "content": None, "tool_calls": "bad"}]},
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "f", "arguments": {}},
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"format": "wav", "data": 1234}}
                    ],
                }
            ]
        },
        {"stream_options": {"include_usage": 1}},
    ],
)
def test_nested_request_types_rejected_before_admission(patch):
    async def case():
        body = payload()
        body.update(patch)
        result, billing, sent = await invoke(body)
        assert result.status_code == 400 and not billing.calls and not sent

    asyncio.run(case())


@pytest.mark.parametrize(
    "ref", ["https://secret.invalid/schema", "file:///etc/passwd", "other.json", 7]
)
@pytest.mark.parametrize("keyword", ["$ref", "$dynamicRef", "$recursiveRef", "$id"])
def test_nonlocal_schema_refs_are_rejected(ref, keyword):
    body = payload()
    body["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {"type": "object", "properties": {"x": {keyword: ref}}},
            },
        }
    ]
    with pytest.raises(GatewayError, match="invalid_request"):
        normalize_chat(body)


def test_local_schema_ref_and_well_typed_options_remain_supported():
    body = payload()
    body.update(temperature=0.7, top_p=0.9, seed=42, stop=["END"], parallel_tool_calls=False)
    body["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "$defs": {"x": {"type": "string"}},
                    "properties": {"x": {"$ref": "#/$defs/x"}},
                },
            },
        }
    ]
    body["tool_choice"] = {"type": "function", "function": {"name": "f"}}
    assert normalize_chat(body)[0]["tools"] == body["tools"]


@pytest.mark.parametrize(
    "raw", [b'{"x":1e999}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1,"x":2}']
)
def test_strict_json_rejects_nonfinite_and_duplicate_values(raw):
    from paid_gateway.client import strict_json

    with pytest.raises(GatewayError, match="invalid_request"):
        strict_json(raw)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "patch",
    [
        {"metrics": "secret-model-payload"},
        {"choices": None},
        {"choices": ["secret-model-payload"]},
        {
            "choices": [
                {
                    "message": "secret-model-payload",
                    "delta": "secret-model-payload",
                    "finish_reason": "stop",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {"audio": "secret-model-payload"},
                    "delta": [],
                    "finish_reason": "stop",
                }
            ]
        },
        {"choices": [{"finish_reason": []}]},
        {"modality": []},
    ],
)
def test_malformed_upstream_is_fixed_error_without_traceback(stream, patch, caplog):
    async def case():
        record = {"metrics": metadata(), "choices": []}
        record.update(patch)
        response = sse([record]) if stream else httpx.Response(200, json=record)
        result, billing, _ = await invoke(payload(stream), response=response)
        assert result.status_code in {502, 503}
        assert result.json()["error"]["code"] in {"upstream_failed", "meter_unavailable"}
        assert "secret-model-payload" not in result.text + caplog.text
        assert KEY not in result.text + caplog.text
        assert all(record.exc_info is None for record in caplog.records)
        assert [op for op, _ in billing.calls] == ["admit", "settle"]

    asyncio.run(case())


@pytest.mark.parametrize("framing", ["line", "multiline", "pending"])
def test_oversized_sse_never_releases_unconfirmed_frames(framing, monkeypatch):
    import paid_gateway.app as gateway_app

    monkeypatch.setattr(gateway_app, "MAX_EVENT", 256)
    if framing == "line":
        data = b"data: " + b"x" * 257 + b"\n\n"
    elif framing == "multiline":
        data = (b"data: " + b"x" * 130 + b"\n") * 2 + b"\n"
    else:
        # Non-billable role frames survive sanitization and fill the pending buffer.
        event = {"modality": "audio", "choices": [{"delta": {"role": "assistant"}}]}
        data = sse([event] * 5).content

    async def case():
        body = payload(True)
        body["stream_options"] = {"include_usage": True}
        result, billing, _ = await invoke(
            body,
            response=httpx.Response(
                200, content=data, headers={"content-type": "text/event-stream"}
            ),
        )
        assert result.status_code == 503 and result.json()["error"]["code"] == "output_limit"
        assert [op for op, _ in billing.calls] == ["admit", "settle"]

    asyncio.run(case())


@pytest.mark.parametrize(
    "tamper", ["request_id", "sequence", "boolean_sequence", "malformed_json"]
)
def test_signed_checkpoint_reply_tampering_fails_closed(tamper):
    # CCS signs requests; replies use HTTPS, with identity/sequence checked by Session.
    from paid_gateway.app import Session

    async def case():
        async def endpoint(request):
            h = request.headers
            canonical = "\n".join(
                (
                    "CCS-BILLING-V1",
                    "POST",
                    request.url.path,
                    h["x-tinfoil-reporter-id"],
                    h["x-tinfoil-timestamp"],
                    h["x-tinfoil-nonce"],
                    hashlib.sha256(request.content).hexdigest(),
                )
            )
            assert hmac.compare_digest(
                h["x-tinfoil-signature"],
                hmac.new(SECRET, canonical.encode(), hashlib.sha256).hexdigest(),
            )
            value = json.loads(request.content)
            if tamper == "malformed_json":
                return httpx.Response(200, content=b'{"secret":NaN}')
            result = {"request_id": value["request_id"], "sequence": value["sequence"]}
            if tamper == "boolean_sequence":
                result["sequence"] = True
            else:
                result[tamper] = "wrong" if tamper == "request_id" else 2
            return httpx.Response(200, json=result)

        client = BillingClient(
            "https://billing.test",
            "model-router",
            SECRET,
            httpx.AsyncClient(transport=httpx.MockTransport(endpoint)),
        )
        session = Session(client, KEY)
        try:
            with pytest.raises(GatewayError, match="billing_unavailable"):
                await session.checkpoint({})
            assert session.sequence == 0 and session.last is None
        finally:
            await client.close()

    asyncio.run(case())


REVIEW_MARKER = "visible-unmetered-marker"
REVIEW_VISIBLE_FIELDS = [
    ("content", REVIEW_MARKER),
    ("reasoning", REVIEW_MARKER),
    ("reasoning_content", REVIEW_MARKER),
    ("refusal", REVIEW_MARKER),
    ("function_call", {"name": "lookup", "arguments": json.dumps({"q": REVIEW_MARKER})}),
    (
        "tool_calls",
        [
            {
                "id": "call_fixture",
                "type": "function",
                "function": {"name": "lookup", "arguments": json.dumps({"q": REVIEW_MARKER})},
            }
        ],
    ),
]


def review_text_request(stream=False):
    body = payload(stream)
    body["modalities"] = ["text"]
    return body


@pytest.mark.parametrize("field,value", REVIEW_VISIBLE_FIELDS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ids", [None, []])
def test_review_visible_fields_require_token_ids(field, value, stream, ids):
    async def case():
        choice = {"delta" if stream else "message": {field: value}, "finish_reason": "stop"}
        if ids is not None:
            choice["token_ids"] = ids
        response = {"model": QWEN, "metrics": metadata(), "choices": [choice]}
        result, billing, _ = await invoke(
            review_text_request(stream),
            response=sse([response]) if stream else httpx.Response(200, json=response),
        )
        assert result.status_code == 503
        assert result.json()["error"]["code"] == "meter_unavailable"
        assert REVIEW_MARKER not in result.text
        assert [op for op, _ in billing.calls] == ["admit", "settle"]
        assert billing.calls[-1][1]["reason"] == "cancelled"

    asyncio.run(case())


@pytest.mark.parametrize("field,value", REVIEW_VISIBLE_FIELDS)
@pytest.mark.parametrize("stream", [False, True])
def test_review_visible_fields_with_ids_are_checkpointed(field, value, stream):
    async def case():
        choice = {
            "delta" if stream else "message": {field: value},
            "finish_reason": "stop",
            "token_ids": [7, 9],
        }
        response = {"model": QWEN, "metrics": metadata(), "choices": [choice]}
        result, billing, _ = await invoke(
            review_text_request(stream),
            response=sse([response]) if stream else httpx.Response(200, json=response),
        )
        assert result.status_code == 200 and REVIEW_MARKER in result.text
        checkpoints = [value for op, value in billing.calls if op == "checkpoint"]
        assert checkpoints[-1]["meter"]["meter"]["usage"]["tokens"]["text_output"] == 2
        assert billing.calls[-1][1]["reason"] == "completed"
        assert "token_ids" not in result.text

    asyncio.run(case())


@pytest.mark.parametrize("finish", ["tool_calls", "function_call"])
@pytest.mark.parametrize("stream", [False, True])
def test_review_tool_call_terminal_reasons_are_supported(finish, stream):
    async def case():
        function = {"name": "lookup", "arguments": '{"q":"fixture"}'}
        value = (
            function
            if finish == "function_call"
            else [{"id": "call_fixture", "type": "function", "function": function}]
        )
        request = review_text_request(stream)
        request["tools"] = [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
            }
        ]
        request["tool_choice"] = "auto"
        choice = {
            "delta" if stream else "message": {finish: value},
            "finish_reason": finish,
            "token_ids": [7, 9],
        }
        response = {"model": QWEN, "metrics": metadata(), "choices": [choice]}
        result, billing, _ = await invoke(
            request, response=sse([response]) if stream else httpx.Response(200, json=response)
        )
        assert result.status_code == 200 and "lookup" in result.text
        checkpoints = [value for op, value in billing.calls if op == "checkpoint"]
        assert checkpoints[-1]["meter"]["meter"]["usage"]["tokens"]["text_output"] == 2
        assert billing.calls[-1][1]["reason"] == "completed"

    asyncio.run(case())


def test_review_late_unmetered_field_retains_only_confirmed_checkpoint():
    async def case():
        response = sse(
            [
                {
                    "modality": "text",
                    "metrics": metadata(),
                    "choices": [{"delta": {"content": "confirmed"}, "token_ids": list(range(8))}],
                },
                {
                    "modality": "text",
                    "choices": [{"delta": {"refusal": REVIEW_MARKER}, "finish_reason": "stop"}],
                },
            ]
        )
        result, billing, _ = await invoke(review_text_request(True), response=response)
        assert result.status_code == 200 and "confirmed" in result.text
        assert "meter_unavailable" in result.text and REVIEW_MARKER not in result.text
        checkpoints = [value for op, value in billing.calls if op == "checkpoint"]
        assert len(checkpoints) == 1
        assert checkpoints[0]["meter"]["meter"]["usage"]["tokens"]["text_output"] == 8
        assert billing.calls[-1][1]["reason"] == "cancelled"

    asyncio.run(case())


def test_review_unlabelled_mixed_output_fails_closed():
    async def case():
        request = review_text_request(True)
        request["modalities"] = ["text", "audio"]
        response = sse(
            [
                audio_record(),
                {
                    "choices": [
                        {"delta": {"reasoning_content": REVIEW_MARKER}, "finish_reason": "stop"}
                    ]
                },
            ]
        )
        result, billing, _ = await invoke(request, response=response)
        assert "meter_unavailable" in result.text and REVIEW_MARKER not in result.text
        assert billing.calls[-1][1]["reason"] == "cancelled"

    asyncio.run(case())


@pytest.mark.parametrize(
    "files,expected",
    [
        ([], ["text"]),
        ([("input_reference", PNG)], ["text", "image"]),
        ([("input_reference", JPEG, "image/jpeg", "frame.jpg")], ["text", "image"]),
    ],
)
def test_review_video_admission_describes_actual_inputs(files, expected):
    async def case():
        fields = cosmos_fields(extra_params='{"guardrails":true}')
        raw = form(fields, files)
        billing = FakeBilling("admit")
        sent = []
        upstream = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: sent.append(request))
        )
        gateway = PaidGateway(downstream, billing, upstream)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(gateway), base_url="https://gateway.test"
            ) as client:
                result = await client.post(
                    "/v1/videos/sync",
                    content=raw,
                    headers={
                        "Authorization": "Bearer " + KEY,
                        "Content-Type": "multipart/form-data; boundary=fixture-boundary",
                    },
                )
            assert result.status_code == 403 and not sent
            assert billing.calls[0][0] == "admit"
            assert billing.calls[0][1]["input_modalities"] == expected
            assert billing.calls[0][1]["output_modalities"] == ["video"]
        finally:
            await upstream.aclose()

    asyncio.run(case())


def review_confirmed_event():
    return {
        "modality": "text",
        "metrics": metadata(),
        "choices": [{"delta": {"content": "confirmed"}, "token_ids": list(range(8))}],
    }


def review_assert_failed_output(result, billing, late, code="meter_unavailable"):
    assert result.status_code == (200 if late else (502 if code == "upstream_failed" else 503))
    assert code in result.text and REVIEW_MARKER not in result.text
    checkpoints = [value for op, value in billing.calls if op == "checkpoint"]
    assert len(checkpoints) == int(late)
    if late:
        assert "confirmed" in result.text
        assert checkpoints[0]["meter"]["meter"]["usage"]["tokens"]["text_output"] == 8
    assert billing.calls[-1][1]["reason"] == "cancelled"


@pytest.mark.parametrize(
    "field,value", [*REVIEW_VISIBLE_FIELDS, ("audio", {"data": REVIEW_MARKER})]
)
@pytest.mark.parametrize("stream,late", [(False, False), (True, False), (True, True)])
def test_review_alternate_container_cannot_release_unmetered_output(field, value, stream, late):
    async def case():
        choice = {
            "delta" if stream else "message": {"content": "metered-shape"},
            "message" if stream else "delta": {field: value},
            "token_ids": [7, 9],
            "finish_reason": "stop",
        }
        response = {"metrics": metadata(), "choices": [choice]}
        events = [review_confirmed_event(), response] if late else [response]
        result, billing, _ = await invoke(
            review_text_request(stream),
            response=sse(events) if stream else httpx.Response(200, json=response),
        )
        review_assert_failed_output(result, billing, late)

    asyncio.run(case())


@pytest.mark.parametrize("late", [False, True])
def test_review_sse_audio_object_cannot_bypass_audio_meter(late):
    async def case():
        response = {
            "metrics": metadata(),
            "choices": [
                {
                    "delta": {"audio": {"data": REVIEW_MARKER}},
                    "token_ids": [7, 9],
                    "finish_reason": "stop",
                }
            ],
        }
        events = [review_confirmed_event(), response] if late else [response]
        result, billing, _ = await invoke(review_text_request(True), response=sse(events))
        review_assert_failed_output(result, billing, late)

    asyncio.run(case())


@pytest.mark.parametrize("reason", ["abort", "error", "content_filter", "unexpected", ""])
@pytest.mark.parametrize("late", [False, True])
def test_review_sse_failure_terminal_does_not_settle_completed(reason, late):
    async def case():
        response = {
            "metrics": metadata(),
            "choices": [
                {
                    "delta": {"content": REVIEW_MARKER},
                    "token_ids": [7, 9],
                    "finish_reason": reason,
                }
            ],
        }
        events = [review_confirmed_event(), response] if late else [response]
        result, billing, _ = await invoke(review_text_request(True), response=sse(events))
        review_assert_failed_output(result, billing, late, "upstream_failed")

    asyncio.run(case())


@pytest.mark.parametrize("reason", ["stop", "length", "tool_calls", "function_call"])
def test_review_valid_terminal_flushes_acknowledged_output_before_eof(reason):
    async def case():
        response = review_confirmed_event()
        response["choices"][0]["token_ids"] = [7]
        response["choices"][0]["finish_reason"] = reason
        result, billing, _ = await invoke(
            review_text_request(True), response=sse([response], done=False)
        )
        assert result.status_code == 200 and "confirmed" in result.text
        assert "upstream_failed" in result.text
        checkpoints = [value for op, value in billing.calls if op == "checkpoint"]
        assert len(checkpoints) == 1
        assert checkpoints[0]["meter"]["meter"]["usage"]["tokens"]["text_output"] == 1
        assert billing.calls[-1][1]["reason"] == "cancelled"

    asyncio.run(case())


def test_billing_client_absent_route_returns_sentinel_without_retry():
    async def case():
        attempts = 0

        def response(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(response))
        billing = BillingClient("https://billing.test", "reporter", SECRET, client)
        try:
            result = await billing.call("admit", {"request_id": "fixture"})
            assert result is _BILLING_ROUTE_ABSENT
            assert attempts == 1
        finally:
            await billing.close()

    asyncio.run(case())


def test_billing_client_server_error_remains_fail_closed():
    async def case():
        attempts = 0

        def response(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(500)

        client = httpx.AsyncClient(transport=httpx.MockTransport(response))
        billing = BillingClient("https://billing.test", "reporter", SECRET, client)
        try:
            with pytest.raises(GatewayError, match="billing_unavailable"):
                await billing.call("admit", {"request_id": "fixture"})
            assert attempts == 3
        finally:
            await billing.close()

    asyncio.run(case())


def test_usage_only_video_skips_billing_and_preserves_video_response(monkeypatch):
    async def case():
        fields = cosmos_fields(size="1280x720", fps="24")
        raw = form(fields, [("input_reference", JPEG, "image/jpeg", "frame.jpg")])
        billing = FakeBilling("absent")
        sent = []

        async def model(request):
            sent.append(request)
            return httpx.Response(200, headers={"content-type": "video/mp4"}, content=b"mp4")

        upstream = httpx.AsyncClient(transport=httpx.MockTransport(model))
        gateway = PaidGateway(downstream, billing, upstream)
        monkeypatch.setattr("paid_gateway.app.mp4_duration", lambda data: Fraction(189, 24))
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(gateway), base_url="https://gateway.test"
            ) as client:
                result = await client.post(
                    "/v1/videos/sync",
                    content=raw,
                    headers={
                        "Authorization": "Bearer " + KEY,
                        "Content-Type": "multipart/form-data; boundary=fixture-boundary",
                    },
                )
            assert result.status_code == 200
            assert result.content == b"mp4"
            assert [op for op, _ in billing.calls] == ["admit"]
            assert len(sent) == 1
            assert str(sent[0].url) == "http://vllm-omni-cosmos:8001/v1/videos/sync"
            assert sent[0].content == raw
            assert sent[0].headers["content-type"] == VIDEO_TYPE
        finally:
            await upstream.aclose()

    from fractions import Fraction

    asyncio.run(case())


GUARDRAIL_BODY = {
    "error": {
        "message": "Input was blocked by Cosmos3 guardrails",
        "type": "BadRequestError",
        "param": None,
        "code": 400,
    }
}


async def invoke_video(response, fail=None):
    billing = FakeBilling(fail)
    sent = []

    async def model(request):
        sent.append(request)
        return response

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(model))
    gateway = PaidGateway(downstream, billing, upstream)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(gateway), base_url="https://gateway.test"
        ) as client:
            result = await client.post(
                "/v1/videos/sync",
                content=form(cosmos_fields()),
                headers={"Authorization": "Bearer " + KEY, "Content-Type": VIDEO_TYPE},
            )
        return result, billing, sent
    finally:
        await upstream.aclose()


@pytest.mark.parametrize(
    "message",
    [
        "Input was blocked by Cosmos3 guardrails",
        "Output video was blocked by GUARDRAIL checks",
    ],
)
def test_cosmos_guardrail_block_is_unbilled_content_policy_violation(message):
    async def case():
        body = copy.deepcopy(GUARDRAIL_BODY)
        body["error"]["message"] = message
        result, billing, sent = await invoke_video(httpx.Response(400, json=body))
        assert len(sent) == 1
        assert result.status_code == 400
        error = result.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "content_policy_violation"
        assert error["message"] == "Request was blocked by the model's safety guardrails."
        assert message not in result.text and "BadRequestError" not in result.text
        assert [op for op, _ in billing.calls] == ["admit", "settle"]
        assert billing.calls[-1][1]["reason"] == "cancelled"

    asyncio.run(case())


def test_usage_only_cosmos_guardrail_block_records_nothing():
    async def case():
        result, billing, _ = await invoke_video(httpx.Response(400, json=GUARDRAIL_BODY), "absent")
        assert result.status_code == 400
        assert result.json()["error"]["code"] == "content_policy_violation"
        assert [op for op, _ in billing.calls] == ["admit"]

    asyncio.run(case())


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            httpx.Response(400, json={"error": {"message": "num_frames must be <= 189"}}),
            id="400-other-message",
        ),
        pytest.param(httpx.Response(400, content=b"guardrails blocked"), id="400-non-json"),
        pytest.param(httpx.Response(400, json=["guardrail"]), id="400-json-not-object"),
        pytest.param(httpx.Response(400, json={"error": "guardrail"}), id="400-error-not-object"),
        pytest.param(
            httpx.Response(400, json={"error": {"message": ["guardrail"]}}),
            id="400-message-not-string",
        ),
        pytest.param(
            httpx.Response(400, json={"detail": "blocked by guardrails"}), id="400-no-error"
        ),
        pytest.param(
            httpx.Response(
                400,
                json={
                    "error": {"message": "Input was blocked by Cosmos3 guardrails"},
                    "pad": "x" * (64 * 1024),
                },
            ),
            id="400-oversize",
        ),
        pytest.param(
            httpx.Response(
                400,
                content=b'{"error":{"message":"guardrail"},"error":{"message":"x"}}',
            ),
            id="400-duplicate-keys",
        ),
        pytest.param(httpx.Response(500, json=GUARDRAIL_BODY), id="500-guardrail-body"),
        pytest.param(httpx.Response(500, content=b"internal error"), id="500"),
        pytest.param(httpx.Response(422, json=GUARDRAIL_BODY), id="422-guardrail-body"),
    ],
)
def test_cosmos_other_upstream_errors_remain_upstream_failed(response):
    async def case():
        result, billing, _ = await invoke_video(response)
        assert result.status_code == 502
        assert result.json()["error"] == {
            "type": "inference_error",
            "code": "upstream_failed",
            "message": "upstream_failed",
        }
        assert "guardrail" not in result.text.lower()
        assert [op for op, _ in billing.calls] == ["admit", "settle"]
        assert billing.calls[-1][1]["reason"] == "cancelled"

    asyncio.run(case())


def test_qwen_upstream_guardrail_400_remains_upstream_failed():
    async def case():
        result, billing, _ = await invoke(
            payload(), response=httpx.Response(400, json=GUARDRAIL_BODY)
        )
        assert result.status_code == 502
        assert result.json()["error"]["code"] == "upstream_failed"
        assert result.json()["error"]["type"] == "inference_error"
        assert [op for op, _ in billing.calls] == ["admit", "settle"]
        assert billing.calls[-1][1]["reason"] == "cancelled"

    asyncio.run(case())


def test_router_injected_priority_is_forwarded_to_qwen():
    # The router adds "priority": 1 once a key passes its soft per-minute limit.
    async def case():
        body = payload()
        body["priority"] = 1
        result, _, sent = await invoke(body)
        assert result.status_code == 200
        assert json.loads(sent[0].content)["priority"] == 1

    asyncio.run(case())


@pytest.mark.parametrize("value", [-1, 11, 1.0, True, "1", None])
def test_out_of_range_priority_is_rejected_before_admission(value):
    async def case():
        body = payload()
        body["priority"] = value
        result, billing, sent = await invoke(body)
        assert result.status_code == 400 and not billing.calls and not sent

    asyncio.run(case())


class _Held:
    """Upstream transport that parks every request until released."""

    def __init__(self, response):
        self.response = response
        self.release = asyncio.Event()
        self.started = 0

    async def handler(self, request):
        self.started += 1
        await self.release.wait()
        return self.response()


async def _flood(gateway, n, post):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(gateway), base_url="https://gateway.test"
    ) as client:
        return await asyncio.gather(*(post(client) for _ in range(n)))


def test_qwen_runs_concurrency_limit_at_once_then_answers_busy():
    from paid_gateway.app import CONCURRENCY, QUEUE_DEPTH

    limit = CONCURRENCY[QWEN] + QUEUE_DEPTH[QWEN]
    assert limit >= 64

    async def case():
        held = _Held(lambda: httpx.Response(200, json=output()))
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(held.handler))
        gateway = PaidGateway(downstream, FakeBilling(), upstream)

        async def post(client):
            return await client.post(
                "/v1/chat/completions",
                json=payload(),
                headers={"Authorization": "Bearer " + KEY},
            )

        task = asyncio.create_task(_flood(gateway, limit + 3, post))
        for _ in range(500):
            if held.started == CONCURRENCY[QWEN]:
                break
            await asyncio.sleep(0.01)
        assert held.started == CONCURRENCY[QWEN]
        held.release.set()
        results = await task
        codes = sorted(r.status_code for r in results)
        assert codes.count(200) == limit and codes.count(429) == 3
        assert gateway.admitted[QWEN] == 0
        await upstream.aclose()

    asyncio.run(case())


def test_cosmos_queues_behind_one_running_video_then_answers_busy():
    from paid_gateway.app import CONCURRENCY, QUEUE_DEPTH

    assert CONCURRENCY[VIDEO] == 1
    queued = QUEUE_DEPTH[VIDEO]

    async def case():
        held = _Held(lambda: httpx.Response(400, json=GUARDRAIL_BODY))
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(held.handler))
        gateway = PaidGateway(downstream, FakeBilling(), upstream)

        async def post(client):
            return await client.post(
                "/v1/videos/sync",
                content=form(cosmos_fields()),
                headers={"Authorization": "Bearer " + KEY, "Content-Type": VIDEO_TYPE},
            )

        task = asyncio.create_task(_flood(gateway, 1 + queued + 2, post))
        for _ in range(500):
            if held.started == 1 and gateway.admitted[VIDEO] == 1 + queued:
                break
            await asyncio.sleep(0.01)
        # Only one video reaches the engine; the rest wait instead of failing.
        assert held.started == 1 and gateway.admitted[VIDEO] == 1 + queued
        held.release.set()
        results = await task
        codes = [r.status_code for r in results]
        assert codes.count(429) == 2
        assert codes.count(400) == 1 + queued
        assert held.started == 1 + queued
        assert gateway.admitted[VIDEO] == 0
        await upstream.aclose()

    asyncio.run(case())
