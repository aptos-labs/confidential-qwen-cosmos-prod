"""Signed, replay-safe calls to CCS. No credentials or request bodies are logged."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import uuid
from datetime import UTC, datetime
from typing import Any, Final
from urllib.parse import urlsplit

import httpx

BILLING_URL: Final = "https://api.inference.aptoslabs.com"
REPORTER_ID: Final = "model-router"
_BILLING_ROUTE_ABSENT: Final = object()


def production_billing() -> BillingClient:
    """Native injected secret only; routing and identity are measured constants."""
    value = os.environ.get("USAGE_REPORTER_SECRET", "")
    if (
        not 32 <= len(value) <= 128
        or not value.isascii()
        or any(not 33 <= ord(c) <= 126 for c in value)
    ):
        raise RuntimeError("USAGE_REPORTER_SECRET is missing or invalid")
    return BillingClient(BILLING_URL, REPORTER_ID, value.encode("ascii"))


class GatewayError(Exception):
    def __init__(self, code: str, status: int = 503) -> None:
        allowed = {
            "invalid_request",
            "unauthorized",
            "busy",
            "billing_unavailable",
            "billing_rejected",
            "meter_unavailable",
            "upstream_failed",
            "output_limit",
            "content_policy_violation",
        }
        if code not in allowed:
            raise ValueError("invalid gateway error code")
        self.code, self.status = code, status
        super().__init__(code)


def strict_json(raw: bytes | str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise GatewayError("invalid_request", 400)
            result[key] = value
        return result

    def invalid_constant(_: str) -> None:
        raise GatewayError("invalid_request", 400)

    def finite_float(raw_value: str) -> float:
        value = float(raw_value)
        if not math.isfinite(value):
            invalid_constant(raw_value)
        return value

    def bounded(value: Any, depth: int = 0) -> None:
        if depth > 64:
            invalid_constant("")
        if isinstance(value, dict):
            for key, item in value.items():
                key.encode("utf-8")
                bounded(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                bounded(item, depth + 1)
        elif isinstance(value, str):
            value.encode("utf-8")

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs, parse_constant=invalid_constant, parse_float=finite_float
        )
        bounded(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise GatewayError("invalid_request", 400) from exc


def integer(value: Any, minimum: int = 0, maximum: int = (1 << 64) - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise GatewayError("meter_unavailable")
    return value


class BillingClient:
    def __init__(
        self,
        base_url: str,
        reporter: str,
        secret: bytes,
        client: httpx.AsyncClient | None = None,
        *,
        test_loopback: bool = False,
    ) -> None:
        url = urlsplit(base_url)
        if (
            url.scheme != "https"
            and not (
                test_loopback
                and url.scheme == "http"
                and url.hostname in {"127.0.0.1", "localhost", "::1"}
            )
        ) or not url.hostname:
            raise ValueError("billing requires HTTPS")
        if url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}:
            raise ValueError("invalid billing base URL")
        if (
            not reporter
            or len(reporter) > 128
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in reporter
            )
        ):
            raise ValueError("invalid reporter identifier")
        if not 32 <= len(secret) <= 128:
            raise ValueError("invalid reporter secret length")
        self.base_url = base_url.rstrip("/")
        self.reporter, self.secret = reporter, secret
        self.client = client or httpx.AsyncClient(
            timeout=10, trust_env=False, follow_redirects=False
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def call(self, operation: str, value: dict[str, Any]) -> dict[str, Any]:
        if operation not in {"admit", "checkpoint", "settle"}:
            raise ValueError("invalid billing operation")
        path = "/api/internal/billing/" + operation
        body = json.dumps(
            value, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
        if len(body) > 65536:
            raise GatewayError("invalid_request", 400)
        for attempt in range(3):
            timestamp = datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
            nonce = uuid.uuid4().hex
            canonical = "\n".join(
                (
                    "CCS-BILLING-V1",
                    "POST",
                    path,
                    self.reporter,
                    timestamp,
                    nonce,
                    hashlib.sha256(body).hexdigest(),
                )
            )
            signature = hmac.new(self.secret, canonical.encode(), hashlib.sha256).hexdigest()
            headers = {
                "Content-Type": "application/json",
                "X-Tinfoil-Reporter-ID": self.reporter,
                "X-Tinfoil-Timestamp": timestamp,
                "X-Tinfoil-Nonce": nonce,
                "X-Tinfoil-Signature": signature,
            }
            try:
                response = await self.client.post(
                    self.base_url + path, content=body, headers=headers
                )
                if response.status_code == 200:
                    if len(response.content) > 65536:
                        raise GatewayError("billing_unavailable")
                    try:
                        result = strict_json(response.content)
                    except GatewayError as exc:
                        raise GatewayError("billing_unavailable") from exc
                    if not isinstance(result, dict):
                        raise GatewayError("billing_unavailable")
                    return result
                if response.status_code == 404:
                    return _BILLING_ROUTE_ABSENT
                if response.status_code < 500:
                    status = (
                        response.status_code
                        if response.status_code in {400, 403, 402, 409, 410, 413}
                        else 503
                    )
                    raise GatewayError("billing_rejected", status)
            except httpx.TransportError:
                pass
            if attempt < 2:
                await asyncio.sleep(0.1 * (attempt + 1))
        raise GatewayError("billing_unavailable")
