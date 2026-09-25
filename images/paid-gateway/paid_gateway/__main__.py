"""Measured, single-process LiteLLM launcher: python -m paid_gateway."""

from __future__ import annotations

import asyncio
import os
import sys

from . import install


def main() -> None:
    # No reload, process spawning, alternate server or runtime routing options:
    # those can re-import an unwrapped proxy app in a different process.
    if len(sys.argv) != 1:
        raise RuntimeError("paid gateway launcher does not accept arguments")
    if os.environ.get("LITELLM_WORKER_STARTUP_HOOKS"):
        raise RuntimeError("paid gateway launcher does not accept worker hooks")

    from litellm.proxy.proxy_server import app  # type: ignore[import-not-found]

    gateway = install(app)
    try:
        from litellm.proxy.proxy_cli import run_server  # type: ignore[import-not-found]

        # The pinned CLI imports this same app and runs Uvicorn in-process with
        # one worker. Keep CLI configuration/initialization, not a second server.
        run_server(
            args=[
                "--config",
                "/tmp/llm.yaml",
                "--port",
                "8080",
                "--host",
                "0.0.0.0",
                "--num_workers",
                "1",
                "--telemetry",
                "False",
            ],
            standalone_mode=False,
        )
    finally:
        # Normally closed by the ASGI lifespan in its own loop. This also covers
        # CLI/config/import failure before lifespan startup (no active sockets).
        if not gateway.closed:
            asyncio.run(gateway.close())


if __name__ == "__main__":
    main()
