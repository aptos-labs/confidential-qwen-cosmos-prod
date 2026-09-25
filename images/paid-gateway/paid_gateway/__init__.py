"""Unconditional paid boundary for the measured production gateway."""

from __future__ import annotations

from starlette.applications import Starlette

from .app import PaidGateway


def install(app: Starlette) -> PaidGateway:
    """Install before the server starts, never from a LiteLLM worker hook.

    Starlette builds its stack BEFORE entering the lifespan. A late wrapper
    would not receive that in-flight lifespan and could not own client cleanup.
    Build and wrap the entire native stack now; later routes remain behind it.
    Invalid secrets abort synchronously, before Uvicorn can open a listener.
    """
    if getattr(app.state, "ccs_paid_gateway", False):
        raise RuntimeError("paid gateway already installed")
    if app.middleware_stack is not None:
        raise RuntimeError("paid gateway must be installed before server startup")
    gateway = PaidGateway(app.build_middleware_stack())
    app.middleware_stack = gateway
    app.state.ccs_paid_gateway = gateway
    return gateway
