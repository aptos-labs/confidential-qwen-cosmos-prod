#!/bin/sh
set -eu
umask 077
PATH=/app/.venv/bin:/usr/local/bin:/usr/bin:/bin
PYTHONPATH=/opt/ccs-gateway
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
export PATH PYTHONPATH PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE
unset PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP
cp /opt/ccs-gateway/litellm.yaml /tmp/llm.yaml
exec /app/.venv/bin/python3 -m paid_gateway "$@"
