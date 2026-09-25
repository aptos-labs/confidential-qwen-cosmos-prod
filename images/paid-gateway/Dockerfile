FROM docker.io/litellm/litellm@sha256:a3715fa7ad8387941ab697259bd2881d68931657247a41984f90fae6d11c62bf AS runtime-base
WORKDIR /opt/ccs-gateway
ENV PYTHONPATH=/opt/ccs-gateway PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
COPY --chmod=0444 paid_gateway/*.py /opt/ccs-gateway/paid_gateway/
COPY --chmod=0444 litellm.yaml /opt/ccs-gateway/litellm.yaml
COPY --chmod=0555 entrypoint.sh /opt/ccs-gateway/entrypoint.sh
RUN chmod 0555 /opt/ccs-gateway /opt/ccs-gateway/paid_gateway
ENTRYPOINT ["/opt/ccs-gateway/entrypoint.sh"]
CMD []

FROM runtime-base AS test
RUN /app/.venv/bin/python3 -m ensurepip --upgrade \
    && /app/.venv/bin/python3 -m pip install --no-cache-dir pytest==9.1.1
RUN --network=none \
    --mount=type=bind,source=tests,target=/opt/ccs-gateway/tests,readonly \
    --mount=type=tmpfs,target=/tmp \
    /app/.venv/bin/python3 -m pytest -q -p no:cacheprovider \
      --confcutdir=/opt/ccs-gateway/tests /opt/ccs-gateway/tests

FROM runtime-base AS runtime
