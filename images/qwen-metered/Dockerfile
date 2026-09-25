FROM docker.io/vllm/vllm-omni@sha256:4780186f168af96634917208596675439fa71cb5ed3440a30e8c21ca1defc451 AS upstream

FROM upstream AS runtime-base
COPY --chmod=0444 model-metering/install.py model-metering/input_meter.py model-metering/manifest.json model-metering/vllm-omni-v0.28.0.patch /opt/model-metering/
RUN chmod 0555 /opt/model-metering
RUN python3 /opt/model-metering/install.py \
    && python3 -c "import sys; sys.path.insert(0, '/opt/model-metering'); from importlib.metadata import distribution; from pathlib import Path; from install import preflight; root=Path(distribution('vllm-omni').locate_file('vllm_omni')).resolve(); assert preflight(root).already_installed, 'meter is not installed'"

FROM runtime-base AS test
RUN python3 -m pip install --no-cache-dir pytest==9.1.1
RUN --network=none \
    --mount=type=bind,from=upstream,source=/usr/local/lib/python3.12/dist-packages/vllm_omni,target=/opt/upstream/vllm_omni,readonly \
    --mount=type=bind,source=model-metering/tests,target=/opt/model-metering/tests,readonly \
    --mount=type=tmpfs,target=/tmp \
    PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
      --confcutdir=/opt/model-metering /opt/model-metering/tests --upstream=/opt/upstream

FROM runtime-base AS runtime
