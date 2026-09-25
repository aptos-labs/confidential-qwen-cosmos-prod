# Confidential Qwen3-Omni + Cosmos3-Super production runtime

This repository is the public, signed Tinfoil release provenance for the production
dual-model confidential-inference CVM on `host0.inference.aptoslabs.com`:

| Model ID | Model | Container | GPUs |
|---|---|---|---|
| `qwen3-omni` | Qwen3-Omni-30B-A3B-Instruct (chat/omni) | `vllm-omni-qwen:8000` | 4–7 |
| `cosmos3-super` | NVIDIA Cosmos3-Super (video generation) | `vllm-omni-cosmos:8001` | 0–3 |

Earlier release history (the retired MiniMax runtime and the first Cosmos releases) is in the
archived `aptos-labs/confidential-qwen-minimax-prod-archive`. `v0.0.1` here publishes the same
measured runtime as that repository's final release, `v0.0.14`, so its images still come from
the original public `confidential-qwen-minimax-*` packages. New images are published as
`confidential-qwen-cosmos-*` (see below).

## What is attested

`tinfoil-config.yml` is the exact measured runtime exported from the production CVM
specification (`cvmctl export-runtime -f vm-qwen-cosmos-prod.yml`, kept in
`aptos-labs/atlas` under `rust/crates/cvmctl/vmconfig/`). It pins both model packs, the paid
gateway, Qwen and vLLM-Omni image digests, resource allocation, and all command lines.

A mandatory paid gateway fronts both models. It admits requests through CCS before calling a
model server, validates output meters, and persists checkpoints before releasing output, with
no free fallback. Video is metered as `generated_video_seconds` (at most 15 s per request).

The CVM uses Tinfoil's signed `extra_large_2d` hardware profile
([`tinfoilsh/hardware-measurements`](https://github.com/tinfoilsh/hardware-measurements)):
32 vCPUs, 512 GiB RAM, eight H200s with four NVSwitch ports, and five disks (root, measured
config, external config and two model volumes). Cosmos3-Super and its guardrail models share
one model volume.

## Cosmos3-Super video API

`POST /v1/videos/sync` with `multipart/form-data`, returning `video/mp4` (H.264, 24 fps; with
`generate_sound=true`, also a 48 kHz stereo AAC track). The gateway forwards the body unchanged after validating it:

- `model` = `cosmos3-super`; `prompt` required; optional `negative_prompt`.
- `size` required: `832x480` or `1280x720`.
- `num_frames` required: 1–189 (at most 7.875 s at 24 fps); `fps`, if sent, must be `24`.
- Optional `num_inference_steps` (1–50), `guidance_scale` and `flow_shift` (0–32), `seed`,
  `max_sequence_length` (1–4096).
- Image-to-video: at most one `input_reference` file, PNG or JPEG.
- Sound: `generate_sound` (`true`/`false`, default off) and, only with `true`, optional
  `sound_duration` in seconds (greater than 0, at most `num_frames / 24`).
- Optional `extra_params` JSON with `use_resolution_template` / `use_duration_template`
  booleans and `guardrails`, which may only be `true`.

**Guardrails are always on.** The server runs without `--no-guardrails`, and the gateway
rejects any request that tries to disable them. The guardrail models
(`nvidia/Cosmos-1.0-Guardrail`, `google/siglip-so400m-patch14-384`,
`Qwen/Qwen3Guard-Gen-0.6B`) load offline from the measured model volume.
Guardrail-blocked requests return HTTP 400 `content_policy_violation` and are not billed.

Staging on the production shape (2026-09-25): 720p/189 frames in about 230–260 s, 480p in
about 75 s; Qwen latency was unaffected while Cosmos generated.

## Sources

- `images/paid-gateway/` contains the mandatory gateway, its baked LiteLLM config,
  entrypoint, and offline tests. Test dependencies stay in the Dockerfile's `test` stage.
- `images/qwen-metered/` installs the hash-checked vLLM-Omni meter patch at image build
  time. Production pins the image digest recorded in `tinfoil-config.yml`.

## Image publication and verification

1. Review and merge the source PR, inspecting every review thread. PR CI runs policy/script
   tests, both Dockerfile `test` targets, and final-runtime smoke checks with all
   capabilities dropped, read-only roots, and no network.
2. Dispatch **Publish runtime images** on the reviewed `main` commit, passing its full
   40-character hash as `source_sha`.
3. Record the workflow's build-output digests for:
   - `ghcr.io/aptos-labs/confidential-qwen-cosmos-paid-gateway`
   - `ghcr.io/aptos-labs/confidential-qwen-cosmos-qwen-metered`
4. New GHCR packages start private, and the CVM pulls images anonymously. A package
   administrator must make each new package Public in its package settings, then rerun the
   anonymous-verification job of the same run.
5. Verify each image's provenance, for example:

   ```bash
   gh attestation verify "oci://$IMAGE_REF" \
     --repo aptos-labs/confidential-qwen-cosmos-prod \
     --signer-workflow aptos-labs/confidential-qwen-cosmos-prod/.github/workflows/publish-runtime-images.yml \
     --source-digest "$SOURCE_SHA" --signer-digest "$SOURCE_SHA"
   ```

Image publication never dispatches a Tinfoil runtime release or deploys a VM.

## Runtime release process

1. Update the production `cvmctl` spec with the verified image digests.
2. Export its exact measured runtime:

   ```bash
   cvmctl export-runtime -f vm-qwen-cosmos-prod.yml > tinfoil-config.yml
   ```

3. Review and commit the generated `tinfoil-config.yml`.
4. Run the **Tinfoil Release** workflow with a new, unused immutable tag. Never reuse a tag.
5. Deploy that exact production specification with its metadata repository and tag set to
   this repository and release.

Any runtime change (CPU/RAM, a model pack, an image digest, or vLLM arguments) requires a new
release before deployment. Before stopping a running VM, populate the protected host-side
copy's secret slots and verify that its exported runtime hash equals the reviewed release.
Never apply the public template's empty secret placeholders.

## Security boundaries

This repository contains no credentials. The certificate authorization token and
`USAGE_REPORTER_SECRET` exist only in the protected external configuration. The gateway's CCS
destination and reporter identity are fixed in measured code. Missing or invalid reporter
credentials fail startup; unavailable admission fails closed.
