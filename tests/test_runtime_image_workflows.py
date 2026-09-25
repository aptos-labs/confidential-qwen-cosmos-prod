"""Runtime workflow policy and executable guard tests (unittest + existing PyYAML).

Run: python3 -m unittest discover -s tests -p 'test_runtime_image_workflows.py' -v
No image contexts, parent scripts, Docker daemon, credentials, or network needed.
Only these new workflows are in scope; the existing Tinfoil workflows are untouched.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PINS = {
    "actions/checkout": "8e8c483db84b4bee98b60c0593521ed34d9990e8",
    "docker/setup-buildx-action": "4d04d5d9486b7bd6fa91e7baf45bbb4f8b9deedd",
    "docker/login-action": "4907a6ddec9925e35a0a9e82d7399ccc52663121",
    "docker/build-push-action": "bcafcacb16a39f128d818304e6c9c0c18556b85f",
    "actions/attest-build-provenance": "977bb373ede98d70efdf65b84cb5f73e068dcc2a",
}
IMAGES = {
    "gateway": (
        "images/paid-gateway",
        "ghcr.io/aptos-labs/confidential-qwen-cosmos-paid-gateway",
        4,
    ),
    "qwen": (
        "images/qwen-metered",
        "ghcr.io/aptos-labs/confidential-qwen-cosmos-qwen-metered",
        24,
    ),
}
DIGEST = "sha256:" + "a" * 64
OTHER_DIGEST = "sha256:" + "b" * 64


class UniqueKeyLoader(yaml.BaseLoader):
    """Keep YAML scalars as strings (not YAML 1.1's on=True); reject key shadowing."""

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"Duplicate workflow key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_workflow(name):
    return yaml.load((WORKFLOWS / name).read_text(), Loader=UniqueKeyLoader)


def step_by_id(job, step_id):
    return next(step for step in job["steps"] if step.get("id") == step_id)


def action_steps(job, action):
    return [step for step in job["steps"] if step.get("uses", "").startswith(action + "@")]


def inline_python(step):
    return step["run"].split("python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


def run_shell(script, directory, env=None):
    # An allowlisted environment prevents accidental inheritance of real tokens/config.
    variables = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(directory),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GITHUB_OUTPUT": str(Path(directory) / "outputs"),
        "GITHUB_ENV": str(Path(directory) / "environment"),
        "GITHUB_STEP_SUMMARY": str(Path(directory) / "summary"),
        "RUNNER_TEMP": str(directory),
    }
    variables.update(env or {})
    return subprocess.run(
        ["bash", "-c", script],
        cwd=directory,
        env=variables,
        text=True,
        capture_output=True,
        timeout=15,
    )


class WorkflowPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validate = load_workflow("validate-runtime-images.yml")
        cls.publish = load_workflow("publish-runtime-images.yml")

    def workflows(self):
        return (self.validate, self.publish)

    def test_yaml_duplicate_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            yaml.load("permissions: read\npermissions: write\n", Loader=UniqueKeyLoader)

    def test_triggers_are_pr_and_explicit_reviewed_dispatch_only(self):
        self.assertEqual(set(self.validate["on"]), {"pull_request"})
        self.assertEqual(set(self.publish["on"]), {"workflow_dispatch"})
        inputs = self.publish["on"]["workflow_dispatch"]["inputs"]
        self.assertEqual(set(inputs), {"source_sha"})
        self.assertEqual(inputs["source_sha"]["required"], "true")
        self.assertEqual(inputs["source_sha"]["type"], "string")
        self.assertNotIn("default", inputs["source_sha"])
        self.assertIn("reviewed", inputs["source_sha"]["description"].lower())
        self.assertEqual(self.publish["env"]["SOURCE_SHA"], "${{ inputs.source_sha }}")
        self.assertEqual(
            self.publish["env"]["SOURCE_URL"], "${{ github.server_url }}/${{ github.repository }}"
        )
        for workflow in self.workflows():
            self.assertNotIn("pull_request_target", json.dumps(workflow))

    def test_workflow_and_job_permissions_are_least_privilege(self):
        for workflow in self.workflows():
            self.assertEqual(workflow["permissions"], {"contents": "read"})
        for job in self.validate["jobs"].values():
            self.assertEqual(
                job.get("permissions", self.validate["permissions"]), {"contents": "read"}
            )
            self.assertNotIn("secrets.", json.dumps(job))
        for kind in IMAGES:
            self.assertEqual(
                self.publish["jobs"][kind]["permissions"],
                {
                    "contents": "read",
                    "packages": "write",
                    "id-token": "write",
                    "attestations": "write",
                },
            )
        self.assertEqual(
            self.publish["jobs"]["anonymous_pull"]["permissions"], {"contents": "read"}
        )

    def test_all_actions_are_exactly_pinned_and_allowlisted(self):
        for workflow in self.workflows():
            for job in workflow["jobs"].values():
                for step in job["steps"]:
                    if "uses" in step:
                        action, sha = step["uses"].split("@")
                        self.assertIn(action, PINS)
                        self.assertEqual(sha, PINS[action])

    def test_image_jobs_are_isolated_on_standard_runners_with_timeouts(self):
        self.assertEqual(set(self.validate["jobs"]), {"policy", "gateway", "qwen"})
        self.assertEqual(set(self.publish["jobs"]), {"gateway", "qwen", "anonymous_pull"})
        for workflow in self.workflows():
            for name, job in workflow["jobs"].items():
                self.assertEqual(job["runs-on"], "ubuntu-latest")
                self.assertGreaterEqual(int(job["timeout-minutes"]), 20)
                self.assertLessEqual(int(job["timeout-minutes"]), 120)
                for key in ("strategy", "container", "services", "uses"):
                    self.assertNotIn(key, job)
                if name in IMAGES:
                    context, image, minimum = IMAGES[name]
                    self.assertNotIn("needs", job)
                    self.assertEqual(job["env"]["IMAGE_CONTEXT"], context)
                    self.assertEqual(int(job["env"]["MIN_FREE_GIB"]), minimum)
                    if workflow is self.publish:
                        self.assertEqual(job["env"]["IMAGE_NAME"], image)
                        self.assertEqual(job["env"]["SMOKE_KIND"], name)

    def test_no_failure_bypasses_or_automatic_release(self):
        for workflow in self.workflows():
            self.assertNotIn("continue-on-error", json.dumps(workflow))
            for job in workflow["jobs"].values():
                self.assertNotIn("if", job)
                for step in job["steps"]:
                    if "if" in step:
                        self.assertEqual(step["if"], "${{ failure() }}")
                        self.assertIn("exit 1", step["run"])
                    script = step.get("run", "")
                    self.assertNotIn(
                        "${{", script, "Pass expressions through env, not shell interpolation"
                    )
                    self.assertNotRegex(script, r"\|\|\s*true|set\s+\+e")
                    self.assertNotRegex(
                        script, r"\b(gh\s+(workflow|release|api)|git\s+push|tinfoil\s+deploy)\b"
                    )
                    self.assertNotRegex(script, r"\b(rm|prune|apt-get|apt|sudo)\b")
        self.assertEqual(self.publish["concurrency"]["cancel-in-progress"], "false")

    def test_checkouts_never_persist_credentials_and_publish_fetches_main_history(self):
        for workflow in self.workflows():
            for job in workflow["jobs"].values():
                checkouts = action_steps(job, "actions/checkout")
                self.assertEqual(len(checkouts), 1)
                self.assertEqual(job["steps"][0], checkouts[0])
                self.assertEqual(checkouts[0]["with"]["persist-credentials"], "false")
                if workflow is self.publish:
                    self.assertEqual(checkouts[0]["with"]["ref"], "${{ inputs.source_sha }}")
                    self.assertEqual(checkouts[0]["with"]["fetch-depth"], "0")

    def test_source_guard_precedes_any_build_or_source_script(self):
        guards = []
        for job in self.publish["jobs"].values():
            guard = step_by_id(job, "source")
            self.assertEqual(job["steps"][1], guard)
            guards.append(guard["run"])
        self.assertEqual(len(set(guards)), 1, "Every checkout must use the same tested guard")
        self.assertIn('git merge-base --is-ancestor "$SOURCE_SHA" origin/main', guards[0])
        self.assertIn("printf 'sha=%s\\n' \"$actual_sha\"", guards[0])

    def test_disk_gate_runs_before_buildx_and_building(self):
        for workflow in self.workflows():
            for kind in IMAGES:
                job = workflow["jobs"][kind]
                gate = step_by_id(job, "disk")
                self.assertLess(
                    job["steps"].index(gate),
                    job["steps"].index(action_steps(job, "docker/setup-buildx-action")[0]),
                )
                self.assertLess(
                    job["steps"].index(gate), job["steps"].index(step_by_id(job, "test"))
                )
                self.assertIn('"/var/lib/docker"', gate["run"])
                self.assertIn('os.environ["GITHUB_WORKSPACE"]', gate["run"])
                self.assertIn('os.environ["RUNNER_TEMP"]', gate["run"])
        gate = self.publish["jobs"]["anonymous_pull"]
        self.assertEqual(
            int(gate["env"]["MIN_FREE_GIB"]), sum(image[2] for image in IMAGES.values())
        )
        self.assertLess(
            gate["steps"].index(step_by_id(gate, "disk")),
            gate["steps"].index(step_by_id(gate, "anonymous_smoke")),
        )

    def test_pr_tests_and_smokes_without_logging_in_or_publishing(self):
        for kind in IMAGES:
            job = self.validate["jobs"][kind]
            self.assertEqual(
                [step["uses"].split("@")[0] for step in job["steps"] if "uses" in step],
                [
                    "actions/checkout",
                    "docker/setup-buildx-action",
                    "docker/build-push-action",
                    "docker/build-push-action",
                ],
            )
            test, runtime, smoke = (step_by_id(job, name) for name in ("test", "runtime", "smoke"))
            self.assertEqual(action_steps(job, "docker/build-push-action"), [test, runtime])
            self.assertEqual(
                runtime["with"],
                {
                    "context": "${{ env.IMAGE_CONTEXT }}",
                    "target": "runtime",
                    "platforms": "linux/amd64",
                    "push": "false",
                    "load": "true",
                    "provenance": "false",
                    "tags": "${{ env.LOCAL_IMAGE }}",
                },
            )
            self.assertLess(job["steps"].index(test), job["steps"].index(runtime))
            self.assertLess(job["steps"].index(runtime), job["steps"].index(smoke))
            self.assertEqual(job["env"]["SMOKE_KIND"], kind)
            self.assertIn(
                'python3 scripts/smoke_image.py "$SMOKE_KIND" "$LOCAL_IMAGE" --local',
                smoke["run"],
            )
            self.assertNotRegex(json.dumps(job), r"docker (push|login)|ghcr\.io|attest-build")

    def test_test_builds_have_no_publishing_and_cannot_reuse_cached_tests(self):
        for workflow in self.workflows():
            for kind in IMAGES:
                build = step_by_id(workflow["jobs"][kind], "test")["with"]
                self.assertEqual(
                    build,
                    {
                        "context": "${{ env.IMAGE_CONTEXT }}",
                        "target": "test",
                        "platforms": "linux/amd64",
                        "no-cache": "true",
                        "push": "false",
                        "load": "false",
                        "provenance": "false",
                    },
                )

    def test_publish_tests_before_login_runtime_push_digest_smoke_and_attestation(self):
        for kind in IMAGES:
            job = self.publish["jobs"][kind]
            test = step_by_id(job, "test")
            build = step_by_id(job, "build")
            (login,) = action_steps(job, "docker/login-action")
            (attest,) = action_steps(job, "actions/attest-build-provenance")
            sequence = [test, login, build, step_by_id(job, "smoke"), attest]
            indices = [job["steps"].index(step) for step in sequence]
            self.assertEqual(indices, sorted(indices))
            self.assertEqual(action_steps(job, "docker/build-push-action"), [test, build])
            self.assertEqual(
                login["with"],
                {
                    "registry": "ghcr.io",
                    "username": "${{ github.actor }}",
                    "password": "${{ secrets.GITHUB_TOKEN }}",
                },
            )
            self.assertEqual(build["with"]["target"], "runtime")
            self.assertEqual(build["with"]["context"], test["with"]["context"])
            self.assertEqual(build["with"]["platforms"], "linux/amd64")
            self.assertEqual(build["with"]["push"], "true")
            self.assertEqual(build["with"]["provenance"], "mode=max")
            self.assertEqual(
                build["with"]["tags"],
                "${{ env.IMAGE_NAME }}:source-${{ steps.source.outputs.sha }}",
            )
            self.assertEqual(job["outputs"], {"digest": "${{ steps.build.outputs.digest }}"})
            self.assertEqual(
                step_by_id(job, "smoke")["env"], {"DIGEST": "${{ steps.build.outputs.digest }}"}
            )
            self.assertEqual(
                attest["with"],
                {
                    "subject-name": "${{ env.IMAGE_NAME }}",
                    "subject-digest": "${{ steps.build.outputs.digest }}",
                    "push-to-registry": "true",
                },
            )

    def test_runtime_metadata_records_checked_out_source_revision_and_pinned_base(self):
        for kind in IMAGES:
            job = self.publish["jobs"][kind]
            labels = step_by_id(job, "build")["with"]["labels"].splitlines()
            self.assertEqual(
                labels,
                [
                    "org.opencontainers.image.source=${{ env.SOURCE_URL }}",
                    "org.opencontainers.image.revision=${{ steps.source.outputs.sha }}",
                    "org.opencontainers.image.base.name=${{ steps.metadata.outputs.base_image }}",
                    "org.opencontainers.image.base.digest="
                    "${{ steps.metadata.outputs.base_digest }}",
                    "org.opencontainers.image.description=${{ env.IMAGE_DESCRIPTION }}",
                ],
            )
            self.assertEqual(
                step_by_id(job, "build")["with"]["annotations"].splitlines(),
                ["index:org.opencontainers.image.description=${{ env.IMAGE_DESCRIPTION }}"],
            )
            description = job["env"]["IMAGE_DESCRIPTION"]
            self.assertTrue(0 < len(description) <= 200, description)
            metadata = step_by_id(job, "metadata")
            self.assertLess(
                job["steps"].index(metadata), job["steps"].index(step_by_id(job, "test"))
            )
            self.assertIn(
                'python3 scripts/source_delta.py >> "$GITHUB_STEP_SUMMARY"', metadata["run"]
            )
            self.assertIn('["git", "rev-parse", "HEAD"]', metadata["run"])

    def test_anonymous_gate_is_fresh_read_only_and_consumes_both_job_digests(self):
        gate = self.publish["jobs"]["anonymous_pull"]
        self.assertEqual(set(gate["needs"]), set(IMAGES))
        self.assertNotIn("outputs", gate)
        self.assertEqual(
            [step["uses"].split("@")[0] for step in gate["steps"] if "uses" in step],
            ["actions/checkout"],
        )
        serialized = json.dumps({"env": self.publish["env"], "job": gate})
        for forbidden in (
            "secrets.",
            "github.token",
            "GITHUB_TOKEN",
            "GH_TOKEN",
            "docker login",
            "download-artifact",
            "cache-from",
            "self-hosted",
        ):
            self.assertNotIn(forbidden, serialized)
        for kind, (_, image, _) in IMAGES.items():
            self.assertEqual(gate["env"][kind.upper() + "_IMAGE"], image)
            self.assertEqual(
                gate["env"][kind.upper() + "_DIGEST"], "${{ needs." + kind + ".outputs.digest }}"
            )
        config = step_by_id(gate, "anonymous_config")
        self.assertLess(
            gate["steps"].index(config), gate["steps"].index(step_by_id(gate, "anonymous_smoke"))
        )
        self.assertIn("Public", config["run"])
        self.assertIn("UI", config["run"])
        self.assertIn("release blocked", config["run"])
        self.assertEqual(gate["steps"][-1]["if"], "${{ failure() }}")
        self.assertIn("BLOCKED", gate["steps"][-1]["run"])


class ExecutableGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.publish = load_workflow("publish-runtime-images.yml")
        cls.validate = load_workflow("validate-runtime-images.yml")

    def test_source_guard_accepts_only_the_exact_workflow_sha_on_main_history(self):
        script = step_by_id(self.publish["jobs"]["gateway"], "source")["run"]
        with tempfile.TemporaryDirectory() as directory:

            def git(*args):
                result = subprocess.run(
                    ["git", *args],
                    cwd=directory,
                    text=True,
                    capture_output=True,
                    check=True,
                    env={
                        "PATH": os.environ.get("PATH", os.defpath),
                        "HOME": directory,
                        "GIT_CONFIG_NOSYSTEM": "1",
                        "GIT_CONFIG_GLOBAL": os.devnull,
                    },
                    timeout=15,
                )
                return result.stdout.strip()

            git("init", "--initial-branch=main")
            git("config", "user.email", "workflow-tests@example.invalid")
            git("config", "user.name", "Workflow tests")
            git("config", "commit.gpgsign", "false")
            git("commit", "--allow-empty", "-m", "base")
            ancestor = git("rev-parse", "HEAD")
            git("commit", "--allow-empty", "-m", "main tip")
            main = git("rev-parse", "HEAD")
            git("update-ref", "refs/remotes/origin/main", main)
            git("checkout", "--detach", ancestor)
            git("commit", "--allow-empty", "-m", "unreviewed side branch")
            side = git("rev-parse", "HEAD")
            cases = [
                ("main tip", main, main, main, True),
                ("main ancestor", ancestor, ancestor, ancestor, True),
                ("checkout differs", ancestor, main, main, False),
                ("workflow differs", main, main, ancestor, False),
                ("input differs", main, ancestor, main, False),
                ("abbreviated input", main, main[:12], main, False),
                ("branch input", main, "main", main, False),
                ("shell metacharacters", main, "$(touch injected)", main, False),
                ("divergent commit", side, side, side, False),
            ]
            for name, head, source, workflow_sha, success in cases:
                with self.subTest(name=name):
                    git("checkout", "--detach", head)
                    output = Path(directory, "outputs")
                    output.unlink(missing_ok=True)
                    result = run_shell(
                        script, directory, {"SOURCE_SHA": source, "GITHUB_SHA": workflow_sha}
                    )
                    self.assertEqual(
                        result.returncode == 0, success, result.stdout + result.stderr
                    )
                    if success:
                        self.assertEqual(output.read_text(encoding="utf-8"), f"sha={source}\n")
                    else:
                        self.assertFalse(output.exists())
                    self.assertFalse(Path(directory, "injected").exists())
            git("checkout", "--detach", main)
            git("update-ref", "-d", "refs/remotes/origin/main")
            result = run_shell(script, directory, {"SOURCE_SHA": main, "GITHUB_SHA": main})
            self.assertNotEqual(result.returncode, 0, "Missing origin/main must fail closed")

    def test_every_disk_gate_enforces_bytes_on_each_storage_path_without_cleanup(self):
        paths = ("/var/lib/docker", "/workspace", "/runner-temp")
        for workflow in (self.validate, self.publish):
            kinds = ("gateway", "qwen")
            if workflow is self.publish:
                kinds += ("anonymous_pull",)
            for kind in kinds:
                job = workflow["jobs"][kind]
                code = inline_python(step_by_id(job, "disk"))
                required = int(job["env"]["MIN_FREE_GIB"]) * 1024**3
                for low_path in (None, *paths):
                    with self.subTest(kind=kind, low_path=low_path):
                        calls = []

                        def disk_usage(path, calls=calls, required=required, low_path=low_path):
                            calls.append(path)
                            return mock.Mock(free=required - 1 if path == low_path else required)

                        with (
                            mock.patch.dict(
                                os.environ,
                                {
                                    "MIN_FREE_GIB": job["env"]["MIN_FREE_GIB"],
                                    "GITHUB_WORKSPACE": paths[1],
                                    "RUNNER_TEMP": paths[2],
                                },
                            ),
                            mock.patch("shutil.disk_usage", side_effect=disk_usage),
                            contextlib.redirect_stdout(io.StringIO()),
                        ):
                            if low_path is None:
                                exec(compile(code, "<workflow disk gate>", "exec"), {})
                                self.assertEqual(calls, list(paths))
                            else:
                                with self.assertRaises(SystemExit) as error:
                                    exec(compile(code, "<workflow disk gate>", "exec"), {})
                                self.assertEqual(error.exception.code, 1)
                                self.assertEqual(calls[-1], low_path)

    def test_metadata_requires_digest_pinning_and_records_actual_revision(self):
        code = inline_python(step_by_id(self.publish["jobs"]["gateway"], "metadata"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / "images" / "example"
            context.mkdir(parents=True)
            output = root / "outputs"
            summary = root / "summary"
            pinned_base = "docker.io/example/runtime@" + DIGEST
            for base, accepted in (
                (pinned_base, True),
                ("docker.io/example/runtime:latest", False),
                ("docker.io/example/runtime@sha256:abcd", False),
            ):
                with self.subTest(base=base):
                    (context / "Dockerfile").write_text(
                        f"FROM {base} AS upstream\nFROM upstream AS test\n"
                    )
                    output.unlink(missing_ok=True)
                    summary.unlink(missing_ok=True)
                    with (
                        mock.patch.dict(
                            os.environ,
                            {
                                "IMAGE_CONTEXT": str(context),
                                "IMAGE_NAME": IMAGES["gateway"][1],
                                "SOURCE_URL": "https://github.com/example/repo",
                                "SOURCE_SHA": "b" * 40,
                                "GITHUB_OUTPUT": str(output),
                                "GITHUB_STEP_SUMMARY": str(summary),
                            },
                        ),
                        mock.patch("subprocess.check_output", return_value="a" * 40 + "\n") as git,
                    ):
                        if accepted:
                            exec(compile(code, "<workflow metadata>", "exec"), {})
                            git.assert_called_once_with(["git", "rev-parse", "HEAD"], text=True)
                            self.assertEqual(
                                output.read_text(),
                                f"base_image={pinned_base}\nbase_digest={DIGEST}\n",
                            )
                            self.assertIn("a" * 40, summary.read_text())
                            self.assertNotIn("b" * 40, summary.read_text())
                            self.assertIn(pinned_base, summary.read_text())
                        else:
                            with self.assertRaises(SystemExit):
                                exec(compile(code, "<workflow metadata>", "exec"), {})
                            self.assertFalse(output.exists())


class DigestSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.publish = load_workflow("publish-runtime-images.yml")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "commands.jsonl"
        binary = self.root / "bin"
        binary.mkdir()
        # These spies NEVER call Docker or the parent's smoke helper.
        spy = (
            f"#!{sys.executable}\n"
            + """import json, os, pathlib, sys
command = pathlib.Path(sys.argv[0]).name
record = {"command": command, "args": sys.argv[1:], "config": os.environ.get("DOCKER_CONFIG")}
with open(os.environ["COMMAND_LOG"], "a") as output:
    output.write(json.dumps(record) + "\\n")
if os.environ.get("FAIL_COMMAND") == command:
    sys.exit(1)
"""
        )
        for name in ("docker", "python3"):
            executable = binary / name
            executable.write_text(spy)
            executable.chmod(0o755)
        self.env = {
            "PATH": str(binary) + os.pathsep + os.environ.get("PATH", os.defpath),
            "COMMAND_LOG": str(self.log),
        }

    def commands(self):
        return (
            [json.loads(line) for line in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )

    def test_publisher_smokes_exact_build_digest_not_source_tag(self):
        for kind, (_, image, _) in IMAGES.items():
            with self.subTest(kind=kind):
                self.log.unlink(missing_ok=True)
                script = step_by_id(self.publish["jobs"][kind], "smoke")["run"]
                result = run_shell(
                    script,
                    self.root,
                    dict(self.env, IMAGE_NAME=image, DIGEST=DIGEST, SMOKE_KIND=kind),
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(
                    [(record["command"], record["args"]) for record in self.commands()],
                    [
                        ("docker", ["pull", image + "@" + DIGEST]),
                        ("python3", ["scripts/smoke_image.py", kind, image + "@" + DIGEST]),
                    ],
                )

    def test_publisher_rejects_missing_malformed_or_tag_outputs_without_pulling(self):
        for digest in ("", "latest", "source-" + "a" * 40, "sha256:abcd", DIGEST + "\n"):
            with self.subTest(digest=digest):
                script = step_by_id(self.publish["jobs"]["gateway"], "smoke")["run"]
                result = run_shell(
                    script,
                    self.root,
                    dict(
                        self.env,
                        IMAGE_NAME=IMAGES["gateway"][1],
                        DIGEST=digest,
                        SMOKE_KIND="gateway",
                    ),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.commands(), [])

    def anonymous_env(self):
        gate = self.publish["jobs"]["anonymous_pull"]
        inherited = self.root / "inherited-docker-config"
        inherited.mkdir(exist_ok=True)
        (inherited / "config.json").write_text(
            '{"auths":{"ghcr.io":{"auth":"NOT_A_REAL_CREDENTIAL"}}}'
        )
        env = dict(
            self.env,
            DOCKER_CONFIG=str(inherited),
            DOCKER_AUTH_CONFIG="unwanted auth",
            DOCKER_CONTEXT="unwanted context",
            DOCKER_HOST="unwanted host",
        )
        result = run_shell(step_by_id(gate, "anonymous_config")["run"], self.root, env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        env.update(
            line.split("=", 1) for line in (self.root / "environment").read_text().splitlines()
        )
        fresh = Path(env["DOCKER_CONFIG"])
        self.assertNotEqual(fresh, inherited)
        self.assertEqual(fresh.parent, self.root)
        self.assertEqual(list(fresh.iterdir()), [])
        self.assertEqual(fresh.stat().st_mode & 0o777, 0o700)
        for key in ("DOCKER_AUTH_CONFIG", "DOCKER_CONTEXT", "DOCKER_HOST"):
            self.assertEqual(env[key], "")
        env.update(
            GATEWAY_IMAGE=IMAGES["gateway"][1],
            QWEN_IMAGE=IMAGES["qwen"][1],
            GATEWAY_DIGEST=DIGEST,
            QWEN_DIGEST=OTHER_DIGEST,
        )
        return env

    def test_anonymous_pull_uses_empty_config_and_both_independent_digest_outputs(self):
        env = self.anonymous_env()
        script = step_by_id(self.publish["jobs"]["anonymous_pull"], "anonymous_smoke")["run"]
        result = run_shell(script, self.root, env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            [(record["command"], record["args"]) for record in self.commands()],
            [
                ("docker", ["pull", IMAGES["gateway"][1] + "@" + DIGEST]),
                (
                    "python3",
                    ["scripts/smoke_image.py", "gateway", IMAGES["gateway"][1] + "@" + DIGEST],
                ),
                ("docker", ["pull", IMAGES["qwen"][1] + "@" + OTHER_DIGEST]),
                (
                    "python3",
                    ["scripts/smoke_image.py", "qwen", IMAGES["qwen"][1] + "@" + OTHER_DIGEST],
                ),
            ],
        )
        for record in self.commands():
            self.assertEqual(record["config"], env["DOCKER_CONFIG"])
        self.assertIn("passed for both", (self.root / "summary").read_text())

    def test_anonymous_gate_refuses_nonempty_config_before_pulling(self):
        env = self.anonymous_env()
        Path(env["DOCKER_CONFIG"], "config.json").write_text("{}", encoding="utf-8")
        script = step_by_id(self.publish["jobs"]["anonymous_pull"], "anonymous_smoke")["run"]
        self.assertNotEqual(run_shell(script, self.root, env).returncode, 0)
        self.assertEqual(self.commands(), [])

    def test_anonymous_gate_rejects_either_invalid_digest_before_pulling(self):
        env = self.anonymous_env()
        script = step_by_id(self.publish["jobs"]["anonymous_pull"], "anonymous_smoke")["run"]
        for kind in IMAGES:
            with self.subTest(kind=kind):
                result = run_shell(
                    script, self.root, dict(env, **{kind.upper() + "_DIGEST": "latest"})
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.commands(), [])

    def test_failed_pull_or_smoke_blocks_publish_and_anonymous_gate(self):
        anon_env = self.anonymous_env()
        cases = [
            (
                step_by_id(self.publish["jobs"]["anonymous_pull"], "anonymous_smoke")["run"],
                anon_env,
            )
        ]
        for kind, (_, image, _) in IMAGES.items():
            cases.append(
                (
                    step_by_id(self.publish["jobs"][kind], "smoke")["run"],
                    dict(self.env, IMAGE_NAME=image, DIGEST=DIGEST, SMOKE_KIND=kind),
                )
            )
        for script, env in cases:
            for command in ("docker", "python3"):
                with self.subTest(script=script.splitlines()[-1], command=command):
                    self.log.unlink(missing_ok=True)
                    (self.root / "summary").unlink(missing_ok=True)
                    result = run_shell(script, self.root, dict(env, FAIL_COMMAND=command))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(len(self.commands()), 1 if command == "docker" else 2)
                    self.assertFalse(
                        (self.root / "summary").exists(),
                        "Failure must not record a successful verification",
                    )


if __name__ == "__main__":
    unittest.main()
