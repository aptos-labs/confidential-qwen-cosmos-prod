"""Check an exact image under read-only/no-network/no-GPU runtime constraints."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "gateway": "ghcr.io/aptos-labs/confidential-qwen-cosmos-paid-gateway",
    "qwen": "ghcr.io/aptos-labs/confidential-qwen-cosmos-qwen-metered",
}
SECRET = "synthetic-image-smoke-reporter-secret-0123456789"
GATEWAY = r"""
import http.client,json,os,pathlib,subprocess,sys,time
assert sys.executable == '/app/.venv/bin/python3',sys.executable
root=pathlib.Path('/opt/ccs-gateway/paid_gateway'); expected=pathlib.Path('/expected')
for directory in (root.parent,root):
 assert directory.stat().st_mode & 0o777 == 0o555,directory
assert {p.name for p in root.glob('*.py')}=={p.name for p in expected.glob('*.py')}
for source in expected.glob('*.py'):
 assert (root/source.name).read_bytes()==source.read_bytes(),source.name
assert not pathlib.Path('/opt/test-env').exists()
assert not pathlib.Path('/opt/ccs-gateway/tests').exists()
scenario=os.environ['SMOKE_CASE']; log=pathlib.Path('/tmp/startup.log')
child_env=dict(os.environ)
if scenario=='overrides':
 child_env.update(PATH='/tmp/untrusted',PYTHONPATH='/tmp/untrusted',
                  PYTHONHOME='/tmp/untrusted',PYTHONUSERBASE='/tmp/untrusted',
                  PYTHONSTARTUP='/tmp/untrusted')
args=['/opt/ccs-gateway/entrypoint.sh']
if scenario=='arguments': args+=['--config','/tmp/untrusted.yaml']
with log.open('wb') as output:
 process=subprocess.Popen(args,env=child_env,stdout=output,stderr=subprocess.STDOUT)
 def request(method,path,body=None,headers=None):
  c=http.client.HTTPConnection('127.0.0.1',8080,timeout=45)
  c.request(method,path,body=body,headers=headers or {})
  r=c.getresponse(); status=r.status; data=r.read(); c.close(); return status,data
 try:
  deadline=time.monotonic()+75
  ready=False
  while time.monotonic()<deadline:
   if process.poll() is not None: break
   try:
    if request('GET','/v1/models')[0]==200: ready=True; break
   except (OSError,http.client.HTTPException): pass
   time.sleep(0.2)
  if scenario not in {'valid','overrides'}:
   assert not ready,'invalid startup became reachable'
   assert process.poll() not in (None,0),'invalid startup did not fail'
   text=log.read_text(errors='replace')
   errors={'hooks':'does not accept worker hooks','arguments':'does not accept arguments'}
   expected_error=errors.get(scenario,'USAGE_REPORTER_SECRET is missing or invalid')
   assert expected_error in text,text[-2000:]
  else:
   assert ready,log.read_text(errors='replace')[-3000:]
   cmdline=pathlib.Path(f'/proc/{process.pid}/cmdline').read_bytes().split(b'\x00')
   assert cmdline[:3]==[b'/app/.venv/bin/python3',b'-m',b'paid_gateway'],cmdline[:3]
   headers={'Content-Type':'application/json','Authorization':'Bearer synthetic-unmapped-key'}
   body=json.dumps({'model':'qwen3-omni','messages':[{'role':'user','content':'fixture'}],'max_tokens':8})
   status,data=request('POST','/v1/chat/completions',body,headers)
   assert status==503 and json.loads(data)['error']['code']=='billing_unavailable',(status,data)
   boundary='smoke-boundary'
   fields={'model':'cosmos3-super','prompt':'fixture','size':'832x480','num_frames':'189',
           'num_inference_steps':'2'}
   body=''.join('--'+boundary+'\r\nContent-Disposition: form-data; name="'
                +k+'"\r\n\r\n'+v+'\r\n' for k,v in fields.items())+'--'+boundary+'--\r\n'
   headers={'Content-Type':'multipart/form-data; boundary='+boundary,
            'Authorization':'Bearer synthetic-unmapped-key'}
   status,data=request('POST','/v1/videos/sync',body,headers)
   assert status==503 and json.loads(data)['error']['code']=='billing_unavailable',(status,data)
  print(json.dumps({'case':scenario,'passed':True}))
 finally:
  if process.poll() is None:
   process.terminate()
   try: process.wait(timeout=20)
   except subprocess.TimeoutExpired: process.kill();process.wait(timeout=10)
"""
QWEN = r"""
import importlib.metadata,json,pathlib,sys
artifact=pathlib.Path('/opt/model-metering'); expected=pathlib.Path('/expected')
assert artifact.stat().st_mode & 0o777 == 0o555,artifact
assert not (artifact/'tests').exists()
for name in ('install.py','input_meter.py','manifest.json','vllm-omni-v0.28.0.patch'):
 assert (artifact/name).read_bytes()==(expected/name).read_bytes(),name
sys.path.insert(0,str(artifact));from install import preflight
assert importlib.metadata.version('vllm-omni')=='0.28.0'
assert importlib.metadata.version('vllm')=='0.28.0'
root=pathlib.Path(importlib.metadata.distribution('vllm-omni').locate_file('vllm_omni')).resolve()
assert preflight(root).already_installed,'pristine image is not a metered image'
import vllm_omni.input_meter
import vllm_omni.engine.messages
import vllm_omni.engine.orchestrator
import vllm_omni.entrypoints.omni_base
print(json.dumps({'case':'qwen-installed-meter','passed':True}))
"""


def run_image(kind: str, image: str, scenario: str) -> None:
    name = "ccs-image-smoke-" + uuid.uuid4().hex
    expected = ROOT / (
        "images/paid-gateway/paid_gateway"
        if kind == "gateway"
        else "images/qwen-metered/model-metering"
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=512m",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "XDG_CACHE_HOME=/tmp/cache",
        "--env",
        "SMOKE_CASE=" + scenario,
        "--mount",
        f"type=bind,source={expected},target=/expected,readonly",
    ]
    if kind == "gateway":
        if scenario in {"valid", "hooks", "arguments", "overrides"}:
            command += ["--env", "USAGE_REPORTER_SECRET=" + SECRET]
        elif scenario == "short":
            command += ["--env", "USAGE_REPORTER_SECRET=short"]
        elif scenario == "whitespace":
            command += ["--env", "USAGE_REPORTER_SECRET=" + SECRET + " "]
        if scenario == "hooks":
            command += ["--env", "LITELLM_WORKER_STARTUP_HOOKS=os:path"]
        command += [
            "--env",
            "CCS_BILLING_REQUIRED=0",
            "--env",
            "CCS_BILLING_URL=https://invalid.example",
        ]
    interpreter = "/app/.venv/bin/python3" if kind == "gateway" else "python3"
    command += ["--entrypoint", interpreter, image, "-c", GATEWAY if kind == "gateway" else QWEN]
    try:
        subprocess.run(command, check=True, timeout=210)
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def image_ref_allowed(kind: str, image: str, local: bool) -> bool:
    if re.fullmatch(re.escape(PACKAGES[kind]) + r"@sha256:[0-9a-f]{64}", image):
        return True
    return (
        local
        and len(image) <= 255
        and re.fullmatch(
            r"[a-z0-9]+(?:[._/-][a-z0-9]+)*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?", image
        )
        is not None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=PACKAGES)
    parser.add_argument("image")
    parser.add_argument("--local", action="store_true", help="allow a development-only local tag")
    args = parser.parse_args()
    if not image_ref_allowed(args.kind, args.image, args.local):
        parser.error("expected a package digest, or a safe local tag with --local")
    info = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image]))[0]
    assert info["Architecture"] == "amd64" and info["Os"] == "linux"
    if args.kind == "gateway":
        assert info["Config"]["Entrypoint"] == ["/opt/ccs-gateway/entrypoint.sh"]
        assert info["Config"].get("Cmd") in (None, [])
    scenarios = (
        ["missing", "short", "whitespace", "hooks", "arguments", "overrides", "valid"]
        if args.kind == "gateway"
        else ["installed"]
    )
    for scenario in scenarios:
        run_image(args.kind, args.image, scenario)


if __name__ == "__main__":
    main()
