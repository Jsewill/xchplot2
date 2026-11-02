#!/usr/bin/env python3
"""Host-only checks for private network fixture transport and namespace guards."""
import base64
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
import time

script = Path(__file__).with_name("compute-network.py")
network = runpy.run_path(str(script))


def rpc(root, op, code=0, **values):
    result = subprocess.run([sys.executable, script, "--rpc"],
                            input=json.dumps(dict(root=str(root), op=op, **values)),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == code, result.stderr
    return json.loads(result.stdout) if code == 0 else result.stderr


with tempfile.TemporaryDirectory(prefix="xchplot2-network-") as directory:
    root = Path(directory)
    data = b"private disposable invitation\x00\xff\n"
    rpc(root, "write", path="invitation", data=base64.b64encode(data).decode())
    assert (root / "invitation").stat().st_mode & 0o777 == 0o600
    assert base64.b64decode(rpc(root, "read", path="invitation")) == data
    assert rpc(root, "digest", path="invitation") == hashlib.sha256(data).hexdigest()
    rpc(root, "write", path="staged-binary", data=base64.b64encode(b"fixture executable").decode(), executable=True)
    assert (root / "staged-binary").stat().st_mode & 0o777 == 0o700
    assert "escaped" in rpc(root, "read", path="../other-file", code=1)
    assert "unit escaped" in rpc(root, "stop_unit", unit="unowned.scope", code=1)
    assert "invalid temporary" in rpc(root.parent / "unowned-root", "cleanup", code=1)
    result = rpc(root, "run", argv=[sys.executable, "-c", "import sys; print('output'); print('diagnostic',file=sys.stderr); sys.exit(7)"], env={})
    assert result == dict(code=7, stdout="output\n", stderr="diagnostic\n")
    pid = rpc(root, "start", argv=[sys.executable, "-c", "print('completed')"], env={}, name="completed")
    assert pid > 0
    until = time.monotonic() + 5
    while rpc(root, "poll", path="completed.exit") is None:
        assert time.monotonic() < until, "fixture watcher never recorded child completion"
        time.sleep(0.02)
    assert rpc(root, "poll", path="completed.exit") == 0
    assert (root / "completed.log").read_text() == "completed\n"
    # A stale/reused PID must never let fixture cleanup signal another command.
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        rpc(root, "signal", pid=unrelated.pid, signal=signal.SIGTERM, unit="unowned-unit")
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)

# Losing one SSH host must not retain another participant's credentials.
with tempfile.TemporaryDirectory(prefix="xchplot2-network-") as directory:
    parent = Path(directory)
    calls = []

    class Participant:
        def __init__(self, name, remote):
            self.name, self.remote, self.root = name, remote, parent / name

        def cleanup(self):
            calls.append(self.name)
            if self.remote:
                raise RuntimeError("fixture SSH host unavailable")

    errors = network["cleanup_nodes"]([Participant("local", False), Participant("remote", True)], local_roots=True)
    assert calls == ["remote", "local"] and errors[0]["participant"] == "remote"
    assert not parent.exists(), "unreachable SSH host prevented local credential cleanup"

try:
    network["Network"](os.readlink("/proc/self/ns/net"), os.readlink("/proc/self/ns/user")).create()
except RuntimeError as error:
    assert "fresh user and network namespaces" in str(error)
else:
    raise AssertionError("harness allowed mutation of its original host namespace")

valid = {"hosts": [dict(ssh="explicit-host", binary="/opt/xchplot2", ipv4="192.0.2.1") for _ in network["NAMES"]]}
assert network["validate_hosts"](valid) == valid["hosts"]
for key, value in (("ssh", "-oProxyCommand=unexpected"), ("binary", "relative"), ("ipv4", "::1"),
                   ("interfaces", ["../../host-interface"]), ("scratch", "relative")):
    invalid = json.loads(json.dumps(valid))
    invalid["hosts"][0][key] = value
    try:
        network["validate_hosts"](invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted invalid explicit host {key}")
print("private network fixture transport, cleanup identity and namespace guards passed")
