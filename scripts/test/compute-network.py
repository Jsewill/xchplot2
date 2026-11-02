#!/usr/bin/env python3
"""Qualify automatic compute on isolated Linux networks or explicit SSH hosts.

Local: python3 scripts/test/compute-network.py /path/to/xchplot2 --logs DIR
Remote: add --hosts hosts.json. The JSON contains a `hosts` list with three
entries (storage A, storage B, worker), each with `ssh`, absolute `binary`,
`ipv4`, optional `ipv6`, optional `interfaces`, and optional `environment`.
Use `local: true` instead of `ssh` for participants on this machine. An
optional `stage_binary` copies a local binary into the private remote root;
`scratch` selects its parent directory. Hosts must already have Python,
systemd user scopes and the binary's runtime installed. No package,
service, interface or firewall changes are made on SSH hosts. Disposable pool
credentials and plots live in private temporary directories and are removed.
Only isolated namespaces exercise firewall and address mutation. Namespace
results never certify separate physical machines or their firewall settings.
The worker entry can set `qualify_gpu: true`, `gpu_backend` (the AdaptiveCpp
visibility mask), `gpu_host_bytes` and `memory_max` for an additional forced
GPU case. Each SSH command has a hard transient scope cap (`2G` by default).
Set top-level `ssh_forwarded: true` to test application mTLS through private
loopback SSH forwards; the summary labels this separately from direct LAN.
"""
import argparse
import base64
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time

FARMER = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
SEED = "ab" * 32
POOL = "42" * 32
NAMES = ("storage-a", "storage-b", "worker")
SSH_CONTROL_PATH = None


def checked(argv, **kwargs):
    result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=240, **kwargs)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {shlex.join(list(map(str, argv)))}\n{result.stdout}{result.stderr}")
    return result.stdout


def ssh_command(host, argv):
    options = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10"]
    if SSH_CONTROL_PATH:
        options += ["-o", "ControlMaster=auto", "-o", "ControlPersist=60", "-o", "ControlPath=" + SSH_CONTROL_PATH]
    return [*options, host, shlex.join(list(map(str, argv)))]


def validate_hosts(value):
    hosts = value["hosts"]
    if len(hosts) != len(NAMES):
        raise ValueError("hosts must contain storage A, storage B and worker entries")
    for host in hosts:
        if host.get("local") and host.get("ssh"):
            raise ValueError("local participants cannot also specify SSH")
        if not host.get("local") and (not isinstance(host["ssh"], str) or not host["ssh"] or host["ssh"].startswith("-")):
            raise ValueError("ssh must name an explicitly authorized host")
        if not Path(host["binary"]).is_absolute():
            raise ValueError("remote binary must be an absolute path")
        if host.get("stage_binary") and not Path(host["stage_binary"]).is_absolute():
            raise ValueError("staged binary must be an absolute local path")
        if host.get("scratch") and not Path(host["scratch"]).is_absolute():
            raise ValueError("remote scratch must be an absolute path")
        if not re.fullmatch(r"[1-9][0-9]*[KMG]", host.get("memory_max", "2G")):
            raise ValueError("memory_max must be a positive systemd K/M/G size")
        if host.get("qualify_gpu") and host.get("gpu_backend") not in ("cuda", "hip", "ze", "ocl"):
            raise ValueError("GPU qualification requires an explicit AdaptiveCpp visibility mask")
        if ipaddress.ip_address(host["ipv4"]).version != 4:
            raise ValueError("ipv4 must be an IPv4 address")
        if host.get("ipv6") and ipaddress.ip_address(host["ipv6"]).version != 6:
            raise ValueError("ipv6 must be an IPv6 address")
        for interface in host.get("interfaces", []):
            if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", interface):
                raise ValueError("invalid Linux interface name")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in host.get("environment", {}).items()):
            raise ValueError("environment must map strings to strings")
    return hosts


def rpc():
    """Small SSH transport; requests and private file contents stay off logs."""
    request = json.load(sys.stdin)
    root = Path(request["root"])
    if not root.is_absolute() or ".." in root.parts or not root.name.startswith("xchplot2-network-"):
        raise ValueError("invalid temporary network fixture root")
    op = request["op"]
    path = root / request.get("path", "")
    if not path.is_relative_to(root) or ".." in path.parts:
        raise ValueError("fixture path escaped its temporary root")
    if op == "run":
        result = subprocess.run(request["argv"], env={**os.environ, **request["env"]},
                                capture_output=True, text=True, timeout=240)
        output = dict(code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    elif op == "start":
        name = request["name"]
        if not re.fullmatch(r"[a-z0-9-]+", name):
            raise ValueError("invalid process label")
        record = root / (name + ".request")
        record.write_text(json.dumps(request))
        record.chmod(0o600)
        process = subprocess.Popen([sys.executable, __file__, "--watch", str(record)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        output = process.pid
    elif op == "poll":
        output = json.loads(path.read_text()) if path.exists() else None
    elif op == "signal":
        try:
            command = Path(f"/proc/{request['pid']}/cmdline").read_bytes()
            if str(root).encode() in command and b"--watch" in command:
                subprocess.run(["systemctl", "--user", "kill", "--signal", str(request["signal"]), request["unit"]],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                os.killpg(request["pid"], request["signal"])
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass
        output = None
    elif op == "read":
        output = None if request.get("missing_ok") and not path.exists() else base64.b64encode(path.read_bytes()).decode()
    elif op == "write":
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700 if request.get("executable") else 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(base64.b64decode(request["data"], validate=True))
        output = None
    elif op == "digest":
        with path.open("rb") as stream:
            output = hashlib.file_digest(stream, "sha256").hexdigest()
    elif op == "identity":
        output = hashlib.sha256(Path("/etc/machine-id").read_bytes()).hexdigest()
    elif op == "stop_unit":
        if not request["unit"].startswith(root.name + "-") or not request["unit"].endswith(".scope"):
            raise ValueError("unit escaped its fixture root")
        subprocess.run(["systemctl", "--user", "stop", request["unit"]], capture_output=True, timeout=15)
        state = subprocess.run(["systemctl", "--user", "show", request["unit"],
                                "--property=LoadState", "--property=ActiveState"],
                               capture_output=True, text=True, timeout=15)
        values = dict(line.split("=", 1) for line in state.stdout.splitlines() if "=" in line)
        if state.returncode or (values.get("LoadState") != "not-found" and values.get("ActiveState") not in ("inactive", "failed")):
            raise RuntimeError("could not confirm fixture scope stopped")
        output = None
    elif op == "port":
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", request.get("port", 0)))
            output = probe.getsockname()[1]
    elif op == "cleanup":
        import shutil
        shutil.rmtree(root)
        output = None
    else:
        raise ValueError("unknown network fixture operation")
    json.dump(output, sys.stdout)


def watch(record):
    request = json.loads(record.read_text())
    root, name = record.parent, request["name"]
    with (root / (name + ".log")).open("w") as stream:
        result = subprocess.run(request["argv"], env={**os.environ, **request["env"]},
                                stdout=stream, stderr=subprocess.STDOUT)
    temporary = root / (name + ".exit.partial")
    temporary.write_text(json.dumps(result.returncode))
    temporary.replace(root / (name + ".exit"))


class Node:
    def __init__(self, name, root, binary, logs, prefix=(), remote=None):
        self.name, self.root, self.binary, self.logs = name, Path(root), str(binary), logs
        self.prefix, self.remote = list(map(str, prefix)), remote
        self.processes = []
        self.units = []
        self.stopped = set()
        self.counter = 0
        self.environment = {**os.environ, "XDG_CONFIG_HOME": str(self.root / "config")}
        self.environment.pop("XCHPLOT2_GROW_FAILPOINT", None)
        if remote:
            self.environment = {**remote.get("environment", {}), "XDG_CONFIG_HOME": str(self.root / "config")}
            with (logs / "cleanup.jsonl").open("a") as record:
                record.write(json.dumps(dict(ssh=remote["ssh"], root=str(self.root))) + "\n")
        self.environment.update({"XDG_CACHE_HOME": str(self.root / "cache"),
                                 "XDG_DATA_HOME": str(self.root / "data"),
                                 "TMPDIR": str(self.root / "tmp"),
                                 "ACPP_RT_NO_JIT_CACHE_POPULATION": "1", "CUDA_CACHE_DISABLE": "1",
                                 "CUDA_CACHE_PATH": str(self.root / "cache/cuda"),
                                 "AMD_COMGR_CACHE_DIR": str(self.root / "cache/comgr"),
                                 "ACPP_SSCP_FAILED_IR_DUMP_DIRECTORY": str(self.root / "jit"),
                                 "ACPP_HCF_DUMP_DIRECTORY": str(self.root / "jit")})
        if not remote:
            self.root.mkdir(mode=0o700)
            for name in ("tmp", "cache", "data", "jit"):
                (self.root / name).mkdir(mode=0o700)

    def unit(self, label):
        return self.root.name + "-" + label + ".scope"

    def remote_command(self, label, command):
        self.units.append(self.unit(label))
        with (self.logs / "cleanup.jsonl").open("a") as record:
            record.write(json.dumps(dict(ssh=self.remote["ssh"], root=str(self.root), unit=self.unit(label))) + "\n")
        return ["systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit", self.unit(label),
                "--property", "MemoryMax=" + self.remote.get("memory_max", "2G"),
                "--property", "MemorySwapMax=0", *command]

    def request(self, op, **values):
        request = dict(op=op, root=str(self.root), **values)
        result = checked(ssh_command(self.remote["ssh"], ["python3", self.root / "harness.py", "--rpc"]),
                         input=json.dumps(request))
        return json.loads(result)

    def run(self, *args):
        command = [self.binary, *map(str, args), "--config", "/dev/null"]
        if self.remote:
            command = self.remote_command(f"command-{self.counter}", command)
            result = self.request("run", argv=command, env=self.environment)
        else:
            done = subprocess.run([*self.prefix, *command], env=self.environment,
                                  capture_output=True, text=True, timeout=240)
            result = dict(code=done.returncode, stdout=done.stdout, stderr=done.stderr)
        self.counter += 1
        self.last_output = result["stdout"] + result["stderr"]
        (self.logs / f"{self.name}-{self.counter:03d}.log").write_text(self.last_output)
        if result["code"]:
            raise RuntimeError(f"{self.name} command failed: {' '.join(map(str, args[:2]))}\n{result['stdout']}{result['stderr']}")
        if "PRIVATE KEY" in result["stdout"] + result["stderr"]:
            raise AssertionError("private identity material exposed in diagnostics")
        return result["stdout"]

    def start(self, label, *args):
        command = [self.binary, *map(str, args), "--config", "/dev/null"]
        if self.remote:
            command = self.remote_command(label, command)
            process = self.request("start", argv=command, env=self.environment, name=label)
        else:
            stream = (self.root / (label + ".log")).open("w")
            process = subprocess.Popen([*self.prefix, *command], env=self.environment,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            stream.close()
        self.processes.append((label, process))
        if self.remote:
            with (self.logs / "cleanup.jsonl").open("a") as record:
                record.write(json.dumps(dict(ssh=self.remote["ssh"], root=str(self.root),
                                             pid=process, unit=self.unit(label))) + "\n")
        return label, process

    def poll(self, job):
        label, process = job
        return self.request("poll", path=label + ".exit") if self.remote else process.poll()

    def signal(self, job, value):
        if self.remote:
            self.request("signal", pid=job[1], signal=value, unit=self.unit(job[0]))
        elif job[1].poll() is None:
            os.killpg(job[1].pid, value)

    def read(self, path, missing_ok=False):
        relative = str(Path(path).relative_to(self.root))
        if self.remote:
            return base64.b64decode(self.request("read", path=relative, missing_ok=missing_ok) or "")
        return b"" if missing_ok and not Path(path).exists() else Path(path).read_bytes()

    def write(self, path, value, executable=False):
        if self.remote:
            self.request("write", path=str(Path(path).relative_to(self.root)),
                         data=base64.b64encode(value).decode(), executable=executable)
        else:
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700 if executable else 0o600), "wb") as stream:
                stream.write(value)

    def digest(self, path):
        if self.remote:
            return self.request("digest", path=str(Path(path).relative_to(self.root)))
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def finish(self, job, timeout=240):
        until = time.monotonic() + timeout
        while self.poll(job) is None:
            if time.monotonic() >= until:
                raise RuntimeError(f"{self.name}/{job[0]} timed out")
            time.sleep(0.1)
        output = self.read(self.root / (job[0] + ".log")).decode()
        (self.logs / (self.name + "-" + job[0] + ".log")).write_text(output)
        if self.poll(job):
            raise RuntimeError(output)
        paths = [Path(line) for line in output.splitlines() if line.endswith(".gplot")]
        if len(paths) != 1 or not paths[0].is_relative_to(self.root):
            raise AssertionError("plot command did not publish exactly one fixture output")
        return paths[0]

    def stop(self, job):
        if job[0] in self.stopped:
            return
        self.stopped.add(job[0])
        if self.poll(job) is not None:
            return
        self.signal(job, signal.SIGTERM)
        until = time.monotonic() + 3
        while self.poll(job) is None and time.monotonic() < until:
            time.sleep(0.1)
        if self.poll(job) is None:
            self.signal(job, signal.SIGKILL)
        if not self.remote:
            job[1].wait(timeout=10)

    def cleanup(self):
        for job in reversed(self.processes):
            self.stop(job)
            try:
                output = self.read(self.root / (job[0] + ".log"))
                (self.logs / (self.name + "-" + job[0] + ".log")).write_bytes(output)
            except (OSError, RuntimeError):
                pass
        if self.remote:
            for unit in reversed(self.units):
                self.request("stop_unit", unit=unit)
            self.request("cleanup")


class Network:
    """All interfaces, routes and firewall rules belong to a fresh namespace."""
    def __init__(self, outer_netns, outer_userns):
        self.holders = []
        self.prefixes = []
        self.outer_netns, self.outer_userns = outer_netns, outer_userns

    def create(self):
        if not self.outer_netns or not self.outer_userns or \
                os.readlink("/proc/self/ns/net") == self.outer_netns or \
                os.readlink("/proc/self/ns/user") == self.outer_userns:
            raise RuntimeError("network changes require fresh user and network namespaces")
        checked(["ip", "link", "set", "lo", "up"])
        for lane in (0, 1):
            checked(["ip", "link", "add", f"bridge{lane}", "type", "bridge"])
            checked(["ip", "link", "set", f"bridge{lane}", "up"])
        for index, name in enumerate(NAMES, 2):
            process = subprocess.Popen(["unshare", "--net", "sleep", "infinity"])
            self.holders.append(process)
            prefix = ["nsenter", "--target", str(process.pid), "--net"]
            # Wait until unshare has switched the holder out of our network.
            until = time.monotonic() + 5
            while os.readlink(f"/proc/{process.pid}/ns/net") == os.readlink("/proc/self/ns/net"):
                if time.monotonic() >= until:
                    raise RuntimeError("child network namespace did not initialize")
                time.sleep(0.02)
            self.prefixes.append(prefix)
            checked([*prefix, "ip", "link", "set", "lo", "up"])
            for lane in (0, 1):
                link, peer = f"v{index}-{lane}", f"p{index}-{lane}"
                checked(["ip", "link", "add", link, "type", "veth", "peer", "name", peer])
                checked(["ip", "link", "set", link, "master", f"bridge{lane}"])
                checked(["ip", "link", "set", link, "up"])
                checked(["ip", "link", "set", peer, "netns", process.pid])
                checked([*prefix, "ip", "link", "set", peer, "name", f"lan{lane}"])
                checked([*prefix, "ip", "addr", "add", f"192.0.{lane + 2}.{index}/24", "dev", f"lan{lane}"])
                checked([*prefix, "ip", "-6", "addr", "add", f"fd42:{lane + 2}::{index}/64", "dev", f"lan{lane}", "nodad"])
                checked([*prefix, "ip", "link", "set", f"lan{lane}", "up"])
        time.sleep(1)

    def command(self, index, *args, **kwargs):
        return checked([*self.prefixes[index], *args], **kwargs)

    def firewall(self, blocked):
        if blocked:
            self.command(2, "nft", "-f", "-", input="""table inet fixture {
chain output { type filter hook output priority 0; policy accept; meta l4proto tcp drop; }
chain input { type filter hook input priority 0; policy accept; meta l4proto tcp drop; }
}
""")
        else:
            self.command(2, "nft", "delete", "table", "inet", "fixture")

    def close(self):
        for process in reversed(self.holders):
            process.terminate()
            process.wait(timeout=10)

    def routed_worker(self):
        # Move the worker behind a router, with no shared multicast LAN. The
        # router is this harness's isolated parent namespace.
        checked(["sysctl", "-qw", "net.ipv4.ip_forward=1", "net.ipv6.conf.all.forwarding=1"])
        checked(["ip", "addr", "add", "192.0.2.1/24", "dev", "bridge0"])
        checked(["ip", "-6", "addr", "add", "fd42:2::1/64", "dev", "bridge0", "nodad"])
        checked(["ip", "link", "add", "router", "type", "veth", "peer", "name", "uplink"])
        checked(["ip", "link", "set", "uplink", "netns", self.holders[2].pid])
        checked(["ip", "addr", "add", "192.0.4.1/24", "dev", "router"])
        checked(["ip", "-6", "addr", "add", "fd42:4::1/64", "dev", "router", "nodad"])
        checked(["ip", "link", "set", "router", "up"])
        self.command(2, "ip", "addr", "add", "192.0.4.4/24", "dev", "uplink")
        self.command(2, "ip", "-6", "addr", "add", "fd42:4::4/64", "dev", "uplink", "nodad")
        self.command(2, "ip", "link", "set", "uplink", "up")
        for lane in ("lan0", "lan1"):
            self.command(2, "ip", "link", "set", lane, "down")
        self.command(2, "ip", "route", "add", "192.0.2.0/24", "via", "192.0.4.1")
        self.command(2, "ip", "-6", "route", "add", "fd42:2::/64", "via", "fd42:4::1")
        for index in (0, 1):
            self.command(index, "ip", "route", "add", "192.0.4.0/24", "via", "192.0.2.1")
            self.command(index, "ip", "-6", "route", "add", "fd42:4::/64", "via", "fd42:2::1")


def enroll(nodes, interfaces=None):
    first = nodes[0]
    first.run("compute", "init", "--name", first.name)
    for node in nodes[1:]:
        invitation = first.root / (node.name + ".invite")
        first.run("compute", "invite", invitation, "--name", node.name)
        target = node.root / "invitation"
        node.write(target, first.read(invitation))
        node.run("compute", "join", target)
    for index, node in enumerate(nodes):
        options = ["--backend", "cpu", "--cpus", 1, "--host-bytes", "512M"]
        for interface in (interfaces[index] if interfaces else []):
            options += ["--interface", interface]
        node.run("compute", "configure", *options)


def plot_options(node, label, seed=SEED, device="cpu"):
    return ["plot", "-k", 18, "-f", FARMER, "--pool-ph", POOL, "-S", seed,
            "--group-members", 3, "--devices", device, "--max-host-ram", "1G", "--workers", 2,
            "--window-bytes", "64K", "--checkpoint-bytes", "64K", "--compute-timeout-ms", 120000,
            "--compute-socket-timeout-ms", 1000, "--quiet", "-o", node.root / label]


def worker_options():
    return ["compute", "worker", "--max-brokers", 8, "--max-frame-bytes", 65536]


def forwarded_port(origin, nodes, logs):
    remotes = {node.remote["ssh"]: node for node in nodes if node.remote}
    for attempt in range(32):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        try:
            for node in remotes.values():
                node.request("port", port=port)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("no unused loopback port is available across the explicit hosts")
    specification = f"127.0.0.1:{port}:127.0.0.1:{port}"
    forwards = []
    if origin.remote:
        forwards.append((origin.remote["ssh"], "-L"))
    for host in remotes:
        if not origin.remote or host != origin.remote["ssh"]:
            forwards.append((host, "-R"))
    for host, direction in forwards:
        checked(["ssh", "-S", SSH_CONTROL_PATH, "-o", "ExitOnForwardFailure=yes",
                 "-O", "forward", direction, specification, host])
        with (logs / "cleanup.jsonl").open("a") as record:
            record.write(json.dumps(dict(ssh=host, forward=specification, direction=direction)) + "\n")
    return port


def qualify(nodes, summary, network=None, hosts=None, forwarded=False):
    enroll(nodes, [host.get("interfaces", []) for host in hosts] if hosts else None)
    references = []
    for node in nodes[:2]:
        profile = node.read(node.root / "config/xchplot2/compute/profile").decode().splitlines()
        owner = shlex.split(profile[1])[1]
        seed = hashlib.sha256(b"xchplot2-pool-storage-owner-v1" + bytes.fromhex(SEED) + owner.encode()).hexdigest()
        output = Path(node.run(*plot_options(node, "reference", seed), "--compute", "off").strip())
        references.append(node.digest(output))

    def compare(node, output, index, case):
        digest = node.digest(output)
        if digest != references[index]:
            raise AssertionError(f"{case}: shared computation changed CPU output bytes")
        node.run("verify", output, "--full", "--trials", 100)
        found = re.search(r"(\d+) full proofs validated", node.last_output.lower())
        if not found or int(found[1]) <= 0:
            raise AssertionError(f"{case}: no solved full proofs")
        summary["cases"].append(dict(case=case, owner=node.name, sha256=digest, full_proofs=int(found[1])))

    def direct_connections():
        # Explicit direct connections work without multicast, independently for
        # each address family. These are routed TLS connections, not DNS identity.
        for family in (("ipv4",) if forwarded else ("ipv4", "ipv6")):
            if hosts and not all(host.get(family) for host in hosts):
                summary["unavailable"].append(f"physical {family} addresses were not supplied")
                continue
            address = (hosts[0][family] if hosts else ("192.0.2.2" if family == "ipv4" else "fd42:2::2"))
            endpoint = f"[{address}]:18444" if family == "ipv6" else f"{address}:18444"
            worker_listen = []
            if forwarded:
                endpoint = "127.0.0.1:" + str(forwarded_port(nodes[0], nodes, nodes[0].logs))
                worker_endpoint = "127.0.0.1:" + str(forwarded_port(nodes[2], nodes, nodes[0].logs))
                worker_listen = ["--listen", worker_endpoint, "--advertise", worker_endpoint]
            broker = nodes[0].start("broker-" + family, "compute", "broker", "--listen", endpoint)
            nodes[1].run("compute", "configure", "--broker", endpoint, "--discovery", "off")
            if forwarded:
                nodes[0].run("compute", "configure", "--broker", endpoint, "--discovery", "off")
            nodes[2].run("compute", "status", "--broker", endpoint, "--timeout-ms", 3000, "--json")
            direct_worker = nodes[2].start("worker-" + family, *worker_options(), *worker_listen,
                                           "--broker", endpoint, "--discovery", "off")
            selected = nodes[:2] if forwarded else nodes[1:2]
            jobs = [(node, node.start("direct-" + family, *plot_options(node, "direct-" + family), "--compute-local", "off"))
                    for node in selected]
            for node, job in jobs:
                case = "SSH-forwarded concurrent CPU owners" if forwarded else "direct " + family + " without multicast"
                compare(node, node.finish(job), nodes.index(node), case)
            nodes[2].stop(direct_worker)
            nodes[0].stop(broker)
        print("PASS SSH-forwarded concurrent storage owners" if forwarded else
              "PASS explicit IPv4/IPv6 brokers with multicast disabled", flush=True)
        if hosts and hosts[2].get("qualify_gpu"):
            worker_node = nodes[2]
            worker_node.environment["ACPP_VISIBILITY_MASK"] = hosts[2]["gpu_backend"]
            worker_node.run("devices")
            endpoint = hosts[0]["ipv4"] + ":18445"
            worker_listen = []
            if forwarded:
                endpoint = "127.0.0.1:" + str(forwarded_port(nodes[0], nodes, nodes[0].logs))
                worker_endpoint = "127.0.0.1:" + str(forwarded_port(nodes[2], nodes, nodes[0].logs))
                worker_listen = ["--listen", worker_endpoint, "--advertise", worker_endpoint]
            broker = nodes[0].start("gpu-broker", "compute", "broker", "--listen", endpoint)
            nodes[1].run("compute", "configure", "--broker", endpoint, "--discovery", "off")
            gpu_worker = worker_node.start("gpu-worker", *worker_options(), *worker_listen, "--broker", endpoint,
                                          "--discovery", "off", "--backend", "gpu", "--tier", "tiny",
                                          "--host-bytes", hosts[2].get("gpu_host_bytes", "8G"),
                                          "--task-timeout-ms", 600000, "--tls-timeout-ms", 60000)
            # Backend initialization can JIT its first kernels before the
            # listener exists. Give startup its own bound before task timing.
            ready_until = time.monotonic() + 600
            while True:
                output = worker_node.read(worker_node.root / "gpu-worker.log", missing_ok=True).decode()
                if "Waiting for plotting work." in output:
                    break
                if worker_node.poll(gpu_worker) is not None or time.monotonic() >= ready_until:
                    raise RuntimeError("GPU worker did not become ready:\n" + output)
                time.sleep(1)
            job = nodes[1].start("remote-gpu", *plot_options(nodes[1], "remote-gpu", device="gpu0"),
                                 "--compute-local", "off", "--compute-timeout-ms", 600000,
                                 "--compute-socket-timeout-ms", 60000)
            compare(nodes[1], nodes[1].finish(job, timeout=900), 1,
                    "forced remote " + hosts[2]["gpu_backend"] + " GPU generation")
            worker_node.stop(gpu_worker)
            nodes[0].stop(broker)
            print("PASS forced remote GPU generation; CPU bytes and full proofs", flush=True)

    if hosts:
        direct_connections()
        if forwarded:
            return
        nodes[1].run("compute", "configure", "--broker", "auto", "--discovery", "on")
        nodes[2].environment["ACPP_VISIBILITY_MASK"] = hosts[2].get("environment", {}).get("ACPP_VISIBILITY_MASK", "omp")

    worker = nodes[2].start("automatic-worker", *worker_options())
    if network:
        status = nodes[0].run("compute", "status")
        if not any(address in status for address in ("fd42:", "fe80:")):
            raise AssertionError("worker was not discovered over IPv6 multicast")
        secondary = json.loads(network.command(0, "ip", "-j", "link", "show", "lan1"))[0]["ifindex"]
        if "192.0.3.4" not in status and "fd42:3::" not in status and \
                f"%{secondary}:" not in status and f"%{secondary}]" not in status:
            raise AssertionError("worker was not discovered over the secondary NIC")
    jobs = [node.start("automatic", *plot_options(node, "automatic"), "--compute-local", "off") for node in nodes[:2]]
    for index, (node, job) in enumerate(zip(nodes, jobs)):
        compare(node, node.finish(job), index, "concurrent automatic brokers")
    print("PASS concurrent owners discover a shared worker; CPU bytes and full proofs", flush=True)

    if network:
        network.command(0, "ip", "addr", "add", "192.0.2.22/24", "dev", "lan0")
        secondary_broker = nodes[0].start("secondary-broker", "compute", "broker", "--listen", "192.0.2.22:18446")
        status = nodes[2].run("compute", "status")
        if "192.0.2.22:18446" not in status:
            raise AssertionError("broker bound to a secondary interface address was not advertised")
        nodes[1].run("compute", "configure", "--broker", "192.0.2.22:18446")
        job = nodes[1].start("secondary-address", *plot_options(nodes[1], "secondary-address"), "--compute-local", "off")
        compare(nodes[1], nodes[1].finish(job), 1, "broker bound to secondary IPv4 address")
        nodes[1].run("compute", "configure", "--broker", "auto")
        nodes[0].stop(secondary_broker)
        # An actual isolated firewall drops both established and new TCP flows.
        # Keep multicast alive so this checks transport recovery separately.
        network.firewall(True)
        job = nodes[0].start("firewall", *plot_options(nodes[0], "firewall"), "--compute-local", "off")
        time.sleep(3)
        if nodes[0].poll(job) is not None:
            raise AssertionError("remote-only plot progressed through a TCP firewall drop")
        network.firewall(False)
        compare(nodes[0], nodes[0].finish(job), 0, "TCP firewall drop and reconnect")
        # Leave the process running across a primary NIC loss and a secondary
        # IPv4 address change; the remaining IPv6 route must keep discovery usable.
        network.command(2, "ip", "link", "set", "lan0", "down")
        network.command(2, "ip", "addr", "del", "192.0.3.4/24", "dev", "lan1")
        network.command(2, "ip", "addr", "add", "192.0.3.44/24", "dev", "lan1")
        time.sleep(6)
        job = nodes[1].start("changed-address", *plot_options(nodes[1], "changed-address"), "--compute-local", "off")
        compare(nodes[1], nodes[1].finish(job), 1, "NIC loss and IPv4 address change")
        network.command(2, "ip", "link", "set", "lan0", "up")
        print("PASS isolated TCP firewall reconnect, NIC loss and address refresh", flush=True)
    nodes[2].stop(worker)
    if network:
        network.routed_worker()
        direct_connections()



def remote_nodes(hosts, logs):
    nodes = []
    try:
        for name, host in zip(NAMES, hosts):
            if host.get("local"):
                parent = Path(tempfile.mkdtemp(prefix="xchplot2-network-", dir=host.get("scratch")))
                node = Node(name, parent / name, host["binary"], logs)
                node.environment.update(host.get("environment", {}))
                nodes.append(node)
                continue
            root = checked(ssh_command(host["ssh"], ["python3", "-c",
                            "import sys,tempfile; print(tempfile.mkdtemp(prefix='xchplot2-network-',dir=sys.argv[1] or None))",
                            host.get("scratch", "")])).strip()
            node = Node(name, root, host["binary"], logs, remote=host)
            # Bootstrap only this harness into the private, freshly made root.
            try:
                checked(ssh_command(host["ssh"], ["python3", "-c",
                         "import os,sys; from pathlib import Path; p=Path(sys.argv[1]); f=os.open(p/'harness.py',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); s=os.fdopen(f,'w'); s.write(sys.stdin.read()); s.close(); [(p/n).mkdir(mode=0o700) for n in ('tmp','cache','data','jit')]",
                         node.root]), input=Path(__file__).read_text())
            except BaseException:
                checked(ssh_command(host["ssh"], ["python3", "-c", "import shutil,sys; shutil.rmtree(sys.argv[1])", node.root]))
                raise
            nodes.append(node)
            if host.get("stage_binary"):
                content = Path(host["stage_binary"]).read_bytes()
                node.binary = str(node.root / "xchplot2")
                node.write(node.binary, content, executable=True)
                if node.digest(node.binary) != hashlib.sha256(content).hexdigest():
                    raise RuntimeError("staged CLI checksum did not match its local source")
        identities = [node.request("identity") if node.remote else hashlib.sha256(Path("/etc/machine-id").read_bytes()).hexdigest()
                      for node in nodes]
        if len(set(identities)) < 2:
            raise ValueError("physical qualification requires separate machine identities")
        return nodes
    except BaseException:
        errors = cleanup_nodes(nodes, local_roots=True)
        if errors:
            (logs / "cleanup-errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        raise


def cleanup_nodes(nodes, local_roots=False):
    errors = []
    for node in reversed(nodes):
        try:
            node.cleanup()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            errors.append(dict(participant=node.name, root=str(node.root), error=str(error)))
        finally:
            if local_roots and not node.remote:
                import shutil
                if not node.root.parent.name.startswith("xchplot2-network-"):
                    raise RuntimeError("refusing to clean an unowned local fixture root")
                shutil.rmtree(node.root.parent)
    return errors


def main():
    global SSH_CONTROL_PATH
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--hosts", type=Path, help="explicitly authorized SSH host definitions")
    parser.add_argument("--scratch", type=Path, help="existing filesystem for disposable plots")
    parser.add_argument("--inside-namespace", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--outer-netns", help=argparse.SUPPRESS)
    parser.add_argument("--outer-userns", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.hosts and not args.inside_namespace:
        # Re-exec before touching interfaces: every mutation is now isolated.
        command = ["unshare", "--user", "--map-root-user", "--net", sys.executable,
                   str(Path(__file__).resolve()), str(args.binary.resolve()), "--logs", str(args.logs.resolve()), "--inside-namespace",
                   "--outer-netns", os.readlink("/proc/self/ns/net"), "--outer-userns", os.readlink("/proc/self/ns/user")]
        if args.scratch:
            command += ["--scratch", str(args.scratch.resolve())]
        return subprocess.call(command)
    os.umask(0o077)
    args.logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    summary = dict(mode="physical SSH hosts" if args.hosts else "isolated Linux network namespaces",
                   status="running", cases=[], unavailable=[], fixture=dict(k=18, members=3, strength=2, trials=100))
    with args.binary.open("rb") as stream:
        summary["binary_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    nodes, network, hosts, control = [], None, [], None
    try:
        if args.hosts:
            configuration = json.loads(args.hosts.read_text())
            hosts = validate_hosts(configuration)
            control = tempfile.TemporaryDirectory(prefix="xchplot2-net-control-")
            SSH_CONTROL_PATH = str(Path(control.name) / "%C")
            nodes = remote_nodes(hosts, args.logs)
            forwarded = bool(configuration.get("ssh_forwarded", False))
            summary["transport"] = "SSH-forwarded application mTLS" if forwarded else "direct LAN mTLS"
            if forwarded:
                summary["unavailable"].append("SSH forwarding does not qualify direct LAN connectivity or multicast discovery")
            qualify(nodes, summary, hosts=hosts, forwarded=forwarded)
            summary["unavailable"].append("remote interface/firewall mutation intentionally not performed")
        else:
            summary["unavailable"].append("separate physical LAN machines were not supplied")
            with tempfile.TemporaryDirectory(prefix="xchplot2-network-", dir=args.scratch) as work:
                network = Network(args.outer_netns, args.outer_userns)
                network.create()
                nodes = [Node(name, Path(work) / name, args.binary.resolve(), args.logs, prefix)
                         for name, prefix in zip(NAMES, network.prefixes)]
                try:
                    qualify(nodes, summary, network)
                finally:
                    errors = cleanup_nodes(nodes)
                    nodes.clear()
                    if errors:
                        raise RuntimeError(f"fixture cleanup failed: {errors}")
        summary["status"] = "passed"
    except BaseException as error:
        summary["status"], summary["error"] = "failed", str(error)
        raise
    finally:
        try:
            errors = cleanup_nodes(nodes, local_roots=bool(args.hosts))
            preparation_errors = args.logs / "cleanup-errors.json"
            if preparation_errors.is_file():
                errors += json.loads(preparation_errors.read_text())
            if errors:
                summary["cleanup_errors"] = errors
                summary["status"] = "failed"
        finally:
            if network:
                network.close()
            if control:
                for host in {host["ssh"] for host in hosts if not host.get("local")}:
                    subprocess.run(["ssh", "-S", SSH_CONTROL_PATH, "-O", "exit", host],
                                   capture_output=True, timeout=10)
                control.cleanup()
                SSH_CONTROL_PATH = None
            (args.logs / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if errors:
            raise RuntimeError("fixture cleanup incomplete; see summary.json and cleanup.jsonl")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--rpc":
        rpc()
    elif len(sys.argv) == 3 and sys.argv[1] == "--watch":
        watch(Path(sys.argv[2]))
    else:
        sys.exit(main())
