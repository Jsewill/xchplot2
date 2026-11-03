#!/usr/bin/env python3
"""Exercise installed group commands, saved jobs, read-only plans and recovery."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import struct
import subprocess
import tempfile
import time

# Public BLS12-381 generator; all keys and output in this check are disposable.
FARMER = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
POOL = "42" * 32
SEED = "ab" * 32
HEADER = struct.Struct("<4sB32sBBHBQB")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def header(path):
    with path.open("rb") as stream:
        magic, version, group, k, strength, count, meta, index, memo_size = HEADER.unpack(stream.read(HEADER.size))
        memo = stream.read(memo_size)
    assert magic == b"PoS2" and version == 2 and count, path
    assert HEADER.size + memo_size < index < path.stat().st_size, path
    # Windows private ACLs are checked by the native file and membership fixtures.
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600, path
    return group.hex(), k, strength, count, meta, memo.hex()


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mode, p.stat().st_size, p.stat().st_mtime_ns,
                                     digest(p) if p.is_file() else None)
            for p in root.rglob("*")}


def check(binary, root, reference=None):
    counter = 0

    def command(*args):
        return [str(binary), *map(str, args), "--config", os.devnull]

    def run(*args, code=0, point=None):
        nonlocal counter
        env = dict(os.environ)
        env.pop("XCHPLOT2_GROW_FAILPOINT", None)
        if point:
            env["XCHPLOT2_GROW_FAILPOINT"] = point
        result = subprocess.run(command(*args), capture_output=True, text=True, encoding="utf-8", env=env, timeout=180)
        counter += 1
        (root / f"{counter:03d}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        assert SEED not in result.stdout + result.stderr, "private seed exposed in diagnostics"
        assert result.returncode == code, (args, result.returncode, result.stdout, result.stderr)
        return result

    def verify(path):
        result = run("verify", path, "--full", "--trials", 100)
        found = re.search(r"(\d+) full proofs validated", (result.stdout + result.stderr).lower())
        assert found and int(found[1]) > 0, result

    keys = ["-f", FARMER, "--pool-ph", POOL]
    construction = ["-d", "cpu", "-M", "512M", "-j", 2,
                    "--window-bytes", "64K", "--checkpoint-bytes", "64K",
                    "--capacity-bytes", "64M", "--headroom-bytes", 0]
    for subcommand in [("plan",), ("plot",), ("fill",), ("grow",),
                       ("compute",), ("compute", "broker"), ("compute", "worker"),
                       *(("compute", command) for command in ("init", "invite", "join", "configure", "status"))]:
        assert "Usage:" in run(*subcommand, "--help").stdout + (root / f"{counter:03d}.log").read_text(encoding="utf-8")

    preview = root / "preview"
    preview_peer = root / "preview peer"
    preview.mkdir()
    preview_peer.mkdir()
    options = [preview, "-D", preview_peer, "-k", 18, "--group-members", 2, "--max-total-members", 3, *construction]
    before = snapshot(preview), snapshot(preview_peer)
    planned = json.loads(run("plan", *options, "--json").stdout)
    filled = json.loads(run("fill", *options, "--plan-only", "--json").stdout)
    assert planned == filled and planned["feasible"]
    assert len(planned["destinations"]) == 2 and all(d["files"] == 2 for d in planned["destinations"])
    assert (snapshot(preview), snapshot(preview_peer)) == before, "preview created or changed files"
    for subcommand in [("plan",), ("fill", "--plan-only")]:
        refused = json.loads(run(*subcommand, *options, "--farming-deadline-ms", 1, "--json", code=2).stdout)
        assert not refused["feasible"] and (snapshot(preview), snapshot(preview_peer)) == before
    print("PASS plan/fill preview equivalence and no writes on infeasibility", flush=True)

    output = root / "grouped files"
    plot = ["plot", "-k", 18, "-n", 2, *keys, "-o", output,
            "--group-members", 3, "--batch-members", 2, "--construction", "staged",
            *construction, "--quiet"]
    made = run(*plot)
    paths = [Path(line) for line in made.stdout.splitlines()]
    assert len(paths) == 2 and all(header(p)[3] == 3 for p in paths), made
    assert not made.stderr, made.stderr
    assert header(paths[0])[0] != header(paths[1])[0]
    job, = output.glob("xchplot2-job-*.tsv")
    saved = job.read_bytes()
    rows = [shlex.split(line) for line in saved.decode().splitlines() if line and not line.startswith("#")]
    assert len(rows) == 2 and all(row[4] == "gplot-v2-grouped" and row[9] == "3" for row in rows)
    if os.name != "nt":
        assert job.stat().st_mode & 0o777 == 0o600
    before = {path: digest(path) for path in paths}
    repeated = run(*plot, "--resume")
    assert set(repeated.stdout.splitlines()) == set(map(str, paths))
    assert job.read_bytes() == saved and all(digest(p) == value for p, value in before.items())
    # batch uses the saved membership even when no group-specific flags are repeated.
    run("batch", job, "--resume", *construction, "--quiet")
    assert all(digest(p) == value for p, value in before.items())
    run(*plot, "--resume", "--group-members", 4, code=2)
    run(*plot, "--group-members", 0, code=1)
    for path in paths:
        verify(path)
    fresh = run(*plot)
    assert not set(fresh.stdout.splitlines()) & set(map(str, paths)), "new jobs reused plot identities"
    print("PASS file counts, membership, private manifests, fresh identities, resume and full proofs", flush=True)

    # A one-member group must match the existing ordinary writer exactly.
    one = root / "one"
    single = root / "single"
    grouped_one = Path(run("plot", "-k", 18, *keys, "-S", SEED, "-o", one,
                           "--group-members", 1, *construction, "--quiet").stdout.strip())
    ordinary_one = Path(run("plot", "-k", 18, *keys, "-S", SEED, "-o", single,
                            "--devices", "cpu", "--cpu-workers", 1, "--quiet").stdout.strip())
    assert digest(grouped_one) == digest(ordinary_one)
    if reference:
        group, k, strength, count, meta, memo = header(paths[0])
        fragments = root / "reference fragments"
        fragments.mkdir()
        raw = []
        for member in range(count):
            name = f"{member}.plot2"
            run("test", k, group, strength, member, meta, "--raw", "-m", memo, "-o", fragments, "-N", name)
            raw.append(fragments / name)
        inputs = fragments / "group.inputs"
        inputs.write_text(f"{k} {strength} {meta} {group} {memo}\n" +
                          "".join(json.dumps(str(p), ensure_ascii=False) + "\n" for p in raw), encoding="utf-8")
        canonical = fragments / "canonical.gplot"
        comparison = subprocess.run([str(reference), "assemble-only", str(inputs), str(canonical), "512"],
                                    capture_output=True, text=True, encoding="utf-8", timeout=180)
        assert comparison.returncode == 0, (comparison.stdout, comparison.stderr)
        assert digest(canonical) == digest(paths[0])
    print("PASS ordinary byte parity", flush=True)

    # The native storage adapter must recover every construction engine to the
    # same ordinary bytes. Each failpoint exits a real child process abruptly.
    expected = None
    scratch = root / "engine scratch"
    scratch.mkdir()
    for mode, point in (("incremental", "initialized"), ("staged", "stage-run-committed"),
                        ("merged", "merge-index-flushed"), ("external", "external-header-flushed")):
        engine = ["plot", "-k", 18, *keys, "-S", SEED, "-o", root / (mode + " recovered"),
                  "--group-members", 3, "--batch-members", 2, "--construction", mode,
                  *construction, "-t", scratch, "--temp-capacity-bytes", "256M", "--quiet"]
        run(*engine, point=point, code=86)
        recovered = Path(run(*engine, "--resume").stdout.strip())
        assert header(recovered)[3] == 3
        if expected is None:
            expected = digest(recovered)
        assert digest(recovered) == expected, mode
        verify(recovered)
    print("PASS all construction engines recover to identical ordinary bytes and full proofs", flush=True)

    run("grow", grouped_one, "--max-members", 3, *construction, "--quiet")
    assert header(grouped_one)[3] == 3
    verify(grouped_one)
    grown = digest(grouped_one)
    run("grow", grouped_one, "--max-members", 3, *construction, "--quiet")
    assert digest(grouped_one) == grown
    run("grow", root / "missing.gplot", "--max-members", 2, *construction, code=2)
    assert not (root / "missing.gplot").exists()

    # Reject a corrupt saved identity before changing any peer output.
    corrupted = root / "mismatch.tsv"
    rows[0][5] = "cc" * 32
    corrupted.write_text("\n".join(" ".join(json.dumps(word, ensure_ascii=False) if index in (6, 7, 8) else word
                                            for index, word in enumerate(row)) for row in rows) + "\n", encoding="utf-8")
    run("batch", corrupted, "--resume", *construction, code=2)
    assert all(digest(p) == value for p, value in before.items())
    print("PASS existing-file growth and identity audit", flush=True)

    interrupted = root / "interrupted"
    recover = ["plot", "-k", 18, "-n", 2, *keys, "-o", interrupted,
               "--group-members", 3, "--batch-members", 1, "--construction", "staged", *construction, "--quiet"]
    run(*recover, point="stage-run-committed", code=86)
    recovery_job, = interrupted.glob("xchplot2-job-*.tsv")
    recovery_manifest = recovery_job.read_bytes()
    completed = run(*recover, "--resume")
    assert len(completed.stdout.splitlines()) == 2 and recovery_job.read_bytes() == recovery_manifest
    for name in completed.stdout.splitlines():
        assert header(Path(name))[3] == 3
        verify(Path(name))

    cancelled = root / "cancelled"
    cancel_args = ["plot", "-k", 22, "-n", 6, *keys, "-o", cancelled,
                   "--group-members", 2, "--batch-members", 1, "--construction", "staged",
                   *construction, "--capacity-bytes", "256M", "--verbose"]
    cancel_log = root / "cancel.log"
    with cancel_log.open("w+") as log:
        process = subprocess.Popen(command(*cancel_args), stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 30
            while ": generating cpu member" not in cancel_log.read_text(encoding="utf-8"):
                assert process.poll() is None, cancel_log.read_text(encoding="utf-8")
                assert time.monotonic() < deadline, cancel_log.read_text(encoding="utf-8")
                time.sleep(0.01)
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            output = process.communicate(timeout=180)[0]
            assert process.returncode == 4, (process.returncode, output, cancel_log.read_text(encoding="utf-8"))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    cancel_job, = cancelled.glob("xchplot2-job-*.tsv")
    cancel_manifest = cancel_job.read_bytes()
    run(*cancel_args, "--resume", "--no-verbose", "--quiet")
    assert cancel_job.read_bytes() == cancel_manifest
    assert len(list(cancelled.glob("*.gplot"))) == 6
    verify(next(cancelled.glob("*.gplot")))
    print("PASS process cancellation leaves a resumable grouped job", flush=True)

    fill = ["fill", *options, *keys, "--seed", SEED, "--quiet"]
    produced = run(*fill)
    farm_outputs = [Path(line) for line in produced.stdout.splitlines()]
    assert sorted(header(p)[3] for p in farm_outputs) == [1, 1, 2, 2]
    assert {p.parent for p in farm_outputs} == {preview, preview_peer}
    assert (preview / "grow-farm.plan").is_file()
    farm_hashes = {p: digest(p) for p in farm_outputs}
    assert set(run(*fill, "-r").stdout.splitlines()) == set(map(str, farm_outputs))
    assert all(digest(p) == value for p, value in farm_hashes.items())
    for path in farm_outputs:
        verify(path)
    print("PASS interrupted construction, directory fill, saved-plan resume and full proofs", flush=True)


def check_compute(binary, root):
    # Reuse the established TLS and process lifecycle fixtures.
    import importlib.util
    spec = importlib.util.spec_from_file_location("group_pool_test", Path(__file__).with_name("group-pool-test.py"))
    pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pool)
    ca, identities = pool.certificates(root / "tls")
    broker_port, worker_port = pool.port(), pool.port()
    endpoint, worker_endpoint = f"127.0.0.1:{broker_port}", f"127.0.0.1:{worker_port}"
    processes = []

    def start(args, name):
        process = pool.Process([binary, *args, "--config", os.devnull], root / (name + ".log"))
        processes.append(process)
        return process

    try:
        cert, key = identities["broker"]
        broker = start(["compute", "broker", "--listen", endpoint, "--tls-ca", ca,
                        "--tls-cert", cert, "--tls-key", key], "broker")
        pool.wait_listener(broker, broker_port)
        cert, key = identities["worker-a"]
        worker = start(["compute", "worker", "--broker", endpoint, "--listen", worker_endpoint,
                        "--advertise", worker_endpoint, "--ca", ca, "--cert", cert, "--key", key,
                        "--cpus", 2, "--slots", 2, "--host-bytes", 1024 * 1024 * 1024,
                        "--generation-threads", 1, "--backend", "cpu"], "worker")
        pool.wait_listener(worker, worker_port)
        common = ["plot", "-k", 18, "-n", 1, "-f", FARMER, "--pool-ph", POOL,
                  "--group-members", 3, "--batch-members", 2, "--backend", "cpu",
                  "--construction", "staged", "--max-host-ram", "512M", "--workers", 2,
                  "--window-bytes", "64K", "--checkpoint-bytes", "64K", "--quiet"]
        owners = []
        for owner in ("owner-a", "owner-b"):
            cert, key = identities[owner]
            network = ["--compute-broker", endpoint, "--compute-owner-id", owner,
                       "--compute-ca", ca, "--compute-cert", cert, "--compute-key", key,
                       "--compute-inflight", 2, "--compute-codec-ram", 0, "--compute-timeout-ms", 120000]
            owners.append((owner, start([*common, "-S", SEED, "-o", root / owner, *network], owner)))
        paths = []
        for owner, process in owners:
            path = Path(process.finish().strip())
            assert path.is_file() and header(path)[3] == 3
            paths.append(path)
            scoped = hashlib.sha256(b"xchplot2-pool-storage-owner-v1" + bytes.fromhex(SEED) + owner.encode()).hexdigest()
            result = pool.run([binary, *common, "-S", scoped, "-o", root / (owner + "-local"), "--config", os.devnull])
            local = Path(result.stdout.strip())
            assert digest(local) == digest(path), "remote group differs from local construction"
            result = pool.run([binary, "verify", path, "--full", "--trials", 100, "--config", os.devnull])
            proof = re.search(r"(\d+) full proofs validated", (result.stdout + result.stderr).lower())
            assert proof and int(proof[1]) > 0
        assert header(paths[0])[0] != header(paths[1])[0], "owners reused a group identity"
        print("PASS two simultaneous plotting owners share one compute worker, byte parity and full proofs", flush=True)
    finally:
        for process in reversed(processes):
            process.stop()


def check_automatic_compute(binary, root):
    """Saved membership, embedded brokers, discovery and shared executors."""
    import socket

    directory = root / "automatic-compute"
    directory.mkdir()
    states = {name: directory / name for name in ("first", "second", "worker")}
    processes = []
    counter = 0

    def command(*args):
        return [str(binary), *map(str, args), "--config", os.devnull]

    def environment(name):
        env = dict(os.environ)
        env["XDG_CONFIG_HOME"] = str(states[name])
        env.pop("XCHPLOT2_GROW_FAILPOINT", None)
        return env

    def run(name, *args, code=0):
        nonlocal counter
        result = subprocess.run(command(*args), env=environment(name), capture_output=True, text=True, encoding="utf-8", timeout=180)
        counter += 1
        (directory / f"{counter:03d}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        assert result.returncode == code, (args, result.returncode, result.stdout, result.stderr)
        assert "PRIVATE KEY" not in result.stdout + result.stderr
        return result

    def start(name, label, *args):
        path = directory / (label + ".log")
        stream = path.open("w")
        process = subprocess.Popen(command(*args), env=environment(name), stdout=stream, stderr=subprocess.STDOUT)
        processes.append((process, stream, path))
        return process, path

    def finish(process, path, service=None):
        deadline = time.monotonic() + 180
        while process.poll() is None:
            if service:
                required, log = service
                assert required.poll() is None, log.read_text(encoding="utf-8")
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(process.args, 180)
            time.sleep(0.1)
        result = process.returncode
        output = path.read_text(encoding="utf-8")
        assert result == 0, (result, output)
        paths = [Path(line) for line in output.splitlines() if line.endswith(".gplot")]
        assert paths and all(p.is_file() for p in paths), output
        return paths

    def stop(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    keys = ["-f", FARMER, "--pool-ph", POOL]
    limits = ["--devices", "cpu", "--max-host-ram", "1G", "--workers", 2,
              "--window-bytes", "64K", "--checkpoint-bytes", "64K",
              "--compute-timeout-ms", 60000, "--quiet"]
    plot = ["plot", "-k", 18, *keys, *limits]
    try:
        run("first", "compute", "init", "--name", "storage A")
        for name in ("second", "worker"):
            invitation = directory / (name + ".invite")
            run("first", "compute", "invite", invitation, "--name", name)
            run(name, "compute", "join", invitation)
            if os.name != "nt":
                assert invitation.stat().st_mode & 0o777 == 0o600
        for name in states:
            run(name, "compute", "configure", "--interface", "lo", "--backend", "cpu",
                "--cpus", 1, "--host-bytes", "512M")
        refused = run("worker", "compute", "worker", "--max-brokers", 128, code=1)
        assert "reduce --max-brokers or increase --host-bytes" in refused.stderr, refused
        run("worker", "compute", "configure", "--max-brokers", 16)
        assert "automatic for each plotting job" in run("second", "compute", "status").stdout
        preview = directory / "preview"
        preview.mkdir()
        before = {name: snapshot(state) for name, state in states.items()}
        plan_options = [preview, "-k", 18, "--group-members", 2, "--max-total-members", 3,
                        "--capacity-bytes", "64M", "--headroom-bytes", 0, *limits, "--json"]
        plan = run("first", "plan", *plan_options)
        dry_fill = run("first", "fill", *plan_options, "--plan-only")
        assert json.loads(plan.stdout) == json.loads(dry_fill.stdout)
        assert not list(preview.iterdir()) and all(snapshot(states[name]) == saved for name, saved in before.items())

        # No worker or dedicated broker is running: a normal plot still works.
        made = run("first", *plot, "--max-host-ram", "512M", "-S", SEED, "-o", directory / "local")
        output = Path(made.stdout.strip())
        assert output.is_file() and header(output)[3] == 1 and not made.stderr
        profile = (states["first"] / "xchplot2/compute/profile").read_text(encoding="utf-8").splitlines()
        node = shlex.split(profile[1])[1]
        scoped = hashlib.sha256(b"xchplot2-pool-storage-owner-v1" + bytes.fromhex(SEED) + node.encode()).hexdigest()
        local = Path(run("first", *plot, "--compute", "off", "-S", scoped, "-o", directory / "reference").stdout.strip())
        assert digest(local) == digest(output), "automatic local executor changed plot bytes"
        run("first", "grow", output, "--max-members", 3, *limits)
        assert header(output)[3] == 3
        run("first", "verify", output, "--full", "--trials", 100)
        fill_dir = directory / "fill"
        fill_dir.mkdir()
        filled = run("first", "fill", fill_dir, "-k", 18, *keys, "-S", SEED,
                     "--group-members", 2, "--max-total-members", 2, "--capacity-bytes", "64M", "--headroom-bytes", 0, *limits)
        fill_paths = [Path(line) for line in filled.stdout.splitlines() if line.endswith(".gplot")]
        assert len(fill_paths) == 1 and header(fill_paths[0])[3] == 2, filled
        print("PASS saved enrollment, read-only plans and automatic local plot/fill/grow", flush=True)

        # Both storage machines broker their own work; a worker can arrive later.
        jobs = [start(name, name + "-remote", *plot, "--group-members", 3, "-S", SEED,
                      "--compute-local", "off", "-o", directory / (name + "-remote"))
                for name in ("first", "second")]
        time.sleep(0.5)
        assert all(process.poll() is None for process, _ in jobs), [path.read_text(encoding="utf-8") for _, path in jobs]
        worker, worker_log = start("worker", "shared-worker", "compute", "worker")
        remote_paths = [finish(process, path, (worker, worker_log))[0] for process, path in jobs]
        assert all(header(path)[3] == 3 for path in remote_paths)
        assert header(remote_paths[0])[0] != header(remote_paths[1])[0]
        assert worker.poll() is None, worker_log.read_text(encoding="utf-8")
        for path in remote_paths:
            run("first", "verify", path, "--full", "--trials", 100)
        print("PASS worker arrival and one worker shared by two automatic brokers", flush=True)

        # A saved dedicated broker replaces the per-job broker.
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        broker, broker_log = start("first", "dedicated-broker", "compute", "broker", "--listen", f"127.0.0.1:{port}",
                                   "--worker-ms", 3000, "--lease-ms", 3000)
        run("second", "compute", "configure", "--broker", f"127.0.0.1:{port}")
        result = run("second", *plot, "--compute-local", "off", "-o", directory / "dedicated", "--no-quiet")
        assert "Using broker 127.0.0.1:" in result.stderr and "Automatic broker" not in result.stderr
        assert Path(result.stdout.strip()).is_file()
        assert broker.poll() is None, broker_log.read_text(encoding="utf-8")
        explicit = [*plot, "--compute-owner-id", "storage-test", "--compute-local", "off",
                    "-S", SEED, "-o", directory / "explicit-owner"]
        owned = Path(run("second", *explicit).stdout.strip())
        before = digest(owned)
        assert Path(run("second", *explicit, "--resume").stdout.strip()) == owned
        assert digest(owned) == before
        scoped_owner = hashlib.sha256(b"xchplot2-pool-storage-owner-v1" + bytes.fromhex(SEED) + b"storage-test").hexdigest()
        reference = Path(run("second", *plot, "--compute", "off", "-S", scoped_owner,
                             "-o", directory / "explicit-reference").stdout.strip())
        assert digest(reference) == before, "explicit owner namespace was not preserved"
        # Kill a frozen worker while the owner is waiting on its lease. A new
        # process with the same enrolled identity takes over after expiration.
        if os.name == "nt":
            worker.kill()
            worker.wait(timeout=10)
        else:
            worker.send_signal(signal.SIGSTOP)
        recovering, recovering_log = start("second", "worker-restart", *plot, "--group-members", 3, "-S", SEED,
                                            "--compute-local", "off", "--compute-socket-timeout-ms", 1000,
                                            "-o", directory / "worker-restart")
        time.sleep(0.3)
        assert recovering.poll() is None, recovering_log.read_text(encoding="utf-8")
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)
        worker, worker_log = start("worker", "restarted-worker", "compute", "worker")
        recovered = finish(recovering, recovering_log, (worker, worker_log))[0]
        assert digest(recovered) == digest(remote_paths[1]), "worker loss changed the recovered group"
        stop(worker)
        stop(broker)
        # A dedicated broker outage still permits the configured local
        # executor to finish through the existing bounded fallback.
        fallback = run("second", *plot, "--compute-socket-timeout-ms", 1000,
                       "-o", directory / "broker-outage")
        run("second", "verify", Path(fallback.stdout.strip()), "--full", "--trials", 100)
        run("second", "compute", "configure", "--broker", "auto")
        print("PASS worker replacement and local progress during dedicated broker outage", flush=True)

        # Two concurrent jobs on the same enrolled machine reuse one executor.
        same_host = [start("first", f"same-host-{i}", *plot, "--group-members", 3, "-n", 2,
                           "-o", directory / f"same-host-{i}") for i in range(2)]
        for process, path in same_host:
            assert len(finish(process, path)) == 2
        print("PASS dedicated broker selection and concurrent local plotting jobs", flush=True)
    finally:
        for process, stream, _ in reversed(processes):
            stop(process)
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--compute", action="store_true", help="also test authenticated loopback compute sharing (needs openssl)")
    parser.add_argument("--automatic-compute", action="store_true", help="test saved enrollment and automatic LAN compute over loopback")
    parser.add_argument("--scratch", type=Path, default=Path.cwd(), help="existing directory on real disk")
    args = parser.parse_args()
    allocated_console = False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        allocated_console = kernel.AllocConsole()
        if not allocated_console and ctypes.get_last_error() != 5:
            raise ctypes.WinError(ctypes.get_last_error())
    try:
        with tempfile.TemporaryDirectory(prefix="xchplot2 group é-", dir=args.scratch.resolve()) as work:
            os.environ["XDG_CONFIG_HOME"] = str(Path(work) / "empty-compute-settings")
            check(args.binary.resolve(), Path(work), args.reference.resolve() if args.reference else None)
            if args.compute:
                check_compute(args.binary.resolve(), Path(work))
            if args.automatic_compute:
                check_automatic_compute(args.binary.resolve(), Path(work))
    finally:
        if allocated_console:
            kernel.FreeConsole()



if __name__ == "__main__":
    main()
