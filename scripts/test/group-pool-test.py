#!/usr/bin/env python3
"""Loopback multi-process compute sharing, ordinary-byte parity and recovery."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time


# Disposable public identities, matching the existing rollover tests.
FARMER = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
SEED = "aa" * 32
POOL = "42" * 32
MIB = 1024 * 1024


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, code=0, point=None, timeout=180):
    env = dict(os.environ)
    env.pop("XCHPLOT2_GROW_FAILPOINT", None)
    if point:
        env["XCHPLOT2_GROW_FAILPOINT"] = point
    result = subprocess.run([str(value) for value in command], capture_output=True,
                            text=True, env=env, timeout=timeout)
    assert result.returncode == code, (command, result.returncode, result.stdout, result.stderr)
    assert SEED not in result.stdout + result.stderr, "seed was exposed in diagnostics"
    return result


def report(result):
    return json.loads(result.stdout.splitlines()[-1])


def port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class Process:
    def __init__(self, command, log, point=None, extra_env=None):
        self.log = log
        self.stream = log.open("w+")
        env = dict(os.environ)
        env.pop("XCHPLOT2_GROW_FAILPOINT", None)
        if point:
            env["XCHPLOT2_GROW_FAILPOINT"] = point
        if extra_env:
            env.update(extra_env)
        self.process = subprocess.Popen([str(value) for value in command], stdout=self.stream,
                                        stderr=subprocess.STDOUT, env=env)
        self.suspended = False

    def finish(self, code=0, timeout=180):
        status = self.process.wait(timeout=timeout)
        self.stream.flush()
        text = self.log.read_text()
        assert status == code, (status, self.log, text)
        assert SEED not in text, "seed was exposed in service diagnostics"
        return text

    def stop(self):
        if self.suspended and self.process.poll() is None:
            self.process.send_signal(signal.SIGCONT)
            self.suspended = False
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if not self.stream.closed:
            self.stream.close()

    def suspend(self):
        self.process.send_signal(signal.SIGSTOP)
        self.suspended = True


def wait_listener(process, number):
    end = time.monotonic() + 15
    while time.monotonic() < end:
        assert process.process.poll() is None, process.log.read_text()
        try:
            with socket.create_connection(("127.0.0.1", number), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"service did not listen: {process.log}")


def wait_stopped(process, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        assert process.process.poll() is None, process.log.read_text()
        status = Path(f"/proc/{process.process.pid}/status").read_text()
        if any(line.startswith("State:") and "T" in line for line in status.splitlines()):
            process.suspended = True
            return
        time.sleep(0.01)
    raise AssertionError(f"worker did not pause after partial transfer: {process.log}")


def certificates(root):
    root.mkdir(mode=0o700)
    ca = root / "ca.crt"
    key = root / "ca.key"
    run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256",
         "-nodes", "-subj", "/CN=grow-pool-test-ca", "-days", 1, "-keyout", key, "-out", ca])
    extensions = root / "node.cnf"
    extensions.write_text("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\n"
                          "extendedKeyUsage=serverAuth,clientAuth\n"
                          "subjectAltName=DNS:localhost,IP:127.0.0.1\n")
    identities = {}
    for serial, name in enumerate(("broker", "worker-a", "worker-b", "owner-a", "owner-b"), 2):
        cert, private, request = (root / (name + suffix) for suffix in (".crt", ".key", ".csr"))
        run(["openssl", "req", "-new", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256",
             "-nodes", "-subj", f"/CN={name}", "-keyout", private, "-out", request])
        run(["openssl", "x509", "-req", "-in", request, "-CA", ca, "-CAkey", key,
             "-set_serial", serial, "-days", 1, "-extfile", extensions, "-out", cert])
        identities[name] = (cert, private)
    return ca, identities


def check(args, root):
    ca, identities = certificates(root / "tls")
    services = []
    broker_port = port()
    endpoint = f"127.0.0.1:{broker_port}"
    results = {"parity": [], "recovery": [], "rejections": [], "proofs": []}

    def service(command, name, number, **options):
        process = Process(command, root / (name + ".log"), **options)
        services.append(process)
        wait_listener(process, number)
        return process

    def broker(name):
        cert, key = identities["broker"]
        return service([args.broker, "--listen", endpoint, "--tls-ca", ca,
                        "--tls-cert", cert, "--tls-key", key, "--timeout-ms", 1500,
                        "--lease-ms", 3000, "--worker-ms", 2500], name, broker_port)

    def worker(identity, name, cpus=2, slots=2, **options):
        number = port()
        cert, key = identities[identity]
        address = f"127.0.0.1:{number}"
        return service([args.worker, "--broker", endpoint, "--listen", address,
                        "--advertise", address, "--ca", ca, "--cert", cert, "--key", key,
                        "--cpus", cpus, "--slots", slots, "--host-bytes", 1024 * MIB,
                        "--generation-threads", 1, "--max-frame-bytes", 65536,
                        "--tls-timeout-ms", 1500, "--task-timeout-ms", 120000], name, number, **options)

    def common(target, members=3, seed=SEED):
        return [args.grow, target, "--seed", seed, "--farmer-pk", FARMER, "--pool-ph", POOL,
                "--k", 18, "--max-members", members, "--batch-members", 2,
                "--capacity-bytes", 64 * MIB, "--max-ram", 1024 * MIB,
                "--window-bytes", 65536, "--checkpoint-bytes", MIB, "--workers", 2,
                "--backend", "cpu"]

    def network(owner):
        cert, key = identities[owner]
        return ["--pool-broker", endpoint, "--pool-owner-id", owner, "--pool-ca", ca,
                "--pool-cert", cert, "--pool-key", key, "--pool-inflight", 2,
                "--pool-frame-bytes", 65536, "--pool-socket-timeout-ms", 1500,
                "--pool-timeout-ms", 120000, "--pool-codec-ram", 0]

    def targets(name):
        directory = root / name
        directory.mkdir(mode=0o700)
        return [directory / "first.gplot", directory / "second.gplot"]

    def verify(path, name):
        proof = run([args.plotter, "verify", path, "--full", "--trials", 100, "--config", os.devnull])
        output = proof.stdout + proof.stderr
        validated = re.search(r"(\d+) full proofs validated", output.lower())
        assert validated and int(validated.group(1)) > 0, output
        (root / (name + ".log")).write_text(output)
        results["proofs"].append({"case": name, "full_proofs_validated": int(validated.group(1))})

    def commands(paths, owner, members=3):
        return [*common(paths[0], members), "--destination", paths[1], *network(owner),
                "--construction", "external", "--destination-temp", "on",
                "--temp-capacity-bytes", 256 * MIB]

    expected = {}
    for owner in ("owner-a", "owner-b"):
        paths = targets("reference-" + owner)
        scoped = hashlib.sha256(b"xchplot2-pool-storage-owner-v1" + bytes.fromhex(SEED) + owner.encode()).hexdigest()
        run([*common(paths[0], seed=scoped), "--destination", paths[1], "--construction", "incremental"])
        expected[owner] = [digest(path) for path in paths]
    assert len(set(expected["owner-a"] + expected["owner-b"])) == 4, "storage-owner identities collided"

    try:
        active_broker = broker("broker")
        first = worker("worker-a", "worker-a-first")
        # Both owners must draw from each of the two worker identities, even
        # when no local compute is offered. A separate phase tests concurrency.
        for identity, active in (("worker-a", first), ("worker-b", None)):
            if active is None:
                time.sleep(2.6)  # Expire the stopped worker's registration.
                active = worker(identity, identity + "-only")
            jobs = []
            for owner in ("owner-a", "owner-b"):
                paths = targets(identity + "-" + owner)
                process = Process(commands(paths, owner), root / (identity + "-" + owner + ".log"))
                services.append(process)
                jobs.append((owner, paths, process))
            for owner, paths, process in jobs:
                value = json.loads(process.finish().splitlines()[-1])
                assert [digest(path) for path in paths] == expected[owner], (identity, owner, value)
                assert value["peak_modeled_bytes"] <= 1024 * MIB, value
                assert value["peak_file_bytes"] <= value["capacity_bytes"], value
                results["parity"].append({"case": identity + " serves " + owner, **value})
            active.stop()
        print("PASS both storage owners use both remote workers with ordinary-byte parity", flush=True)

        time.sleep(2.6)
        shared_workers = [worker("worker-a", "worker-a-shared"), worker("worker-b", "worker-b-shared")]
        jobs = []
        for owner in ("owner-a", "owner-b"):
            paths = targets("shared-" + owner)
            process = Process(commands(paths, owner), root / ("shared-" + owner + ".log"))
            services.append(process)
            jobs.append((owner, paths, process))
        for owner, paths, process in jobs:
            value = json.loads(process.finish().splitlines()[-1])
            assert [digest(path) for path in paths] == expected[owner], value
            assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths), "output permissions changed"
            before = [(digest(path), path.stat().st_mtime_ns) for path in paths]
            idle = report(run([*commands(paths, owner), "--resume"]))
            assert idle["generated_members"] == 0 and before == [(digest(path), path.stat().st_mtime_ns) for path in paths], idle
            results["parity"].append({"case": "shared pool " + owner, **value})
        print("PASS concurrent owners share the pool and completed resume preserves outputs", flush=True)

        for storage in ("compact", "raw"):
            path = targets("packed-" + storage)[0]
            value = report(run([*common(path), *network("owner-a"), "--member-storage", storage,
                                "--construction", "incremental"]))
            assert digest(path) == expected["owner-a"][0], value
            results["parity"].append({"case": "packed stream " + storage, **value})

        # The focused owner fixture queues one actual successor generation
        # after source preparation, then delegates all measured profitability,
        # transport and validation to the production remote codec provider.
        publication = root / "remote-final-codec"
        publication.mkdir(mode=0o700)
        value = report(run([args.owner_test, "--codec-publication", publication], timeout=300))
        remote = publication / "remote.gplot"
        assert digest(remote) == digest(publication / "reference.gplot"), value
        assert remote.stat().st_size == value["final_bytes"], value
        assert value["pool_codec_windows"] > 0 and value["pool_codec_chunks"] > 0, value
        assert value["pool_codec_calibration_windows"] > 0, value
        if args.plotter:
            verify(remote, "remote-final-codec-proofs")
        results["parity"].append({"case": "accepted remote final encoding through publication", **value})

        codec_reference = targets("remote-final-codec-reference")
        scoped = hashlib.sha256(b"xchplot2-pool-storage-owner-v1" + bytes.fromhex(SEED) + b"owner-a").hexdigest()
        run([*common(codec_reference[0], 8, scoped), "--destination", codec_reference[1],
             "--construction", "incremental"])

        paths = targets("grouped-contributions")
        value = report(run([*commands(paths, "owner-a", 8), "--batch-members", 8]))
        assert [digest(path) for path in paths] == [digest(path) for path in codec_reference], value
        assert value["pool_peak_generation_members"] > 1, "pool never formed a learned contribution batch"
        assert value["peak_modeled_bytes"] <= 1024 * MIB, value
        results["parity"].append({"case": "adaptive grouped remote contributions", **value})

        # Use the ordinary owner's complete file extent to choose the cap.
        # Cutting a compressed group must eventually regenerate the exact
        # fitting prefix, rather than stopping at the previous group boundary.
        # A longer source gives the observed policy room to form grouped runs
        # before exact capacity admission reaches its ordinary fallback.
        capacity_members = 32
        full = targets("capacity-full-reference")[0]
        full_value = report(run([*common(full, capacity_members, scoped), "--construction", "incremental",
                                 "--batch-members", 8]))
        capacity = full_value["final_bytes"] - 1
        reference = targets("capacity-prefix-reference")[0]
        expected_prefix = report(run([*common(reference, capacity_members, scoped), "--construction", "incremental",
                                      "--batch-members", 8, "--capacity-bytes", capacity]))
        path = targets("capacity-grouped-owner")[0]
        scratch = root / "capacity-grouped-scratch"
        scratch.mkdir(mode=0o700)
        value = report(run([*common(path, capacity_members), *network("owner-a"), "--construction", "external",
                            "--temp-dir", scratch, "--batch-members", 8,
                            "--capacity-bytes", capacity, "--temp-capacity-bytes", 256 * MIB]))
        assert value["members"] == expected_prefix["members"] and digest(path) == digest(reference), value
        assert value["stop_reason"] == expected_prefix["stop_reason"] == "capacity", (value, expected_prefix)
        assert value["pool_peak_generation_members"] > 1, value
        results["parity"].append({"case": "grouped remote capacity frontier keeps exact fitting prefix", **value})

        pair_full = targets("capacity-full-pair-reference")
        run([*common(pair_full[0], capacity_members, scoped), "--destination", pair_full[1],
             "--construction", "incremental", "--batch-members", 8])
        pair_capacity = 2 * min(path.stat().st_size for path in pair_full) - 1
        references = targets("capacity-prefix-pair-reference")
        expected_pair = report(run([*common(references[0], capacity_members, scoped), "--destination", references[1],
                                    "--construction", "incremental", "--batch-members", 8,
                                    "--capacity-bytes", pair_capacity]))
        paths = targets("capacity-grouped-pair")
        value = report(run([*commands(paths, "owner-a", capacity_members), "--batch-members", 8,
                            "--capacity-bytes", pair_capacity]))
        assert value["members"] == expected_pair["members"], (value, expected_pair)
        assert [digest(path) for path in paths] == [digest(path) for path in references], value
        assert value["stop_reason"] == expected_pair["stop_reason"] == "capacity", (value, expected_pair)
        assert value["pool_peak_generation_members"] > 1, value
        results["parity"].append({"case": "multi-destination grouped capacity frontier", **value})

        for point in ("external-contribution-input-sealed", "external-run-before-checkpoint", "external-run-after-checkpoint"):
            paths = targets("resume-" + point)
            command = commands(paths, "owner-a")
            run(command, code=86, point=point)
            value = report(run([*command, "--resume"]))
            assert [digest(path) for path in paths] == expected["owner-a"], (point, value)
            results["recovery"].append({"point": point, **value})
        print("PASS remote contributions recover before and after durable acceptance", flush=True)

        paths = targets("wrong-owner")
        run(commands(paths, "owner-a"))
        before = [(digest(path), path.stat().st_mtime_ns) for path in paths]
        run([*commands(paths, "owner-b"), "--resume"], code=1)
        assert before == [(digest(path), path.stat().st_mtime_ns) for path in paths], "owner mismatch modified outputs"
        results["rejections"].append("changed owner namespace preserves all existing destinations")

        if args.plotter:
            verify(paths[0], "full-proofs")

        # Existing workers must register with the replacement broker's epoch.
        active_broker.stop()
        active_broker = broker("broker-restarted")
        paths = targets("broker-restart")
        value = report(run(commands(paths, "owner-a")))
        assert [digest(path) for path in paths] == expected["owner-a"], value
        results["recovery"].append({"point": "broker process restarted", **value})
        for process in shared_workers:
            process.stop()
        time.sleep(2.6)

        # Pause a real worker only after its first contribution Data frame was
        # sent. Kill it with an owner holding partially received provisional
        # input, then require a fresh worker to retry the exact member batch.
        partial = worker("worker-a", "worker-partial", cpus=1, slots=1,
                         point="worker-contribution-paused")
        paths = targets("worker-partial-owner")
        interrupted = Process(commands(paths, "owner-a"), root / "worker-partial-owner.log")
        services.append(interrupted)
        wait_stopped(partial)
        partial.process.kill()
        partial.finish(code=-signal.SIGKILL)
        replacement = worker("worker-b", "worker-partial-replacement")
        value = json.loads(interrupted.finish().splitlines()[-1])
        assert value["pool_retries"] > 0 and value["pool_failed_tasks"] > 0, value
        assert [digest(path) for path in paths] == expected["owner-a"], value
        results["recovery"].append({"point": "worker killed after partial contribution transfer", **value})
        replacement.stop()
        time.sleep(2.6)

        # Repeat the partial transfer across a broker epoch change, interrupt
        # the owner immediately after successful replacement input is sealed,
        # and recover that provisional state through ordinary --resume.
        partial = worker("worker-a", "worker-partial-broker", cpus=1, slots=1,
                         point="worker-contribution-paused")
        paths = targets("broker-partial-owner")
        command = commands(paths, "owner-a")
        interrupted = Process(command, root / "broker-partial-owner.log",
                              point="external-contribution-input-sealed")
        services.append(interrupted)
        wait_stopped(partial)
        active_broker.stop()
        partial.process.kill()
        partial.finish(code=-signal.SIGKILL)
        active_broker = broker("broker-partial-restarted")
        replacement = worker("worker-b", "broker-partial-replacement")
        interrupted.finish(code=86)
        value = report(run([*command, "--resume"]))
        assert [digest(path) for path in paths] == expected["owner-a"], value
        results["recovery"].append({"point": "partial transfer broker restart and sealed-input resume", **value})
        replacement.stop()
        time.sleep(2.6)

        # Both packed RAM and partial scratch input must leave the sole owner
        # controller free to retire another destination on the same drive.
        # Its exact publication must precede the slow task's socket deadline.
        expected_codec = [digest(path) for path in codec_reference]
        for construction, point in (("incremental", "worker-generation-paused"),
                                    ("external", "worker-contribution-paused")):
            label = "heterogeneous-" + construction
            slow = worker("worker-a", label + "-slow", cpus=1, slots=1, point=point)
            paths = targets(label + "-owner")
            command = [*common(paths[0], 8), "--destination", paths[1], *network("owner-a"),
                       "--construction", construction, "--batch-members", 8,
                       "--pool-socket-timeout-ms", 15000]
            if construction == "external":
                command.extend(["--destination-temp", "on", "--temp-capacity-bytes", 256 * MIB])
            heterogeneous = Process(command, root / (label + "-owner.log"),
                                    extra_env={"POS2GPU_PHASE_TIMING": "1"})
            services.append(heterogeneous)
            wait_stopped(slow)
            fast = worker("worker-b", label + "-fast", cpus=1, slots=1)
            deadline = time.monotonic() + 12
            published = False
            while time.monotonic() < deadline:
                assert heterogeneous.process.poll() is None, heterogeneous.log.read_text()
                # Executor assignment races between the two admitted owners;
                # either may own the deliberately paused task.
                complete = [path.is_file() and digest(path) == expected_hash
                            for path, expected_hash in zip(paths, expected_codec)]
                published = any(complete)
                if published:
                    break
                time.sleep(0.02)
            assert published and slow.process.poll() is None, heterogeneous.log.read_text()
            assert complete.count(True) == 1, heterogeneous.log.read_text()
            assert heterogeneous.log.read_text().count("[grow-pool-task]") > 2, heterogeneous.log.read_text()
            slow.process.kill()
            slow.finish(code=-signal.SIGKILL)
            value = json.loads(heterogeneous.finish().splitlines()[-1])
            assert [digest(path) for path in paths] == expected_codec, value
            assert value["pool_retries"] > 0, value
            assert value["peak_modeled_bytes"] <= 1024 * MIB, value
            results["parity"].append({"case": construction + " fast destination publishes while slow worker is paused", **value})
            fast.stop()
            time.sleep(2.6)

        # A nonresponsive, still-registered executor must not trap a retry on
        # its old lease. Introduce the replacement after the first TLS deadline.
        stalled = worker("worker-a", "worker-stalled")
        time.sleep(0.2)
        stalled.suspend()
        paths = targets("worker-timeout")
        retry = Process(commands(paths, "owner-a"), root / "worker-timeout-owner.log")
        services.append(retry)
        time.sleep(2.0)
        replacement = worker("worker-b", "worker-replacement")
        value = json.loads(retry.finish().splitlines()[-1])
        assert value["pool_retries"] > 0, value
        assert [digest(path) for path in paths] == expected["owner-a"], value
        results["recovery"].append({"point": "stalled worker replaced", **value})
        stalled.stop()
        replacement.stop()
        time.sleep(2.6)
        print("PASS broker restart and stalled-worker retry preserve ordinary bytes", flush=True)

        # With no standalone workers, a storage owner's offered executor must
        # serve another owner while its own destination is being constructed.
        number = port()
        address = f"127.0.0.1:{number}"
        offer = ["--pool-listen", address, "--pool-advertise", address,
                 "--pool-offer-ram", 512 * MIB, "--pool-offer-cpus", 1,
                 "--pool-offer-slots", 2, "--pool-offer-backend", "cpu"]
        local = targets("embedded-owner")[0]
        command = [*common(local, 32), *network("owner-a"), *offer, "--construction", "incremental"]
        embedded = Process(command, root / "embedded-owner.log")
        services.append(embedded)
        wait_listener(embedded, number)
        remote = targets("embedded-guest")
        value = report(run(commands(remote, "owner-b")))
        assert [digest(path) for path in remote] == expected["owner-b"], value
        results["parity"].append({"case": "storage owner offers compute to another owner", **value})
        own_value = json.loads(embedded.finish().splitlines()[-1])
        assert own_value["peak_modeled_bytes"] <= 1024 * MIB, own_value
        assert own_value["pool_local_direct_tasks"] > 0, own_value
        assert own_value["pool_local_fallbacks"] == 0, own_value
        reference = targets("embedded-reference")[0]
        scoped = hashlib.sha256(b"xchplot2-pool-storage-owner-v1" + bytes.fromhex(SEED) + b"owner-a").hexdigest()
        run([*common(reference, 32, scoped), "--construction", "incremental"])
        assert digest(local) == digest(reference), own_value
        results["parity"].append({"case": "embedded owner shares its compute budget", **own_value})

        # The embedded service remains usable when no broker is listening, and
        # fallback uses its same admission gates and generation executor.
        active_broker.stop()
        before = [(digest(path), path.stat().st_mtime_ns) for path in remote]
        idle = report(run([*commands(remote, "owner-b"), "--resume",
                           "--pool-ca", root / "unavailable-ca.pem"], timeout=10))
        assert idle["generated_members"] == 0 and before == [(digest(path), path.stat().st_mtime_ns) for path in remote], idle
        results["recovery"].append({"point": "completed resume without broker or TLS files", **idle})
        path = targets("local-fallback")[0]
        value = report(run([*common(path), *network("owner-a"), *offer, "--construction", "incremental"]))
        assert value["pool_local_fallbacks"] > 0 and value["peak_modeled_bytes"] <= 1024 * MIB, value
        assert digest(path) == expected["owner-a"][0], value
        results["recovery"].append({"point": "broker unavailable with embedded executor", **value})
        print("PASS embedded compute sharing and local fallback obey owner budgets", flush=True)
        return results
    finally:
        for process in reversed(services):
            process.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grow", type=Path)
    parser.add_argument("broker", type=Path)
    parser.add_argument("worker", type=Path)
    parser.add_argument("--owner-test", type=Path, required=True,
                        help="focused owner fixture used for measured accepted-codec publication")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--plotter", type=Path)
    parser.add_argument("--keep-work", action="store_true", help="retain successful fixture files and logs")
    args = parser.parse_args()
    for name in ("grow", "broker", "worker", "owner_test", "plotter"):
        value = getattr(args, name)
        if value:
            setattr(args, name, value.resolve())
    os.umask(0o077)
    work = Path(tempfile.mkdtemp(prefix="grow-pool-", dir=args.scratch))
    try:
        results = check(args, work)
    except BaseException:
        print(f"network pool fixture retained after failure: {work}", file=sys.stderr, flush=True)
        raise
    if not args.keep_work:
        shutil.rmtree(work)
    args.results.write_text(json.dumps(results, indent=2) + "\n")
    print("network compute sharing, parity, recovery and rejection checks passed")


if __name__ == "__main__":
    main()
