#!/usr/bin/env python3
"""Installed compute membership, live control and user-service integration."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("group_pool_test", Path(__file__).with_name("group-pool-test.py"))
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def check(binary, root, service_manager=False):
    services = []
    sequence = 0
    directories = {name: root / name for name in ("authority", "worker", "owner")}

    def command(name, *args):
        return [str(binary), "compute", *map(str, args), "--directory", str(directories[name]), "--config", os.devnull]

    def run(name, *args, code=0, env=None):
        nonlocal sequence
        result = subprocess.run(command(name, *args), capture_output=True, text=True, encoding="utf-8", timeout=30, env=env)
        sequence += 1
        (root / f"{sequence:03d}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        assert result.returncode == code, (args, result.returncode, result.stdout, result.stderr)
        assert "PRIVATE KEY" not in result.stdout + result.stderr
        return result

    def status(name, *args):
        return json.loads(run(name, "status", *args, "--json", "--timeout-ms", 2000).stdout)

    def until(check, timeout=20):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            value = check()
            if value:
                return value
            time.sleep(0.1)
        raise AssertionError("compute state did not converge")

    def start(name, *args, port):
        process = pool.Process(command(name, *args), root / f"{name}-{len(services)}.log")
        services.append(process)
        pool.wait_listener(process, port)
        return process

    def managed_service():
        # The service name contains this disposable node's random stable ID.
        # Always remove it, including when registration succeeds but start fails.
        try:
            run("authority", "service", "install")
            node = status("authority")["node_id"]
            until(lambda: any(w["node_id"] == node for w in status("owner")["workers"]))
            for _ in range(3):
                for action in ("status", "stop", "start", "restart"):
                    run("authority", "service", action)
                until(lambda: any(w["node_id"] == node for w in status("owner")["workers"]))
        except Exception:
            try:
                result = run("authority", "service", "status")
                print("Service status after failure:\n" + result.stdout + result.stderr, file=sys.stderr, flush=True)
                print("Compute status after service failure:\n" + json.dumps(status("owner")), file=sys.stderr, flush=True)
            except Exception as error:
                print("Cannot inspect service after failure:", error, file=sys.stderr, flush=True)
            raise
        finally:
            run("authority", "service", "remove")
        print("PASS native user service worker startup, control and removal", flush=True)

    try:
        for name in ("members", "revoke", "renew-request", "renew-sign", "renew-import", "policy-export", "policy-import",
                     "status", "pause", "drain", "resume", "service"):
            assert "Usage:" in run("authority", name, "--help").stdout
        assert not status("authority")["configured"]
        run("authority", "init", "--name", "authority")
        for name in ("worker", "owner"):
            invitation = root / (name + ".invite")
            run("authority", "invite", invitation, "--name", name)
            run(name, "join", invitation)
        broker_port, worker_port = pool.port(), pool.port()
        broker_address, worker_address = f"127.0.0.1:{broker_port}", f"127.0.0.1:{worker_port}"
        for name in directories:
            run(name, "configure", "--discovery", "off", "--broker", broker_address,
                "--backend", "cpu", "--cpus", 2, "--host-bytes", "1G", "--max-brokers", 32)
        broker = start("authority", "broker", "--listen", broker_address, "--worker-ms", 3000,
                       "--lease-ms", 3000, port=broker_port)
        worker_args = ("worker", "--listen", worker_address, "--slots", 1, "--generation-threads", 1,
                       "--max-frame-bytes", "64K", "--max-task-bytes", "16M")
        worker = start("worker", *worker_args, port=worker_port)
        live = until(lambda: (value if (value := status("owner", "--worker", worker_address))["workers"] else None))
        worker_node = live["workers"][0]["node_id"]
        assert live["workers"][0]["broker_limit"] == 32
        run("owner", "pause", "--worker", worker_address, code=1)
        paused = json.loads(run("worker", "pause", "--worker", worker_address, "--json").stdout)
        assert paused["changed"] == 1 and paused["workers"][0]["admission"] == "paused"

        output = root / "plots"
        plot = pool.Process([binary, "plot", "-k", 18, "--group-members", 3, "-f", pool.FARMER,
                             "--pool-ph", pool.POOL, "-S", pool.SEED, "-o", output,
                             "--devices", "cpu", "--compute-directory", directories["owner"],
                             "--compute-local", "off", "--max-host-ram", "512M", "--workers", 2,
                             "--window-bytes", "64K", "--checkpoint-bytes", "64K", "--quiet", "--config", os.devnull],
                            root / "owner-plot.log")
        services.append(plot)
        def queued_jobs():
            value = status("owner", "--broker", broker_address)
            return value if any(b["summary"]["queued"] for b in value["brokers"]) else None

        queued = until(queued_jobs)
        jobs = queued["brokers"][0]["jobs"]
        assert any(job["state"] == "queued" and job["waiting"] == "worker_paused" for job in jobs), jobs
        encoded = json.dumps(queued)
        assert pool.SEED not in encoded and "lease_token" not in encoded and "owner_nonce" not in encoded
        run("worker", "resume", "--worker", worker_address)
        plot.finish(timeout=180)
        paths = list(output.glob("*.gplot"))
        assert len(paths) == 1, paths
        proof = pool.run([binary, "verify", paths[0], "--full", "--trials", 100, "--config", os.devnull])
        match = re.search(r"(\d+) full proofs validated", (proof.stdout + proof.stderr).lower())
        assert match and int(match[1]), proof
        drained = json.loads(run("worker", "drain", "--worker", worker_address, "--json").stdout)
        assert drained["workers"][0]["admission"] == "drained"
        run("worker", "resume", "--worker", worker_address)
        print("PASS authenticated live jobs/resources, own-worker controls, queued resume and full proofs", flush=True)

        # Replace one enrolled leaf without changing its stable storage namespace.
        before = live["workers"][0]["identity"]
        request, replacement = root / "renew.request", root / "renew.response"
        run("worker", "renew-request", request)
        run("authority", "renew-sign", request, replacement)
        run("worker", "renew-import", replacement)
        worker.stop()
        worker = start("worker", *worker_args, port=worker_port)
        renewed = until(lambda: (value if (value := status("worker", "--worker", worker_address))["workers"] else None))
        assert renewed["workers"][0]["node_id"] == worker_node
        assert renewed["workers"][0]["identity"] != before
        # A second service for the renewed stable identity cannot offer capacity.
        result = subprocess.run(command("worker", "worker", "--listen", f"127.0.0.1:{pool.port()}"),
                                capture_output=True, text=True, encoding="utf-8", timeout=15)
        assert result.returncode != 0 and "already offers" in result.stderr, result
        run("authority", "revoke", worker_node)
        policy = root / "policy"
        run("authority", "policy-export", policy)
        run("owner", "policy-import", policy)
        rejected = subprocess.run(command("owner", "status", "--worker", worker_address, "--json"),
                                  capture_output=True, text=True, encoding="utf-8", timeout=15)
        rejection = json.loads(rejected.stdout)
        assert not rejection["workers"] and rejection["errors"], rejection
        print("PASS installed renewal preserves identity, duplicate executor fencing and propagated revocation", flush=True)

        if os.name == "nt":
            printed = run("authority", "service", "print").stdout
            task = ET.fromstring(printed.encode("utf-8"))
            # Registration passes the same XML to Task Scheduler as a BSTR.
            ET.fromstring(printed.encode("utf-16"))
            ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
            assert task.findtext("t:Actions/t:Exec/t:Command", namespaces=ns) == str(binary)
            arguments = task.findtext("t:Actions/t:Exec/t:Arguments", namespaces=ns)
            assert "compute worker --directory" in arguments and str(directories["authority"]) in arguments
            assert task.findtext("t:Principals/t:Principal/t:LogonType", namespaces=ns) == "InteractiveToken"
            assert task.findtext("t:Principals/t:Principal/t:RunLevel", namespaces=ns) == "LeastPrivilege"
            assert task.findtext("t:Settings/t:ExecutionTimeLimit", namespaces=ns) == "PT0S"
            assert 1 <= int(task.findtext("t:Settings/t:RestartOnFailure/t:Count", namespaces=ns)) <= 255
            print("PASS Windows task XML, Unicode executable/membership arguments and user context", flush=True)
            if service_manager:
                managed_service()
            return

        # Exercise real publication/removal and exact manager argv without changing
        # the invoking user's systemd state. A generated unit is also printable.
        environment = dict(os.environ)
        environment["XDG_CONFIG_HOME"] = str(root / "user config with % and $ characters")
        fake = root / "manager-bin"
        fake.mkdir()
        calls = root / "manager.jsonl"
        manager = fake / "systemctl"
        manager.write_text("#!" + sys.executable + "\nimport json, sys\nfrom pathlib import Path\n"
                           f"with Path({str(calls)!r}).open('a') as out: out.write(json.dumps(sys.argv[1:]) + '\\n')\n")
        manager.chmod(0o700)
        environment["PATH"] = str(fake) + os.pathsep + environment["PATH"]
        printed = run("authority", "service", "print", env=environment).stdout
        assert "Type=exec" in printed and "Restart=on-failure" in printed and " compute worker " in printed
        assert not calls.exists()
        run("authority", "service", "install", env=environment)
        units = list((Path(environment["XDG_CONFIG_HOME"]) / "systemd/user").glob("*.service"))
        assert len(units) == 1 and units[0].read_text() == printed
        for action in ("start", "status", "stop", "restart"):
            run("authority", "service", action, env=environment)
        run("authority", "service", "remove", env=environment)
        assert not units[0].exists()
        recorded = [json.loads(line) for line in calls.read_text().splitlines()]
        assert all(call[:2] == ["--user", "--no-pager"] for call in recorded)
        assert recorded[1][2:] == ["enable", "--now", units[0].name]
        assert recorded[-2][2:] == ["disable", "--now", units[0].name]
        # Refuse to overwrite a unit supplied by the user.
        units[0].write_text("[Service]\nExecStart=/bin/true\n")
        run("authority", "service", "install", env=environment, code=1)
        assert units[0].read_text() == "[Service]\nExecStart=/bin/true\n"
        print("PASS generated user service, exact manager actions and unrelated-unit preservation", flush=True)
        if service_manager:
            managed_service()
    finally:
        for service in reversed(services):
            service.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--scratch", type=Path, default=Path.cwd())
    parser.add_argument("--service-manager", action="store_true",
                        help="exercise the native user service manager and remove the disposable identity's service")
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compute cli é-", dir=args.scratch) as directory:
        check(args.binary.resolve(), Path(directory).resolve(), args.service_manager)


if __name__ == "__main__":
    main()
