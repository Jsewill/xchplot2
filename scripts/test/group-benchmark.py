#!/usr/bin/env python3
"""Measure bounded same-file growth against canonical CPU bytes (Linux)."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import select
import shlex
import shutil
import subprocess
import sys
import time


# Keep filesystem, allocated-space and post-exec RSS measurements consistent
# with the existing external-assembly benchmark.
spec = importlib.util.spec_from_file_location("group_benchmark", Path(__file__).with_name("group-benchmark.py"))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def gpu_memory(pid):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=10)
    # Do not retain another application's PID, name, activity or memory use.
    values = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid) and fields[1].isdigit():
            values.append(int(fields[1]) * 1024 * 1024)
    return sum(values) if values else None


def measure(command, log_path, work, interval, nvidia):
    peak_rss, peak_gpu = None, None
    peak_allocated = 0
    counters = {}
    samples = 0
    started = time.monotonic()
    with log_path.open("x") as log:
        process = subprocess.Popen([str(value) for value in command], stdout=log, stderr=subprocess.STDOUT)
        pidfd = None
        try:
            pidfd = os.pidfd_open(process.pid)
            while True:
                memory, io, space = common.sample(process.pid, Path(command[0]), work)
                if memory is not None:
                    peak_rss = max(peak_rss or 0, memory)
                peak_allocated = max(peak_allocated, space)
                counters.update(io)
                if os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT):
                    _, io, space = common.sample(process.pid, Path(command[0]), work)
                    counters.update(io)
                    peak_allocated = max(peak_allocated, space)
                    _, status, usage = os.wait4(process.pid, 0)
                    process.returncode = os.waitstatus_to_exitcode(status)
                    break
                if nvidia:
                    memory = gpu_memory(process.pid)
                    if memory is not None:
                        peak_gpu = max(peak_gpu or 0, memory)
                        samples += 1
                select.select([pidfd], [], [], interval)
        finally:
            if pidfd is not None:
                os.close(pidfd)
            if process.returncode is None:
                process.kill()
                process.wait()
    if process.returncode:
        raise RuntimeError(f"growth failed ({process.returncode}); output retained; see {log_path}")
    metrics = next((json.loads(line) for line in reversed(log_path.read_text().splitlines()) if line.startswith('{"mode":')), None)
    if metrics is None:
        raise RuntimeError(f"growth did not report metrics: {log_path}")
    if nvidia and metrics["mode"] == "gpu" and not samples:
        raise RuntimeError("no matching GPU PID observed; run outside a PID namespace or omit --nvidia")
    return {
        "seconds": time.monotonic() - started,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "sampled_peak_rss_bytes": peak_rss,
        "sampled_peak_vram_bytes": peak_gpu,
        "gpu_samples": samples,
        "sampled_peak_allocated_bytes": peak_allocated,
        "process_io": counters,
        "growth": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grow", type=Path)
    parser.add_argument("group", type=Path, help="pinned pos2_pr118_group reference assembler")
    parser.add_argument("plotter", type=Path, help="xchplot2 with full-proof verification")
    parser.add_argument("inputs", type=Path, help="existing index-zero CPU raw-member fixture")
    parser.add_argument("work_dir", type=Path, help="new directory on the storage device under test")
    parser.add_argument("--members", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--batch-members", type=int, default=1)
    parser.add_argument("--modes", nargs="+", choices=("auto", "incremental", "staged", "merged"), default=["incremental", "merged", "auto"])
    parser.add_argument("--backend", choices=("fixture", "cpu", "gpu"), default="fixture")
    parser.add_argument("--tier", default="auto")
    parser.add_argument("--member-storage", choices=("compact", "raw"), default="compact")
    parser.add_argument("--generation-overlap", choices=("on", "off"), default="on")
    parser.add_argument("--max-ram", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--window-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--checkpoint-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--capacity-bytes", type=int)
    parser.add_argument("--max-work-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-ms", type=int, default=100)
    parser.add_argument("--nvidia", action="store_true", help="sample only the measured process's NVIDIA memory")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Linux /proc and wait4 are required")
    if min(*args.members, args.batch_members, args.max_ram, args.window_bytes, args.checkpoint_bytes,
           args.workers, args.max_work_bytes, args.repeats, args.sample_ms) <= 0:
        parser.error("sizes and counts must be positive")
    if args.nvidia and (args.backend != "gpu" or not shutil.which("nvidia-smi")):
        parser.error("--nvidia requires --backend gpu and nvidia-smi")
    for name in ("grow", "group", "plotter", "inputs"):
        value = getattr(args, name).resolve()
        if not value.is_file():
            parser.error(f"missing {name}: {value}")
        setattr(args, name, value)
    lines = args.inputs.read_text().splitlines()
    parameters = shlex.split(lines[0]) if lines else []
    raw = [Path(shlex.split(line)[0]).resolve() for line in lines[1:] if line.strip()]
    if len(parameters) not in (5, 6) or (len(parameters) == 6 and parameters[5] != "0") or max(args.members) > len(raw):
        parser.error("an index-zero fixture covering all requested members is required")
    root = args.work_dir.resolve()
    root.mkdir(mode=0o700)
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "filesystem": common.filesystem(root),
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "executables": {name: common.digest(getattr(args, name)) for name in ("grow", "group", "plotter")},
        "measurement": {
            "timing": "whole process including generation/loading, admission, durable construction and final publication",
            "reference": "pinned CPU assembly and 100 full-proof trials outside timed growth; every output must match its bytes",
            "memory": "post-exec sampled RSS plus grow's actual process RSS high-water mark; excludes filesystem cache",
            "vram": "optional sampled memory for this process only, summed across devices; brief peaks may be missed",
            "io": "grow counters count destination syscalls; /proc counters include other process I/O; neither is physical device traffic",
            "space": "logical capacity bounded; sampled st_blocks excludes filesystem metadata and CoW snapshots",
            "cache": "uncontrolled; no system-wide cache drop; method order rotates each repeat",
            "cleanup": "verified reference and byte-matching growth payloads removed immediately; input fixtures retained; failures retained",
        },
        "runs": [],
    }
    for count in args.members:
        # Raw stubs use k-2 bits per fragment; canonical low/FSE fields use
        # at most k+4 plus unary quotients. The latter total less than 93 bits
        # per member/chunk (range / (178 << (k-8))). Include framing, FSE
        # states and the size index before creating an untimed reference.
        chunks = 1 << (int(parameters[0]) - 6)
        reference_bound = 2 * sum(path.stat().st_size for path in raw[:count]) + chunks * (12 * count + 32) + 4096
        if reference_bound > args.max_work_bytes:
            raise RuntimeError("raw fixture and index bound exceeds --max-work-bytes before reference assembly")
        inputs = root / f"{count}.inputs"
        inputs.write_text(lines[0] + "\n" + "".join('"' + str(path).replace('\\', '\\\\').replace('"', '\\"') + '"\n' for path in raw[:count]))
        reference = root / "reference.gplot"
        with (root / f"{count}-reference.log").open("x") as log:
            subprocess.run([str(args.group), "assemble-only", str(inputs), str(reference), "512"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([str(args.plotter), "verify", str(reference), "--full", "--trials", "100"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        expected, size = common.digest(reference), reference.stat().st_size
        reference.unlink()
        capacity = args.capacity_bytes if args.capacity_bytes is not None else size * 2 + 4 * args.checkpoint_bytes + 16 * args.window_bytes
        if not size <= capacity <= args.max_work_bytes:
            raise RuntimeError("requested capacity exceeds --max-work-bytes or cannot hold the reference")
        for repeat in range(args.repeats):
            order = args.modes[repeat % len(args.modes):] + args.modes[:repeat % len(args.modes)]
            for mode in order:
                label = f"{count}-{repeat}-{mode}"
                print(label, flush=True)
                case = root / label
                case.mkdir(mode=0o700)
                output = case / "group.gplot"
                command = [args.grow, output, "--construction", mode, "--max-members", count,
                           "--batch-members", args.batch_members, "--capacity-bytes", capacity,
                           "--max-ram", args.max_ram, "--window-bytes", args.window_bytes,
                           "--checkpoint-bytes", args.checkpoint_bytes, "--workers", args.workers,
                           "--member-storage", args.member_storage, "--generation-overlap", args.generation_overlap]
                if args.backend == "fixture":
                    command += ["--fixture-inputs", inputs]
                else:
                    k, strength, meta, group, memo = parameters[:5]
                    command += ["--backend", args.backend, "--tier", args.tier, "--k", k, "--strength", strength,
                                "--meta-group", meta, "--group-id", group, "--memo", memo]
                result = measure(command, root / f"{label}.log", case, args.sample_ms / 1000, args.nvidia)
                if result["growth"]["members"] != count:
                    raise RuntimeError(f"{label} stopped at {result['growth']['members']} members; increase --capacity-bytes "
                                       "within --max-work-bytes; output retained")
                if common.digest(output) != expected:
                    raise RuntimeError(f"canonical bytes differ for {label}; output retained")
                if result["growth"]["peak_file_bytes"] > capacity or result["growth"]["peak_modeled_bytes"] > args.max_ram:
                    raise RuntimeError(f"admitted budget exceeded for {label}; output retained")
                result.update(members=count, construction=mode, repeat=repeat, sha256=expected, final_bytes=size)
                report["runs"].append(result)
                output.unlink()
                case.rmdir()
                (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("canonical parity and full-proof verification passed; benchmark outputs removed")


if __name__ == "__main__":
    main()
