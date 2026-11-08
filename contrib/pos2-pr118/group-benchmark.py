#!/usr/bin/env python3
"""Compare direct and streaming assembly separately from identical verification (Linux)."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import stat
import statistics
import subprocess
import sys
import threading
import time


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def filesystem(path):
    matches = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        mount, storage = line.split(" - ", 1)
        fields = mount.split()
        target = fields[4]
        for escaped, plain in [(r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")]:
            target = target.replace(escaped, plain)
        if path.is_relative_to(target):
            kind, source, options = storage.split()[:3]
            matches.append({"mount": target, "type": kind, "source": source,
                            "options": fields[5] + "," + options})
    return max(matches, key=lambda entry: len(entry["mount"]))


def allocated(paths, transient_paths=()):
    files = {}
    for sources, transient in ((paths, False), (transient_paths, True)):
        for path in sources:
            try:
                info = path.stat()
            except (FileNotFoundError, ProcessLookupError):
                continue  # A consumed segment or checkpoint temporary was removed.
            except PermissionError:
                if not transient:
                    raise
                continue  # Access to a sampled /proc fd changed after enumeration.
            if stat.S_ISREG(info.st_mode):
                files[info.st_dev, info.st_ino] = info.st_blocks * 512
    return sum(files.values())


def sample(pid, executable, work):
    proc = Path(f"/proc/{pid}")
    memory = None
    io = {}
    try:
        # VmHWM belongs to the new address space after exec. Unlike wait4's
        # ru_maxrss, this avoids the Python parent's inherited pre-exec peak.
        if (proc / "exe").samefile(executable):
            for line in (proc / "status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    memory = int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    # A zombie has no exe/mm. Final I/O counters can remain readable, but
    # some kernels deny access after exit; unavailable samples stay absent.
    try:
        io = {key: int(value) for key, value in
              (line.split(":", 1) for line in (proc / "io").read_text().splitlines())}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    paths = list(work.rglob("*"))
    # Include the direct assembler's open, unlinked index spool, once per inode.
    descriptors = []
    try:
        for fd in (proc / "fd").iterdir():
            try:
                target = os.readlink(fd)
                if target.startswith(str(work) + "/"):
                    descriptors.append(fd)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return memory, io, allocated(paths, descriptors)


def sync_output(path):
    with path.open("rb") as file:
        os.fsync(file.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ProcessExit:
    """Observe exit independently of potentially slow sampling; leave a zombie for final I/O."""

    def __init__(self, pid):
        self.completed = threading.Event()
        self.observed = None
        self.error = None
        self.thread = threading.Thread(target=self._wait, args=(pid,))
        self.thread.start()

    def _wait(self, pid):
        try:
            os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
            self.observed = time.monotonic()
        except OSError as error:
            self.error = error
        finally:
            self.completed.set()

    def join(self):
        self.thread.join()
        if self.error is not None:
            raise self.error


def measure(command, log_path, work, interval, output=None):
    command = [str(arg) for arg in command]
    peak_rss = None
    peak_allocated = 0
    counters = {}
    with log_path.open("w") as log:
        started = time.monotonic()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        observer = None
        try:
            observer = ProcessExit(process.pid)
            while not observer.completed.is_set():
                memory, io, space = sample(process.pid, Path(command[0]), work)
                if memory is not None:
                    peak_rss = max(peak_rss or 0, memory)
                peak_allocated = max(peak_allocated, space)
                counters.update(io)
                observer.completed.wait(interval)
            observer.join()
            # Sample final I/O without reaping the process first. Its exit
            # timestamp is already fixed, so sampling cannot inflate it.
            _, io, space = sample(process.pid, Path(command[0]), work)
            counters.update(io)
            peak_allocated = max(peak_allocated, space)
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        finally:
            if process.returncode is None:
                process.kill()
                if observer is not None:
                    observer.thread.join()
                process.wait()
    process_seconds = observer.observed - started
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}); see {log_path}")
    synced = time.monotonic()
    if output is not None:
        sync_output(output)
    sync_seconds = time.monotonic() - synced if output is not None else 0
    return {
        "command": command,
        "seconds": process_seconds + sync_seconds,
        "measurement_seconds": time.monotonic() - started,
        "process_seconds": process_seconds,
        "output_sync_seconds": sync_seconds,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "sampled_peak_rss_bytes": peak_rss,
        "sampled_peak_work_allocated_bytes": peak_allocated,
        "process_io": counters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direct", type=Path, help="pos2_pr118_group with assemble-only")
    parser.add_argument("stream", type=Path, help="pos2_pr118_stream")
    parser.add_argument("inputs", type=Path, help="existing group.inputs; raw members are retained")
    parser.add_argument("work_dir", type=Path, help="new directory for report and logs")
    parser.add_argument("--baseline", type=Path, help="optional older assembler with assemble-only")
    parser.add_argument("--ram-mib", type=int, default=512, help="same assembly budget for all methods")
    parser.add_argument("--verify-ram-mib", type=int, default=512)
    parser.add_argument("--batch-members", type=int, default=16)
    parser.add_argument("--fan-in", type=int, default=4)
    parser.add_argument("--buffer-kib", type=int, default=64)
    parser.add_argument("--segment-mib", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-ms", type=int, default=50)
    parser.add_argument("--cache", choices=("read", "unchanged"), default="read",
                        help="read all inputs before each assembly, or leave cache uncontrolled")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this benchmark requires Linux /proc and wait4")
    if args.warmups < 0 or min(args.repeats, args.sample_ms, args.batch_members,
                               args.fan_in, args.buffer_kib, args.segment_mib) < 1:
        parser.error("counts and sizes must be positive; warmups may be zero")
    for name in ("direct", "stream", "inputs", "baseline"):
        path = getattr(args, name)
        if path is not None:
            path = path.resolve()
            if not path.is_file():
                parser.error(f"{name} does not exist: {path}")
            setattr(args, name, path)
    lines = args.inputs.read_text().splitlines()
    parameters = shlex.split(lines[0]) if lines else []
    raw = [Path(shlex.split(line)[0]).resolve() for line in lines[1:] if line.strip()]
    if not raw or len(parameters) not in (5, 6) or (len(parameters) == 6 and parameters[5] != "0"):
        parser.error("streaming comparison requires an ordinary group starting at member zero")
    stamps = {path: (path.stat().st_dev, path.stat().st_ino, path.stat().st_size,
                     path.stat().st_mtime_ns) for path in raw}
    root = args.work_dir.resolve()
    root.mkdir(mode=0o700)
    tools = {"direct": args.direct, "stream": args.stream}
    if args.baseline:
        tools = {"baseline": args.baseline, **tools}
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "cpu_model": next((line.split(":", 1)[1].strip() for line in
                           Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), "unknown"),
        "output_filesystem": filesystem(root),
        "input_filesystems": {str(path): filesystem(path) for path in sorted({p.parent for p in raw})},
        "inputs": str(args.inputs), "inputs_sha256": digest(args.inputs),
        "k": int(parameters[0]), "members": len(raw),
        "raw_bytes": sum(info[2] for info in stamps.values()),
        "raw_allocated_bytes": allocated(raw),
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "executables": {name: {"path": str(path), "sha256": digest(path)}
                        for name, path in tools.items()},
        "measurement": {
            "assembly": "launch through independently observed exit plus output file/directory fsync; final sampling excluded",
            "sampling": "RSS/I/O/allocation sampling remains active during assembly and may contend for CPU/storage; total measurement time is reported separately",
            "verification": "same direct verifier once per method, outside assembly timing; every repeat must match verified bytes",
            "cache": "input pre-read is outside timing; residency is not guaranteed; no global cache drop",
            "rss": "sampled post-exec VmHWM; may miss the final peak; excludes filesystem cache",
            "allocation": "sampled st_blocks, including open unlinked work files; excludes FS metadata and snapshot/shared-extent accounting",
            "io": "Linux per-process counters; include runtime/log/checkpoint I/O, not physical device traffic",
            "durability": "stream retains its checkpoint syncs; direct syncs only completed output",
            "order": "serial methods rotated each round; warmups excluded from medians",
            "outputs": "raw inputs retained; each byte-checked output removed before the next method",
        },
        "runs": [],
    }
    expected = None
    verified = {}
    names = list(tools)
    for round_number in range(args.warmups + args.repeats):
        order = names[round_number % len(names):] + names[:round_number % len(names)]
        for name in order:
            label = f"{round_number:02d}-{name}"
            print(f"Assembly {label}", flush=True)
            case = root / label
            case.mkdir(mode=0o700)
            if args.cache == "read":
                for path in raw:
                    with path.open("rb") as file:
                        while file.read(1024 * 1024):
                            pass
            if name == "stream":
                work = case / "work"
                output = work / "group.gplot"
                command = [tools[name], args.inputs, work, args.ram_mib, args.batch_members,
                           args.fan_in, args.buffer_kib, args.segment_mib]
            else:
                output = case / "group.gplot"
                command = [tools[name], "assemble-only", args.inputs, output, args.ram_mib]
            assembly = measure(command, root / f"{label}-assembly.log", case,
                               args.sample_ms / 1000, output)
            actual = digest(output)
            if expected is None:
                expected = actual
            if actual != expected:
                raise RuntimeError(f"output bytes differ for {label}")
            verification = None
            if name not in verified:
                print(f"Verification {label}", flush=True)
                verification = measure([args.direct, "verify", args.inputs, output, args.verify_ram_mib],
                                       root / f"{label}-verify.log", case, args.sample_ms / 1000)
                verified[name] = label
            run = {"method": name, "round": round_number, "warmup": round_number < args.warmups,
                   "assembly": assembly, "verification": verification,
                   "verified_by_run": verified[name],
                   "group_bytes": output.stat().st_size, "group_sha256": actual}
            if name == "stream":
                run["stream_metrics"] = json.loads((work / "metrics.json").read_text())
            report["runs"].append(run)
            (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            shutil.rmtree(case)
    for path, before in stamps.items():
        info = path.stat()
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != before:
            raise RuntimeError(f"raw input changed during benchmark: {path}")
    report["median_assembly_seconds"] = {
        name: statistics.median(run["assembly"]["seconds"] for run in report["runs"]
                                if run["method"] == name and not run["warmup"])
        for name in names}
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["median_assembly_seconds"], indent=2))
    print(f"PASS identical bytes and full verification; report: {root / 'results.json'}")


if __name__ == "__main__":
    main()
