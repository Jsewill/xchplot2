#!/usr/bin/env python3
"""Crash/reclaim checks using disposable copies of a small existing raw fixture."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import tempfile

from group import quoted


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def run(command, code=0, point=None, **kwargs):
    env = dict(os.environ)
    env.pop("XCHPLOT2_STREAM_FAILPOINT", None)
    if point:
        env["XCHPLOT2_STREAM_FAILPOINT"] = point
    result = subprocess.run([str(arg) for arg in command], env=env, capture_output=True, text=True, **kwargs)
    assert result.returncode == code, result.stdout + result.stderr
    return result


def check(args, root):
    lines = args.inputs.read_text().splitlines()
    parameters = shlex.split(lines[0])
    originals = [Path(shlex.split(line)[0]).resolve() for line in lines[1:] if line.strip()]
    assert int(parameters[0]) == 18 and 5 <= len(originals) <= 64, "use a k18 fixture with 5..64 members"
    before = {path: digest(path) for path in originals}
    expected = digest(args.reference)
    results = []

    def fixture(name):
        case = root / name
        raw = case / "raw"
        raw.mkdir(parents=True, mode=0o700)
        paths = [raw / f"{i}.plot2" for i in range(len(originals))]
        for source, target in zip(originals, paths):
            shutil.copy2(source, target)
        inputs = case / "group.inputs"
        inputs.write_text(lines[0] + "\n" + "".join(quoted(path) + "\n" for path in paths))
        work = case / "work"
        command = [args.stream, inputs, work, 16, 1, 2, 64, 1, "--consume-raw"]
        return work, paths, command

    def complete(work, paths, command):
        run(command + ["--resume"])
        output = work / "group.gplot"
        assert digest(output) == expected, "recovered output differs from reference"
        assert not any(path.exists() for path in paths), "committed raw inputs were not reclaimed"
        assert not list(work.glob("*.tmp")) and not list(work.glob("*.partial*"))
        stats = json.loads((work / "metrics.json").read_text())
        assert stats["remaining_raw_bytes"] == 0 and stats["reclaimed_raw_bytes"] == stats["input_bytes"]
        assert stats["remaining_work_file_bytes"] == output.stat().st_size
        unchanged = output.stat().st_ino, output.stat().st_mtime_ns
        run(command + ["--resume"])
        assert unchanged == (output.stat().st_ino, output.stat().st_mtime_ns)
        assert digest(output) == expected

    points = ["stage-after-flush", "stage-before-checkpoint", "stage-after-checkpoint", "raw-after-unlink",
              "merge-after-flush", "merge-before-checkpoint", "merge-after-checkpoint", "segment-after-unlink",
              "final-after-flush", "final-before-checkpoint", "final-after-checkpoint:2",
              "final-tail-after-flush", "ready-after-checkpoint", "publish-after-link",
              "publish-after-unlink", "done-after-checkpoint"]
    for point in points:
        work, paths, command = fixture(point.replace(":", "-"))
        run(command, 86, point)
        if point.startswith(("merge-", "segment-", "final-", "ready-", "publish-", "done-")):
            assert not any(path.exists() for path in paths), "test still has raw recovery copies"
        if point in ("stage-after-flush", "stage-before-checkpoint"):
            assert all(path.exists() for path in paths), "raw deleted before checkpoint"
        complete(work, paths, command)
        results.append(point)
        print("PASS recovery:", point, flush=True)

    work, paths, command = fixture("repeated-interruption")
    run(command, 86, "raw-after-unlink")
    run(command + ["--resume"], 86, "merge-after-checkpoint")
    run(command + ["--resume"], 86, "final-after-checkpoint")
    run(command + ["--resume"], 86, "publish-after-link")
    complete(work, paths, command)

    work, paths, command = fixture("reject-changes")
    run(command, 86, "stage-after-checkpoint")
    checkpoint = work / "checkpoint"
    saved = checkpoint.read_bytes()
    raw_before = {path: digest(path) for path in paths}
    assert "settings mismatch" in run(command[:-1] + ["--resume"], 1).stderr
    changed = command.copy()
    changed[4] = 2
    assert "settings mismatch" in run(changed + ["--resume"], 1).stderr
    inputs = Path(command[1])
    original_inputs = inputs.read_text()
    inputs.write_text(original_inputs.replace(parameters[3], "ff" * 32, 1))
    assert "identity mismatch" in run(command + ["--resume"], 1).stderr
    inputs.write_text(original_inputs)
    checkpoint.write_bytes(saved[:-1])
    assert "checksum mismatch" in run(command + ["--resume"], 1).stderr
    body = saved.split(b"\n", 1)[1].splitlines(keepends=True)
    body[1] = b"0" * 64 + b"\n"
    body = b"".join(body)
    checkpoint.write_bytes(hashlib.sha256(body).hexdigest().encode() + b"\n" + body)
    assert "source/version mismatch" in run(command + ["--resume"], 1).stderr
    # A valid checksum does not make inconsistent membership safe to reclaim.
    body = saved.split(b"\n", 1)[1].splitlines(keepends=True)
    first_run = 8 + len(originals)
    fields = body[first_run].split()
    fields[1] = b"1"
    body[first_run] = b" ".join(fields) + b"\n"
    body = b"".join(body)
    checkpoint.write_bytes(hashlib.sha256(body).hexdigest().encode() + b"\n" + body)
    assert "run coverage" in run(command + ["--resume"], 1).stderr
    checkpoint.write_bytes(saved)
    segment = next(work.glob("run-*.tmp"))
    data = segment.read_bytes()
    segment.write_bytes(data[:-1])
    assert "segment is truncated" in run(command + ["--resume"], 1).stderr
    assert all(digest(path) == raw_before[path] for path in paths), "invalid replacement caused raw deletion"
    segment.write_bytes(data)
    with (work / "lock").open("r+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert "already in use" in run(command + ["--resume"], 1).stderr
    # The memory cap can be increased without changing the persisted geometry.
    command[3] = 32
    complete(work, paths, command)

    work, paths, command = fixture("truncated-final-prefix")
    run(command, 86, "final-after-checkpoint:2")
    assert not any(path.exists() for path in paths)
    for name in ("group.gplot.partial", "index.tmp"):
        prefix = work / name
        data = prefix.read_bytes()
        assert data
        prefix.write_bytes(data[:-1])
        assert "committed output is truncated" in run(command + ["--resume"], 1).stderr
        prefix.write_bytes(data)
    complete(work, paths, command)

    work, paths, command = fixture("changed-raw")
    run(command, 86, "init-after-checkpoint")
    parked = paths[0].with_suffix(".original")
    paths[0].rename(parked)
    shutil.copy2(parked, paths[0])
    assert "raw member changed" in run(command + ["--resume"], 1).stderr
    assert all(path.exists() for path in paths)
    paths[0].unlink()
    parked.rename(paths[0])
    complete(work, paths, command)

    work, paths, command = fixture("occupied-publication")
    run(command, 86, "ready-after-checkpoint")
    occupied = work / "group.gplot"
    occupied.write_bytes(b"preserve unrelated output")
    assert "completed group was changed" in run(command + ["--resume"], 1).stderr
    assert occupied.read_bytes() == b"preserve unrelated output"
    occupied.unlink()
    complete(work, paths, command)
    old_stat = occupied.stat()
    with occupied.open("r+b") as file:
        file.seek(55)
        byte = file.read(1)
        file.seek(55)
        file.write(bytes([byte[0] ^ 1]))
    damaged = digest(occupied)
    assert "completed group was changed" in run(command + ["--resume"], 1).stderr
    assert digest(occupied) == damaged
    with occupied.open("r+b") as file:
        file.seek(55)
        file.write(byte)
    os.utime(occupied, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    complete(work, paths, command)

    # A real failed write leaves the previous checkpoint recoverable, too.
    work, paths, command = fixture("write-failure")
    def limit_file():
        resource.setrlimit(resource.RLIMIT_FSIZE, (327680, 327680))
    failed = subprocess.run([str(arg) for arg in command], capture_output=True, preexec_fn=limit_file)
    assert failed.returncode != 0 and all(path.exists() for path in paths)
    complete(work, paths, command)
    assert all(digest(path) == before[path] for path in originals), "original fixture was modified"
    (root / "results.json").write_text(json.dumps({"crash_boundaries": results,
        "additional_checks": ["repeated crashes", "identity/settings/source/checksum/coverage rejection", "truncation",
                              "raw replacement", "concurrent lock", "occupied/corrupt output", "write failure"]}, indent=2) + "\n")
    print("PASS durable resume and raw reclamation; original fixture preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", type=Path)
    parser.add_argument("inputs", type=Path, help="existing small group.inputs; originals are never consumed")
    parser.add_argument("reference", type=Path, help="matching reference .gplot")
    parser.add_argument("--work-dir", type=Path, help="new directory for retained test artifacts")
    args = parser.parse_args()
    for name in ("stream", "inputs", "reference"):
        setattr(args, name, getattr(args, name).resolve())
    if args.work_dir:
        root = args.work_dir.resolve()
        root.mkdir(mode=0o700)
        check(args, root)
    else:
        with tempfile.TemporaryDirectory(prefix="xchplot2-stream-recovery-") as temporary:
            check(args, Path(temporary))


if __name__ == "__main__":
    main()
