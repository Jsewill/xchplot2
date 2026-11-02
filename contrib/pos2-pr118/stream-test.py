#!/usr/bin/env python3
"""Byte parity, upstream proving, bounded merges and scratch reclamation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time

from group import quoted


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def run(command, success=True):
    result = subprocess.run([str(arg) for arg in command], capture_output=True, text=True)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


def check(args, root):
    raw_dir = root / "raw members"
    raw_dir.mkdir(mode=0o700)
    group = bytes(range(32)).hex()
    memo = "01020304"
    params = f"{args.k} 2 7 {group} {memo}"
    raw = [raw_dir / f"{member}.plot2" for member in range(args.members)]
    config = root / "empty.conf"
    config.write_text("")
    manifest = root / "batch.tsv"
    manifest.write_text("".join(
        f"{args.k} 2 {member} 7 raw-v2 {group} {memo} {quoted(raw_dir)} {path.name}\n"
        for member, path in enumerate(raw)) +
        f"{args.k} 2 0 7 gplot-v2 {group} {memo} {quoted(root)} native.gplot\n")
    print(f"Plotting {args.members} raw members at k={args.k}", flush=True)
    workers = ["--cpu-workers", "1"] if args.devices == "cpu" else []
    run([args.plotter, "batch", manifest, "--config", config, "--devices", args.devices,
         "--tier", "plain", "--quiet", "--no-progress", *workers])
    before = {path: digest(path) for path in raw}
    inputs = root / "group.inputs"
    inputs.write_text(params + " 0\n" + "".join(quoted(path) + "\n" for path in raw))
    reference = root / "reference.gplot"
    print("Assembling and proving the reference", flush=True)
    result = run([args.reference, "assemble", inputs, reference, 512])
    assert all(f"PASS group member={member} " in result.stdout for member in range(args.members))
    expected = digest(reference)

    cases = [(1, 2, 16), (3, 3, 16), (args.members, 2, 128)]
    metrics = []
    for index, (batch, fan, ram) in enumerate(cases):
        work = root / f"stream-{index}"
        print(f"Stream batch={batch}, fan-in={fan}, RAM={ram} MiB", flush=True)
        run([args.stream, inputs, work, ram, batch, fan, 64, 1])
        output = work / "group.gplot"
        assert digest(output) == expected, "streamed group differs from reference bytes"
        with output.open("rb") as file:
            assert file.read(5) == b"PoS2\x02", "output is not the upstream v2 format"
        report = json.loads((work / "metrics.json").read_text())
        assert report["group_bytes"] == output.stat().st_size
        assert report["input_bytes"] == sum(path.stat().st_size for path in raw)
        assert report["remaining_work_file_bytes"] == report["group_bytes"]
        assert report["run_read_bytes"] == report["staged_bytes"] + report["merge_written_bytes"]
        assert report["max_active_members"] <= batch
        assert report["max_active_runs"] <= fan
        assert report["peak_modeled_bytes"] <= ram * 1024 * 1024
        if index == 0 and args.members > 4:
            assert report["merge_passes"] > 1, "fixture did not exercise multiple merge passes"
        assert not list(work.glob("*.tmp")), "consumed scratch files were not reclaimed"
        assert not list(work.glob("*.partial"))
        metrics.append(report)
    # The production verifier uses the unmodified upstream reader/prover.
    result = run([args.plotter, "verify", root / "stream-0/group.gplot", "--full", "--trials", 32])
    (root / "full-proof-check.txt").write_text(result.stdout + result.stderr)

    kept = root / "keep-runs"
    run([args.stream, inputs, kept, 16, 1, 2, 64, 1, "--keep-runs"])
    assert digest(kept / "group.gplot") == expected
    kept_stats = json.loads((kept / "metrics.json").read_text())
    segments = list(kept.glob("run-*.tmp"))
    assert segments and all(0 < path.stat().st_size <= 1024 * 1024 for path in segments)
    assert kept_stats["peak_work_file_bytes"] > metrics[0]["peak_work_file_bytes"]
    assert kept_stats["remaining_work_file_bytes"] == sum(
        path.stat().st_size for path in [kept / "group.gplot", *segments])

    single = root / "single.inputs"
    single.write_text(params + "\n" + quoted(raw[0]) + "\n")
    run([args.stream, single, root / "single", 16, 1, 2, 64, 1])
    assert digest(root / "single/group.gplot") == digest(root / "native.gplot")

    # Reject incompatible identities, malformed file-controlled lengths, and
    # budget violations without publishing a group or changing source bytes.
    wrong_base = root / "wrong-base.inputs"
    wrong_base.write_text(inputs.read_text().replace(params + " 0", params + " 1", 1))
    bad = run([args.stream, wrong_base, root / "wrong-base", 16, 1, 2, 64, 1], False)
    assert "index zero" in bad.stderr
    assert not (root / "wrong-base").exists()
    bad = run([args.stream, inputs, root / "over-budget", 16, 1, 2, 65536, 1], False)
    assert "RAM budget" in bad.stderr
    assert not (root / "over-budget").exists()
    if args.members >= 32:
        bad = run([args.stream, inputs, root / "batch-over-budget", 16, args.members, 2, 64, 1], False)
        assert "RAM budget" in bad.stderr
        assert not (root / "batch-over-budget/group.gplot").exists()
    if len(raw) > 1:
        wrong_order = root / "wrong-order.inputs"
        wrong_order.write_text(params + "\n" + "".join(quoted(path) + "\n" for path in reversed(raw)))
        run([args.stream, wrong_order, root / "wrong-order", 16, 1, 2, 64, 1], False)
        assert not (root / "wrong-order/group.gplot").exists()
    with raw[0].open("r+b") as file:
        file.seek(51 + len(bytes.fromhex(memo)))
        offset = struct.unpack("<Q", file.read(8))[0]
        file.seek(offset + 8)
        original = file.read(4)
        file.seek(offset + 8)
        file.write(b"\xff" * 4)
    try:
        bad = run([args.stream, inputs, root / "corrupt", 16, 1, 2, 64, 1], False)
        assert "raw chunk lengths" in bad.stderr
        assert not (root / "corrupt/group.gplot").exists()
    finally:
        with raw[0].open("r+b") as file:
            file.seek(offset + 8)
            file.write(original)
    occupied = root / "occupied"
    occupied.mkdir()
    sentinel = occupied / "group.gplot"
    sentinel.write_bytes(b"preserve existing file")
    run([args.stream, inputs, occupied, 16, 1, 2, 64, 1], False)
    assert sentinel.read_bytes() == b"preserve existing file"

    # Interrupt a fresh build once staging has started, then resume using the
    # saved checkpoint and unchanged raw inputs.
    work = root / "interrupted"
    with (root / "interrupted.log").open("w") as log:
        process = subprocess.Popen([str(arg) for arg in
            [args.stream, inputs, work, 16, 1, 2, 4, 1]], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while not list(work.glob("run-*.tmp")):
                assert process.poll() is None, "process finished before interruption check"
                assert time.monotonic() < deadline, "staging did not start"
                time.sleep(0.001)
            process.terminate()
            assert process.wait(timeout=30) != 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert not (work / "group.gplot").exists()
    assert all(digest(path) == before[path] for path in raw), "raw source changed"
    run([args.stream, inputs, work, 16, 1, 2, 4, 1, "--resume"])
    assert digest(work / "group.gplot") == expected
    assert all(digest(path) == before[path] for path in raw), "resume changed raw source"
    if os.name == "posix":
        assert (root / "stream-0").stat().st_mode & 0o777 == 0o700
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in
                   [root / "stream-0/group.gplot", root / "stream-0/metrics.json", *segments])
    (root / "comparison.json").write_text(json.dumps({"stream": metrics, "keep_runs": kept_stats}, indent=2) + "\n")
    print(f"PASS k={args.k}, members={args.members}: byte parity, full proofs, merge/reclaim, failures and resume")
    print(json.dumps(metrics[0], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plotter", type=Path)
    parser.add_argument("reference", type=Path, help="existing pos2_pr118_group")
    parser.add_argument("stream", type=Path, help="pos2_pr118_stream prototype")
    parser.add_argument("--k", type=int, choices=(18, 22, 28), default=18)
    parser.add_argument("--members", type=int, default=9)
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--work-dir", type=Path, help="new directory to retain fixtures and measurements")
    args = parser.parse_args()
    if not 1 <= args.members <= 64:
        parser.error("--members must be 1..64 for these reference fixtures")
    for name in ("plotter", "reference", "stream"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"{name} executable does not exist")
        setattr(args, name, path)
    if args.work_dir:
        args.work_dir = args.work_dir.resolve()
        args.work_dir.mkdir(mode=0o700)
        check(args, args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="xchplot2-stream-") as temporary:
            check(args, Path(temporary))


if __name__ == "__main__":
    main()
