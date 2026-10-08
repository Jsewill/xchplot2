#!/usr/bin/env python3
"""Runnable CPU/GPU multi-member, durable resume, and corruption checks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plotter", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--tier", default="plain")
    parser.add_argument("--large-group", type=int, default=64, help="additional cardinality fixture; 0 skips")
    args = parser.parse_args()
    farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
    seed = "aa" * 32
    tool = Path(__file__).with_name("group.py")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

    def run(command, success=True):
        result = subprocess.run([str(value) for value in command], capture_output=True, text=True)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        if not success:
            assert seed not in result.stdout + result.stderr, "private seed exposed by error"
        return result

    with tempfile.TemporaryDirectory(prefix="xchplot2-groups-") as temporary:
        root = Path(temporary)
        for mode, count, strength, meta, base in [("ph", 3, 4, 7, 0), ("pk", 2, 2, 255, 65534)]:
            output = root / f"{mode}.gplot"
            command = [sys.executable, tool, args.plotter.resolve(), args.reference.resolve(),
                       "--out", output, "--farmer-pk", farmer,
                       f"--pool-{mode}", "42" * 32 if mode == "ph" else farmer,
                       "--seed", seed, "--k", 18, "--strength", strength,
                       "--meta-group", meta, "--group-size", count, "--plot-index", base,
                       "--devices", args.devices, "--tier", args.tier]
            result = run(command)
            assert all(f"PASS group member={base + i} " in result.stdout for i in range(count))
            job_path = Path(str(output) + ".job.json")
            raw_dir = Path(str(output) + ".raw")
            job = json.loads(job_path.read_text())
            assert "source-sha256:" in job["request"]["revision"]
            raw = [raw_dir / f"{identity}.plot2" for identity in job["derived"]["members"]]
            before = {path: (digest(path), path.stat().st_mtime_ns) for path in raw}
            group_hash = digest(output)
            if os.name == "posix":
                assert raw_dir.stat().st_mode & 0o777 == 0o700
                assert all(path.stat().st_mode & 0o777 == 0o600
                           for path in [output, job_path, *raw, raw_dir / "batch.tsv", raw_dir / "group.inputs"])
            run(command + ["--resume"])
            assert digest(output) == group_hash
            assert all((digest(path), path.stat().st_mtime_ns) == before[path] for path in raw)
            run(command + ["--testnet"], False)
            run(command + ["--seed", ""], False)
            run(command + ["--resume", "--seed", " ".join(["AA"] * 32)])
            run(command + ["--resume", "--meta-group", 1], False)

            # Recover a partially completed job with the original shared keys.
            output.unlink()
            raw[0].unlink()
            run(command + ["--resume"])
            assert digest(output) == group_hash
            assert all((digest(path), path.stat().st_mtime_ns) == before[path] for path in raw[1:])
            original = output.read_bytes()
            for corrupt in [original[:5] + bytes([original[5] ^ 1]) + original[6:], original[:-1]]:
                output.write_bytes(corrupt)
                run(command + ["--resume"], False)
                assert output.read_bytes() == corrupt
                assert all(digest(path) == before[path][0] for path in raw)
            output.write_bytes(original)
            raw_bytes = raw[0].read_bytes()
            raw[0].write_bytes(raw_bytes[:5] + bytes([raw_bytes[5] ^ 1]) + raw_bytes[6:])
            run(command + ["--resume"], False)
            assert digest(output) == group_hash
            raw[0].write_bytes(raw_bytes)
            edited = json.loads(job_path.read_text())
            edited["derived"]["members"].reverse()
            job_path.write_text(json.dumps(edited))
            run(command + ["--resume"], False)
            edited = json.loads(job_path.read_text())
            edited["request"]["revision"] = "different-reference"
            job_path.write_text(json.dumps(edited))
            run(command + ["--resume"], False)

            # Reject an index larger than the modeled budget before raw reads.
            inputs = raw_dir / "group.inputs"
            limited = root / "over-budget.inputs"
            lines = inputs.read_text().splitlines()
            lines[0] = "28" + lines[0][2:]
            limited.write_text("\n".join(lines) + "\n")
            partial = root / "over-budget.gplot"
            result = run([args.reference.resolve(), "assemble", limited, partial, 16], False)
            assert "RAM budget" in result.stderr
            assert not partial.exists()
        # Files share keys, but every member in both unseeded jobs is unique.
        fresh = []
        members = set()
        for i in range(2):
            output = root / f"random-{i}.gplot"
            outputs = [output, root / f"random-{i}-001.gplot"]
            command = [sys.executable, tool, args.plotter.resolve(), args.reference.resolve(),
                       "--out", output, "--farmer-pk", farmer, "--pool-ph", "42" * 32,
                       "--k", 18, "--group-size", 2, "--files", 2,
                       "--devices", args.devices, "--tier", args.tier]
            result = run(command)
            assert all(f"PASS group member={member} " in result.stdout for member in range(4))
            job_paths = [Path(str(path) + ".job.json") for path in outputs]
            saved = [path.read_bytes() for path in job_paths]
            jobs = [json.loads(data) for data in saved]
            assert jobs[0]["request"]["files"] == 2
            assert [job["request"]["meta_group"] for job in jobs] == [0, 0]
            assert [job["request"].get("plot_index", 0) for job in jobs] == [0, 2]
            assert jobs[0]["seed"] == jobs[1]["seed"]
            assert jobs[0]["derived"]["group_id"] == jobs[1]["derived"]["group_id"]
            assert jobs[0]["derived"]["memo"] == jobs[1]["derived"]["memo"]
            identities = [identity for job in jobs for identity in job["derived"]["members"]]
            assert len(set(identities)) == 4 and members.isdisjoint(identities)
            members.update(identities)
            before = {path: (digest(path), path.stat().st_mtime_ns) for path in outputs}
            run(command + ["--resume"])
            assert [path.read_bytes() for path in job_paths] == saved
            assert all((digest(path), path.stat().st_mtime_ns) == before[path] for path in outputs)
            fresh.append(jobs[0]["derived"])
            run(command + ["--resume", "--files", 1], False)
            for options in [["--files", 0], ["--files", 32769], ["--plot-index", 65533],
                            ["--plot-index", -1], ["--plot-index", 65536], ["--meta-group", 256]]:
                run(command + options, False)

            # An interruption before the next file's job exists keeps the shared seed.
            outputs[1].unlink()
            job_paths[1].unlink()
            shutil.rmtree(Path(str(outputs[1]) + ".raw"))
            run(command + ["--resume"])
            assert [path.read_bytes() for path in job_paths] == saved
            assert all(digest(path) == before[path][0] for path in outputs)
            assert (digest(output), output.stat().st_mtime_ns) == before[output]

            # Never accept another group's saved keys or a different file's output.
            edited = json.loads(saved[1])
            edited["seed"] = "cc" * 32
            job_paths[1].write_text(json.dumps(edited))
            failed = run(command + ["--resume"], False)
            assert jobs[0]["seed"] not in failed.stdout + failed.stderr
            assert all(digest(path) == before[path][0] for path in outputs)
            job_paths[1].write_bytes(saved[1])
            original = outputs[1].read_bytes()
            assert original[4] == 0x82 and int.from_bytes(original[51:53], "little") == 2
            assert output.read_bytes()[4] == 2
            for corrupt in [original[:51] + b"\x03\x00" + original[53:], original[:52]]:
                outputs[1].write_bytes(corrupt)
                run(command + ["--resume"], False)
                assert outputs[1].read_bytes() == corrupt
            outputs[1].write_bytes(output.read_bytes())
            run(command + ["--resume"], False)
            assert outputs[1].read_bytes() == output.read_bytes()
            outputs[1].write_bytes(original)
            job_paths[1].unlink()
            run(command + ["--resume"], False)
            assert digest(outputs[1]) == before[outputs[1]][0]
            job_paths[1].write_bytes(saved[1])
        assert fresh[0]["group_id"] != fresh[1]["group_id"]
        assert fresh[0]["memo"] != fresh[1]["memo"]
        # A later occupied destination is rejected before starting any file.
        occupied = root / "occupied-001.gplot"
        occupied.write_bytes(b"preserve existing output")
        run(command + ["--out", root / "occupied.gplot"], False)
        assert not (root / "occupied.gplot.job.json").exists()
        assert occupied.read_bytes() == b"preserve existing output"
        if args.large_group:
            output = root / "cardinality.gplot"
            command = [sys.executable, tool, args.plotter.resolve(), args.reference.resolve(),
                       "--out", output, "--farmer-pk", farmer, "--pool-ph", "42" * 32,
                       "--seed", "bb" * 32, "--k", 18, "--group-size", args.large_group,
                       "--devices", args.devices, "--tier", args.tier]
            result = run(command)
            assert all(f"PASS group member={i} " in result.stdout for i in range(args.large_group))
            before = digest(output)
            run(command + ["--resume"])
            assert digest(output) == before
            if args.large_group >= 64:
                result = run(command + ["--resume", "--max-group-ram", 16], False)
                assert "RAM budget" in result.stderr
                assert digest(output) == before
        # FFI failures also keep the private preparation seed out of diagnostics.
        failed = run([sys.executable, tool, args.plotter.resolve(), args.reference.resolve(),
                      "--out", root / "bad.gplot", "--farmer-pk", "00" * 48,
                      "--pool-ph", "42" * 32, "--seed", seed, "--k", 18, "--group-size", 2], False)
        assert "stage failed" in failed.stderr
        assert not (root / "bad.gplot.job.json").exists()
        for option in ["--pool-pk", "--pool-ph"]:
            run([sys.executable, tool, args.plotter.resolve(), args.reference.resolve(),
                 "--out", root / "empty.gplot", "--farmer-pk", farmer, option, "",
                 "--k", 18, "--group-size", 2], False)
            assert not (root / "empty.gplot.job.json").exists()
    print("Multi-member/file groups, unique identities, private jobs, resume, corruption and RAM rejection passed.")


if __name__ == "__main__":
    main()
