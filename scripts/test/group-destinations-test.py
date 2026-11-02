#!/usr/bin/env python3
"""Bounded multi-destination generation, physical-drive ordering and recovery."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile


# Disposable public test identities, matching group-rollover-test.py.
FARMER = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
SEED = "aa" * 32
POOL = "42" * 32


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(paths):
    return [(digest(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]


def run(command, code=0, point=None, **kwargs):
    env = dict(os.environ)
    env.pop("XCHPLOT2_GROW_FAILPOINT", None)
    if point:
        env["XCHPLOT2_GROW_FAILPOINT"] = point
    result = subprocess.run([str(value) for value in command], capture_output=True,
                            text=True, env=env, **kwargs)
    assert result.returncode == code, (command, result.returncode, result.stdout, result.stderr)
    assert SEED not in result.stdout + result.stderr, "seed was exposed in diagnostics"
    return result


def report(result):
    return json.loads(result.stdout.splitlines()[-1])


def check(args, root):
    common = [args.grow, "--seed", SEED, "--farmer-pk", FARMER, "--pool-ph", POOL,
              "--k", 18, "--max-members", 3, "--batch-members", 1,
              "--capacity-bytes", 64 * 1024 * 1024, "--max-ram", args.max_ram,
              "--window-bytes", 65536, "--checkpoint-bytes", 1048576, "--workers", 2,
              "--backend", args.backend]
    results = {"parity": [], "recovery": [], "rejections": []}

    def targets(name):
        directory = root / name
        directory.mkdir(mode=0o700)
        paths = [directory / f"drive {i} é.gplot" for i in range(3)]
        options = [paths[0]]
        for path in paths[1:]:
            options += ["--destination", path]
        return paths, options

    with tempfile.TemporaryDirectory(prefix="serial-", dir=root) as work:
        baseline, options = targets(Path(work).name + "-outputs")
        result = report(run([*common, *options, "--construction", "incremental", "--generation-overlap", "off"]))
        expected = [digest(path) for path in baseline]
        assert result["drive_overlap_batches"] == 0, result
        assert len(set(expected)) == len(expected), "destination identities were reused"
        results["parity"].append({"case": "serial", **result})
        for path in baseline:
            path.unlink()
        baseline[0].parent.rmdir()

    # A later ordinary destination can retain authenticated external metadata.
    # Audit it before growing any earlier output, even without --temp-dir.
    paths, options = targets("retired-no-temp")
    scratch = root / "retired-no-temp-temp"
    scratch.mkdir(mode=0o700)
    ordinary = [*common, *options, "--construction", "incremental", "--backend", "cpu",
                "--generation-overlap", "off"]
    run([*ordinary, "--max-members", 1])
    paths[-1].unlink()
    run([*ordinary, "--resume", "--max-members", 1, "--construction", "external", "--temp-dir", scratch],
        code=86, point="external-handoff-retired")
    holder, = paths[-1].parent.glob(paths[-1].name + ".external.retired-*")
    checkpoint = holder / "checkpoint"
    saved_checkpoint = checkpoint.read_bytes()
    before = snapshot(paths)
    checkpoint.write_bytes(b"!" + saved_checkpoint[1:])
    run([*ordinary, "--resume"], code=1)
    assert snapshot(paths) == before, "later retired corruption was found after modifying an earlier output"
    assert checkpoint.read_bytes() == b"!" + saved_checkpoint[1:]
    checkpoint.write_bytes(saved_checkpoint)
    forged = Path(str(paths[-1]) + ".external.retired-" + "f" * 64)
    forged.mkdir(mode=0o700)
    run([*ordinary, "--resume"], code=1)
    assert snapshot(paths) == before and not list(forged.iterdir()), "forged retired metadata changed an output or was removed"
    forged.rmdir()
    result = report(run([*ordinary, "--resume"]))
    assert result["members"] == len(paths) * 3 and result["peak_modeled_bytes"] <= args.max_ram, result
    assert result["peak_file_bytes"] <= result["capacity_bytes"], result
    assert [digest(path) for path in paths] == expected, "no-temp continuation differs from serial reference"
    assert set(paths[0].parent.iterdir()) == {*paths, holder}
    assert all(path.is_dir() and not list(path.iterdir()) for path in scratch.iterdir()), "retired scratch still has unread contributions"
    before = snapshot(paths)
    idle = report(run([*ordinary, "--resume", "--backend", "gpu"]))
    assert idle["mode"] == "idle" and idle["generated_members"] == 0 and snapshot(paths) == before, idle
    results["parity"].append({"case": "no-temp continuation with retained external ownership", **result})
    results["rejections"].append("later corrupt/forged retired ownership preserves earlier outputs without temp options")
    print("PASS no-temp retired ownership audit and ordinary destination continuation", flush=True)

    # Recover the first destination from its explicitly selected scratch,
    # while new destinations must not inherit those saved temporary roots.
    paths, options = targets("saved-temp-new-destinations")
    scratch = root / "saved-temp-new-destinations-temp"
    scratch.mkdir(mode=0o700)
    run([*common, paths[0], "--construction", "external", "--temp-dir", scratch],
        code=86, point="external-run-after-checkpoint")
    before = snapshot([path for path in paths[0].parent.rglob("*") if path.is_file()])
    rejected = run([*common, *options, "--resume", "--construction", "external"], code=1)
    assert "fresh external construction requires --temp-dir" in rejected.stderr, rejected.stderr
    assert snapshot([path for path in paths[0].parent.rglob("*") if path.is_file()]) == before
    assert not any(path.exists() for path in paths[1:]), "missing temp option created a destination"
    result = report(run([*common, *options, "--resume"], point="external-initialized-after-checkpoint"))
    assert result["members"] == len(paths) * 3 and result["recovered"], result
    assert result["peak_modeled_bytes"] <= args.max_ram and result["peak_file_bytes"] <= result["capacity_bytes"], result
    assert [digest(path) for path in paths] == expected, "saved-only temporary roots changed ordinary output"
    assert set(paths[0].parent.iterdir()) == set(paths) and not list(scratch.iterdir())
    results["recovery"].append({"point": "saved temp paths do not enable new external destinations", **result})
    results["rejections"].append("fresh external destinations require explicit temp paths during resume")
    print("PASS saved temporary roots remain limited to their existing jobs", flush=True)

    for policy in ("explicit", "on"):
        paths, options = targets("destination-temp-" + policy)
        temporary = ["--temp-dir", paths[0].parent] if policy == "explicit" else ["--destination-temp", policy]
        result = report(run([*common, *options, *temporary, "--temp-capacity-bytes", 64 * 1024 * 1024]))
        assert result["construction_selected"] == "external" and result["scratch_peak_bytes"] > 0, result
        assert result["peak_modeled_bytes"] <= args.max_ram and result["peak_file_bytes"] <= result["capacity_bytes"], result
        assert [digest(path) for path in paths] == expected, "enabled destination scratch changed output bytes"
        assert set(paths[0].parent.iterdir()) == set(paths), "enabled destination scratch was not cleaned up"
        results["parity"].append({"case": policy + " destination temporary storage", **result})

    paths, options = targets("destination-temp-capacity")
    command = [*common, *options, "--destination-temp", "on", "--temp-capacity-bytes", 1]
    run([*command, "--construction", "external"], code=1)
    assert not list(paths[0].parent.iterdir()), "rejected destination scratch created outputs"
    result = report(run(command, point="external-initialized-after-checkpoint"))
    assert result["scratch_peak_bytes"] == 0 and [digest(path) for path in paths] == expected, result
    results["parity"].append({"case": "destination temporary capacity ordinary fallback", **result})

    paths, _ = targets("destination-temp-single")
    result = report(run([*common, paths[0], "--destination-temp", "on"]))
    assert result["construction_selected"] == "external" and result["scratch_peak_bytes"] > 0, result
    assert digest(paths[0]) == expected[0] and set(paths[0].parent.iterdir()) == {paths[0]}
    results["parity"].append({"case": "single destination temporary policy", **result})

    directory = root / "destination-temp-rollover"
    directory.mkdir(mode=0o700)
    result = report(run([*common, directory, "--max-members", 65535, "--group-members", 3,
                         "--max-total-members", 6, "--destination-temp", "on"]))
    files = sorted(directory.glob("grow-group-*.gplot"))
    assert result["members"] == 6 and result["scratch_peak_bytes"] > 0, result
    assert result["peak_modeled_bytes"] <= args.max_ram and result["peak_file_bytes"] <= result["capacity_bytes"], result
    assert [digest(path) for path in files] == expected[:2] and set(directory.iterdir()) == set(files)
    results["parity"].append({"case": "rollover destination temporary policy", **result})

    paths, options = targets("destination-temp-resume")
    scratch = root / "destination-temp-resume-original"
    scratch.mkdir(mode=0o700)
    run([*common, paths[0], "--construction", "external", "--temp-dir", scratch],
        code=86, point="external-run-after-checkpoint")
    run([*common, *options, "--resume", "--destination-temp", "on"],
        code=86, point="external-initialized-after-checkpoint")
    assert not list(scratch.iterdir()), "new job reused scratch outside the enabled destinations"
    assert any(paths[0].parent.glob(".xchplot2-external-*")), "new job did not use enabled destination scratch"
    result = report(run([*common, *options, "--resume"], point="external-initialized-after-checkpoint"))
    assert result["members"] == len(paths) * 3 and result["recovered"], result
    assert [digest(path) for path in paths] == expected and set(paths[0].parent.iterdir()) == set(paths)
    results["recovery"].append({"point": "destination policy changes preserve saved roots without enabling others", **result})
    print("PASS destination temporary opt-in, capacity, single-file, rollover and recovery policy", flush=True)

    for mode in ("incremental", "merged", "auto"):
        paths, options = targets(mode)
        command = [*common, *options, "--construction", mode, "--generation-overlap", "on"]
        result = report(run(command, point="external-initialized-after-checkpoint"))
        assert result["destinations"] == 3 and result["writer_groups"] == 1, result
        assert result["members"] == 9 and result["queued_peak_bytes"] > 0, result
        assert result["peak_modeled_bytes"] <= args.max_ram, result
        if args.backend == "cpu":
            assert result["generation_overlap_batches"] == 0, result
        assert result["peak_file_bytes"] <= result["capacity_bytes"], result
        assert result["written_bytes"] >= result["final_bytes"] and result["peak_rss_bytes"] > 0, result
        assert result["scratch_peak_bytes"] == 0 and result["construction_selected"] != "external", result
        assert [digest(path) for path in paths] == expected, "destination output differs from serial reference"
        assert set(paths[0].parent.iterdir()) == set(paths), "destination created a sidecar"
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)
        before = snapshot(paths)
        idle = report(run([*command, "--resume", "--backend", "gpu"]))
        assert idle["generated_members"] == 0 and idle["mode"] == "idle" and snapshot(paths) == before, idle
        run([*command, "--resume", "--seed", "bb" * 32], code=1)
        assert snapshot(paths) == before, "identity rejection changed a destination"
        if mode == "merged":
            bounded = report(run([*command, "--resume", "--max-ram", 32 * 1024 * 1024]))
            assert bounded["mode"] == "idle" and bounded["written_bytes"] == 0 and snapshot(paths) == before
            results["parity"].append({"case": "completed destinations audited without active workspaces", **bounded})
        saved_last = paths[-1].read_bytes()
        paths[-1].write_bytes(b"corrupt saved destination")
        unchanged = snapshot(paths[:-1])
        run([*command, "--resume", "--max-members", 4], code=1)
        assert snapshot(paths[:-1]) == unchanged, "late destination corruption was found after modifying earlier outputs"
        paths[-1].write_bytes(saved_last)
        results["rejections"].append(f"{mode} all-destination audit before modification")
        results["parity"].append({"case": mode, **result})
        results["rejections"].append(f"{mode} wrong seed and completed unavailable-backend resume")
        for path in paths:
            path.unlink()

    for point in ("merge-bootstrap", "merge-initialized-committed", "merge-run-committed:2", "merge-copy-committed:2", "merge-pack-committed:2",
                  "merge-layout-partial", "merge-finalizing", "merge-retired"):
        paths, options = targets("restart-" + point.replace(":", "-"))
        command = [*common, *options, "--construction", "merged", "--generation-overlap", "on"]
        run(command, code=86, point=point)
        # Execution grants are runtime choices; a saved contribution must also
        # resume with a different CPU worker count without changing its bytes.
        resume_workers = ["--workers", 1] if point == "merge-run-committed:2" else []
        result = report(run([*command, "--resume", *resume_workers]))
        assert [digest(path) for path in paths] == expected, (point, result)
        assert result["peak_modeled_bytes"] <= args.max_ram, result
        results["recovery"].append({"point": point, **result})
        for path in paths:
            path.unlink()

    paths, options = targets("saved-batch")
    command = [*common, *options, "--construction", "incremental", "--max-members", 5,
               "--batch-members", 3, "--generation-overlap", "on"]
    run(command, code=86, point="initialized")
    result = report(run([*command, "--resume", "--batch-members", 1]))
    assert result["members"] == 15 and result["max_batch_members"] == 1, result
    assert result["peak_modeled_bytes"] <= args.max_ram, result
    results["recovery"].append({"point": "saved batch with smaller subsequent bound", **result})
    for path in paths:
        path.unlink()

    # Exercise automatic handoffs while other destinations hold queued data.
    # A whole-batch CPU reference also checks all GPU-generated member bytes.
    references, options = targets("automatic-reference")
    report(run([*common, *options, "--max-members", 16, "--batch-members", 16,
                "--backend", "cpu", "--construction", "incremental", "--generation-overlap", "off"]))
    expected_many = [digest(path) for path in references]
    near_capacity = sum(path.stat().st_size + 2 * 1024 * 1024 for path in references)
    for path in references:
        path.unlink()
    for name, options_extra in (("automatic", []),
                                ("automatic-io", []),
                                ("automatic-batch", ["--batch-members", 65535]),
                                ("automatic-raw", ["--member-storage", "raw", "--generation-overlap", "off"]),
                                ("automatic-capacity", ["--capacity-bytes", near_capacity])):
        paths, options = targets(name)
        command = [*common, *options, "--max-members", 16, "--generation-overlap", "on", *options_extra]
        if name == "automatic-io":
            for option in ("--window-bytes", "--checkpoint-bytes"):
                index = command.index(option)
                del command[index:index + 2]
        result = report(run(command))
        assert result["construction_requested"] == "auto", result
        if name == "automatic-batch":
            assert result["fresh_single_pass_batches"] == len(paths) and result["max_batch_members"] == 16, result
        else:
            assert result["automatic_merges"] > 0, result
        assert result["members"] == 48 and result["peak_modeled_bytes"] <= args.max_ram, result
        assert result["peak_file_bytes"] <= result["capacity_bytes"], result
        assert [digest(path) for path in paths] == expected_many
        if name == "automatic-capacity":
            assert 0 < result["automatic_fallbacks"] <= result["automatic_merges"] <= 3, result
        if name == "automatic-io":
            assert result["tuning_changes"] > 0 and result["selected_window_bytes"] <= 16777216, result
            assert result["selected_checkpoint_bytes"] <= 268435456, result
        else:
            assert result["selected_window_bytes"] == 65536 and result["selected_checkpoint_bytes"] == 1048576, result
        before = snapshot(paths)
        idle_ram = args.max_ram if name == "automatic-io" else 32 * 1024 * 1024
        idle = report(run([*command, "--resume", "--backend", "gpu", "--max-ram", idle_ram]))
        assert idle["mode"] == "idle" and idle["written_bytes"] == 0 and snapshot(paths) == before, idle
        results["parity"].append({"case": name, **result})
        for path in paths:
            path.unlink()
    for point in ("merge-copy-committed", "merge-pack-committed", "merge-retired"):
        paths, options = targets("automatic-restart-" + point)
        command = [*common, *options, "--max-members", 16, "--generation-overlap", "on"]
        run(command, code=86, point=point)
        result = report(run([*command, "--resume"]))
        assert [digest(path) for path in paths] == expected_many
        assert result["peak_modeled_bytes"] <= args.max_ram, result
        results["recovery"].append({"point": "automatic-" + point, **result})
        for path in paths:
            path.unlink()

    # Many names on one device must retain cheap identity/path guards while
    # admitting only the live writer's workspace. This also exercises every
    # engine release and completed-file audit without requiring many drives.
    directory = root / "many-destinations"
    directory.mkdir(mode=0o700)
    paths = [directory / f"drive-{index}.gplot" for index in range(64)]
    options = [paths[0]]
    for path in paths[1:]:
        options += ["--destination", path]
    bounded_ram = min(args.max_ram, 256 * 1024 * 1024)
    command = [*common, *options, "--construction", "incremental", "--max-members", 1,
               "--capacity-bytes", 64 * 64 * 1024 * 1024, "--max-ram", bounded_ram,
               "--generation-overlap", "on"]
    result = report(run(command))
    assert result["destinations"] == len(paths) and result["members"] == len(paths), result
    assert result["writer_groups"] == 1 and result["peak_modeled_bytes"] <= bounded_ram, result
    before = snapshot(paths)
    idle = report(run([*command, "--resume", "--backend", "gpu"]))
    assert idle["mode"] == "idle" and idle["written_bytes"] == 0 and snapshot(paths) == before, idle
    results["parity"].append({"case": "many lazy destinations", **result})
    for path in paths:
        path.unlink()

    paths, options = targets("budget")
    run([*common, *options, "--construction", "merged", "--max-ram", 16 * 1024 * 1024], code=1)
    assert not list(paths[0].parent.iterdir()), "budget rejection left empty destinations"
    results["rejections"].append("combined engine budget before generation")
    run([*common, *options, "--destination", paths[0]], code=1)
    assert not list(paths[0].parent.iterdir()), "duplicate target rejection created files"
    results["rejections"].append("duplicate destination")
    run([*common, *options], code=1,
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024)))
    assert not list(paths[0].parent.iterdir()), "file-size rejection happened after creation"
    results["rejections"].append("early process file-size limit")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grow", type=Path)
    parser.add_argument("--scratch", type=Path, required=True, help="existing directory on the test device")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--max-ram", type=int, default=1024 * 1024 * 1024)
    args = parser.parse_args()
    args.grow = args.grow.resolve()
    with tempfile.TemporaryDirectory(prefix="grow-destinations-", dir=args.scratch) as work:
        results = check(args, Path(work))
    args.results.write_text(json.dumps(results, indent=2) + "\n")
    print("multi-destination parity, bounded queues, recovery and rejection checks passed")


if __name__ == "__main__":
    main()
