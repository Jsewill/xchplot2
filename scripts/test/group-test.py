#!/usr/bin/env python3
"""Same-file growth parity and recovery using read-only reference fixtures."""

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


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def run(command, code=0, point=None, **kwargs):
    env = dict(os.environ)
    env.pop("XCHPLOT2_GROW_FAILPOINT", None)
    if point:
        env["XCHPLOT2_GROW_FAILPOINT"] = point
    result = subprocess.run([str(arg) for arg in command], env=env,
                            capture_output=True, text=True, **kwargs)
    assert result.returncode == code, (command, result.returncode, result.stdout, result.stderr)
    return result


def report(result):
    return json.loads(result.stdout.splitlines()[-1])


def check(args, root):
    fixture = args.fixture_dir
    inputs = fixture / "group.inputs"
    reference = fixture / "reference.gplot"
    native = fixture / "native.gplot"
    lines = inputs.read_text().splitlines()
    parameters = shlex.split(lines[0])
    k, strength, meta = map(int, parameters[:3])
    group, memo = parameters[3:5]
    assert len(parameters) == 5 or parameters[5:] == ["0"], "use an ordinary index-zero fixture"
    originals = [Path(shlex.split(line)[0]).resolve() for line in lines[1:] if line.strip()]
    assert len(originals) >= 2, "fixture needs multiple members"
    if args.modes:
        assert k == 18, "use the small k18 fixture for construction mode checks"
    source_hashes = {path: digest(path) for path in [*originals, reference, native]}
    expected = source_hashes[reference]
    capacity = reference.stat().st_size * 2 + 1048576
    common_options = ["--fixture-inputs", inputs, "--max-members", len(originals),
              "--capacity-bytes", capacity, "--max-ram", args.max_ram,
              "--window-bytes", 262144, "--workers", 2]
    common = [args.grow, "--construction", "incremental", *common_options]
    results = {"k": k, "members": len(originals), "parity": [], "recovery": [], "rejections": []}

    def target(name, source=None):
        directory = root / name
        directory.mkdir(mode=0o700)
        path = directory / "group.gplot"
        if source:
            # Deliberately copy bytes: reflinking a fixture would test the
            # shared-extent refusal instead of the in-place protocol.
            with source.open("rb") as old, path.open("xb") as new:
                shutil.copyfileobj(old, new, 1048576)
                new.flush()
                os.fsync(new.fileno())
            path.chmod(0o600)
        return path

    def complete(path, command, resume=False, loaded_members=None, retired=()):
        result = run(command + (["--resume"] if resume else []))
        metrics = report(result)
        assert metrics["members"] == len(originals), metrics
        assert metrics["final_bytes"] == reference.stat().st_size == path.stat().st_size, metrics
        assert metrics["peak_file_bytes"] <= metrics["capacity_bytes"], metrics
        assert metrics["peak_modeled_bytes"] <= args.max_ram, metrics
        assert digest(path) == expected, "growth differs from canonical reference"
        assert set(path.parent.iterdir()) == {path, *retired}, "construction created unexpected sidecar or temp files"
        assert path.stat().st_mode & 0o777 == 0o600, "private plot permissions lost"
        if loaded_members is not None:
            loaded = [int(line.rsplit(" ", 1)[1]) for line in result.stderr.splitlines()
                      if line.startswith(("loading fixture member ", "prefetching fixture member "))]
            assert loaded == list(loaded_members), (loaded, loaded_members, result.stderr)
            assert metrics["generated_members"] == len(loaded), metrics
        before = path.stat().st_mtime_ns, digest(path)
        again = report(run(command + ["--resume"]))
        assert again["members"] == metrics["members"]
        assert (path.stat().st_mtime_ns, digest(path)) == before, "completed resume rewrote the file"
        return metrics

    # Unavailable committed members make accidental regeneration fail instead
    # of being hidden by a matching final digest.
    def resume_inputs(name, committed):
        manifest = root / (name + ".inputs")
        paths = [root / (name + f"-unavailable-{i}.plot2") if i < committed else path
                 for i, path in enumerate(originals)]
        manifest.write_text(lines[0] + "\n" +
                            "".join(json.dumps(str(path)) + "\n" for path in paths))
        return manifest

    for name, batch, workers in [("whole-batch", len(originals), 2),
                                  ("small-batches", 3, 3), ("single-members", 1, 1),
                                  ("serial-generation", 3, 3), ("default-generation", 3, 3),
                                  ("automatic-generation", 3, 3)]:
        path = target(name)
        command = [*common, path, "--batch-members", batch, "--workers", workers]
        if name == "serial-generation":
            command += ["--generation-overlap", "off"]
        elif name in ("small-batches", "single-members"):
            command += ["--generation-overlap", "on"]
        elif name == "automatic-generation":
            command += ["--generation-overlap", "auto"]
        metrics = complete(path, command)
        if name == "serial-generation":
            assert metrics["generation_overlap_batches"] == 0, metrics
        elif name in ("small-batches", "single-members") and k == 18 and batch < len(originals):
            assert metrics["generation_overlap_batches"] > 0, metrics
        assert metrics["sized_chunks"] == metrics["exact_plan_batches"] * (1 << (k - 6)), metrics
        results["parity"].append({"case": name, **metrics})
        print("PASS canonical same-file construction:", name, flush=True)

    if k == 18 and len(originals) <= 64:
        fresh = results["parity"][0]
        assert fresh["fresh_single_pass_batches"] == fresh["batches"] == 1, fresh
        assert fresh["encode_chunks"] == 1 << (k - 6), fresh
        # Bound codec admission so this small fixture spans multiple windows.
        # Compare checkpoint grouping against the same serial configuration;
        # a larger codec budget may admit the entire fixture in one window.
        checkpoint_ram = min(args.max_ram, 268435456)
        checkpoint_common = [*common, "--batch-members", len(originals),
                             "--io-overlap", "off", "--max-ram", checkpoint_ram]
        path = target("unoverlapped-control")
        control = complete(path, [*checkpoint_common, path])
        assert control["overlap_windows"] == 0, control
        assert control["encode_chunks"] == fresh["encode_chunks"], control
        assert control["peak_modeled_bytes"] <= checkpoint_ram, control
        results["parity"].append({"case": "unoverlapped-control", **control})
        path = target("unoverlapped-checkpoints")
        command = [*checkpoint_common, path, "--checkpoint-bytes", 1]
        metrics = complete(path, command)
        assert metrics["overlap_windows"] == 0, metrics
        assert metrics["encode_chunks"] == fresh["encode_chunks"], metrics
        assert metrics["peak_modeled_bytes"] <= checkpoint_ram, metrics
        assert metrics["checkpoints"] > max(2, control["checkpoints"]), (metrics, control)
        assert metrics["syncs"] > control["syncs"], (metrics, control)
        results["parity"].append({"case": "unoverlapped-checkpoints", **metrics})
        print("PASS single-pass creation and coalesced checkpoints", flush=True)

    path = target("existing-group", native)
    inode = path.stat().st_ino
    command = [*common, path, "--batch-members", len(originals)]
    metrics = complete(path, command, True)
    assert path.stat().st_ino == inode, "growth replaced the original inode"
    results["parity"].append({"case": "existing-group", **metrics})
    proof = run([args.plotter, "verify", path, "--full", "--trials", 100])
    assert "full proofs validated" in (proof.stdout + proof.stderr).lower(), proof.stdout + proof.stderr
    (root / "full-proofs.log").write_text(proof.stdout + proof.stderr)

    if args.cpu:
        path = target("cpu-from-empty")
        cpu_options = ["--group-id", group, "--memo", memo,
                       "--k", k, "--strength", strength, "--meta-group", meta,
                       "--max-members", len(originals), "--batch-members", 3,
                       "--max-ram", args.max_ram, "--capacity-bytes", capacity,
                       "--workers", 2, "--window-bytes", 262144]
        command = [args.grow, path, *cpu_options]
        metrics = complete(path, command)
        assert metrics["mode"] == "cpu" and metrics["cpu_generation_peak_bytes"] > 0
        results["parity"].append({"case": "cpu-from-empty", **metrics})
        print("PASS CPU-generated members without input files", flush=True)
        if args.modes:
            for storage in ["raw", "compact"]:
                name = "cpu-from-empty-" + storage
                path = target(name)
                command = [args.grow, path, *cpu_options,
                           "--member-storage", storage, "--backend", "cpu"]
                metrics = complete(path, command)
                assert metrics["mode"] == "cpu" and metrics["member_storage"] == storage, metrics
                assert metrics["generated_members"] == len(originals), metrics
                results["parity"].append({"case": name, **metrics})
                print("PASS CPU-generated member storage:", storage, flush=True)

    if args.modes:
        mode_common = [*common, "--generation-overlap", "off"]
        for storage in ["raw", "compact"]:
            name = "member-storage-" + storage
            path = target(name)
            command = [*mode_common, path, "--batch-members", 3,
                       "--member-storage", storage, "--cache-release", "on"]
            metrics = complete(path, command, loaded_members=range(len(originals)))
            assert metrics["member_storage"] == storage and metrics["cache_release"] is True, metrics
            assert metrics["construction_requested"] == "incremental", metrics
            results["parity"].append({"case": name, **metrics})
            print("PASS retained member storage and cache release:", storage, flush=True)

        stage_batch = min(3, len(originals) - 1)
        staged_common = [*mode_common, "--construction", "staged", "--member-storage", "compact",
                         "--batch-members", stage_batch, "--cache-release", "on",
                         "--checkpoint-bytes", 65536,
                         "--capacity-bytes", reference.stat().st_size * 8 + 1048576]
        for fresh in [True, False]:
            name = "staged-empty" if fresh else "staged-existing"
            path = target(name, None if fresh else native)
            inode = path.stat().st_ino if path.exists() else None
            command = [*staged_common, path]
            metrics = complete(path, command, not fresh,
                               loaded_members=range(0 if fresh else 1, len(originals)))
            assert metrics["construction_requested"] == "staged" and metrics["staged_batches"] > 0, metrics
            assert metrics["staging_fallbacks"] == 0 and metrics["cache_release"] is True, metrics
            if inode is not None:
                assert path.stat().st_ino == inode, "staged growth replaced the original inode"
            if fresh:
                staged_peak = metrics["peak_file_bytes"]
            results["parity"].append({"case": name, **metrics})
            print("PASS canonical same-file staged construction:", name, flush=True)

        # Admit earlier staged runs but reject the last staging reservation.
        # The fallback must retry the resident candidate batch, without loading
        # it again or dropping members that ordinary exact admission can fit.
        path = target("staged-capacity-fallback")
        command = [*staged_common, path, "--capacity-bytes", staged_peak - 1]
        metrics = complete(path, command, loaded_members=range(len(originals)))
        assert metrics["staging_fallbacks"] == 1 and metrics["staged_batches"] > 0, metrics
        assert metrics["exact_plan_batches"] > 0, metrics
        results["parity"].append({"case": "staged-capacity-fallback", **metrics})
        results["rejections"].append("staging capacity fallback reuses the unchanged batch")
        print("PASS staging capacity fallback without member regeneration", flush=True)

        # Temporary roots stay on the test filesystem; RAM-backed directories
        # cannot exercise external scratch admission or durable recovery.
        external_common = [*mode_common, "--construction", "external", "--batch-members", stage_batch,
                           "--checkpoint-bytes", 65536, "--capacity-bytes", 67108864,
                           "--temp-capacity-bytes", 67108864]
        for mode, storage in (("external", "compact"), ("auto", "raw")):
            name = "temporary-" + mode
            path = target(name)
            scratch = root / (name + "-temp")
            scratch.mkdir(mode=0o700)
            sentinel = scratch / "unrelated.txt"
            sentinel.write_text("preserve unrelated temporary files\n")
            command = [*external_common, path, "--temp-dir", scratch,
                       "--construction", mode, "--member-storage", storage]
            metrics = complete(path, command, loaded_members=range(len(originals)))
            assert metrics["construction_requested"] == mode and metrics["construction_selected"] == "external", metrics
            assert metrics["staged_batches"] > 0 and metrics["scratch_peak_bytes"] > 0, metrics
            assert metrics["staging_fallbacks"] == 0 and metrics["layout_bytes"] == 0, metrics
            assert list(scratch.iterdir()) == [sentinel] and sentinel.read_text() == "preserve unrelated temporary files\n"
            results["parity"].append({"case": name, **metrics})
            print("PASS temporary contribution construction and cleanup:", mode, flush=True)

        path = target("temporary-capacity-fallback")
        scratch = root / "temporary-capacity-fallback-temp"
        scratch.mkdir(mode=0o700)
        command = [*external_common, path, "--temp-dir", scratch, "--temp-capacity-bytes", 1]
        run(command, 1)
        assert not list(path.parent.iterdir()) and not list(scratch.iterdir()), "rejected external admission created construction state"
        command += ["--construction", "auto"]
        metrics = complete(path, command, loaded_members=range(len(originals)))
        assert metrics["construction_requested"] == "auto" and metrics["construction_selected"] != "external", metrics
        assert metrics["scratch_peak_bytes"] == 0, metrics
        assert not list(scratch.iterdir()), "scratch fallback left a private namespace"
        results["parity"].append({"case": "temporary-capacity-fallback", **metrics})
        results["rejections"].append("external capacity rejection and automatic ordinary fallback")

        path = target("temporary-completion-extension")
        scratch = root / "temporary-completion-extension-temp"
        scratch.mkdir(mode=0o700)
        command = [*external_common, path, "--temp-dir", scratch]
        partial = report(run([*command, "--max-members", stage_batch]))
        assert partial["members"] == stage_batch and partial["construction_selected"] == "external", partial
        assert list(path.parent.iterdir()) == [path] and not list(scratch.iterdir()), "completed external state was retained"
        inode = path.stat().st_ino
        metrics = complete(path, command, resume=True, loaded_members=range(stage_batch, len(originals)))
        assert path.stat().st_ino == inode and metrics["construction_selected"] == "incremental", metrics
        assert not list(scratch.iterdir()), "ordinary extension recreated external scratch"
        results["parity"].append({"case": "temporary-completion-extension", **metrics})

        for point, committed, limit in (("external-run-after-checkpoint", stage_batch, len(originals)),
                                        ("external-pack-after-checkpoint", len(originals), len(originals)),
                                        ("external-published-after-checkpoint", stage_batch, stage_batch)):
            name = "temporary-recovery-" + point
            path = target(name)
            scratch = root / (name + "-temp")
            scratch.mkdir(mode=0o700)
            run([*external_common, path, "--temp-dir", scratch, "--max-members", limit], 86, point)
            manifest = resume_inputs(name, committed)
            # Saved roots and construction mode must be discovered without
            # repeating the temporary options or regenerating committed data.
            command = [*mode_common, path, "--fixture-inputs", manifest,
                       "--member-storage", "raw", "--batch-members", stage_batch,
                       "--capacity-bytes", 67108864, "--checkpoint-bytes", 65536]
            metrics = complete(path, command, resume=True, loaded_members=range(committed, len(originals)))
            assert metrics["recovered"] is True and not list(scratch.iterdir()), metrics
            results["recovery"].append({"case": name, **metrics})
            print("PASS saved temporary contribution recovery:", point, flush=True)

        # An ordinary resume must audit and charge retained external state even
        # when no temporary options are repeated. Keep a checkpoint so its
        # occupied bytes are positive on the test filesystem.
        path = target("retired-no-temp")
        scratch = root / "retired-no-temp-temp"
        scratch.mkdir(mode=0o700)
        run([*external_common, path, "--temp-dir", scratch, "--max-members", stage_batch],
            86, "external-handoff-retired")
        holder, = path.parent.glob(path.name + ".external.retired-*")
        checkpoint = holder / "checkpoint"
        assert checkpoint.is_file() and checkpoint.stat().st_size > 0
        saved_checkpoint = checkpoint.read_bytes()
        before = digest(path), path.stat().st_size, path.stat().st_mtime_ns
        manifest = resume_inputs("retired-no-temp", stage_batch)
        command = [*mode_common, path, "--fixture-inputs", manifest, "--batch-members", stage_batch,
                   "--capacity-bytes", 67108864, "--checkpoint-bytes", 65536, "--resume"]
        checkpoint.write_bytes(b"!" + saved_checkpoint[1:])
        run(command, 1)
        assert (digest(path), path.stat().st_size, path.stat().st_mtime_ns) == before
        assert checkpoint.read_bytes() == b"!" + saved_checkpoint[1:], "rejection changed retired metadata"
        checkpoint.write_bytes(saved_checkpoint)
        forged = Path(str(path) + ".external.retired-" + "f" * 64)
        forged.mkdir(mode=0o700)
        run(command, 1)
        assert (digest(path), path.stat().st_size, path.stat().st_mtime_ns) == before and not list(forged.iterdir())
        forged.rmdir()
        # Establish that the cap admits the ordinary file alone, then restore
        # its retired holder: the identical no-op cap must include both.
        charge = max(path.stat().st_size, path.stat().st_blocks * 512)
        no_op = [*command, "--max-members", stage_batch, "--capacity-bytes", charge]
        saved_holder = scratch / "saved-retired-holder"
        holder.rename(saved_holder)
        idle = report(run(no_op))
        assert idle["generated_members"] == 0 and (digest(path), path.stat().st_size, path.stat().st_mtime_ns) == before
        saved_holder.rename(holder)
        run(no_op, 1)
        assert (digest(path), path.stat().st_size, path.stat().st_mtime_ns) == before
        metrics = complete(path, command, loaded_members=range(stage_batch, len(originals)), retired=[holder])
        results["parity"].append({"case": "retired-no-temp", **metrics})
        results["rejections"].append("no-temp retired ownership and fixed capacity preserve ordinary output")
        print("PASS no-temp retired metadata audit, capacity and ordinary continuation", flush=True)

        # Fixture manifests cannot be combined with --destination. Reuse the
        # disposable public identities from the destinations/rollover harness.
        farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
        for replacement in (False, True):
            name = "temporary-root-" + ("replaced" if replacement else "missing")
            directory = root / name
            directory.mkdir(mode=0o700)
            paths = [directory / "first.gplot", directory / "second.gplot"]
            scratch = root / (name + "-temp")
            scratch.mkdir(mode=0o700)
            command = [args.grow, paths[0], "--destination", paths[1], "--seed", "aa" * 32,
                       "--farmer-pk", farmer, "--pool-ph", "42" * 32, "--k", 18,
                       "--backend", "cpu", "--max-members", 2, "--batch-members", 1,
                       "--workers", 2, "--window-bytes", 262144, "--checkpoint-bytes", 65536,
                       "--capacity-bytes", 67108864, "--max-ram", args.max_ram,
                       "--generation-overlap", "off", "--construction", "external", "--temp-dir", scratch]
            run([*command, "--construction", "incremental", "--max-members", 1])
            paths[1].unlink()
            # Keep an ordinary first output eligible for more growth while a
            # later output requires its saved external scratch to resume.
            run([*command, "--resume"], 86, "external-run-after-checkpoint")
            assert Path(str(paths[1]) + ".external").is_dir(), "audit fixture did not save the later external destination"
            resuming = [*command, "--resume", "--max-members", 3]
            before = {path: (digest(path), path.stat().st_size, path.stat().st_mtime_ns)
                      for path in directory.rglob("*") if path.is_file()}
            saved = scratch.with_name(scratch.name + "-saved")
            scratch.rename(saved)
            if replacement:
                scratch.mkdir(mode=0o700)
                sentinel = scratch / "unrelated.txt"
                sentinel.write_text("preserve replacement directory\n")
            run(resuming, 1)
            after = {path: (digest(path), path.stat().st_size, path.stat().st_mtime_ns)
                     for path in directory.rglob("*") if path.is_file()}
            assert after == before, "invalid temporary root was detected after modifying a destination or checkpoint"
            if replacement:
                assert list(scratch.iterdir()) == [sentinel] and sentinel.read_text() == "preserve replacement directory\n"
                sentinel.unlink()
                scratch.rmdir()
            saved.rename(scratch)
            metrics = report(run(resuming))
            assert metrics["members"] == len(paths) * 3 and metrics["recovered"] is True, metrics
            assert set(directory.iterdir()) == set(paths) and not list(scratch.iterdir()), "external audit recovery left construction state"
            results["rejections"].append(name + " rejects before any destination modification")
        print("PASS temporary capacity fallback, ordinary extension and all-destination scratch audit", flush=True)

        if len(originals) >= 8:
            automatic = [args.grow, *common_options, "--generation-overlap", "off",
                         "--window-bytes", 65536, "--checkpoint-bytes", 1048576,
                         "--capacity-bytes", 67108864]
            for storage in ("raw", "compact"):
                name = "automatic-many-batches-" + storage
                path = target(name)
                command = [*automatic, path, "--batch-members", 1, "--member-storage", storage]
                metrics = complete(path, command, loaded_members=range(len(originals)))
                assert metrics["construction_requested"] == "auto" and metrics["automatic_merges"] > 0, metrics
                assert metrics["construction_selected"] == "merged", metrics
                assert metrics["selected_window_bytes"] == 65536 and metrics["selected_checkpoint_bytes"] == 1048576, metrics
                assert metrics["contribution_workers"] > 1 and metrics["codec_seconds"] > 0, metrics
                results["parity"].append({"case": name, **metrics})
                path.unlink()
            for name, options in (
                ("automatic-one-batch", ["--batch-members", len(originals)]),
                ("automatic-memory", ["--batch-members", 1, "--max-ram", 16777216, "--window-bytes", 524288]),
                ("automatic-reserve", ["--batch-members", 1, "--capacity-bytes", capacity, "--window-bytes", 4194304]),
            ):
                path = target(name)
                metrics = complete(path, [*automatic, path, *options])
                assert metrics["automatic_merges"] == 0 and metrics["construction_selected"] == "incremental", metrics
                if name == "automatic-memory":
                    assert metrics["peak_modeled_bytes"] <= 16777216, metrics
                results["parity"].append({"case": name, **metrics})
                path.unlink()
            path = target("automatic-capacity-fallback")
            metrics = complete(path, [*automatic, path, "--batch-members", 1,
                "--capacity-bytes", reference.stat().st_size + 1048576], loaded_members=range(len(originals)))
            assert metrics["automatic_fallbacks"] == metrics["automatic_merges"] == 1, metrics
            results["parity"].append({"case": "automatic-capacity-fallback", **metrics})
            path.unlink()
            # Omitted sizes are tuned between completed transactions. Explicit
            # sizes above remain fixed, including during automatic selection.
            path = target("automatic-io-budgets")
            command = [args.grow, path, "--fixture-inputs", inputs, "--max-members", len(originals),
                       "--batch-members", 1, "--capacity-bytes", 67108864,
                       "--max-ram", args.max_ram, "--workers", 2]
            metrics = complete(path, command, loaded_members=range(len(originals)))
            assert metrics["tuning_changes"] > 0 and metrics["selected_window_bytes"] <= 16777216, metrics
            assert metrics["selected_checkpoint_bytes"] <= 268435456 and metrics["max_batch_members"] == 1, metrics
            assert set(metrics["phases"]) == {"plan", "encode", "merge", "pack", "layout"}, metrics
            assert metrics["phases"]["encode"]["codec_seconds"] > 0, metrics
            results["parity"].append({"case": "automatic-io-budgets", **metrics})
            path.unlink()
            path = target("automatic-redo-capacity")
            command[1] = path
            metrics = complete(path, [*command, "--construction", "incremental",
                "--capacity-bytes", reference.stat().st_size + 2 * 1024 * 1024], loaded_members=range(len(originals)))
            assert metrics["final_bytes"] + metrics["selected_window_bytes"] + 8192 <= metrics["capacity_bytes"], metrics
            results["parity"].append({"case": "automatic-redo-capacity", **metrics})
            path.unlink()
            for point in ("merge-run-committed", "merge-pack-committed", "merge-retired"):
                path = target("automatic-recovery-" + point)
                command = [*automatic, path, "--batch-members", 1]
                run(command, code=86, point=point)
                metrics = complete(path, command, resume=True)
                results["recovery"].append({"case": "automatic-" + point, **metrics})
                path.unlink()
            print("PASS automatic construction, memory/space choices, capacity fallback and recovery", flush=True)

    path = target("capacity-rejection", native)
    before = digest(path), path.stat().st_size
    command = [*common, path, "--resume", "--capacity-bytes", path.stat().st_size]
    metrics = report(run(command))
    assert metrics["members"] == 1 and metrics["stop_reason"] == "capacity", metrics
    assert (digest(path), path.stat().st_size) == before, "capacity rejection changed old group"
    results["rejections"].append("exact capacity preserves original")

    path = target("wrong-request", native)
    before = digest(path)
    wrong_group = "ff" * 32 if group != "ff" * 32 else "00" * 32
    for options in [["--group-id", wrong_group], ["--meta-group", (meta + 1) % 256],
                    ["--memo", "ff" if memo != "ff" else "00"],
                    ["--max-ram", 16777216, "--window-bytes", 33554432]]:
        run([*common, path, "--resume", *options], 1)
        assert digest(path) == before, "rejected request changed the target"
    run([*common, path], 1)
    with path.open("r+b") as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run([*common, path, "--resume"], 1)
    assert digest(path) == before
    results["rejections"].extend(["identity", "RAM", "occupied path", "exclusive lock"])

    path = target("truncated-group", native)
    with path.open("r+b") as file:
        file.truncate(path.stat().st_size - 1)
    before = digest(path)
    run([*common, path, "--resume"], 1)
    assert digest(path) == before
    results["rejections"].append("truncation preserves damaged input")

    if args.recovery:
        assert k == 18, "use the small k18 fixture for repeated fault injection"
        points = ["allocated", "checkpoint-partial", "initialized", "marked",
                  "direct-target-partial", "direct-target", "direct-committed",
                  "payload", "final-header-partial", "finalizing", "retire-partial", "retire-partial:2",
                  "retired", "index-partial", "index",
                  "truncated", "published"]
        for fresh in [False, True]:
            extra = ["bootstrap"] if fresh else ["redo-written", "redo", "pending",
                                                "target-partial", "target", "committed"]
            for point in points + extra:
                name = ("create-" if fresh else "grow-") + point.replace(":", "-")
                path = target(name, None if fresh else native)
                command = [*common, path, "--batch-members", len(originals)]
                failed = command + ([] if fresh else ["--resume"])
                run(failed, 86, point)
                complete(path, command, True)
                results["recovery"].append(name)
                print("PASS recovery:", name, flush=True)

        path = target("repeated-crash", native)
        command = [*common, path, "--batch-members", len(originals)]
        run(command + ["--resume"], 86, "pending")
        run(command + ["--resume"], 86, "replay-partial")
        run(command + ["--resume"], 86, "replayed")
        complete(path, command, True)
        results["recovery"].append("repeated redo interruption")

        for point in ["allocated", "pending", "finalizing"]:
            path = target("pending-mismatch-" + point, native)
            command = [*common, path, "--batch-members", len(originals)]
            run(command + ["--resume"], 86, point)
            before = digest(path)
            run(command + ["--resume", "--group-id", wrong_group], 1)
            assert digest(path) == before, "identity mismatch modified interrupted file"
            run(command + ["--resume", "--max-ram", 16777216, "--window-bytes", 33554432], 1)
            assert digest(path) == before, "budget rejection modified interrupted file"
            complete(path, command, True)
        results["rejections"].append("interrupted identity/RAM rejection preserves bytes")

        # A real allocation/write size failure occurs before old-data mutation.
        path = target("file-size-failure", native)
        command = [*common, path, "--batch-members", len(originals), "--resume"]
        before = digest(path)

        def limit_file():
            resource.setrlimit(resource.RLIMIT_FSIZE, (native.stat().st_size, native.stat().st_size))

        failed = subprocess.run([str(arg) for arg in command], capture_output=True, preexec_fn=limit_file)
        assert failed.returncode != 0
        assert digest(path) == before, "failed allocation changed the old group"
        complete(path, command[:-1], True)
        results["rejections"].append("real file-size allocation failure")

        if args.modes:
            stage_points = ["stage-run-partial-flushed", "stage-run-flushed", "stage-run-committed",
                            "stage-pack-initialized", "stage-pack-flushed", "stage-pack-committed",
                            "stage-tail-committed", "stage-compact-partial", "stage-compact-committed",
                            "stage-final-header-partial", "stage-retired", "stage-published"]
            for fresh in [True, False]:
                for point in stage_points:
                    name = ("stage-create-" if fresh else "stage-grow-") + point.removeprefix("stage-")
                    path = target(name, None if fresh else native)
                    command = [*staged_common, path]
                    run(command + ([] if fresh else ["--resume"]), 86, point)
                    if point in ["stage-run-partial-flushed", "stage-run-flushed"]:
                        committed = 0 if fresh else 1
                    elif point == "stage-run-committed":
                        committed = min(len(originals), (0 if fresh else 1) + stage_batch)
                    else:
                        committed = len(originals)
                    manifest = resume_inputs(name, committed)
                    # Dispatch must recognize persisted staging even when the
                    # requested strategy and retained representation change.
                    resume = [*command, "--fixture-inputs", manifest,
                              "--construction", "incremental", "--member-storage", "raw"]
                    metrics = complete(path, resume, True,
                                       loaded_members=range(committed, len(originals)))
                    assert metrics["member_storage"] == "raw", metrics
                    results["recovery"].append(name)
                    print("PASS staged recovery without committed-member generation:", name, flush=True)

            path = target("staged-pending-mismatch", native)
            command = [*staged_common, path, "--resume"]
            run(command, 86, "stage-run-committed")
            before = digest(path), path.stat().st_size
            for options in [["--group-id", wrong_group], ["--max-members", 1],
                            ["--max-ram", 16777216, "--window-bytes", 33554432]]:
                run([*command, *options], 1)
                assert (digest(path), path.stat().st_size) == before, \
                    "rejected staged request changed the interrupted file"
            committed = min(len(originals), 1 + stage_batch)
            manifest = resume_inputs("staged-pending-mismatch", committed)
            complete(path, [*command, "--fixture-inputs", manifest],
                     loaded_members=range(committed, len(originals)))
            results["rejections"].append("staged interrupted identity/membership/RAM rejection preserves bytes")

    assert all(digest(path) == expected_hash for path, expected_hash in source_hashes.items()), \
        "original fixture was modified"
    results["scope"] = ("Canonical bytes, proof samples, software budgets and process fault injection; "
                        "not HDD throughput, physical 16 GB hardware, ENOSPC or power-loss qualification")
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("PASS same-file growth; original fixtures preserved", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grow", type=Path)
    parser.add_argument("plotter", type=Path, help="normal CLI with the unmodified upstream verifier")
    parser.add_argument("fixture_dir", type=Path, help="group.inputs, native.gplot and reference.gplot")
    parser.add_argument("--max-ram", type=int, default=14000000000)
    parser.add_argument("--cpu", action="store_true", help="also regenerate the fixture entirely in RAM")
    parser.add_argument("--recovery", action="store_true", help="repeat all fault boundaries on a k18 fixture")
    parser.add_argument("--modes", action="store_true",
                        help="also check raw/compact storage, staged/automatic construction and cache release on k18")
    parser.add_argument("--work-dir", type=Path, help="new directory for retained test artifacts")
    args = parser.parse_args()
    for name in ["grow", "plotter", "fixture_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    if args.work_dir:
        root = args.work_dir.resolve()
        root.mkdir(mode=0o700)
        check(args, root)
    else:
        # Keep tests on the destination filesystem, not a possible tmpfs /tmp.
        with tempfile.TemporaryDirectory(prefix="xchplot2-grow-", dir=Path.cwd()) as temporary:
            check(args, Path(temporary))


if __name__ == "__main__":
    main()
