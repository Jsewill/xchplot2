#!/usr/bin/env python3
"""Directory farming plans, ordinary-file parity, bounded fill and saved recovery."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile


# Disposable test identities, shared with group-rollover-test.py.
FARMER = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
SEED = "aa" * 32
OTHER_SEED = "bb" * 32
POOL = "42" * 32
HEADER = struct.Struct("<4sB32sBBHBQB")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def farming_copy(path):
    value = path.read_bytes()
    assert len(value) >= 65 and value.endswith(b"\n"), path
    assert hashlib.sha256(value[:-65]).hexdigest().encode() == value[-65:-1], path
    assert path.stat().st_mode & 0o777 == 0o600 and path.stat().st_nlink == 1, path
    return value


def run(command, code=0, point=None, **kwargs):
    env = dict(os.environ)
    env.pop("XCHPLOT2_GROW_FAILPOINT", None)
    if point:
        env["XCHPLOT2_GROW_FAILPOINT"] = point
    result = subprocess.run([str(value) for value in command], capture_output=True,
                            text=True, env=env, **kwargs)
    output = result.stdout + result.stderr
    assert all(seed not in output for seed in (SEED, OTHER_SEED)), "seed exposed in diagnostics"
    assert result.returncode == code, (result.returncode, result.stdout, result.stderr)
    return result


def reports(result):
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]


def plan(result):
    values = [value for value in reports(result) if value.get("farming_plan")]
    assert len(values) == 1, result.stdout
    value = values[0]
    assert not any(name.startswith(("trace_", "validation_")) for name in value), value
    assert "deadline_risk" not in value and "tail_seconds" not in value, value
    if value["feasible"]:
        assert 0 <= value["response_bound_seconds"] <= min(value["deadline_seconds"], value["challenge_seconds"]), value
        assert value["certified_max_eligible_groups"] <= value["max_eligible_groups"], value
    else:
        assert value["infeasibility_reason"], value
    return value


def frozen_plan(value):
    # The source label changes on resume; the saved inputs and initial layout
    # stay fixed while live extent admission has its own assessment.
    result = {name: field for name, field in value.items() if name not in ("extent_assessment", "profile_input")}
    result["destinations"] = [{name: field for name, field in lane.items()
                               if name not in ("admitted_files", "sealed_files", "fill_stop")}
                              for lane in value["destinations"]]
    return result


def metrics(result):
    value = reports(result)[-1]
    assert not value.get("farming_plan"), result.stdout
    return value


def outputs(directory):
    return sorted(directory.glob("grow-group-*.gplot"))


def snapshot(directories):
    result = {}
    for directory in directories:
        for path in directory.rglob("*"):
            status = path.lstat()
            result[str(path)] = (digest(path) if path.is_file() else None,
                                 status.st_size, status.st_mtime_ns, status.st_ino,
                                 status.st_mode, os.readlink(path) if path.is_symlink() else None)
    return result


def header(path):
    with path.open("rb") as stream:
        magic, version, group, k, strength, count, meta, index, memo_size = HEADER.unpack(
            stream.read(HEADER.size))
        memo = stream.read(memo_size)
    assert (magic, version, k, strength) == (b"PoS2", 2, 18, 2), path
    assert count and HEADER.size + memo_size < index < path.stat().st_size, path
    assert path.stat().st_mode & 0o777 == 0o600 and path.stat().st_nlink == 1, path
    return group.hex(), k, strength, count, meta, memo.hex()


def charge(path):
    return max(path.stat().st_size, path.stat().st_blocks * 512)


def bounded(result, directories, maximum_ram):
    assert result["peak_modeled_bytes"] <= maximum_ram, result
    assert result["peak_file_bytes"] <= result["capacity_bytes"], result
    actual = sum(charge(path) for directory in directories for path in directory.iterdir()
                 if path.is_file())
    assert actual <= result["capacity_bytes"], (actual, result)
    assert actual <= result["accounted_file_bytes"] <= result["capacity_bytes"], (actual, result)


def memberships(directories, forecast, expected=None):
    identities = set()
    families = {}
    counts = []
    for lane, directory in enumerate(directories):
        files = outputs(directory)
        lane_counts = []
        for ordinal, path in enumerate(files):
            assert path.name == f"grow-group-{ordinal:020d}.gplot", path
            group, _, _, count, meta, _ = header(path)
            assert count <= forecast["destinations"][lane]["group_members"], (path, count, forecast)
            for member in range(count):
                identity = (group, meta, member)
                assert identity not in identities, "farming files duplicate a member identity"
                identities.add(identity)
            families.setdefault(group, []).append(meta)
            lane_counts.append(count)
        counts.append(lane_counts)
    for values in families.values():
        assert len(values) == len(set(values)), "shared family reused a meta value"
        for bits in (forecast["filter_bits"], forecast["future_filter_bits"]):
            assert len(values) == len({meta & ((1 << bits) - 1) for meta in values}), \
                "shared family meta values alias under a forecast filter"
    if expected is not None:
        assert counts == expected, (counts, expected, forecast)
    return counts, families


def canonical(args, path, work):
    group, k, strength, count, meta, memo = header(path)
    work.mkdir(mode=0o700)
    raw = []
    for member in range(count):
        name = f"{member}.plot2"
        run([args.plotter, "test", k, group, strength, member, meta, 0,
             "--raw", "--out", work, "--out-name", name, "--memo", memo,
             "--config", os.devnull])
        raw.append(work / name)
    inputs = work / "group.inputs"
    inputs.write_text(f"{k} {strength} {meta} {group} {memo}\n" +
                      "".join(json.dumps(str(fragment)) + "\n" for fragment in raw))
    reference = work / "reference.gplot"
    run([args.reference, "assemble-only", inputs, reference, 512])
    assert digest(path) == digest(reference), "planned file differs from independent ordinary assembly"
    proof = run([args.plotter, "verify", path, "--full", "--trials", 100,
                 "--config", os.devnull])
    proof_output = proof.stdout + proof.stderr
    validated = re.search(r"(\d+) full proofs validated", proof_output.lower())
    assert validated and int(validated.group(1)) > 0, proof_output
    (work / "proof.log").write_text(proof_output)
    return {"members": count, "meta_group": meta, "sha256": digest(path),
            "full_proofs_validated": int(validated.group(1))}


def saved_external_pairs(args, root, references, results, construction):
    keys = ["--seed", SEED, "--farmer-pk", FARMER, "--pool-ph", POOL]

    def identity(ordinal, seed=SEED):
        # This fixture has one resource/lane and a fixed meta, so each ordinal
        # is its own family. Derive through the existing reference key boundary.
        farming_seed = hashlib.sha256(b"xchplot2 directory farming v1" + bytes.fromhex(seed)).digest()
        subseed = hashlib.sha256(farming_seed + ordinal.to_bytes(8, "little")).hexdigest()
        return run([args.reference, "prepare", 18, 2, 0, 1],
                   input=f"{subseed} {FARMER} {POOL}\n").stdout.splitlines()[0].split()

    def stage(directory, scratch, ordinal, count, point=None, seed=SEED):
        group, memo = identity(ordinal, seed)
        path = directory / f"grow-group-{ordinal:020d}.gplot"
        command = [args.grow, path, *construction, "--meta-group", 0,
                   "--group-id", group, "--memo", memo, "--construction", "external",
                   "--temp-dir", scratch, "--max-members", max(1, count),
                   "--batch-members", max(1, count)]
        run(command, 86, point or ("external-run-after-checkpoint" if count else
                                  "external-initialized-after-checkpoint"))
        return path

    def fixture(name, prior=2, following=1, groups=2, point=None, seed=SEED):
        parent = root / name
        parent.mkdir(mode=0o700)
        directory = parent / "outputs é"
        directory.mkdir(mode=0o700)
        scratch = [parent / f"scratch-{ordinal}" for ordinal in range(groups)]
        for path in scratch:
            path.mkdir(mode=0o700)
        command = [args.grow, directory, *keys, *construction, "--meta-group", 0,
                   "--group-members", 2, "--max-groups", groups, "--max-total-members", groups * 2,
                   "--farming-max-eligible-groups", groups, "--construction", "external",
                   "--temp-dir", scratch[0], "--batch-members", prior]
        run(command, 86, "external-run-after-checkpoint")
        forecast = plan(run([*command, "--resume", "--plan-only"]))
        lane, = forecast["destinations"]
        assert [lane[key] for key in ("resource", "resource_rank", "resource_lanes", "meta_slots")] == [0, 0, 1, 1]
        next_path = stage(directory, scratch[1], 1, following, point, seed)
        resume = [*command, "--resume"]
        for path in scratch[1:]:
            resume += ["--temp-dir", path]
        return directory, scratch, next_path, resume

    def reject_unchanged(command, directories):
        before = snapshot(directories)
        run(command, 1)
        assert snapshot(directories) == before, "saved-pair rejection modified owned files"

    for following in (1, 0):
        name = "external-pair-" + ("partial" if following else "empty")
        directory, scratch, next_path, command = fixture(name, following=following)
        work = [directory, *scratch]
        manifest = directory / "grow-farm.plan"
        original_plan = digest(manifest)
        before = snapshot(work)
        assert plan(run([*command, "--plan-only"]))["feasible"]
        assert snapshot(work) == before, "saved-pair preview modified owned files"
        reject_unchanged([*command, "--max-total-members", 2], work)
        reject_unchanged([*command, "--max-groups", 1], work)
        capacity = sum(charge(path) for path in directory.iterdir() if path.is_file()) - 1
        reject_unchanged([*command, "--capacity-bytes", capacity], work)

        def next_snapshot():
            return {path: stamp for path, stamp in snapshot(work).items()
                    if Path(path) == next_path or path.startswith(str(next_path) + ".external") or
                    Path(path).is_relative_to(scratch[1])}

        untouched_next = next_snapshot()
        run(command, 86, "external-published-after-checkpoint")
        assert next_snapshot() == untouched_next, "resume changed the successor before publishing its prior group"
        recovered = run(command)
        assert digest(manifest) == original_plan
        bounded(metrics(recovered), [directory], args.max_ram)
        memberships([directory], plan(recovered), [[2, 2]])
        assert metrics(recovered)["generated_members"] == 2 - following, "resume regenerated a committed external prefix"
        assert all(not list(path.iterdir()) for path in scratch), "saved-pair resume retained owned scratch"
        for ordinal, path in enumerate(outputs(directory)):
            results["parity"].append({"case": name, "file": ordinal,
                **canonical(args, path, references / f"{name}-{ordinal}")})
        before = snapshot(work)
        idle = run(command)
        assert metrics(idle)["generated_members"] == 0 and snapshot(work) == before
        results["recovery"].append({"case": name, "members": metrics(recovered)["members"]})

    for name, options in (
            ("external-pair-partial-prior", {"prior": 1}),
            ("external-pair-finishing-next", {"point": "external-pack-initialized-after-checkpoint"}),
            ("external-pair-wrong-next-identity", {"seed": OTHER_SEED}),
            ("external-pair-third-pending", {"groups": 3, "following": 2})):
        directory, scratch, _, command = fixture(name, **options)
        if options.get("groups") == 3:
            stage(directory, scratch[2], 2, 1)
        reject_unchanged(command, [directory, *scratch])
        results["rejections"].append(name)
    print("PASS consecutive external saved-pair audit, prior-first recovery and bounds", flush=True)


def compatible_builds(original, replacement, root, maximum_ram):
    """Use genuinely different producers with the same explicit farm semantics."""
    root.mkdir(mode=0o700)
    original_job, replacement_job = root / "original job", root / "replacement job"
    original_job.mkdir(mode=0o700)
    replacement_job.mkdir(mode=0o700)
    limits = ["--k", 18, "--strength", 2, "--backend", "cpu", "--max-ram", maximum_ram,
              "--workers", 2, "--window-bytes", 65536, "--checkpoint-bytes", 65536,
              "--capacity-bytes", 100000000, "--headroom-bytes", 0, "--construction", "incremental"]
    policy = ["--group-members", "auto", "--max-total-members", 1, "--max-groups", 1]
    keys = ["--farmer-pk", FARMER, "--pool-ph", POOL]
    run([original, original_job, *limits, *policy, *keys, "--seed", SEED])
    run([replacement, replacement_job, *limits, *policy, *keys, "--seed", OTHER_SEED])
    manifests = [directory / "grow-farm.plan" for directory in (original_job, replacement_job)]
    records = [farming_copy(path) for path in manifests]
    producers = [json.loads(record.splitlines()[1]) for record in records]
    assert producers[0] != producers[1], "compatibility fixture requires different source build identities"
    assert all(record.startswith(b"ordinary-group-farming-v6\n") for record in records)
    before = snapshot([original_job, replacement_job])
    resumed = run([replacement, original_job, "--resume", *limits, *policy])
    assert metrics(resumed)["generated_members"] == 0
    assert snapshot([original_job, replacement_job]) == before, "compatible build changed completed job bytes or metadata"
    assert farming_copy(manifests[0]) == records[0], "compatible build rewrote provenance or canonical digest"

    # This same-file checkpoint has no farming manifest. Its exact construction
    # identity must independently refuse the replacement build before writes.
    pending = root / "pending.gplot"
    pending_options = [pending, *limits, "--group-id", "ab" * 32, "--memo", "00" * 112,
                       "--max-members", 1]
    run([original, *pending_options], 86, "initialized")
    before = snapshot([root])
    refused = run([replacement, *pending_options, "--resume"], 1)
    assert "same codec/build identity" in refused.stderr, refused.stderr
    assert snapshot([root]) == before, "incompatible checkpoint inspection modified pending data"
    run([original, *pending_options, "--resume"])
    completed = digest(pending)
    ordinary = run([replacement, *pending_options, "--resume"])
    assert metrics(ordinary)["generated_members"] == 0 and digest(pending) == completed
    print("PASS compatible completed-job recovery, canonical provenance and strict transient checkpoint identity", flush=True)
    return {"original_producer": producers[0], "replacement_producer": producers[1],
            "original_binary": digest(original), "replacement_binary": digest(replacement)}


def check(args, root):
    results = {"backend": args.backend, "plans": [], "parity": [], "recovery": [], "rejections": []}
    if args.replacement_grow:
        results["compatibility"] = compatible_builds(args.grow, args.replacement_grow, root / "compatible builds", args.max_ram)
    keys = ["--seed", SEED, "--farmer-pk", FARMER, "--pool-ph", POOL]
    construction = ["--k", 18, "--strength", 2, "--backend", args.backend,
                    "--max-ram", args.max_ram, "--window-bytes", 65536,
                    "--checkpoint-bytes", 65536, "--workers", 2, "--batch-members", 2,
                    "--capacity-bytes", 100000000, "--headroom-bytes", 0]

    def directories(name, count=1):
        parent = root / name
        parent.mkdir(mode=0o700)
        value = [parent / f"drive {index} é" for index in range(count)]
        for directory in value:
            directory.mkdir(mode=0o700)
        return value

    def options(value):
        command = [value[0]]
        for directory in value[1:]:
            command += ["--destination", directory]
        return command

    # Preview needs no keys, producer, lock, checkpoint or output artifact.
    preview = directories("preview", 2)
    preview_command = [args.grow, *options(preview), "--plan-only", "--k", 18,
                       "--group-members", 2, "--max-total-members", 5,
                       "--capacity-bytes", 100000000, "--headroom-bytes", 0,
                       "--backend", "gpu", "--device", 2147483647]
    before = snapshot(preview)
    forecast = plan(run(preview_command))
    assert snapshot(preview) == before and all(not list(directory.iterdir()) for directory in preview)
    assert forecast["destinations"][0]["resource"] == forecast["destinations"][1]["resource"], forecast
    assert sum(lane["capacity_bytes"] for lane in forecast["destinations"]) <= 100000000, forecast
    assert all(lane["group_members"] == 2 and lane["members"] == 5 for lane in forecast["destinations"]), forecast
    results["plans"].append({"case": "keyless same-resource shared-capacity preview", **forecast})

    # Fixed layout makes these quantitative comparisons about reader work,
    # rather than asserting a particular heuristic's chosen group size.
    profiles = directories("profiles")
    profile_command = [args.grow, profiles[0], "--plan-only", "--k", 18,
                       "--group-members", 16, "--capacity-bytes", 1000000000,
                       "--headroom-bytes", 0]
    low = plan(run([*profile_command, "--farming-cores", 1,
                    "--farming-filter-bits", 9, "--farming-future-filter-bits", 9]))
    cores = plan(run([*profile_command, "--farming-cores", 8,
                      "--farming-filter-bits", 9, "--farming-future-filter-bits", 9]))
    filtered = plan(run([*profile_command, "--farming-cores", 1,
                         "--farming-filter-bits", 10, "--farming-future-filter-bits", 10]))
    workers = plan(run([*profile_command, "--farming-cores", 1, "--farming-lookup-workers", 8,
                        "--farming-filter-bits", 9, "--farming-future-filter-bits", 9]))
    faster = plan(run([*profile_command, "--farming-cores", 1, "--farming-chain-us-per-member", 1000,
                       "--farming-filter-bits", 9, "--farming-future-filter-bits", 9]))
    cached = plan(run([*profile_command, "--farming-cores", 1,
                       "--farming-filter-bits", 9, "--farming-future-filter-bits", 9,
                       "--farming-reader", "cached", "--farming-index-cache-bytes", 268435456]))
    assert 0 < cores["expected_cpu_utilization"] < low["expected_cpu_utilization"], (low, cores)
    assert 0 < filtered["expected_cpu_utilization"] < low["expected_cpu_utilization"], (low, filtered)
    assert 0 < workers["expected_worker_utilization"] < low["expected_worker_utilization"], (low, workers)
    assert 0 < faster["expected_cpu_utilization"] < low["expected_cpu_utilization"], (low, faster)
    assert cached["expected_cpu_utilization"] < low["expected_cpu_utilization"], (low, cached)
    assert cached["reader"] == "cached" and cached["index_cache_bytes"] > 0, cached
    for value in (low, cores, filtered, workers, faster, cached):
        lane = value["destinations"][0]
        assert lane["members"] >= lane["files"] > 0 and lane["group_members"] == 16, value
        assert value["feasible"] and math.isfinite(value["expected_cpu_utilization"]), value
    assert not list(profiles[0].iterdir())
    results["plans"].extend({"case": name, **value} for name, value in
                            (("single core", low), ("shared CPU budget", cores),
                             ("stronger filter", filtered), ("lookup worker budget", workers),
                             ("faster individual core", faster), ("persistent index cache", cached)))
    print("PASS keyless plans, shared capacity and quantitative reader-profile sensitivity", flush=True)

    # Neither preference may turn a response deadline into a soft objective.
    # The invalid GPU selection proves refusal precedes backend startup too.
    for objective in ("energy", "latency"):
        refused = directories("infeasible-" + objective, 2)
        command = [args.grow, *options(refused), *keys, *construction,
                   "--group-members", 1, "--max-total-members", 1,
                   "--farming-objective", objective, "--farming-deadline-ms", 1,
                   "--backend", "gpu", "--device", 2147483647]
        before = snapshot(refused)
        preview = plan(run([*command, "--plan-only"], 1))
        applied = plan(run(command, 1))
        assert not preview["feasible"] and applied == preview, (preview, applied)
        assert snapshot(refused) == before and all(not list(directory.iterdir()) for directory in refused)
        results["rejections"].append("infeasible " + objective + " preview/apply precedes publication and backend")

    # A farming option explicitly opts a fixed-N single directory into the
    # planner; the legacy fixed-N case below still has no farming options.
    opted = directories("fixed-planner-opt-in")
    opted_command = [args.grow, *options(opted), *construction, "--group-members", 2,
                     "--max-total-members", 1, "--max-groups", 1,
                     "--farming-objective", "latency", "--farming-cpu-work-weight", 2,
                     "--farming-io-work-weight", 3, "--farming-max-eligible-groups", 2]
    opted_preview = plan(run([*opted_command, "--plan-only"]))
    opted_result = run([*opted_command, *keys])
    assert plan(opted_result) == opted_preview and (opted[0] / "grow-farm.plan").exists()
    assert opted_preview["objective"] == "latency" and opted_preview["cpu_work_weight"] == 2
    assert opted_preview["io_work_weight"] == 3 and opted_preview["max_eligible_groups"] == 2
    before = snapshot(opted)
    for changed in (["--farming-objective", "energy"], ["--farming-cpu-work-weight", 3],
                    ["--farming-io-work-weight", 4], ["--farming-max-eligible-groups", 3]):
        run([*opted_command, *keys, "--resume", *changed], 1)
        assert snapshot(opted) == before
    omitted = [args.grow, *options(opted), *keys, *construction, "--group-members", 2,
               "--max-total-members", 1, "--max-groups", 1, "--resume"]
    repeated = run([*omitted, "--backend", "gpu", "--device", 2147483647])
    assert plan(repeated) == {**opted_preview, "profile_input": "saved job"}
    assert metrics(repeated)["generated_members"] == 0
    assert snapshot(opted) == before
    run([*omitted, "--group-members", 3], 1)
    assert snapshot(opted) == before
    results["recovery"].append("fixed-N planner opt-in and frozen objective/weights/eligible-group limit")

    pending = directories("single-pending-fixed-plan")
    pending_command = [args.grow, *options(pending), *keys, *construction,
                       "--group-members", 2, "--max-total-members", 1, "--max-groups", 1]
    run([*pending_command, "--farming-objective", "latency"], 86, "farm-plan-before-publication")
    before = snapshot(pending)
    run([*pending_command, "--resume", "--group-members", 3], 1)
    assert snapshot(pending) == before, "pending saved protocol was modified by mismatching fixed N"
    recovered = run([*pending_command, "--resume"])
    assert plan(recovered)["objective"] == "latency" and header(outputs(pending[0])[0])[3] == 1
    assert (pending[0] / "grow-farm.plan").exists() and not (pending[0] / "grow-farm.plan.next").exists()
    results["recovery"].append("published and pending fixed-N farm resumes retain omitted saved farming options")

    # Each extra family would fit separately, but the simultaneous requested
    # extensions must count against the same physical resource together.
    joint = directories("joint-family-constraint", 2)
    joint_command = [args.grow, *options(joint), *keys, *construction,
                     "--group-members", 1, "--meta-group", 0,
                     "--max-total-members", 1, "--max-groups", 2,
                     "--farming-max-eligible-groups", 3]
    original = plan(run(joint_command))
    assert original["certified_max_eligible_groups"] == 2, original
    before = snapshot(joint)
    preview = plan(run([*joint_command, "--resume", "--max-total-members", 2, "--plan-only"], 1))
    applied = plan(run([*joint_command, "--resume", "--max-total-members", 2,
                        "--backend", "gpu", "--device", 2147483647], 1))
    assert not applied["feasible"] and applied == preview, (preview, applied)
    assert applied["extent_assessment"]["certified_max_eligible_groups"] == 4, applied
    assert snapshot(joint) == before and all(len(outputs(directory)) == 1 for directory in joint)
    results["rejections"].append("joint shared-resource new-family extensions exceed the frozen eligible-group limit")
    print("PASS hard-deadline refusal, fixed-N planner opt-in and joint family admission", flush=True)

    # Omitting group-members selects the new directory default. Its actual
    # output must follow the reported plan and ordinary upstream file format.
    single = directories("automatic-single")
    single_command = [args.grow, *options(single), *keys, *construction,
                      "--max-total-members", 3, "--max-groups", 3]
    first = run(single_command)
    single_plan = plan(first)
    bounded(metrics(first), single, args.max_ram)
    counts, _ = memberships(single, single_plan)
    assert sum(counts[0]) == 3 and (single[0] / "grow-farm.plan").is_file()
    assert (single[0] / "grow-farm.plan").stat().st_mode & 0o777 == 0o600
    assert SEED in (single[0] / "grow-farm.plan").read_text()

    automatic = directories("automatic-multiple", 2)
    automatic_result = run([args.grow, *options(automatic), *keys, *construction,
                            "--max-total-members", 3, "--max-groups", 3])
    automatic_plan = plan(automatic_result)
    automatic_counts, _ = memberships(automatic, automatic_plan)
    assert all(sum(lane) == 3 for lane in automatic_counts)
    bounded(metrics(automatic_result), automatic, args.max_ram)

    # Existing fixed-size single-directory rollover remains the old protocol.
    legacy = directories("legacy-fixed")
    old = run([args.grow, *options(legacy), *keys, *construction,
               "--group-members", 2, "--max-total-members", 3])
    assert not any(value.get("farming_plan") for value in reports(old))
    assert [header(path)[3] for path in outputs(legacy[0])] == [2, 1]
    assert not (legacy[0] / "grow-farm.plan").exists()
    explicit_file = root / "explicit.gplot"
    file_result = run([args.grow, explicit_file, *keys, *construction, "--max-members", 1])
    assert not any(value.get("farming_plan") for value in reports(file_result))
    assert header(explicit_file)[3] == 1
    results["recovery"].append("explicit file and legacy fixed rollover preserve their protocols")

    # Multiple directories use planned rollover even with an explicit file
    # size. On one resource, shared families must have unique meta values.
    multiple = directories("multiple", 2)
    command = [args.grow, *options(multiple), *keys, *construction,
               "--group-members", 2, "--max-total-members", 5, "--max-groups", 3]
    grown = run(command)
    saved_plan = plan(grown)
    bounded(metrics(grown), multiple, args.max_ram)
    counts, families = memberships(multiple, saved_plan, [[2, 2, 1], [2, 2, 1]])
    assert saved_plan["destinations"][0]["resource"] == saved_plan["destinations"][1]["resource"]
    if saved_plan["destinations"][0]["meta_slots"] > 1:
        assert any(len(values) > 1 for values in families.values()), "planned shared families were not emitted"
    for directory in multiple:
        assert set(directory.iterdir()) == {*outputs(directory), directory / "grow-farm.plan"}
        assert (directory / "grow-farm.plan").read_bytes() == (multiple[0] / "grow-farm.plan").read_bytes()
    before = snapshot(multiple)
    repeated = run([*command, "--resume", "--backend", "gpu", "--device", 2147483647])
    assert metrics(repeated)["generated_members"] == 0 and snapshot(multiple) == before
    assert plan(repeated) == {**saved_plan, "profile_input": "saved job"}
    results["recovery"].append("planned multi-group fill, terminal partial and producer-free no-op")

    # Preview and apply must reserve the same manifest space. Resume with no
    # remaining capacity must also stay idle when a scratch pool is present.
    for scratch_enabled in (False, True):
        mode = "scratch" if scratch_enabled else "no-scratch"
        boundary = directories("exact-capacity-" + mode, 2)
        scratch = root / ("scratch-" + mode)
        scratch.mkdir(mode=0o700)
        extra = ["--temp-dir", scratch] if scratch_enabled else []
        boundary_command = [args.grow, *options(boundary), *construction, *extra,
                            "--group-members", 1, "--max-total-members", 1,
                            "--max-groups", 1, "--construction", "incremental"]
        preview = plan(run([*boundary_command, "--plan-only"]))
        applied = run([*boundary_command, *keys])
        assert plan(applied) == preview, "keyless preview differs from applied plan"
        before = snapshot(boundary)
        exact_capacity = sum(charge(path) for directory in boundary for path in directory.iterdir()
                             if path.is_file())
        idle = run([*boundary_command, *keys, "--resume", "--capacity-bytes", exact_capacity,
                    "--max-total-members", 3, "--max-groups", 3,
                    "--backend", "gpu", "--device", 2147483647])
        assert metrics(idle)["generated_members"] == 0 and metrics(idle)["stop_reason"] == "capacity"
        bounded(metrics(idle), boundary, args.max_ram)
        assert snapshot(boundary) == before and not list(scratch.iterdir())
        for changed in (["--k", 20], ["--strength", 3]):
            run([*boundary_command, "--plan-only", *changed], 1)
            assert snapshot(boundary) == before, "incompatible preview changed outputs"

        # Resume one completed lane beside one unfinished lane. The completed
        # lane needs no engine, including when a scratch pool is available.
        mixed = directories("idle-active-" + mode, 2)
        mixed_command = [args.grow, *options(mixed), *construction, *extra,
                         "--group-members", 1, "--max-total-members", 1,
                         "--max-groups", 1, "--construction", "incremental"]
        mixed_plan = plan(run([*mixed_command, "--plan-only"]))
        run([*mixed_command, *keys], 86, "published")
        completed = {path: digest(path) for directory in mixed for path in outputs(directory)
                     if path.read_bytes()[4] == 2}
        assert len(completed) == 1
        result = run([*mixed_command, *keys, "--resume"])
        bounded(metrics(result), mixed, args.max_ram)
        assert plan(result) == {**mixed_plan, "profile_input": "saved job"}
        assert all(digest(path) == saved for path, saved in completed.items())
        assert metrics(result)["generated_members"] > 0
        assert not list(scratch.iterdir())
        results["recovery"].append("preview, exact capacity and idle lane with " + mode)

    limited = directories("capacity", 2)
    capacity = sum(charge(path) for directory in multiple for path in directory.iterdir()
                   if path.is_file()) - min(charge(outputs(directory)[-1]) for directory in multiple)
    limited_result = run([args.grow, *options(limited), *keys, *construction,
                          "--group-members", 2, "--max-total-members", 5,
                          "--max-groups", 3, "--capacity-bytes", capacity])
    limited_metrics = metrics(limited_result)
    bounded(limited_metrics, limited, args.max_ram)
    assert limited_metrics["members"] < sum(sum(lane) for lane in counts), limited_metrics
    assert limited_metrics["stop_reason"] == "capacity", limited_metrics
    before = snapshot(limited)
    stopped = run([args.grow, *options(limited), *keys, *construction,
                   "--group-members", 2, "--max-total-members", 5,
                   "--max-groups", 3, "--capacity-bytes", capacity, "--resume"])
    assert metrics(stopped)["stop_reason"] == "capacity" and snapshot(limited) == before
    results["rejections"].append("shared aggregate capacity includes immutable plan and completed files")

    references = root / "references"
    references.mkdir(mode=0o700)
    for lane, directory in enumerate([*single, *automatic, *multiple, *limited]):
        for ordinal, path in enumerate(outputs(directory)):
            results["parity"].append({"lane": lane, "file": ordinal,
                                       **canonical(args, path, references / f"{lane}-{ordinal}")})
    print("PASS planned ordinary files, independent CPU byte parity and full proofs", flush=True)

    # Save a partial job, then change plotting RAM and available filesystem
    # space. The immutable plan must survive while the last file can extend.
    partial = directories("partial", 2)
    partial_command = [args.grow, *options(partial), *keys, *construction,
                       "--group-members", 2, "--meta-group", 0, "--farming-max-eligible-groups", 16,
                       "--max-total-members", 1, "--max-groups", 2]
    initial = run(partial_command)
    initial_plan = plan(initial)
    manifest = partial[0] / "grow-farm.plan"
    saved_manifest = digest(manifest), manifest.stat().st_mtime_ns
    bounded_ram = min(args.max_ram, 268435456)
    unrelated = root / "unrelated-allocation"
    with unrelated.open("xb") as stream:
        stream.write(b"x" * 1048576)
        stream.flush()
        os.fsync(stream.fileno())
    continued = run([*partial_command, "--resume", "--max-total-members", 3,
                     "--max-ram", bounded_ram, "--capacity-bytes", 90000000])
    assert frozen_plan(plan(continued)) == frozen_plan(initial_plan)
    assert plan(continued)["extent_assessment"]["feasible"], continued.stdout
    assert metrics(continued)["farming_certified_max_eligible_groups"] == 4, continued.stdout
    assert metrics(continued)["farming_response_bound_seconds"] <= min(initial_plan["deadline_seconds"], initial_plan["challenge_seconds"])
    assert (digest(manifest), manifest.stat().st_mtime_ns) == saved_manifest
    memberships(partial, initial_plan, [[2, 1], [2, 1]])
    bounded(metrics(continued), partial, bounded_ram)
    for lane, directory in enumerate(partial):
        for ordinal, path in enumerate(outputs(directory)):
            results["parity"].append({"case": "partial continuation", "lane": lane, "file": ordinal,
                                       **canonical(args, path, references / f"partial-{lane}-{ordinal}")})
    results["recovery"].append("shared-resource frozen layout extends partial files and new families beyond forecast")

    # Every saved output is audited before an earlier lane can grow.
    continuation = [*partial_command, "--resume", "--max-total-members", 4]
    for extra in (["--seed", OTHER_SEED], ["--farming-cores", initial_plan["harvester_cores"] + 1],
                  ["--farming-lookup-workers", initial_plan["lookup_workers"] + 1],
                  ["--farming-memory-bytes", initial_plan["memory_bytes"] + 1],
                  ["--farming-chain-us-per-member", 1000], ["--farming-index-ns-per-entry", 100],
                  ["--farming-reader", "cached"], ["--group-members", 3],
                  ["--meta-group", 7]):
        before = snapshot(partial)
        run([*continuation, *extra], 1)
        assert snapshot(partial) == before, "incompatible resume mutated a saved output"
    before = snapshot(partial)
    reordered = run([args.grow, *options(list(reversed(partial))), *keys, *construction,
                     "--group-members", 2, "--max-total-members", 3, "--resume"])
    assert frozen_plan(plan(reordered)) == frozen_plan(initial_plan)
    assert snapshot(partial) == before, "reordered destinations mutated outputs"
    late = outputs(partial[-1])[-1]
    with late.open("r+b") as stream:
        original = stream.read(1)
        stream.seek(0)
        stream.write(b"!")
    before = snapshot(partial)
    run(continuation, 1)
    assert snapshot(partial) == before, "late corruption was detected after an earlier mutation"
    with late.open("r+b") as stream:
        stream.write(original)
    saved = manifest.read_bytes()
    manifest.write_bytes(b"!" + saved[1:])
    before = snapshot(partial)
    run(continuation, 1)
    assert snapshot(partial) == before, "invalid plan changed output files"
    manifest.write_bytes(saved)
    results["rejections"].extend(["changed keys/profile/layout", "changed destination order",
                                  "late corrupt output before mutation", "corrupt plan checksum"])

    existing_charge = sum(charge(path) for directory in partial for path in directory.iterdir()
                          if path.is_file())
    before = snapshot(partial)
    run([*continuation, "--capacity-bytes", existing_charge - 1], 1)
    assert snapshot(partial) == before, "lowered capacity below saved charges changed outputs"
    narrow_capacity = existing_charge + sum(charge(outputs(directory)[0]) for directory in partial)
    prior_hashes = {path: digest(path) for directory in partial for path in outputs(directory)}
    narrowed = run([*continuation, "--max-total-members", 7, "--max-groups", 4,
                    "--capacity-bytes", narrow_capacity])
    narrowed_metrics = metrics(narrowed)
    bounded(narrowed_metrics, partial, args.max_ram)
    assert narrowed_metrics["capacity_bytes"] <= narrow_capacity, narrowed_metrics
    assert 6 <= narrowed_metrics["members"] < 14, narrowed_metrics
    assert narrowed_metrics["stop_reason"] == "capacity", narrowed_metrics
    assert frozen_plan(plan(narrowed)) == frozen_plan(initial_plan) and digest(manifest) == hashlib.sha256(saved).hexdigest()
    for lane, directory in enumerate(partial):
        for ordinal, path in enumerate(outputs(directory)):
            if prior_hashes.get(path) != digest(path):
                results["parity"].append({"case": "narrow capacity rollover", "lane": lane, "file": ordinal,
                                           **canonical(args, path, references / f"narrow-{lane}-{ordinal}")})
    before = snapshot(partial)
    unchanged = run([*continuation, "--max-total-members", 7, "--max-groups", 4,
                     "--capacity-bytes", narrow_capacity])
    assert metrics(unchanged)["stop_reason"] == "capacity" and snapshot(partial) == before
    results["rejections"].extend(["resume cap below existing charges", "narrow aggregate continuation across rollover"])
    print("PASS immutable layout resume and complete pre-mutation audits", flush=True)

    # The ordinary in-file durability protocols remain responsible for live
    # progress. Resume each interrupted stage using the same saved farm plan.
    for point in ("stage-run-committed", "stage-pack-committed", "published:2"):
        interrupted = directories(point.replace(":", "-"), 2)
        recovery_command = [args.grow, *options(interrupted), *keys, *construction,
                            "--group-members", 2, "--max-total-members", 3,
                            "--max-groups", 2, "--batch-members", 1,
                            "--construction", "staged" if point.startswith("stage-") else "incremental"]
        run(recovery_command, 86, point)
        manifest = interrupted[0] / "grow-farm.plan"
        original_plan = digest(manifest)
        recovered = run([*recovery_command, "--resume"])
        assert digest(manifest) == original_plan
        bounded(metrics(recovered), interrupted, args.max_ram)
        memberships(interrupted, plan(recovered), [[2, 1], [2, 1]])
        # Same seed, resource and layout give the same ordinary files as the
        # independently checked earlier multi-directory run.
        for lane, directory in enumerate(interrupted):
            for ordinal, path in enumerate(outputs(directory)):
                control = outputs(multiple[lane])[ordinal]
                if header(control) == header(path):
                    assert digest(path) == digest(control), (point, lane, ordinal)
                else:
                    results["parity"].append({"point": point, "lane": lane, "file": ordinal,
                                               **canonical(args, path, references / f"{point.replace(':', '-')}-{lane}-{ordinal}")})
        before = snapshot(interrupted)
        idle = run([*recovery_command, "--resume"])
        assert metrics(idle)["generated_members"] == 0 and snapshot(interrupted) == before
        results["recovery"].append({"point": point, "members": metrics(recovered)["members"]})
    print("PASS staged, packing and ordinary publication interruption recovery", flush=True)

    saved_external_pairs(args, root, references, results, construction)

    for point in ("farm-plan-before-publication", "farm-plan-all-staged", "farm-plan-published"):
        interrupted = directories(point, 2)
        recovery_command = [args.grow, *options(interrupted), *keys, *construction,
                            "--group-members", 2, "--max-total-members", 1, "--max-groups", 1]
        run(recovery_command, 86, point)
        assert all(not outputs(directory) for directory in interrupted), "plan publication started plotting early"
        published = interrupted[0] / "grow-farm.plan"
        if point != "farm-plan-published":
            pending = interrupted[0] / "grow-farm.plan.next"
            assert pending.is_file() and SEED in pending.read_text()
            assert pending.stat().st_mode & 0o777 == 0o600
            assert not published.exists()
        else:
            assert published.is_file() and (interrupted[1] / "grow-farm.plan.next").read_bytes() == published.read_bytes()
        if point == "farm-plan-all-staged":
            copies = [farming_copy(directory / "grow-farm.plan.next") for directory in interrupted]
            assert all(value == copies[0] for value in copies), "all-staged barrier omitted a complete plan replica"
        recovered = run([*recovery_command, "--resume"])
        assert all((directory / "grow-farm.plan").read_bytes() == published.read_bytes() and
                   not (directory / "grow-farm.plan.next").exists() for directory in interrupted)
        memberships(interrupted, plan(recovered), [[1], [1]])
        bounded(metrics(recovered), interrupted, args.max_ram)
        for lane, directory in enumerate(interrupted):
            results["parity"].append({"point": point, "lane": lane,
                                       **canonical(args, outputs(directory)[0], references / f"{point}-{lane}")})
        results["recovery"].append({"point": point, "members": metrics(recovered)["members"]})
    print("PASS durable farming-plan publication interruption recovery", flush=True)

    for point in ("farm-extent-all-staged", "farm-extent-one-published", "farm-extent-published"):
        interrupted = directories(point, 2)
        recovery_command = [args.plotter, "fill", *interrupted, "--compute", "off", "--json", *keys, *construction,
                            "--max-total-members", 2, "--max-groups", 2]
        forecast = plan(run([*recovery_command, "--plan-only"]))
        assert forecast["fill_policy"] == "capacity-with-joint-admission", forecast
        run(recovery_command, 86, point)
        published = [directory / "grow-farm.plan" for directory in interrupted]
        pending = [directory / "grow-farm.plan.next" for directory in interrupted]
        current = [farming_copy(path) for path in published]
        if point == "farm-extent-all-staged":
            staged = [farming_copy(path) for path in pending]
            assert all(value == current[0] for value in current), "extent staging replaced a published parent early"
            assert all(value == staged[0] for value in staged) and staged[0] != current[0], \
                "extent all-staged barrier omitted the same complete revision on every destination"
        elif point == "farm-extent-one-published":
            assert not pending[0].exists() and farming_copy(pending[1]) == current[0]
            assert current[0] != current[1], "extent interruption did not preserve its unpublished parent replica"
        else:
            assert all(value == current[0] for value in current) and all(not path.exists() for path in pending)
        recovered = run([*recovery_command, "--resume"])
        assert frozen_plan(plan(recovered)) == frozen_plan(forecast)
        copies = [farming_copy(path) for path in published]
        assert all(value == copies[0] for value in copies) and all(not path.exists() for path in pending)
        bounded(metrics(recovered), interrupted, args.max_ram)
        counts, _ = memberships(interrupted, plan(recovered))
        assert all(sum(lane) == 2 for lane in counts), counts
        for lane, directory in enumerate(interrupted):
            for ordinal, path in enumerate(outputs(directory)):
                results["parity"].append({"point": point, "lane": lane, "file": ordinal,
                                           **canonical(args, path, references / f"{point}-{lane}-{ordinal}")})
        before = snapshot(interrupted)
        idle = run([*recovery_command, "--resume"])
        assert metrics(idle)["generated_members"] == 0 and snapshot(interrupted) == before
        results["recovery"].append({"point": point, "members": metrics(recovered)["members"]})
    print("PASS replicated capacity-extent interruption recovery and ordinary parity", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grow", type=Path)
    parser.add_argument("plotter", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--backend", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--max-ram", type=int, default=536870912)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--replacement-grow", type=Path,
                        help="also check a compatible reference build with a different source identity")
    args = parser.parse_args()
    for name in ("grow", "plotter", "reference"):
        setattr(args, name, getattr(args, name).resolve())
    if args.replacement_grow:
        args.replacement_grow = args.replacement_grow.resolve()
    if args.work_dir:
        directory = args.work_dir.resolve()
        directory.mkdir(mode=0o700)
        results = check(args, directory)
        destination = args.results or directory / "results.json"
        destination.write_text(json.dumps(results, indent=2) + "\n")
    else:
        with tempfile.TemporaryDirectory(prefix="xchplot2-grow-farming-") as temporary:
            results = check(args, Path(temporary))
        if args.results:
            args.results.write_text(json.dumps(results, indent=2) + "\n")
    print("directory farming plans, ordinary parity, bounds and recovery checks passed")


if __name__ == "__main__":
    main()
