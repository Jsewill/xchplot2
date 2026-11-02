#!/usr/bin/env python3
"""Key-derived rollover, aggregate capacity, private outputs and durable resume."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


# Public BLS12-381 generator and disposable deterministic test keys, as in the
# existing group and CLI recovery checks. No user key material is needed.
FARMER = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
SEED = "aa" * 32
POOL = "42" * 32
HEADER = struct.Struct("<4sB32sBBHBQB")


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def run(command, code=0, point=None, **kwargs):
    env = dict(os.environ)
    env.pop("XCHPLOT2_GROW_FAILPOINT", None)
    if point:
        env["XCHPLOT2_GROW_FAILPOINT"] = point
    result = subprocess.run([str(arg) for arg in command], capture_output=True,
                            text=True, env=env, **kwargs)
    assert result.returncode == code, (result.returncode, result.stdout, result.stderr)
    assert SEED not in result.stdout + result.stderr, "private seed exposed in output"
    return result


def report(result):
    return json.loads(result.stdout.splitlines()[-1])


def outputs(directory):
    return sorted(directory.glob("grow-group-*.gplot"))


def snapshot(directory):
    return {path.name: (digest(path), path.stat().st_size, path.stat().st_mtime_ns,
                        path.stat().st_ino) for path in outputs(directory)}


def clone(source, target):
    target.mkdir(mode=0o700)
    for path in outputs(source):
        # Reflinks would exercise the shared-extent guard rather than rollover.
        copy = target / path.name
        with path.open("rb") as old, copy.open("xb") as new:
            shutil.copyfileobj(old, new, 1048576)
            new.flush()
            os.fsync(new.fileno())
        copy.chmod(0o600)
    return target


def check_memberships(args, directory):
    counts = []
    ids = []
    for ordinal, path in enumerate(outputs(directory)):
        assert path.name == f"grow-group-{ordinal:020d}.gplot"
        with path.open("rb") as file:
            magic, version, group, k, strength, count, meta, index, memo_size = HEADER.unpack(
                file.read(HEADER.size))
            memo = file.read(memo_size)
        assert (magic, version, k, strength, meta) == (b"PoS2", 2, 18, 2, 7)
        assert 0 < count <= 2 and HEADER.size + memo_size < index < path.stat().st_size
        # Match the production C ABI's documented base_seed || idx_le_u64
        # derivation, then ask its existing group tool for the identity.
        subseed = hashlib.sha256(bytes.fromhex(SEED) + ordinal.to_bytes(8, "little")).hexdigest()
        expected = run([args.reference, "prepare", 18, 2, 7, count],
                       input=f"{subseed} {FARMER} {POOL}\n").stdout.splitlines()[0].split()
        assert [group.hex(), memo.hex()] == expected, "rollover identity differs from key derivation"
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.stat().st_nlink == 1
        ids.append(group)
        counts.append(count)
    assert len(set(ids)) == len(ids), "rollover duplicated a group identity"
    assert set(directory.iterdir()) == set(outputs(directory)), "rollover created external state"
    return counts


def check_bounded(args, metrics):
    assert metrics["rollover"] is True
    assert metrics["accounted_file_bytes"] <= metrics["capacity_bytes"], metrics
    assert metrics["peak_file_bytes"] <= metrics["capacity_bytes"], metrics
    assert metrics["peak_modeled_bytes"] <= args.max_ram, metrics
    if args.backend == "cpu":
        assert metrics["generation_overlap_batches"] == 0, metrics


def check_canonical(args, path, work):
    with path.open("rb") as file:
        _, _, group, _, _, count, _, _, memo_size = HEADER.unpack(file.read(HEADER.size))
        memo = file.read(memo_size).hex()
    work.mkdir(mode=0o700)
    raw = []
    for member in range(count):
        name = f"{member}.plot2"
        run([args.plotter, "test", 18, group.hex(), 2, member, 7, 0,
             "--raw", "--out", work, "--out-name", name, "--memo", memo,
             "--config", os.devnull])
        raw.append(work / name)
    inputs = work / "group.inputs"
    inputs.write_text(f"18 2 7 {group.hex()} {memo}\n" +
                      "".join(json.dumps(str(p)) + "\n" for p in raw))
    reference = work / "reference.gplot"
    run([args.reference, "assemble-only", inputs, reference, 512])
    assert digest(path) == digest(reference), "rollover bytes differ from normal assembly"
    proof = run([args.plotter, "verify", path, "--full", "--trials", 32,
                 "--config", os.devnull])
    assert "full proofs validated" in (proof.stdout + proof.stderr).lower()
    (work / "proof.log").write_text(proof.stdout + proof.stderr)
    return count, digest(path)


def check_pending_pairs(args, root):
    common = [args.grow, "--group-members", 2, "--seed", SEED,
              "--farmer-pk", FARMER, "--pool-ph", POOL, "--k", 18,
              "--strength", 2, "--meta-group", 7, "--backend", "cpu",
              "--batch-members", 2, "--workers", 2, "--max-ram", args.max_ram,
              "--capacity-bytes", 67108864, "--construction", "external"]
    root.mkdir(mode=0o700)

    def stage_next(directory, scratch, ordinal, point, identity_ordinal=None):
        index = ordinal if identity_ordinal is None else identity_ordinal
        subseed = hashlib.sha256(bytes.fromhex(SEED) + index.to_bytes(8, "little")).hexdigest()
        identity = run([args.reference, "prepare", 18, 2, 7, 1],
                       input=f"{subseed} {FARMER} {POOL}\n").stdout.splitlines()[0].split()
        run([args.grow, directory / f"grow-group-{ordinal:020d}.gplot", "--group-id", identity[0],
             "--memo", identity[1], "--k", 18, "--strength", 2, "--meta-group", 7,
             "--max-members", 1, "--batch-members", 1, "--construction", "external",
             "--temp-dir", scratch, "--backend", "cpu", "--workers", 1,
             "--capacity-bytes", 33554432, "--max-ram", args.max_ram], code=86, point=point)

    def saved(case, members=2, next_point="external-run-after-checkpoint", wrong=False, third=False):
        work = root / case
        work.mkdir(mode=0o700)
        directory = work / "outputs"
        directory.mkdir(mode=0o700)
        current = work / "current-scratch"
        current.mkdir(mode=0o700)
        future = work / "next-scratch"
        future.mkdir(mode=0o700)
        run([*common, directory, "--max-total-members", members, "--temp-dir", current],
            code=86, point="external-run-after-checkpoint")
        stage_next(directory, future, 1, next_point, 9 if wrong else None)
        if third:
            extra = work / "third-scratch"
            extra.mkdir(mode=0o700)
            stage_next(directory, extra, 2, "external-run-after-checkpoint")
        return directory, current, future

    def tree(work):
        return {str(path.relative_to(work)): (digest(path), path.stat().st_size,
                                             path.stat().st_mtime_ns, path.stat().st_ino)
                for path in work.rglob("*") if path.is_file()}

    reference = root / "reference"
    reference.mkdir(mode=0o700)
    run([*common, reference, "--max-total-members", 4, "--construction", "incremental"])
    expected = [digest(path) for path in outputs(reference)]
    checked = []
    for case, next_point, acknowledged in (
            ("partial", "external-run-after-checkpoint", 1),
            ("empty", "external-initialized-after-checkpoint", 0)):
        directory, current, future = saved(case, next_point=next_point)
        result = report(run([*common, directory, "--resume", "--max-total-members", 4,
                             "--temp-dir", current, "--temp-dir", future, "--workers", 1]))
        assert result["generated_members"] == 2 - acknowledged, result
        assert result["groups"] == 2 and result["members"] == 4 and result["recovered"], result
        assert result["peak_modeled_bytes"] <= args.max_ram, result
        assert result["accounted_file_bytes"] <= result["capacity_bytes"], result
        assert check_memberships(args, directory) == [2, 2]
        assert [digest(path) for path in outputs(directory)] == expected
        checked.append(case + " adjacent durable successor resumes after full pending prefix")
    for case, options in (
            ("incomplete-prior", {"members": 1}),
            ("packing-next", {"next_point": "external-pack-initialized-after-checkpoint"}),
            ("wrong-next-identity", {"wrong": True}),
            ("third-pending", {"third": True})):
        directory, current, future = saved(case, **options)
        before = tree(directory.parent)
        run([*common, directory, "--resume", "--max-total-members", 6,
             "--temp-dir", current, "--temp-dir", future], code=1)
        assert tree(directory.parent) == before, "invalid pending pair changed acknowledged outputs or scratch"
        checked.append(case + " rejected before mutation")
    return checked


def check(args, root):
    common = [args.grow, "--group-members", 2, "--seed", SEED,
              "--farmer-pk", FARMER, "--pool-ph", POOL, "--k", 18,
              "--strength", 2, "--meta-group", 7, "--backend", args.backend,
              "--batch-members", 2, "--workers", 2, "--max-ram", args.max_ram,
              "--capacity-bytes", 67108864]
    results = {"backend": args.backend, "parity": [], "recovery": [], "rejections": []}

    def memberships(directory):
        return check_memberships(args, directory)

    def bounded(metrics):
        check_bounded(args, metrics)

    main = root / "partial-then-resume"
    main.mkdir(mode=0o700)
    first = report(run([*common, main, "--max-total-members", 3]))
    bounded(first)
    assert memberships(main) == [2, 1]
    preserved = snapshot(main)
    resumed = report(run([*common, main, "--max-total-members", 5, "--resume"]))
    bounded(resumed)
    assert memberships(main) == [2, 2, 1]
    assert snapshot(main)[outputs(main)[0].name] == preserved[outputs(main)[0].name]
    before = snapshot(main)
    repeat = report(run([*common, main, "--max-total-members", 5, "--resume"]))
    bounded(repeat)
    assert snapshot(main) == before, "completed resume changed output"
    idle = report(run([*common, main, "--max-total-members", 5, "--resume",
                       "--backend", "gpu", "--device", 2147483647]))
    assert idle["mode"] == "idle" and idle["generated_members"] == 0, idle
    assert idle["read_bytes"] > 0 and idle["written_bytes"] == 0, "read-only rollover audit is missing from I/O metrics"
    assert snapshot(main) == before, "unavailable-GPU no-op changed output"
    results["recovery"].extend(["partial last group extends", "completed groups preserved", "no-op resume"])
    results["recovery"].append("completed resume needs no GPU")

    # Compare each completed ordinary group to the existing raw-member writer
    # and direct assembler, then exercise the unmodified reader with full proofs.
    references = root / "references"
    references.mkdir(mode=0o700)
    for ordinal, path in enumerate(outputs(main)):
        count, sha256 = check_canonical(args, path, references / str(ordinal))
        results["parity"].append({"group": ordinal, "members": count, "sha256": sha256})
    print("PASS rollover key derivation, byte parity and full proofs", flush=True)

    # Exact aggregate capacity cannot admit another member and must preserve
    # already completed output, including a partial final file.
    limited = clone(main, root / "capacity")
    cap = sum(max(p.stat().st_size, p.stat().st_blocks * 512) for p in outputs(limited))
    before = snapshot(limited)
    stopped = report(run([*common, limited, "--resume", "--max-total-members", 6,
                          "--capacity-bytes", cap]))
    assert stopped["stop_reason"] == "capacity" and stopped["members"] == 5, stopped
    bounded(stopped)
    assert snapshot(limited) == before and memberships(limited) == [2, 2, 1]
    results["rejections"].append("aggregate capacity preserves every output")

    # Invalid requests must be rejected before any existing output changes.
    before = snapshot(main)
    for options in [["--seed", "bb" * 32], ["--meta-group", 8],
                    ["--group-members", 3],
                    ["--max-ram", 16777216, "--window-bytes", 33554432],
                    ["--max-total-members", 4]]:
        run([*common, main, "--resume", *options], 1)
        assert snapshot(main) == before, "rejected rollover request changed output"
    run([*common, main], 1)
    directory = os.open(main, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run([*common, main, "--resume", "--max-total-members", 6], 1)
    finally:
        os.close(directory)
    assert snapshot(main) == before
    results["rejections"].extend(["wrong identity", "RAM", "member limit", "occupied output", "directory lock"])

    hole = clone(main, root / "hole")
    outputs(hole)[1].unlink()
    before = snapshot(hole)
    run([*common, hole, "--resume", "--max-total-members", 6], 1)
    assert snapshot(hole) == before
    results["rejections"].append("missing group ordinal")

    linked = clone(main, root / "symlink")
    first_path = outputs(linked)[0]
    first_path.unlink()
    first_path.symlink_to(outputs(main)[0])
    before = snapshot(main)
    run([*common, linked, "--resume", "--max-total-members", 6], 1)
    assert first_path.is_symlink() and snapshot(main) == before
    results["rejections"].append("symlink target preserved")

    group_limit = root / "group-limit"
    group_limit.mkdir(mode=0o700)
    limited_groups = report(run([*common, group_limit, "--max-groups", 1,
                                  "--max-total-members", 3]))
    bounded(limited_groups)
    assert limited_groups["groups"] == 1 and limited_groups["members"] == 2
    assert memberships(group_limit) == [2]
    completed = outputs(group_limit)[0]
    before = digest(completed), completed.stat().st_mtime_ns
    pending_command = [*common, group_limit, "--resume", "--max-groups", 2,
                       "--max-total-members", 4]
    run(pending_command, 86, "direct-target-partial")
    assert (digest(completed), completed.stat().st_mtime_ns) == before
    resumed_group = report(run(pending_command))
    bounded(resumed_group)
    assert memberships(group_limit) == [2, 2]
    assert (digest(completed), completed.stat().st_mtime_ns) == before
    assert all(digest(left) == digest(right) for left, right in zip(outputs(group_limit), outputs(main)))
    results["recovery"].append("pending later group resumes after file limit is raised")

    interrupted = root / "interrupted"
    interrupted.mkdir(mode=0o700)
    command = [*common, interrupted, "--max-groups", 2, "--max-total-members", 4]
    run(command, 86, "published:2")
    completed = outputs(interrupted)[0]
    before = digest(completed), completed.stat().st_mtime_ns
    recovered = report(run([*command, "--resume"]))
    bounded(recovered)
    assert memberships(interrupted) == [2, 2]
    assert (digest(completed), completed.stat().st_mtime_ns) == before
    assert all(digest(left) == digest(right) for left, right in zip(outputs(interrupted), outputs(main)))
    results["recovery"].append("interruption during later group publication")

    # Finishing an already encoded payload must not initialize a producer.
    finalizing = root / "payload-complete"
    finalizing.mkdir(mode=0o700)
    command = [*common, finalizing, "--max-groups", 1, "--max-total-members", 2]
    run(command, 86, "payload")
    finalized = report(run([*command, "--resume", "--backend", "gpu",
                            "--device", 2147483647]))
    bounded(finalized)
    assert finalized["mode"] == "idle" and finalized["generated_members"] == 0, finalized
    assert finalized["encode_chunks"] == 0, finalized
    assert memberships(finalizing) == [2]
    assert digest(outputs(finalizing)[0]) == digest(outputs(main)[0])
    results["recovery"].append("payload-complete recovery needs no GPU")
    results["recovery"].extend(check_pending_pairs(args, root / "pending-pairs"))
    print("PASS rollover limits, rejection preservation and interrupted resume", flush=True)
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")


def check_mode(args, root, mode):
    construction, storage = mode.split("-", 1)
    common = [args.grow, "--group-members", 2, "--seed", SEED,
              "--farmer-pk", FARMER, "--pool-ph", POOL, "--k", 18,
              "--strength", 2, "--meta-group", 7, "--backend", args.backend,
              "--batch-members", 1, "--workers", 2, "--max-ram", args.max_ram,
              "--capacity-bytes", 67108864, "--construction", construction,
              "--member-storage", storage]
    if construction == "merged":
        common += ["--window-bytes", 65536, "--checkpoint-bytes", 1048576]
    results = {"mode": mode, "backend": args.backend, "parity": [], "recovery": [], "rejections": []}
    main = root / "complete"
    main.mkdir(mode=0o700)
    complete = report(run([*common, main, "--max-total-members", 4]))
    check_bounded(args, complete)
    assert complete["staged_batches"] > 0 and complete["staging_fallbacks"] == 0, complete
    assert complete["member_storage"] == storage and complete["construction_requested"] == construction, complete
    assert check_memberships(args, main) == [2, 2]
    expected = {path.name: digest(path) for path in outputs(main)}
    references = root / "references"
    references.mkdir(mode=0o700)
    for ordinal, path in enumerate(outputs(main)):
        count, sha256 = check_canonical(args, path, references / str(ordinal))
        results["parity"].append({"group": ordinal, "members": count, "sha256": sha256})

    # Reuse a completed prior group for every interrupted second-group case.
    # The exact total limit is already reached by committed staged coverage;
    # publication must still run, without creating or regenerating any member.
    prior = root / "prior"
    clone(main, prior)
    outputs(prior)[1].unlink()
    points = [("stage-run-flushed:2", 1), ("stage-run-committed:2", 0),
              ("stage-pack-committed", 0), ("stage-compact-partial", 0),
              ("stage-final-header-partial", 0), ("stage-retired", 0),
              ("stage-published", 0)]
    if construction == "merged":
        points = [("merge-run-flushed:2", 1), ("merge-run-committed:2", 0),
                  ("merge-pack-committed", 0),
                  ("merge-layout-partial", 0), ("merge-final-header-partial", 0),
                  ("merge-retired", 0), ("merge-published", 0)]
    for point, generated in points:
        directory = clone(prior, root / ("recovery-" + point.replace(":", "-")))
        preserved = snapshot(directory)
        command = [*common, directory, "--resume", "--max-total-members", 4]
        run(command, 86, point)
        assert snapshot(directory)[outputs(directory)[0].name] == preserved[outputs(directory)[0].name]
        resume = [*command, "--construction", "incremental"]
        if not generated:
            resume += ["--backend", "gpu", "--device", 2147483647]
        recovered = report(run(resume))
        check_bounded(args, recovered)
        assert recovered["generated_members"] == generated and recovered["members"] == 4, recovered
        if not generated:
            assert recovered["mode"] == "idle", recovered
        assert check_memberships(args, directory) == [2, 2]
        assert snapshot(directory)[outputs(directory)[0].name] == preserved[outputs(directory)[0].name]
        assert {path.name: digest(path) for path in outputs(directory)} == expected, point
        before = snapshot(directory)
        repeated = report(run(resume))
        assert repeated["generated_members"] == 0 and snapshot(directory) == before, repeated
        results["recovery"].append({"point": point, "generated_members": generated})

    # Leave the final group partial, then allow ordinary exact growth while
    # refusing the larger simultaneous compressed-run/canonical-tail workspace.
    partial = clone(prior, root / "partial")
    report(run([*common, partial, "--resume", "--max-total-members", 3,
                "--construction", "incremental"]))
    preserved = snapshot(partial)
    cap = sum(max(path.stat().st_size, path.stat().st_blocks * 512) for path in outputs(main)) + 1048576
    fallback = report(run([*common, partial, "--resume", "--max-total-members", 4,
                           "--capacity-bytes", cap]))
    check_bounded(args, fallback)
    if construction == "staged":
        assert fallback["staging_fallbacks"] > 0 and fallback["members"] == 4, fallback
        results["recovery"].append("aggregate capacity falls back to exact growth")
    else:
        assert fallback["stop_reason"] == "capacity" and fallback["members"] == 3, fallback
        assert snapshot(partial) == preserved, "merged capacity rejection changed ordinary prefix"
        restored = report(run([*common, partial, "--resume", "--max-total-members", 4]))
        assert restored["members"] == 4, restored
        results["recovery"].append("merged aggregate capacity preserves prefix; larger cap extends it")
    assert check_memberships(args, partial) == [2, 2]
    assert snapshot(partial)[outputs(partial)[0].name] == preserved[outputs(partial)[0].name]
    assert {path.name: digest(path) for path in outputs(partial)} == expected

    cap = sum(max(path.stat().st_size, path.stat().st_blocks * 512) for path in outputs(partial))
    before = snapshot(partial)
    stopped = report(run([*common, partial, "--resume", "--max-total-members", 5,
                          "--capacity-bytes", cap]))
    check_bounded(args, stopped)
    assert stopped["stop_reason"] == "capacity" and stopped["members"] == 4, stopped
    assert check_memberships(args, partial) == [2, 2] and snapshot(partial) == before
    results["rejections"].append("exact aggregate capacity preserves completed groups")
    print(f"PASS {mode} rollover parity, exact-limit publication, persisted recovery and capacity checks", flush=True)
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")


def checks(args, root):
    if args.modes == ["default"]:
        check(args, root)
        return
    for mode in args.modes:
        directory = root / mode
        directory.mkdir(mode=0o700)
        if mode == "default":
            check(args, directory)
        else:
            check_mode(args, directory, mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grow", type=Path)
    parser.add_argument("plotter", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--max-ram", type=int, default=536870912)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--modes", nargs="+", choices=("default", "staged-raw", "staged-compact", "merged-raw", "merged-compact"),
                        default=["default"], help="opt-in retained-member and same-file staging integration checks")
    args = parser.parse_args()
    for name in ("grow", "plotter", "reference"):
        setattr(args, name, getattr(args, name).resolve())
    if args.work_dir:
        args.work_dir = args.work_dir.resolve()
        args.work_dir.mkdir(mode=0o700)
        checks(args, args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="xchplot2-grow-rollover-") as temporary:
            checks(args, Path(temporary))


if __name__ == "__main__":
    main()
