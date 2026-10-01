#!/usr/bin/env python3
"""GPU correctness, production VRAM boundaries, and physical small-card checks."""

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import filecmp
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from release import extract_archive

MIB = 1 << 20
TIERS = ("tiny", "pinned", "minimal", "compact", "plain")
MASKS = {"cuda": "cuda", "hip": "hip", "level_zero": "ze"}


def plot_vectors(k):
    # The legacy file header cannot select testnet parameters for `verify`.
    # Keep full-plot vectors mainnet; existing CTest kernels cover testnet.
    return [dict(name="baseline", k=k, strength=2, plot_index=0, meta_group=0,
                 testnet=False, plot_id="ab" * 32, memo="00" * 112),
            dict(name="index-meta-max", k=18, strength=3, plot_index=65535, meta_group=255,
                 testnet=False, plot_id="ff" * 32, memo=bytes(range(255)).hex()),
            dict(name="strength4", k=18, strength=4, plot_index=1, meta_group=1,
                 testnet=False, plot_id=bytes(range(32)).hex(), memo="00" * 112)]


def gpu_process_snapshot(logs, label, backend):
    # Optional evidence, not a memory attribution or a qualification gate.
    if backend == "cuda" and shutil.which("nvidia-smi"):
        try:
            with (logs / f"nvidia-smi-{label}.log").open("w") as log:
                subprocess.run(["nvidia-smi", "-q", "-x"], stdout=log,
                               stderr=subprocess.STDOUT, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"GPU process snapshot unavailable: {error}", flush=True)


@contextmanager
def qualification_result(logs, summary):
    def save():
        (logs / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save()
    try:
        yield
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        summary["status"] = "failed"
        summary["failure"] = dict(type=type(error).__name__, message=str(error))
        if "vector" in summary:
            summary["failure"]["vector"] = summary["vector"]
            for vector in summary["vectors"]:
                if vector["name"] == summary["vector"]:
                    vector["status"] = "failed"
        if isinstance(error, subprocess.CalledProcessError):
            summary["failure"].update(command=error.cmd, exit_code=error.returncode)
        gpu_process_snapshot(logs, "failure", summary["backend"])
        raise
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


def test_environment(source, backend, max_host_ram_gib=None, strict_memory=True):
    # Keep driver/toolchain paths, but remove plotting overrides and bypasses.
    env = {k: v for k, v in source.items()
           if not k.startswith(("POS2GPU_", "XCHPLOT2_"))}
    if "POS2GPU_VRAM_MARGIN_MB" in source:
        env["POS2GPU_VRAM_MARGIN_MB"] = source["POS2GPU_VRAM_MARGIN_MB"]
    if max_host_ram_gib is not None:
        env["XCHPLOT2_MAX_HOST_RAM"] = f"{max_host_ram_gib}G"
    env.update(ACPP_VISIBILITY_MASK=MASKS[backend], POS2GPU_ASSERT_VRAM="1" if strict_memory else "0",
               POS2GPU_STREAMING_STATS="1")
    return env


def parse_inventory(text, backend):
    info = dict(line.split("=", 1) for line in text.splitlines())
    if info.pop("backend") != backend:
        raise ValueError("GPU backend does not match this CI lane")
    spill_tiers = info.pop("spill_tiers").split(",")
    info = {k: int(v) for k, v in info.items()}
    required = {"total_bytes", "free_bytes", "margin_bytes", "tiny", "minimal", "compact", "plain"}
    if not required <= info.keys() or info.keys() - required - {"pinned"}:
        raise ValueError("Incomplete or unexpected GPU inventory")
    if any(v <= 0 for v in info.values()) or info["free_bytes"] > info["total_bytes"]:
        raise ValueError("Invalid GPU memory inventory")
    if info["margin_bytes"] % MIB:
        raise ValueError("VRAM margin must be a whole number of MiB")
    if not set(spill_tiers) <= info.keys() & (set(TIERS) - {"plain"}):
        raise ValueError("Invalid spill tier inventory")
    info["spill_tiers"] = spill_tiers
    return info


def tier_caps(info, physical_mib=0):
    if physical_mib:
        if physical_mib not in (2048, 4096, 6144, 8192):
            raise ValueError("Physical lanes require a 2, 4, 6, or 8 GiB card")
        # Allow driver-reserved memory; a capped large card cannot pass this.
        if not physical_mib * MIB * 7 // 8 <= info["total_bytes"] <= (physical_mib + 64) * MIB:
            raise ValueError("Actual GPU capacity does not match the physical CI lane")
    caps = {tier: (info[tier] + MIB - 1) // MIB + info["margin_bytes"] // MIB
            for tier in TIERS if tier in info}
    fits = {tier: cap for tier, cap in caps.items() if cap * MIB <= info["free_bytes"]}
    if not fits or (not physical_mib and fits != caps):
        raise ValueError("Insufficient physical free VRAM for the requested k=28 tiers")
    return fits


def archive_binary(archive, build, work):
    package, digest = extract_archive(archive, work)
    buildinfo = (package / "BUILDINFO.txt").read_bytes()
    if not buildinfo or (build / "BUILDINFO.txt").read_bytes() != buildinfo:
        raise ValueError("Archive BUILDINFO.txt does not match the inventory/parity build")
    binary = package / "bin/xchplot2"
    if not binary.is_file():
        raise ValueError("Archive has no Linux bin/xchplot2 executable")
    metadata = dict(name=archive.name, sha256=digest, buildinfo=buildinfo.decode("utf-8"),
                    started_at=datetime.now(timezone.utc).isoformat(), platform=platform.platform())
    return binary, metadata


def archive_environment(source, binary):
    runtime = str(binary.parent.parent / "lib")
    return dict(source, LD_LIBRARY_PATH=runtime + (os.pathsep + source["LD_LIBRARY_PATH"]
                                                  if source.get("LD_LIBRARY_PATH") else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    artifact = parser.add_mutually_exclusive_group()
    artifact.add_argument("--binary", type=Path, help="Test an extracted release executable")
    artifact.add_argument("--archive", type=Path, help="Qualify an archive using its matching packaged build")
    parser.add_argument("--backend", choices=MASKS, required=True)
    parser.add_argument("--suite", choices=("quick", "correctness", "vram", "physical"), default="quick")
    parser.add_argument("--physical-vram-mib", type=int, default=0)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path.cwd(),
                        help="Existing directory on real disk for plots and spill files")
    parser.add_argument("--max-host-ram-gib", type=int,
                        help="Explicit host RAM budget for boundary and normal tier plots; spill cases use min")
    args = parser.parse_args()
    if args.max_host_ram_gib is not None and not 0 < args.max_host_ram_gib < 2**34:
        parser.error("--max-host-ram-gib must be a positive integer below 2**34")
    if (args.suite == "physical") != bool(args.physical_vram_mib):
        parser.error("--physical-vram-mib is required only for the physical suite")
    build, logs = args.build.resolve(), args.logs.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    strict_memory = args.suite in ("vram", "physical")
    summary = dict(backend=args.backend, suite=args.suite, status="running", stage="setup",
                   memory_assertion=strict_memory,
                   started_at=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                   build=str(build),
                   host_ram_policy=dict(max_host_ram_gib=args.max_host_ram_gib, spill_cases="min"),
                   memory_measurement="watchdog samples device-wide free-memory deltas; other processes can contribute")
    with qualification_result(logs, summary):
        env = test_environment(os.environ, args.backend, args.max_host_ram_gib, strict_memory)
        scratch = args.scratch.resolve()
        if not scratch.is_dir():
            raise ValueError("--scratch must be an existing directory on real disk")
        env["TMPDIR"] = str(scratch)
        binary = args.binary.resolve() if args.binary else build / "tools/xchplot2/xchplot2"

        def run(label, command, run_env=None, cwd=None):
            summary["stage"] = label
            summary["log"] = f"{label}.log"
            print(f"Running {label}", flush=True)
            with (logs / f"{label}.log").open("w") as log:
                subprocess.run([str(v) for v in command], env=run_env or env, cwd=cwd,
                               stdout=log, stderr=subprocess.STDOUT, check=True)

        def verify(label, plot):
            run(label, [binary, "verify", plot, "--full", "--trials", "100", "--config", "/dev/null"])
            summary["stage"] = f"{label}-proof-count"
            counts = re.findall(r"^\[verify\] ([0-9]+) full proofs validated$",
                                (logs / f"{label}.log").read_text(), re.MULTILINE)
            if len(counts) != 1 or int(counts[0]) == 0:
                raise RuntimeError(f"{label}: missing or zero validated full proofs")
            return int(counts[0])

        # ponytail: one GPU suite per host; use per-device locks for a larger fleet.
        summary["stage"] = "gpu-lock"
        lock = os.open("/tmp/xchplot2-gpu-ci.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock, "w"), (tempfile.TemporaryDirectory(prefix=".gpu-archive-", dir=scratch)
                                   if args.archive else nullcontext()) as archive_directory:
            if args.archive:
                summary["stage"] = "archive"
                summary["archive"] = dict(name=args.archive.name, path=str(args.archive.resolve()))
                binary, archive_metadata = archive_binary(args.archive.resolve(), build, Path(archive_directory))
                env = archive_environment(env, binary)
                summary["archive"] = archive_metadata
            print("Waiting for exclusive GPU test access", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            gpu_process_snapshot(logs, "start", args.backend)
            summary["stage"] = "inventory"
            summary["log"] = "inventory.log"
            with (logs / "inventory.log").open("w") as log:
                result = subprocess.run([str(build / "tools/sanity/gpu_ci_info"), args.backend],
                                        env=env, text=True, stdout=subprocess.PIPE, stderr=log, check=False)
            (logs / "inventory.txt").write_text(result.stdout)
            result.check_returncode()
            summary["stage"] = "inventory-validation"
            info = parse_inventory(result.stdout, args.backend)
            summary["inventory"] = info
            summary["stage"] = "capacity-validation"
            caps = tier_caps(info, args.physical_vram_mib) if args.suite != "quick" else {}
            tiers = list(caps) if caps else [tier for tier in TIERS if tier in info]
            k = 18 if args.suite == "quick" else 28
            summary.update(k=k, tiers=tiers)
            summary["vectors"] = [dict(vector, status="pending") for vector in plot_vectors(k)]
            (logs / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            run("devices", [binary, "devices", "--config", "/dev/null"])
            run("ctest", ["ctest", "--test-dir", build, "--output-on-failure", "--no-tests=error",
                          "--parallel", "1", "--timeout", "900", "--output-junit", logs / "ctest.xml"])
            if caps and strict_memory:
                specs = [f"{tier}:{cap - info['margin_bytes'] // MIB}" for tier, cap in caps.items()]
                run("vram-boundaries", [Path(__file__).with_name("vram-tiers.sh"), binary, "0", *specs],
                    dict(env, XCHPLOT2_TEST_LOG_DIR=str(logs / "boundaries")))

            with tempfile.TemporaryDirectory(prefix=".gpu-ci-", dir=scratch) as directory:
                work = Path(directory)
                for vector in summary["vectors"]:
                    summary["vector"] = vector["name"]
                    vector["status"] = "running"
                    prefix = "" if vector["name"] == "baseline" else vector["name"] + "-"
                    run(f"{prefix}cpu-reference",
                        [binary, "test", vector["k"], vector["plot_id"], vector["strength"],
                         vector["plot_index"], vector["meta_group"], "-m", vector["memo"],
                         "-o", work, "-N", "reference.plot2", "--config", "/dev/null"])
                    reference = work / "reference.plot2"
                    summary["stage"] = f"{prefix}cpu-reference-hash"
                    with reference.open("rb") as source:
                        vector["reference_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
                    if vector["name"] == "baseline":
                        summary["reference_sha256"] = vector["reference_sha256"]
                    vector["full_proofs_validated"] = {}
                    (logs / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                    for tier in tiers:
                        for spill in (False, True) if tier in info["spill_tiers"] else (False,):
                            label = prefix + tier + ("-disk" if spill else "")
                            plot = work / f"{label}.plot2"
                            manifest = work / "manifest.tsv"
                            manifest.write_text(
                                f"{vector['k']} {vector['strength']} {vector['plot_index']} "
                                f"{vector['meta_group']} 0 {vector['plot_id']} {vector['memo']} . {plot.name}\n")
                            command = [binary, "batch", manifest, "--devices", "0", "--tier", tier,
                                       "--no-progress", "--config", "/dev/null"]
                            if spill:
                                command += ["--max-host-ram", "min", "--temp-dir", work]
                            plot_env = (dict(env, POS2GPU_MAX_VRAM_MB=str(caps[tier]))
                                        if args.suite in ("correctness", "vram") else env)
                            run(label, command, plot_env, cwd=work)
                            summary["stage"] = f"{label}-parity"
                            trace = (logs / f"{label}.log").read_text()
                            if f"streaming tier: {tier} (" not in trace:
                                raise RuntimeError(f"{label}: requested GPU tier did not run")
                            if spill and "-> disk" not in trace and "-> mmap" not in trace:
                                raise RuntimeError(f"{label}: no disk spill was exercised")
                            if not filecmp.cmp(reference, plot, shallow=False):
                                with plot.open("rb") as source:
                                    actual = hashlib.file_digest(source, "sha256").hexdigest()
                                raise RuntimeError(
                                    f"{label}: plot bytes differ from the CPU reference; "
                                    f"expected {reference.stat().st_size} bytes, SHA256 {vector['reference_sha256']}; "
                                    f"got {plot.stat().st_size} bytes, SHA256 {actual}")
                            vector["full_proofs_validated"][label] = verify(f"{label}-proofs", plot)
                            plot.unlink()
                    reference.unlink()
                    vector["status"] = "passed"
                summary.pop("vector")
                if args.suite == "physical":
                    # No software cap: exercise automatic selection on the real card.
                    run("physical-auto", [binary, "bench", "--devices", "0", "-k", "28", "-n", "3",
                                          "--warmup", "0", "--keep", "--out", work, "--config", "/dev/null"])
                    summary["stage"] = "physical-auto-plots"
                    plots = sorted(work.glob("bench-*.plot2"))
                    if len(plots) != 3:
                        raise RuntimeError("Physical auto-tier run did not produce three plots")
                    for i, plot in enumerate(plots):
                        verify(f"physical-auto-proofs-{i}", plot)
            summary.update(status="passed", stage="complete")
            summary.pop("log", None)
            (logs / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"PASS: {args.backend} {args.suite}; {len(tiers)} tiers, "
                  f"{len(summary['vectors'])} vectors, CPU byte parity and full proofs", flush=True)


if __name__ == "__main__":
    main()
