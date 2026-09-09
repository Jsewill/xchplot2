#!/usr/bin/env python3
"""GPU correctness, production VRAM boundaries, and physical small-card checks."""

import argparse
import fcntl
import filecmp
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

MIB = 1 << 20
TIERS = ("tiny", "pinned", "minimal", "compact", "plain")
MASKS = {"cuda": "cuda", "hip": "hip", "level_zero": "ze"}


def test_environment(source, backend):
    # Keep driver/toolchain paths, but remove plotting overrides and bypasses.
    env = {k: v for k, v in source.items()
           if not k.startswith(("POS2GPU_", "XCHPLOT2_"))}
    if "POS2GPU_VRAM_MARGIN_MB" in source:
        env["POS2GPU_VRAM_MARGIN_MB"] = source["POS2GPU_VRAM_MARGIN_MB"]
    env.update(ACPP_VISIBILITY_MASK=MASKS[backend], POS2GPU_ASSERT_VRAM="1",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("--backend", choices=MASKS, required=True)
    parser.add_argument("--suite", choices=("quick", "vram", "physical"), default="quick")
    parser.add_argument("--physical-vram-mib", type=int, default=0)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path.cwd(),
                        help="Existing directory on real disk for plots and spill files")
    args = parser.parse_args()
    if (args.suite == "physical") != bool(args.physical_vram_mib):
        parser.error("--physical-vram-mib is required only for the physical suite")
    build, logs = args.build.resolve(), args.logs.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    env = test_environment(os.environ, args.backend)
    scratch = args.scratch.resolve()
    if not scratch.is_dir():
        parser.error("--scratch must be an existing directory on real disk")
    env["TMPDIR"] = str(scratch)
    binary = build / "tools/xchplot2/xchplot2"

    def run(label, command, run_env=None, cwd=None):
        print(f"Running {label}", flush=True)
        with (logs / f"{label}.log").open("w") as log:
            subprocess.run([str(v) for v in command], env=run_env or env, cwd=cwd,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    # ponytail: one GPU suite per host; use per-device locks for a larger fleet.
    lock = os.open("/tmp/xchplot2-gpu-ci.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock, "w"):
        print("Waiting for exclusive GPU test access", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (logs / "inventory.log").open("w") as log:
            result = subprocess.run([str(build / "tools/sanity/gpu_ci_info"), args.backend],
                                    env=env, text=True, stdout=subprocess.PIPE, stderr=log, check=True)
        (logs / "inventory.txt").write_text(result.stdout)
        info = parse_inventory(result.stdout, args.backend)
        caps = tier_caps(info, args.physical_vram_mib) if args.suite != "quick" else {}
        tiers = list(caps) if caps else [tier for tier in TIERS if tier in info]
        k = 18 if args.suite == "quick" else 28
        summary = dict(backend=args.backend, suite=args.suite, k=k, tiers=tiers, inventory=info)
        (logs / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        run("devices", [binary, "devices", "--config", "/dev/null"])
        run("ctest", ["ctest", "--test-dir", build, "--output-on-failure", "--no-tests=error",
                      "--parallel", "1", "--timeout", "900", "--output-junit", logs / "ctest.xml"])
        if caps:
            specs = [f"{tier}:{cap - info['margin_bytes'] // MIB}" for tier, cap in caps.items()]
            run("vram-boundaries", [Path(__file__).with_name("vram-tiers.sh"), binary, "0", *specs],
                dict(env, XCHPLOT2_TEST_LOG_DIR=str(logs / "boundaries")))

        with tempfile.TemporaryDirectory(prefix=".gpu-ci-", dir=scratch) as directory:
            work = Path(directory)
            plot_id, memo = "ab" * 32, "00" * 112
            run("cpu-reference", [binary, "test", k, plot_id, "2", "0", "0", "-m", memo,
                                  "-o", work, "-N", "reference.plot2", "--config", "/dev/null"])
            reference = work / "reference.plot2"
            with reference.open("rb") as source:
                summary["reference_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
            (logs / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            for tier in tiers:
                for spill in (False, True) if tier in info["spill_tiers"] else (False,):
                    label = tier + ("-disk" if spill else "")
                    plot = work / f"{label}.plot2"
                    manifest = work / "manifest.tsv"
                    manifest.write_text(f"{k} 2 0 0 0 {plot_id} {memo} . {plot.name}\n")
                    command = [binary, "batch", manifest, "--devices", "0", "--tier", tier,
                               "--no-progress", "--config", "/dev/null"]
                    if spill:
                        command += ["--max-host-ram", "min", "--temp-dir", work]
                    plot_env = dict(env, POS2GPU_MAX_VRAM_MB=str(caps[tier])) if args.suite == "vram" else env
                    run(label, command, plot_env, cwd=work)
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
                            f"expected {reference.stat().st_size} bytes, SHA256 {summary['reference_sha256']}; "
                            f"got {plot.stat().st_size} bytes, SHA256 {actual}")
                    run(f"{label}-proofs", [binary, "verify", plot, "--full", "--trials", "100",
                                           "--config", "/dev/null"])
                    plot.unlink()
            if args.suite == "physical":
                # No software cap: exercise automatic selection on the real card.
                run("physical-auto", [binary, "bench", "--devices", "0", "-k", "28", "-n", "3",
                                      "--warmup", "0", "--keep", "--out", work, "--config", "/dev/null"])
                plots = sorted(work.glob("bench-*.plot2"))
                if len(plots) != 3:
                    raise RuntimeError("Physical auto-tier run did not produce three plots")
                for i, plot in enumerate(plots):
                    run(f"physical-auto-proofs-{i}", [binary, "verify", plot, "--full", "--trials", "100",
                                                      "--config", "/dev/null"])
        summary["status"] = "passed"
        (logs / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"PASS: {args.backend} {args.suite}; {len(tiers)} tiers, CPU byte parity and full proofs", flush=True)


if __name__ == "__main__":
    main()
