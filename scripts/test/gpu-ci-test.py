#!/usr/bin/env python3
"""Host-only checks for GPU CI's backend and capacity gates."""
from pathlib import Path
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ci = runpy.run_path(str(Path(__file__).with_name("gpu-ci.py")))
MIB = ci["MIB"]
parse = ci["parse_inventory"]
select = ci["tier_caps"]
vectors = ci["plot_vectors"](28)
assert [(v["k"], v["strength"], v["plot_index"], v["meta_group"], v["testnet"])
        for v in vectors] == [(28, 2, 0, 0, False), (18, 3, 65535, 255, False), (18, 4, 1, 1, False)]
assert vectors[0]["plot_id"] == "ab" * 32 and vectors[0]["memo"] == "00" * 112
assert vectors[1]["plot_id"] == "ff" * 32 and bytes.fromhex(vectors[1]["memo"]) == bytes(range(255))
assert bytes.fromhex(vectors[2]["plot_id"]) == bytes(range(32))
assert ci["plot_vectors"](18)[0]["k"] == 18


def rejects(function, *args):
    try:
        function(*args)
    except (ValueError, KeyError):
        return
    raise AssertionError(f"Accepted invalid CI input: {args}")


info = dict(total_bytes=8192 * MIB, free_bytes=7900 * MIB, margin_bytes=128 * MIB,
            tiny=1100 * MIB, pinned=1150 * MIB, minimal=3900 * MIB,
            compact=5200 * MIB, plain=7290 * MIB)
text = "backend=cuda\nspill_tiers=tiny,pinned,minimal,compact\n" + "".join(f"{key}={value}\n" for key, value in info.items())
info["spill_tiers"] = ["tiny", "pinned", "minimal", "compact"]
assert parse(text, "cuda") == info
rejects(parse, text, "hip")
rejects(parse, "backend=cuda\ntotal_bytes=0\n", "cuda")
rejects(parse, text + "unknown=1\n", "cuda")
rejects(parse, text.replace("spill_tiers=tiny,pinned,minimal,compact", "spill_tiers=plain"), "cuda")
assert select(info)["tiny"] == 1228
assert select(dict(info, tiny=1100 * MIB + 1))["tiny"] == 1229
assert select(info, 8192) == select(info)
small = dict(info, total_bytes=2048 * MIB, free_bytes=1600 * MIB)
assert set(select(small, 2048)) == {"tiny", "pinned"}
rejects(select, small)  # A nightly lane cannot quietly omit larger tiers.
rejects(select, info, 2048)  # A large card cannot stand in for a small one.
rejects(select, info, 24576)
rejects(select, dict(small, free_bytes=500 * MIB), 2048)
assert select(dict(info, free_bytes=1228 * MIB, total_bytes=2048 * MIB), 2048) == {"tiny": 1228}
rejects(select, dict(small, free_bytes=1228 * MIB - 1), 2048)

env = ci["test_environment"]({"PATH": "/bin", "LD_LIBRARY_PATH": "/toolchain",
    "POS2GPU_MAX_VRAM_MB": "2048", "POS2GPU_SKIP_SELFTEST": "1",
    "POS2GPU_ASSERT_VRAM": "0", "POS2GPU_VRAM_MARGIN_MB": "256",
    "XCHPLOT2_MAX_HOST_RAM": "1G", "XCHPLOT2_SYCL_CPU_BENCH": "1", "ACPP_VISIBILITY_MASK": "omp"}, "level_zero")
assert env["PATH"] == "/bin" and env["LD_LIBRARY_PATH"] == "/toolchain"
assert env["ACPP_VISIBILITY_MASK"] == "ze" and env["POS2GPU_ASSERT_VRAM"] == "1"
assert env["POS2GPU_VRAM_MARGIN_MB"] == "256"
assert "POS2GPU_MAX_VRAM_MB" not in env and "POS2GPU_SKIP_SELFTEST" not in env
assert "XCHPLOT2_SYCL_CPU_BENCH" not in env
assert "XCHPLOT2_MAX_HOST_RAM" not in env
bounded = ci["test_environment"]({"XCHPLOT2_MAX_HOST_RAM": "1G", "POS2GPU_ASSERT_VRAM": "0"}, "cuda", 18)
assert bounded["XCHPLOT2_MAX_HOST_RAM"] == "18G" and bounded["POS2GPU_ASSERT_VRAM"] == "1"
for budget in ("0", "-1", str(2**34)):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("gpu-ci.py")), ".",
                             "--backend", "cuda", "--logs", "/invalid-unused-log-path",
                             "--max-host-ram-gib", budget], capture_output=True, text=True)
    assert result.returncode == 2 and "must be a positive integer below" in result.stderr


# Archive qualification must select the packaged binary and reject checksum or
# source/toolchain mismatches before any GPU inventory or executable runs.
with tempfile.TemporaryDirectory(prefix="xchplot2-archive-check-") as directory:
    root = Path(directory)
    build = root / "build"
    build.mkdir()
    buildinfo = "Source: public-fixture\nC++: fixture compiler\n"
    (build / "BUILDINFO.txt").write_text(buildinfo)
    archive = root / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("fixture/BUILDINFO.txt", buildinfo)
        package.writestr("fixture/bin/xchplot2", "public fixture executable")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = Path(str(archive) + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n")
    work = root / "extracted"
    work.mkdir()
    binary, metadata = ci["archive_binary"](archive, build, work)
    assert binary == work / "fixture/bin/xchplot2" and binary.is_file()
    assert metadata["name"] == archive.name and metadata["sha256"] == digest
    assert metadata["buildinfo"] == buildinfo
    assert metadata["started_at"] and metadata["platform"]
    archive_env = ci["archive_environment"](env, binary)
    assert archive_env["LD_LIBRARY_PATH"] == str(work / "fixture/lib") + os.pathsep + "/toolchain"
    assert env["LD_LIBRARY_PATH"] == "/toolchain"
    assert ci["archive_environment"]({}, binary)["LD_LIBRARY_PATH"] == str(work / "fixture/lib")
    linux_archive = root / "fixture.tar.gz"
    with tarfile.open(linux_archive, "w:gz") as package:
        package.add(work / "fixture", arcname="fixture")
    linux_digest = hashlib.sha256(linux_archive.read_bytes()).hexdigest()
    Path(str(linux_archive) + ".sha256").write_text(f"{linux_digest}  {linux_archive.name}\n")
    linux_binary, linux_metadata = ci["archive_binary"](linux_archive, build, root / "linux")
    assert linux_binary == root / "linux/fixture/bin/xchplot2"
    assert linux_metadata["sha256"] == linux_digest
    checksum.write_text(f"{'0' * 64}  {archive.name}\n")
    rejects(ci["archive_binary"], archive, build, root / "bad-checksum")
    assert not (root / "bad-checksum").exists()
    checksum.write_text(f"{digest}  {archive.name}\n")
    (build / "BUILDINFO.txt").write_text("Source: different fixture\n")
    rejects(ci["archive_binary"], archive, build, root / "bad-build")
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("gpu-ci.py")), str(build),
                             "--backend", "cuda", "--logs", str(root / "mismatch-logs"),
                             "--archive", str(archive)], capture_output=True, text=True)
    assert result.returncode != 0 and "BUILDINFO.txt does not match" in result.stderr
    assert not (root / "mismatch-logs/inventory.log").exists()
    summary = json.loads((root / "mismatch-logs/summary.json").read_text())
    assert summary["status"] == "failed" and summary["stage"] == "archive"
    assert summary["archive"]["name"] == archive.name
    assert summary["failure"]["type"] == "ValueError" and summary["finished_at"]
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("gpu-ci.py")), str(build),
                             "--backend", "cuda", "--logs", str(root / "logs"),
                             "--binary", str(binary), "--archive", str(archive)],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "not allowed with argument" in result.stderr

# Exercise real qualification failure paths without GPU execution.
with tempfile.TemporaryDirectory(prefix="xchplot2-result-check-") as directory:
    root = Path(directory)
    build = root / "build"
    (build / "tools/sanity").mkdir(parents=True)
    commands = root / "commands"
    commands.mkdir()
    inventory = build / "tools/sanity/gpu_ci_info"
    inventory.write_text('''#!/usr/bin/env python3
import os, sys
print(os.environ["CI_INVENTORY"], end="")
sys.exit(7 if os.environ["CI_FAIL"] == "inventory" else 0)
''')
    binary = commands / "xchplot2"
    binary.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
kind = sys.argv[1]
failure = os.environ["CI_FAIL"]
with open(os.environ["CI_CALLS"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if failure == kind:
    sys.exit(17)
if kind == "test":
    params = [*sys.argv[2:7], sys.argv[sys.argv.index("-m") + 1]]
    assert "--testnet" not in sys.argv and "-T" not in sys.argv
    (Path(sys.argv[sys.argv.index("-o") + 1]) / "reference.plot2").write_bytes(json.dumps(params).encode())
if kind == "batch":
    tier = sys.argv[sys.argv.index("--tier") + 1]
    print(f"streaming tier: {tier} (")
    print("-> disk")
    k, strength, index, meta, testnet, plot_id, memo, directory, name = Path(sys.argv[2]).read_text().split()
    assert testnet == "0" and directory == "."
    params = [k, plot_id, strength, index, meta, memo]
    corrupt = failure == "parity" or (failure == "vector-parity" and name.startswith("index-meta-max-"))
    Path(name).write_bytes(b"different" if corrupt else json.dumps(params).encode())
if kind == "verify":
    assert "--full" in sys.argv and sys.argv[sys.argv.index("--trials") + 1] == "100"
    count = 0 if failure == "zero-proofs" else 2
    if failure != "missing-proofs":
        print(f"[verify] {count} full proofs validated")
''')
    ctest = commands / "ctest"
    ctest.write_text('#!/bin/sh\n[ "$CI_FAIL" != ctest ]\n')
    smi = commands / "nvidia-smi"
    smi.write_text('#!/bin/sh\nprintf "<process_info><pid>123</pid><used_memory>3062 MiB</used_memory></process_info>\\n"\n')
    for executable in (inventory, binary, ctest, smi):
        executable.chmod(0o755)
    setup_logs = root / "setup"
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("gpu-ci.py")), str(build),
                             "--backend", "cuda", "--logs", str(setup_logs),
                             "--scratch", str(root / "missing")],
        env=dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"]),
        capture_output=True, text=True)
    summary = json.loads((setup_logs / "summary.json").read_text())
    assert result.returncode != 0 and summary["status"] == "failed" and summary["stage"] == "setup"
    assert summary["failure"] == dict(type="ValueError", message="--scratch must be an existing directory on real disk")
    assert summary["started_at"] and summary["finished_at"] and not (setup_logs / "inventory.log").exists()
    cases = (("inventory", "inventory", "CalledProcessError", 7),
             ("invalid", "inventory-validation", "ValueError", None),
             ("capacity", "capacity-validation", "ValueError", None),
             ("devices", "devices", "CalledProcessError", 17),
             ("ctest", "ctest", "CalledProcessError", 1),
             ("test", "cpu-reference", "CalledProcessError", 17),
             ("batch", "tiny", "CalledProcessError", 17),
             ("parity", "tiny-parity", "RuntimeError", None),
             ("verify", "tiny-proofs", "CalledProcessError", 17),
             ("zero-proofs", "tiny-proofs-proof-count", "RuntimeError", None),
             ("missing-proofs", "tiny-proofs-proof-count", "RuntimeError", None),
             ("vector-parity", "index-meta-max-tiny-parity", "RuntimeError", None),
             ("none", "complete", None, None))
    for failure, stage, error_type, exit_code in cases:
        logs = root / failure
        fixture = text.replace("backend=cuda", "backend=hip" if failure == "invalid" else "backend=cuda")
        if failure == "capacity":
            fixture = fixture.replace(f"free_bytes={7900 * MIB}", f"free_bytes={500 * MIB}")
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("gpu-ci.py")), str(build),
                                 "--backend", "cuda", "--suite", "vram" if failure == "capacity" else "quick",
                                 "--binary", str(binary), "--logs", str(logs), "--scratch", str(root)],
            env=dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"],
                     CI_FAIL=failure, CI_INVENTORY=fixture, CI_CALLS=str(root / f"{failure}-calls.jsonl")),
            capture_output=True, text=True)
        summary = json.loads((logs / "summary.json").read_text())
        assert summary["stage"] == stage, result.stdout + result.stderr
        assert summary["status"] == ("failed" if error_type else "passed")
        assert summary["started_at"] and summary["finished_at"]
        assert (result.returncode != 0) == bool(error_type)
        assert "3062 MiB" in (logs / "nvidia-smi-start.log").read_text()
        if error_type:
            assert summary["failure"]["type"] == error_type and summary["failure"]["message"]
            assert "3062 MiB" in (logs / "nvidia-smi-failure.log").read_text()
        if exit_code is not None:
            assert summary["failure"]["exit_code"] == exit_code
            assert summary["failure"]["command"] and (logs / summary["log"]).is_file()
        if failure == "inventory":
            assert (logs / "inventory.txt").read_text().strip() == fixture.strip()
        if failure == "vector-parity":
            assert summary["failure"]["vector"] == "index-meta-max"
            assert [v["status"] for v in summary["vectors"]] == ["passed", "failed", "pending"]
        if failure == "none":
            assert "vector" not in summary and len(summary["vectors"]) == 3
            assert summary["reference_sha256"] == summary["vectors"][0]["reference_sha256"]
            calls = [json.loads(line) for line in (root / "none-calls.jsonl").read_text().splitlines()]
            assert sum(command[0] == "test" for command in calls) == 3
            assert sum(command[0] == "batch" for command in calls) == 27
            assert sum(command[0] == "verify" for command in calls) == 27
            for vector in summary["vectors"]:
                prefix = "" if vector["name"] == "baseline" else vector["name"] + "-"
                labels = {prefix + tier + suffix for tier in ci["TIERS"]
                          for suffix in (("", "-disk") if tier in info["spill_tiers"] else ("",))}
                assert vector["status"] == "passed" and set(vector["full_proofs_validated"]) == labels
                assert set(vector["full_proofs_validated"].values()) == {2}
                params = [str(vector["k"]), vector["plot_id"], str(vector["strength"]),
                          str(vector["plot_index"]), str(vector["meta_group"]), vector["memo"]]
                assert vector["reference_sha256"] == hashlib.sha256(json.dumps(params).encode()).hexdigest()

# Exercise the actual boundary script: missing/oversized driver measurements
# must fail even if the plot command itself succeeds.
with tempfile.TemporaryDirectory(prefix="xchplot2-ci-check-") as directory:
    root = Path(directory)
    fake = root / "xchplot2"
    fake.write_text('''#!/usr/bin/env python3
import os, sys
assert sys.argv[1] == "bench" and sys.argv[sys.argv.index("--devices") + 1] == "0"
if int(os.environ["POS2GPU_MAX_VRAM_MB"]) < 138:
    print("tier does not fit, including the VRAM buffer")
    sys.exit(1)
assert sys.argv[sys.argv.index("-n") + 1] == "3"
peak = os.environ["CI_FAKE_PEAK"]
if peak != "missing":
    print(f"[batch:gpu0] vram: peak {peak} MiB of 2000 free")
''')
    fake.chmod(0o755)
    for peak, success in (("100", True), ("139", False), ("missing", False)):
        result = subprocess.run([str(Path(__file__).with_name("vram-tiers.sh")), str(fake), "0", "tiny:10"],
            env=dict(os.environ, POS2GPU_VRAM_MARGIN_MB="128", CI_FAKE_PEAK=peak,
                     XCHPLOT2_TEST_LOG_DIR=str(root / peak)), capture_output=True, text=True)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        assert (root / peak / "tiny-boundary.log").exists()
    assert (root / "100/tiny-rejection.log").exists()
print("GPU CI archive, backend, capacity, rounding, environment, and driver-peak checks passed")
