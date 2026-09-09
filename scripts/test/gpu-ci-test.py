#!/usr/bin/env python3
"""Host-only checks for GPU CI's backend and capacity gates."""
from pathlib import Path
import os
import runpy
import subprocess
import tempfile

ci = runpy.run_path(str(Path(__file__).with_name("gpu-ci.py")))
MIB = ci["MIB"]
parse = ci["parse_inventory"]
select = ci["tier_caps"]


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
    "XCHPLOT2_SYCL_CPU_BENCH": "1", "ACPP_VISIBILITY_MASK": "omp"}, "level_zero")
assert env["PATH"] == "/bin" and env["LD_LIBRARY_PATH"] == "/toolchain"
assert env["ACPP_VISIBILITY_MASK"] == "ze" and env["POS2GPU_ASSERT_VRAM"] == "1"
assert env["POS2GPU_VRAM_MARGIN_MB"] == "256"
assert "POS2GPU_MAX_VRAM_MB" not in env and "POS2GPU_SKIP_SELFTEST" not in env
assert "XCHPLOT2_SYCL_CPU_BENCH" not in env

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
print("GPU CI backend, capacity, rounding, environment, and driver-peak checks passed")
