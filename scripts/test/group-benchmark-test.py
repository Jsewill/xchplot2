#!/usr/bin/env python3
"""Host-only checks for growth benchmark loading, exit timing and sampler cleanup."""

import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch


script = Path(__file__).with_name("group-benchmark.py")
benchmark = runpy.run_path(str(script))
common = benchmark["common"]
measure = benchmark["measure"]
assert Path(common.__file__).resolve() == script.resolve().parents[2] / "contrib/pos2-pr118/group-benchmark.py"
subprocess.run([sys.executable, "-B", script, "--help"], stdout=subprocess.DEVNULL, check=True, timeout=10)

with tempfile.TemporaryDirectory(prefix="xchplot2-benchmark-") as directory:
    root = Path(directory)
    # /proc can deny a child's exe, final I/O and fd samples after it exits.
    # Keep allocation measurements and report unavailable counters as absent.
    original_read_text, original_iterdir = Path.read_text, Path.iterdir
    proc = Path(f"/proc/{os.getpid()}")
    sampled_file = root / "allocated"
    sampled_file.write_bytes(b"sample" * 1024)

    def restricted_read_text(path, *args, **kwargs):
        if path == proc / "io":
            raise PermissionError("process counters unavailable")
        return original_read_text(path, *args, **kwargs)

    def restricted_iterdir(path):
        if path == proc / "fd":
            raise PermissionError("process descriptors unavailable")
        return original_iterdir(path)

    for error in (FileNotFoundError, PermissionError):
        with patch.object(Path, "samefile", side_effect=error("process executable unavailable")), \
             patch.object(Path, "read_text", restricted_read_text), \
             patch.object(Path, "iterdir", restricted_iterdir):
            memory, io, space = common.sample(os.getpid(), Path(sys.executable), root)
        assert memory is None and io == {} and space == sampled_file.stat().st_blocks * 512

    # Descriptor access can also disappear between enumeration and stat.
    # A permission failure on the benchmark's own files must still fail.
    original_stat = Path.stat
    descriptor = proc / "fd/123456"

    def restricted_stat(path, *args, **kwargs):
        if path == descriptor:
            raise PermissionError("process descriptor unavailable")
        return original_stat(path, *args, **kwargs)

    with patch.object(Path, "stat", restricted_stat):
        assert common.allocated([sampled_file], [descriptor]) == sampled_file.stat().st_blocks * 512
    with patch.object(Path, "stat", side_effect=PermissionError("work file unavailable")):
        try:
            common.allocated([sampled_file])
        except PermissionError:
            pass
        else:
            raise AssertionError("benchmark hid a file allocation error")
    sampled_file.unlink()

    marker = root / "finish"
    fixture = root / "fixture.py"
    fixture.write_text("""import json
from pathlib import Path
import sys
import time
while not Path(sys.argv[1]).exists():
    time.sleep(0.01)
print(json.dumps({"mode": "gpu"}))
""")
    observers = []
    original_exit = common.ProcessExit

    def observe(pid):
        observer = original_exit(pid)
        observers.append((pid, observer))
        return observer

    def slow_gpu(pid):
        assert pid == observers[-1][0]
        marker.touch()
        assert observers[-1][1].completed.wait(10), "fixture did not exit during sampling"
        time.sleep(0.05)
        return 4096

    with patch.object(common, "ProcessExit", observe), patch.dict(measure.__globals__, gpu_memory=slow_gpu):
        result = measure([Path(sys.executable), fixture, marker], root / "sampled.log", root, 0.01, True)
    assert result["gpu_samples"] == 1 and result["sampled_peak_vram_bytes"] == 4096
    assert result["measurement_seconds"] - result["seconds"] >= 0.05, "slow sampling inflated process exit time"
    assert result["sampling_wall_seconds"] >= result["gpu_query_seconds"] >= 0.05

    marker.unlink()
    original_sample = common.sample

    def slow_sample(pid, executable, work):
        marker.touch()
        assert observers[-1][1].completed.wait(10), "fixture did not exit during allocation sampling"
        time.sleep(0.05)
        return original_sample(pid, executable, work)

    with patch.object(common, "ProcessExit", observe), patch.object(common, "sample", slow_sample):
        assembly = common.measure([Path(sys.executable), fixture, marker], root / "assembly.log", root, 0.01)
    assert assembly["seconds"] == assembly["process_seconds"]
    assert assembly["measurement_seconds"] - assembly["process_seconds"] >= 0.05

    marker.unlink()
    with patch.object(common, "ProcessExit", observe), patch.object(common, "sample", side_effect=RuntimeError("injected sample failure")):
        try:
            measure([Path(sys.executable), fixture, marker], root / "failed.log", root, 0.01, False)
        except RuntimeError as error:
            assert str(error) == "injected sample failure"
        else:
            raise AssertionError("benchmark hid a sampling failure")
    try:
        os.waitpid(observers[-1][0], os.WNOHANG)
    except ChildProcessError:
        pass
    else:
        raise AssertionError("benchmark did not reap its child after a sampling failure")

print("growth benchmark helper, exit attribution and sampler cleanup checks passed")
