#!/usr/bin/env python3
"""Check native Windows cancellation and durable recovery with real CPU plots."""
import ctypes
import hashlib
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import sys
import tempfile
import threading


def main():
    assert os.name == "nt", "Run this check on native Windows"
    binary = str(Path(sys.argv[1]).resolve())
    # Public BLS12-381 generator; all keys and plots in this check are disposable.
    farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    allocated_console = kernel.AllocConsole()
    if not allocated_console and ctypes.get_last_error() != 5:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        with tempfile.TemporaryDirectory(prefix="xchplot2 Windows é-") as directory:
            out = Path(directory)
            args = [binary, "plot", "--config", os.devnull, "-k", "22", "-n", "12",
                    "-f", farmer, "--pool-ph", "42" * 32, "-o", str(out),
                    "--devices", "cpu", "--cpu-workers", "2", "--quiet", "--no-progress"]
            with (out / "check.log").open("w", encoding="utf-8") as log:
                def run(command, check=True):
                    return subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True,
                                          encoding="utf-8", check=check, timeout=180)

                process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=log, text=True,
                    encoding="utf-8", creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                first_line = queue.Queue()
                threading.Thread(target=lambda: first_line.put(process.stdout.readline()), daemon=True).start()
                try:
                    first = first_line.get(timeout=90)
                    assert first, "No output was published before exit"
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                    rest = process.communicate(timeout=180)[0]
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                paths = (first + rest).splitlines()
                assert process.returncode == 4 and 0 < len(paths) < 12, (process.returncode, paths)
                assert {Path(p) for p in paths} == set(out.glob("*.plot2"))
                job, = out.glob("xchplot2-job-*.tsv")
                saved = job.read_bytes()
                entries = [shlex.split(line) for line in saved.decode("utf-8").splitlines()
                           if not line.startswith("#")]
                expected = {str(Path(e[7]) / e[8]) for e in entries}
                hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
                result = run(args + ["--resume"])
                assert set(result.stdout.splitlines()) == expected
                assert job.read_bytes() == saved
                assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
                print("Windows Ctrl-Break: drained current plots; resume preserved keys and completed files")

                failed = sorted(expected)[0]
                Path(failed).unlink()
                Path(failed).mkdir()  # force a real publication failure
                result = run(args + ["--resume"], check=False)
                assert result.returncode == 3 and set(result.stdout.splitlines()) == expected - {failed}
                assert not list(out.glob("*.partial.*"))
                Path(failed).rmdir()
                run(args + ["--resume"])
                assert job.read_bytes() == saved
                run([binary, "verify", failed, "--full", "--trials", "100", "--config", os.devnull])
                print("Windows publication failure: correct status, partial cleanup, recovery, and full proofs")

                pool_pk = out / "pool public key"
                result = run([binary, "plot", "--config", os.devnull, "-k", "18", "-n", "2",
                    "-f", farmer, "-p", farmer, "-o", str(pool_pk), "--devices", "cpu",
                    "--cpu-workers", "2", "--quiet", "--no-progress"])
                assert len(result.stdout.splitlines()) == 2
                for path in result.stdout.splitlines():
                    run([binary, "verify", path, "--full", "--trials", "100", "--config", os.devnull])
                print("Windows keygen: pool public key memos and parallel CPU plotting passed")

                bench = out / "benchmark é"
                run([binary, "bench", "--config", os.devnull, "-k", "18", "-n", "1",
                     "--warmup", "0", "--cpu", "--cpu-workers", "1", "--keep",
                     "--compute-only", "-o", str(bench)])
                log.flush()
                output = (out / "check.log").read_text(encoding="utf-8")
                kept = [line.removeprefix("[bench] kept ") for line in output.splitlines()
                        if line.startswith("[bench] kept ")]
                assert len(kept) == 2 and all(Path(p).is_file() for p in kept), kept
                assert "no usable tmpfs" in output and "compute+cache" in output
                print("Windows benchmark: cache fallback and Unicode kept paths passed")
    finally:
        if allocated_console:
            kernel.FreeConsole()


if __name__ == "__main__":
    main()
