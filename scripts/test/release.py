#!/usr/bin/env python3
"""Exercise an extracted binary archive without a compiler or GPU toolkit."""
import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=pathlib.Path)
    parser.add_argument("--sycl-probe", type=pathlib.Path,
                        help="Matching build/tools/sanity/hellosycl executable")
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.archive.open("rb") as archive:
        for block in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(block)
    checksum = pathlib.Path(str(args.archive) + ".sha256").read_text().split()
    assert checksum == [digest.hexdigest(), args.archive.name], "Archive checksum mismatch"
    with tempfile.TemporaryDirectory(prefix="xchplot2 release é-") as temporary:
        work = pathlib.Path(temporary)
        if args.archive.suffix == ".zip":
            with zipfile.ZipFile(args.archive) as archive:
                archive.extractall(work)
        else:
            with tarfile.open(args.archive) as archive:
                archive.extractall(work, filter="data")
        packages = list(work.iterdir())
        assert len(packages) == 1 and packages[0].is_dir(), "Expected one package directory"
        package = packages[0]
        for name in ("BUILDINFO.txt", "README.txt", "licenses/LICENSE", "licenses/rust.txt",
                     "licenses/pos2-chip.txt", "licenses/fse.txt", "licenses/aes.txt",
                     "licenses/adaptivecpp.txt", "licenses/adaptivecpp-third-party.txt",
                     "licenses/llvm.txt", "licenses/llvm-third-party.txt",
                     "licenses/rust-standard-library/COPYRIGHT-library.html"):
            assert (package / name).stat().st_size > 0, f"Missing or empty {name}"
        notices = (package / "licenses/adaptivecpp-third-party.txt").read_text(encoding="utf-8")
        revision = (package / "licenses/adaptivecpp-revision.txt").read_text().strip()
        assert revision and revision in notices, "AdaptiveCpp notices do not match the packaged revision"
        for author in ("Mike Loomis", "Martin Leitner-Ankerl", "Facebook Inc.",
                       "Georgia Institute of Technology", "Google LLC", "Mike Pall",
                       "Universidad Rey Juan Carlos", "Pekka Jääskeläinen"):
            assert author in notices, f"Missing AdaptiveCpp third-party notice: {author}"
        llvm_notices = (package / "licenses/llvm-third-party.txt").read_text(encoding="utf-8")
        for author in ("Yann Collet", "Henry Spencer", "Todd C. Miller", "Unicode, Inc.",
                       "2019 Intel Corporation"):
            assert author in llvm_notices, f"Missing LLVM third-party notice: {author}"
        if os.name == "nt":
            for name in ("acpp-rt.dll", "acpp-common.dll", "libomp.dll", "cudart64_12.dll",
                         "hipSYCL/rt-backend-omp.dll", "hipSYCL/rt-backend-cuda.dll"):
                assert (package / "bin" / name).stat().st_size > 0, f"Missing {name}"
            # ctypes keeps DLLs loaded; let a child exit before removing the archive.
            subprocess.run([sys.executable, "-c", """
import ctypes, os, pathlib, sys
directory = pathlib.Path(sys.argv[1])
# The CUDA plugin imports nvcuda.dll from the user's driver. Hosted runners
# have no driver; still load the bundled CUDA runtime there.
try:
    ctypes.WinDLL("nvcuda.dll")
    cuda_driver = True
except OSError as error:
    if error.winerror != 126:
        raise
    cuda_driver = False
    print("CUDA backend DLL load check requires an NVIDIA driver")
with os.add_dll_directory(str(directory)):
    libraries = [ctypes.WinDLL(str(path)) for path in directory.rglob("*.dll")
                 if cuda_driver or path.name != "rt-backend-cuda.dll"]
    assert libraries, "No packaged Windows runtime DLLs"
""", str(package / "bin")], check=True, timeout=60)
        binary = package / ("bin/xchplot2.exe" if os.name == "nt" else "bin/xchplot2")
        subprocess.run([binary, "--help", "--config", os.devnull], check=True, timeout=30)
        # The existing probe runs a real SSCP kernel through the packaged JIT.
        # Its build-tree RPATH cannot resolve inside the clean runtime image.
        if args.sycl_probe:
            env = dict(os.environ, ACPP_VISIBILITY_MASK="omp", LD_LIBRARY_PATH=str(package / "lib"))
            probe = args.sycl_probe.resolve()
            if os.name == "nt":
                # The Windows loader searches beside the EXE before PATH.
                # Copy the matching probe beside the packaged DLLs.
                probe = package / "bin/hellosycl.exe"
                shutil.copy2(args.sycl_probe, probe)
            subprocess.run([probe], cwd=work, env=env, check=True, timeout=180)
            print("Packaged SYCL JIT check passed")
        plot_id, memo = "ab" * 32, "00" * 112
        manifest = work / "cpu.tsv"
        manifest.write_text(f"18 2 0 0 0 {plot_id} {memo} . cpu.plot2\n")
        subprocess.run([binary, "batch", manifest, "--devices", "cpu", "--cpu-workers", "2",
                        "--config", os.devnull],
                       cwd=work, check=True, timeout=180)
        subprocess.run([binary, "verify", work / "cpu.plot2", "--full", "--trials", "100",
                        "--config", os.devnull],
                       check=True, timeout=180)
        if os.name == "nt":
            assert (package / "licenses/microsoft-runtime.txt").stat().st_size > 0
            assert (package / "licenses/adaptivecpp-windows.txt").stat().st_size > 0
            subprocess.run([sys.executable, pathlib.Path(__file__).with_name("windows.py"), binary],
                           check=True, timeout=360)
        print("Release archive: extraction, CLI, CPU plotting, and full proofs passed")


if __name__ == "__main__":
    main()
