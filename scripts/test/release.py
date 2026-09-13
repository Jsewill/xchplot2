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
                         "hiprtc0604.dll", "hiprtc-builtins0604.dll", "amd_comgr0604.dll", "ze_loader.dll",
                         "msvcp140.dll", "msvcp140_atomic_wait.dll", "vcruntime140.dll", "vcruntime140_1.dll",
                         "hipSYCL/rt-backend-omp.dll", "hipSYCL/rt-backend-cuda.dll",
                         "hipSYCL/rt-backend-hip.dll", "hipSYCL/rt-backend-ze.dll",
                         "hipSYCL/ext/bitcode/amdgcn/oclc_isa_version_1031.bc",
                         "hipSYCL/bitcode/libkernel-sscp-spirv-full.bc",
                         "hipSYCL/ext/llvm-spirv/bin/llvm-spirv.exe"):
                assert (package / "bin" / name).stat().st_size > 0, f"Missing {name}"
            for name in ("amd-runtime.txt", "hip-headers.txt", "hiprtc.txt", "rocm-comgr.txt",
                         "rocm-device-libs.txt", "level-zero-license.txt", "llvm-spirv-license.txt",
                         "spirv-headers-license.txt", "cuda.txt", "cuda-cccl.txt"):
                assert (package / "licenses" / name).stat().st_size > 0, f"Missing {name}"
            for directory in (package / "bin", package / "bin/hipSYCL/ext/llvm/bin",
                              package / "bin/hipSYCL/ext/llvm-spirv/bin"):
                for name in ("msvcp140.dll", "msvcp140_atomic_wait.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
                    assert (directory / name).is_file(), f"Missing app-local runtime: {directory / name}"
            assert not list(package.rglob("*.lib")), "Import/static libraries are not runtime dependencies"
            assert not list((package / "bin").glob("amdhip64*.dll")), "Use the HIP runtime supplied by the AMD driver"
            # ctypes keeps DLLs loaded; let a child exit before removing the archive.
            subprocess.run([sys.executable, "-c", """
import ctypes, os, pathlib, sys
directory = pathlib.Path(sys.argv[1])
# NVIDIA and AMD runtime plugins import DLLs supplied by their graphics
# drivers. Hosted runners have neither; every bundled dependency still loads.
skip = set()
for driver, backend in (("nvcuda.dll", "cuda"), ("amdhip64_6.dll", "hip")):
    try:
        ctypes.WinDLL(driver)
    except FileNotFoundError:
        skip.add(f"rt-backend-{backend}.dll")
        print(f"{backend} backend DLL load check requires its graphics driver")
with os.add_dll_directory(str(directory)):
    libraries = [ctypes.WinDLL(str(path)) for path in directory.rglob("*.dll") if path.name not in skip]
    assert libraries, "No packaged Windows runtime DLLs"
    # Compile a kernel for the reported RX 6700 XT without any GPU or SDK.
    rtc = ctypes.WinDLL(str(directory / "hiprtc0604.dll"))
    rtc.hiprtcGetErrorString.restype = ctypes.c_char_p
    def check(result):
        assert result == 0, rtc.hiprtcGetErrorString(result).decode()
    program = ctypes.c_void_p()
    source = b'extern "C" __global__ void probe(int* out) { out[0] = 42; }'
    check(rtc.hiprtcCreateProgram(ctypes.byref(program), source, b"probe.hip", 0, None, None))
    try:
        options = (ctypes.c_char_p * 1)(b"--gpu-architecture=gfx1031")
        check(rtc.hiprtcCompileProgram(program, len(options), options))
        size = ctypes.c_size_t()
        check(rtc.hiprtcGetCodeSize(program, ctypes.byref(size)))
        assert size.value > 0, "HIP RTC returned no GPU code"
    finally:
        check(rtc.hiprtcDestroyProgram(ctypes.byref(program)))
    print("Packaged HIP RTC compiled gfx1031 kernel without an SDK")
""", str(package / "bin")], check=True, timeout=60)
            for tool in ("opt.exe", "llc.exe", "lld-link.exe"):
                subprocess.run([package / "bin/hipSYCL/ext/llvm/bin" / tool, "--version"],
                               check=True, timeout=30)
            # Exercise Intel's translator with actual kernel IR, not only --version.
            spirv_ir = work / "probe.ll"
            spirv_ir.write_text('target triple = "spir64-unknown-unknown"\n'
                                'define spir_kernel void @probe() { ret void }\n')
            spirv_bc, spirv_out = work / "probe.bc", work / "probe.spv"
            subprocess.run([package / "bin/hipSYCL/ext/llvm/bin/opt.exe", spirv_ir, "-o", spirv_bc],
                           check=True, timeout=30)
            subprocess.run([package / "bin/hipSYCL/ext/llvm-spirv/bin/llvm-spirv.exe", spirv_bc, "-o", spirv_out],
                           check=True, timeout=30)
            assert spirv_out.read_bytes()[:4] == b"\x03\x02\x23\x07", "Invalid SPIR-V output"
        binary = package / ("bin/xchplot2.exe" if os.name == "nt" else "bin/xchplot2")
        subprocess.run([binary, "--help", "--config", os.devnull], check=True, timeout=30)
        if os.name == "nt":
            devices = subprocess.run([binary, "devices", "--config", os.devnull],
                                     check=True, capture_output=True, text=True, timeout=30)
            assert "nvidia-smi" not in devices.stdout and "rocminfo" not in devices.stdout
        # The existing probe runs a real SYCL kernel through the packaged runtime.
        # Linux uses SSCP JIT; Windows includes a precompiled OpenMP CPU path.
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
            print("Packaged SYCL CPU kernel check passed" if os.name == "nt"
                  else "Packaged SYCL JIT check passed")
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
