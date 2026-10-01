#!/usr/bin/env python3
"""Exercise an extracted binary archive without a compiler or GPU toolkit."""
import argparse
import hashlib
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def extract_archive(archive_path, work):
    digest = hashlib.sha256()
    with archive_path.open("rb") as archive:
        for block in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(block)
    checksum = pathlib.Path(str(archive_path) + ".sha256").read_text().split()
    if checksum != [digest.hexdigest(), archive_path.name]:
        raise ValueError("Archive checksum mismatch")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(work)
    else:
        with tarfile.open(archive_path) as archive:
            archive.extractall(work, filter="data")
    packages = list(work.iterdir())
    if len(packages) != 1 or not packages[0].is_dir():
        raise ValueError("Expected one package directory")
    return packages[0], digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=pathlib.Path)
    parser.add_argument("--sycl-probe", type=pathlib.Path,
                        help="Matching build/tools/sanity/hellosycl executable")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="xchplot2 release é-") as temporary:
        work = pathlib.Path(temporary)
        package, _ = extract_archive(args.archive, work)
        arm64 = platform.machine().lower() in ("aarch64", "arm64")
        for name in ("BUILDINFO.txt", "README.txt", "licenses/LICENSE", "licenses/rust.txt",
                     "licenses/pos2-chip.txt", "licenses/fse.txt", "licenses/aes.txt",
                     "licenses/adaptivecpp.txt", "licenses/adaptivecpp-third-party.txt",
                     "licenses/adaptivecpp-cuda-llvm20.txt",
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
        runtime = package / ("bin" if os.name == "nt" else "lib")
        executable_suffix = ".exe" if os.name == "nt" else ""
        if os.name == "nt":
            gpu_libraries = (("cudart64_13.dll", "OpenCL.dll", "hipSYCL/rt-backend-ocl.dll") if arm64 else
                             ("cudart64_12.dll", "hiprtc0604.dll", "hiprtc-builtins0604.dll",
                              "amd_comgr0604.dll", "hipSYCL/rt-backend-hip.dll",
                              "hipSYCL/ext/bitcode/amdgcn/oclc_isa_version_1031.bc"))
            crt_libraries = ("msvcp140.dll", "msvcp140_atomic_wait.dll", "vcruntime140.dll")
            if not arm64:
                crt_libraries += ("vcruntime140_1.dll",)
            for name in ("acpp-rt.dll", "acpp-common.dll", "libomp.dll", "ze_loader.dll",
                         *crt_libraries, *gpu_libraries,
                         "hipSYCL/rt-backend-omp.dll", "hipSYCL/rt-backend-cuda.dll",
                         "hipSYCL/rt-backend-ze.dll",
                         "hipSYCL/bitcode/libkernel-sscp-spirv-full.bc",
                         "hipSYCL/ext/llvm-spirv/bin/llvm-spirv.exe"):
                assert (package / "bin" / name).stat().st_size > 0, f"Missing {name}"
            gpu_notices = (("opencl-loader-license.txt", "opencl-headers-license.txt",
                            "opencl-backend-headers-license.txt", "opencl-cxx-headers-license.txt") if arm64 else
                           ("amd-runtime.txt", "hip-headers.txt", "hiprtc.txt", "rocm-comgr.txt", "rocm-device-libs.txt"))
            for name in (*gpu_notices, "level-zero-license.txt", "llvm-spirv-license.txt",
                         "spirv-headers-license.txt", "cuda.txt", "cuda-cccl.txt"):
                assert (package / "licenses" / name).stat().st_size > 0, f"Missing {name}"
            for directory in (package / "bin", package / "bin/hipSYCL/ext/llvm/bin",
                              package / "bin/hipSYCL/ext/llvm-spirv/bin"):
                for name in crt_libraries:
                    assert (directory / name).is_file(), f"Missing app-local runtime: {directory / name}"
            assert not list(package.rglob("*.lib")), "Import/static libraries are not runtime dependencies"
            assert not list((package / "bin").glob("amdhip64*.dll")), "Use the HIP runtime supplied by the AMD driver"
        else:
            for name in ("libacpp-rt.so", "libacpp-common.so", "libcudart.so.12",
                         "libamdhip64.so.7", "libhiprtc.so.7", "libamd_comgr.so.3",
                         "libhsa-runtime64.so.1", "libze_loader.so.1",
                         "hipSYCL/librt-backend-omp.so", "hipSYCL/librt-backend-cuda.so",
                         "hipSYCL/librt-backend-hip.so", "hipSYCL/librt-backend-ze.so",
                         "hipSYCL/ext/bitcode/amdgcn/ocml.bc",
                         "hipSYCL/bitcode/libkernel-sscp-spirv-full.bc",
                         "hipSYCL/ext/llvm-spirv/bin/llvm-spirv"):
                assert (runtime / name).stat().st_size > 0, f"Missing {name}"
            rocm_notices = (tuple(f"rocm/{name}.txt" for name in
                                 ("libamdhip64-7", "libhiprtc7", "libhsa-runtime64-1",
                                  "libhsakmt1", "libamd-comgr3", "rocm-device-libs-21"))
                            if arm64 else ("rocm/hip/LICENSE.md", "rocm/amd_comgr/LICENSE.txt",
                                           "rocm/hsakmt/LICENSE.md", "rocm/ROCm-Device-Libs/LICENSE.TXT",
                                           "rocm/rocm-llvm/LICENSE.TXT"))
            for name in ("cuda.txt", "cuda-cccl.txt", "level-zero.txt", "llvm-spirv.txt", *rocm_notices):
                assert (package / "licenses" / name).stat().st_size > 0, f"Missing {name}"
            if arm64:
                for name in ("libLLVM.so.21.1", "libhiprtc-builtins.so.7", "libOpenCL.so.1",
                             "libhsakmt.so.1", "hipSYCL/librt-backend-ocl.so"):
                    assert (runtime / name).stat().st_size > 0, f"Missing {name}"
                for name in ("libllvm21.txt", "libhiprtc-builtins7.txt", "ocl-icd-libopencl1.txt",
                             "opencl-backend-headers-license.txt", "opencl-cxx-headers-license.txt"):
                    assert (package / "licenses" / name).stat().st_size > 0, f"Missing {name}"
            assert not list(runtime.glob("libcuda.so*")), "Use the NVIDIA driver supplied by the system"
        # ctypes keeps libraries loaded; let a child exit before removing the archive.
        subprocess.run([sys.executable, "-c", """
import contextlib, ctypes, os, pathlib, sys
directory = pathlib.Path(sys.argv[1])
load = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
# Driver libraries remain system prerequisites. Linux bundles the HIP runtime;
# Windows obtains it from the AMD graphics driver.
drivers = (("nvcuda.dll", "rt-backend-cuda.dll"), ("amdhip64_6.dll", "rt-backend-hip.dll")) \
    if os.name == "nt" else (("libcuda.so.1", "librt-backend-cuda.so"),)
skip = set()
for driver, backend in drivers:
    if not (directory / "hipSYCL" / backend).is_file():
        continue
    try:
        load(driver)
    except (FileNotFoundError if os.name == "nt" else OSError):
        skip.add(backend)
        print(f"{backend} load check requires its graphics driver")
with os.add_dll_directory(str(directory)) if os.name == "nt" else contextlib.nullcontext():
    libraries = [load(str(path)) for path in directory.rglob("*.dll" if os.name == "nt" else "*.so*")
                 if path.name not in skip]
    assert libraries, "No packaged runtime libraries"
    if sys.argv[2] == "True":
        # Compile a kernel for the reported RX 6700 XT without any GPU or SDK.
        rtc = load(str(directory / ("hiprtc0604.dll" if os.name == "nt" else "libhiprtc.so.7")))
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
""", str(runtime), str(not (os.name == "nt" and arm64))], check=True, timeout=60)
        for tool in ("opt", "llc", "lld-link" if os.name == "nt" else "ld.lld"):
            subprocess.run([runtime / "hipSYCL/ext/llvm/bin" / (tool + executable_suffix), "--version"],
                           check=True, timeout=30)
        # Exercise Intel's translator with actual kernel IR, not only --version.
        spirv_ir = work / "probe.ll"
        spirv_ir.write_text('target triple = "spir64-unknown-unknown"\n'
                            'define spir_kernel void @probe() { ret void }\n')
        spirv_bc, spirv_out = work / "probe.bc", work / "probe.spv"
        subprocess.run([runtime / "hipSYCL/ext/llvm/bin" / ("opt" + executable_suffix), spirv_ir, "-o", spirv_bc],
                       check=True, timeout=30)
        subprocess.run([runtime / "hipSYCL/ext/llvm-spirv/bin" / ("llvm-spirv" + executable_suffix), spirv_bc, "-o", spirv_out],
                       check=True, timeout=30)
        assert spirv_out.read_bytes()[:4] == b"\x03\x02\x23\x07", "Invalid SPIR-V output"
        binary = package / ("bin/xchplot2.exe" if os.name == "nt" else "bin/xchplot2")
        if os.name == "nt":
            with binary.open("rb") as executable:
                header = executable.read(64)
                assert header[:2] == b"MZ", "Expected a Windows executable"
                executable.seek(int.from_bytes(header[60:64], "little"))
                pe = executable.read(6)
            assert pe[:4] == b"PE\0\0" and int.from_bytes(pe[4:6], "little") == (0xAA64 if arm64 else 0x8664), "Archive CPU architecture mismatch"
        else:
            with binary.open("rb") as executable:
                header = executable.read(20)
            assert header[:6] == b"\x7fELF\x02\x01", "Expected a 64-bit little-endian ELF executable"
            assert int.from_bytes(header[18:20], "little") == (183 if arm64 else 62), "Archive CPU architecture mismatch"
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
        # Both single plotting and the SYCL batch pipeline must work without
        # an NVIDIA driver, even when CUDA acceleration is compiled in.
        subprocess.run([binary, "test", "18", plot_id, "2", "0", "0", "-m", memo,
                        "-o", work, "-N", "reference.plot2", "--config", os.devnull],
                       check=True, timeout=180)
        reference = (work / "reference.plot2").read_bytes()
        assert (work / "cpu.plot2").read_bytes() == reference, "CPU batch differs from single plot"
        manifest.write_text(f"18 2 0 0 0 {plot_id} {memo} . sycl.plot2\n")
        subprocess.run([binary, "batch", manifest, "--devices", "cpu", "--cpu-workers", "1",
                        "--no-progress", "--config", os.devnull],
                       cwd=work, env=dict(os.environ, ACPP_VISIBILITY_MASK="omp",
                                          XCHPLOT2_SYCL_CPU_BENCH="1"),
                       check=True, timeout=180)
        assert (work / "sycl.plot2").read_bytes() == reference, "SYCL plot differs from CPU reference"
        if os.name == "nt":
            assert (package / "licenses/microsoft-runtime.txt").stat().st_size > 0
            assert (package / "licenses/adaptivecpp-windows.txt").stat().st_size > 0
        subprocess.run([sys.executable, pathlib.Path(__file__).with_name("recovery.py"), binary],
                       check=True, timeout=720)
        print("Release archive: extraction, CLI, CPU/SYCL plotting parity, and full proofs passed")


if __name__ == "__main__":
    main()
