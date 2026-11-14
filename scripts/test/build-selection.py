#!/usr/bin/env python3
"""GPU-free checks for Cargo's script mode and standalone CMake's include mode."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def run(args, env):
    return subprocess.run(["cmake", *args], env=env, text=True, capture_output=True)


with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    env = os.environ.copy()
    for name in ("XCHPLOT2_FORCE_GFX_SPOOF", "XCHPLOT2_NO_GFX_SPOOF"):
        env.pop(name, None)
    policy = ROOT / "cmake/TargetSelection.cmake"
    include = work / "include.cmake"
    include.write_text(f'include("{policy}")\n')
    cases = [
        ({}, ("generic", "OFF")),
        ({"X2_HAVE_NVCC": "ON"}, ("generic", "ON")),
        ({"X2_HAVE_NVIDIA": "true", "X2_HAVE_AMD": "false", "X2_HAVE_INTEL": "false"}, ("generic", "ON")),
        ({"X2_HAVE_AMD": "ON", "X2_AMD_GFX": "gfx1100", "X2_HAVE_NVCC": "ON"}, ("hip:gfx1100", "OFF")),
        ({"X2_HAVE_AMD": "ON", "X2_HAVE_INTEL": "ON", "X2_AMD_GFX": "gfx1100"}, ("generic", "OFF")),
        ({"X2_HAVE_AMD": "ON", "X2_HAVE_NVIDIA": "ON", "X2_AMD_GFX": "gfx1100"}, ("generic", "ON")),
        ({"X2_HAVE_AMD": "ON", "X2_AMD_GFX": "gfx1010"}, ("generic", "OFF")),
        ({"ACPP_TARGETS": "generic;omp", "XCHPLOT2_BUILD_CUDA": "OFF", "X2_HAVE_NVIDIA": "ON"}, ("generic;omp", "OFF")),
        ({"XCHPLOT2_BUILD_CUDA": "true"}, ("generic", "ON")),
        ({"XCHPLOT2_BUILD_CUDA": "0", "X2_HAVE_NVIDIA": "ON"}, ("generic", "OFF")),
        ({"ACPP_TARGETS": "hip:gfx1010", "XCHPLOT2_BUILD_CUDA": "ON"}, ("hip:gfx1010", "ON")),
    ]
    for facts, expected in cases:
        for script in (policy, include):
            output = work / "selection.txt"
            result = run([*[f"-D{k}={v}" for k, v in facts.items()],
                          f"-DX2_SELECTION_OUTPUT={output}", "-P", str(script)], env)
            assert result.returncode == 0, result.stderr
            assert tuple(output.read_text().splitlines()) == expected, (facts, result.stdout)
    for flag, expected in (("XCHPLOT2_FORCE_GFX_SPOOF", "hip:gfx1013"),
                           ("XCHPLOT2_NO_GFX_SPOOF", "hip:gfx1011")):
        override_env = dict(env, **{flag: "1"})
        output = work / "selection.txt"
        result = run(["-DX2_HAVE_AMD=ON", "-DX2_AMD_GFX=gfx1011",
                      f"-DX2_SELECTION_OUTPUT={output}", "-P", str(policy)], override_env)
        assert result.returncode == 0, result.stderr
        assert output.read_text().splitlines() == [expected, "OFF"]

    # Fake nvcc exercises toolkit compatibility before enable_language(CUDA).
    nvcc = work / "nvcc"
    arch_script = work / "arches.cmake"
    output = work / "arches.txt"
    arch_script.write_text(f'include("{ROOT / "cmake/CudaArchitectures.cmake"}")\n'
                           f'file(WRITE "{output}" "${{CMAKE_CUDA_ARCHITECTURES}}")\n')
    for version, supported, requested, expected in (
        ("12.9", "compute_90\ncompute_75\ncompute_89", "61", "61"),
        ("13.0", "compute_120\ncompute_75\ncompute_103", "61", None),
        ("12.4", "compute_90\ncompute_75\ncompute_89", "89;120", "89;90-virtual"),
        ("13.0", "compute_120\ncompute_75\ncompute_103", "120", "120"),
        ("unknown", "", "120", "120"),
    ):
        nvcc.write_text('#!/bin/sh\ncase "$1" in\n'
                        f'--version) printf "Cuda compilation tools, release {version}\\n" ;;\n'
                        f'--list-gpu-arch) printf "{supported}\\n" ;;\nesac\n')
        nvcc.chmod(0o755)
        result = run([f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DCMAKE_CUDA_ARCHITECTURES={requested}",
                      "-P", str(arch_script)], env)
        if expected is None:
            assert result.returncode != 0 and "dropped codegen" in result.stderr, result.stderr
        else:
            assert result.returncode == 0, result.stderr
            assert output.read_text() == expected, (requested, output.read_text())
    # Exercise the dependency bootstrap without downloading AdaptiveCpp or
    # installing packages. A loader without headers must not select Intel on
    # a mixed host, while an explicit Intel request must explain the failure.
    compiler_root = work / "llvm"
    (compiler_root / "bin").mkdir(parents=True)
    compiler = compiler_root / "bin/clang++"
    dependencies = work / "dependencies.cmake"
    dependencies.write_text(
        f'include("{ROOT / "cmake/AdaptiveCppDependencies.cmake"}")\n'
        f'xchplot2_check_adaptivecpp_dependencies("{compiler_root}" 20)\n'
        f'file(WRITE "{output}" "${{WITH_LEVEL_ZERO_BACKEND}}")\n')
    for cxx, omp, ze, facts, expected, missing in (
        (0, 0, 1, {"X2_HAVE_NVIDIA": "ON", "X2_HAVE_INTEL": "ON"}, "OFF", ()),
        (0, 0, 1, {"X2_HAVE_INTEL": "ON"}, None, ("Level Zero loader and headers",)),
        (0, 0, 1, {"WITH_LEVEL_ZERO_BACKEND": "ON"}, None, ("Level Zero loader and headers",)),
        (0, 0, 1, {"WITH_LEVEL_ZERO_BACKEND": "OFF", "X2_HAVE_INTEL": "ON"}, "OFF", ()),
        (0, 0, 0, {}, "ON", ()),
        (0, 0, 0, {"ACPP_TARGETS": "omp"}, "OFF", ()),
        (1, 1, 1, {"WITH_LEVEL_ZERO_BACKEND": "ON"}, None,
         ("C++20 headers and libraries", "LLVM OpenMP headers and runtime", "Level Zero loader and headers")),
    ):
        compiler.write_text('#!/bin/sh\ncase "$*" in\n'
                            f'*openmp.cpp*) echo "OpenMP probe" >&2; exit {omp} ;;\n'
                            f'*level_zero.cpp*) echo "Level Zero header/link probe" >&2; exit {ze} ;;\n'
                            f'*) echo "C++ library probe" >&2; exit {cxx} ;;\nesac\n')
        compiler.chmod(0o755)
        values = {"ACPP_TARGETS": "generic", **facts}
        result = run([*[f"-D{k}={v}" for k, v in values.items()],
                      f"-DXCHPLOT2_PREREQUISITES_DIR={work / 'probes'}",
                      "-P", str(dependencies)], env)
        if expected is None:
            assert result.returncode != 0, result.stdout
            assert all(message in result.stderr for message in missing), result.stderr
            assert "install-deps.sh" in result.stderr, result.stderr
        else:
            assert result.returncode == 0, result.stderr
            assert output.read_text() == expected, (facts, result.stdout)
print("build selection: target policy, toolchain compatibility and dependency checks passed")
