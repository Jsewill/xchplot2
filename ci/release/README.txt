xchplot2 — SYCL/AdaptiveCpp binary

This archive includes NVIDIA, AMD, and Intel GPU backends together.
Keep bin/ and lib/ together when moving the extracted directory.
Run ./bin/xchplot2 --help or ./bin/xchplot2 devices from that directory.

Requirements:
  Linux x86_64 with glibc 2.39+ (Ubuntu 24.04 or a compatible system).
  Or Linux ARM64 with glibc 2.43+ (Ubuntu 26.04 or a compatible system).
  x86_64 with AES, SSSE3, and SSE4.1, or ARMv8-A with crypto extensions.
  Standard OS libraries; GPU plotting also needs a compatible driver.

On Ubuntu 24.04, install the OS libraries with:
  sudo apt install libstdc++6 libnuma1 libelf1t64

NVIDIA: Maxwell or newer; driver 575.57.08+ recommended for CUDA 12.9.1.
AMD: hardware supported by the bundled ROCm 7.1.1 runtime (Ubuntu HIP 7.1.0
and compiler libraries 7.1.1 on ARM64) and a compatible
amdgpu kernel driver; libdrm2 and libdrm-amdgpu1 must be installed.
Intel: an Intel GPU compute driver providing libze_intel_gpu.so.1.
See the installation guide for driver requirements and known limitations.
ARM64 includes CUDA for Linux SBSA and an OpenCL backend for devices with
SPIR-V, Intel USM or fine-grained system SVM, and 64-bit atomics.
Every backend needs a matching ARM64 GPU driver. GPU plotting on ARM64
remains experimental.

AdaptiveCpp, LLVM, CUDA and HIP runtimes, the Level Zero loader, and the
SPIR-V translator are included under lib/.
No CUDA, ROCm, LLVM, or AdaptiveCpp development installation is needed.
Dependency licenses are under licenses/.
GPU drivers and core OS libraries are not bundled.

Usage: https://github.com/Jsewill/xchplot2
Installation: https://github.com/Jsewill/xchplot2/blob/main/INSTALL.md
Commands and shared compute: https://github.com/Jsewill/xchplot2/blob/main/REFERENCE.md
Intel settings: https://github.com/Jsewill/xchplot2/blob/main/REFERENCE.md#troubleshooting
Report issues: https://github.com/Jsewill/xchplot2/issues
