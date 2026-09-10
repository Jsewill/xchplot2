xchplot2 — SYCL/AdaptiveCpp binary

Choose the archive for your GPU: sycl-nvidia, sycl-amd, or sycl-intel.
Keep bin/ and lib/ together when moving the extracted directory.
Run ./bin/xchplot2 --help or ./bin/xchplot2 devices from that directory.

Requirements:
  Linux x86_64 with glibc 2.39+ (Ubuntu 24.04 or a compatible system).
  CPU with AES, SSSE3, and SSE4.1 instructions.
  Standard OS libraries and a compatible GPU driver.

On Ubuntu 24.04, install the OS libraries with:
  sudo apt install libstdc++6 libnuma1 libelf1t64

NVIDIA: Maxwell or newer; driver 575.57.08+ recommended for CUDA 12.9.1.
AMD: hardware supported by the bundled ROCm 6.2 runtime and a compatible
amdgpu kernel driver; libdrm2 and libdrm-amdgpu1 must be installed.
Intel: an Intel GPU compute driver providing libze_intel_gpu.so.1.
Consult the release notes for tested hardware and driver versions.

AdaptiveCpp, LLVM, and the selected backend runtime are included under lib/.
No CUDA, ROCm, LLVM, or AdaptiveCpp development installation is needed.
BUILDINFO.txt records the source and toolchain revisions. Dependency licenses
are under licenses/. GPU drivers and core OS libraries are not bundled.

Usage: https://github.com/Jsewill/xchplot2
Installation: https://github.com/Jsewill/xchplot2/blob/main/INSTALL.md
Intel settings: https://github.com/Jsewill/xchplot2/blob/main/REFERENCE.md#troubleshooting
Report issues: https://github.com/Jsewill/xchplot2/issues
