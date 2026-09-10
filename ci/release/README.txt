xchplot2 — native CUDA binary

Requirements:
  Linux x86_64 with glibc 2.35+ and libstdc++.so.6 with GLIBCXX_3.4.30.
  Updated Ubuntu 22.04 or newer provides these runtime libraries.
  CPU with AES, SSSE3, and SSE4.1 instructions.
  NVIDIA Maxwell or newer GPU with a compatible driver.
  NVIDIA driver 575.57.08 or newer is recommended for CUDA 12.9.1.

Extract the archive, then run:
  ./bin/xchplot2 --help
  ./bin/xchplot2 devices

The CUDA runtime is linked into the executable. No CUDA toolkit is needed.
BUILDINFO.txt records the source revisions, compilers, and GPU targets.
GPU targets describe compiled coverage; consult the benchmark report for
hardware measurements and the release notes for qualification results.

Usage and installation: https://github.com/Jsewill/xchplot2/tree/cuda-only
Command reference: https://github.com/Jsewill/xchplot2/blob/cuda-only/REFERENCE.md
Benchmarks: https://github.com/Jsewill/xchplot2/blob/cuda-only/BENCHMARKS.md
Report issues: https://github.com/Jsewill/xchplot2/issues

Dependency licenses are in licenses/. The bundled CPU solver includes the
growing-buffer correction in contrib/pos2-solver-candidates.patch.
