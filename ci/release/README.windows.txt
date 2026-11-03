xchplot2 — Windows SYCL/AdaptiveCpp binary

Run .\bin\xchplot2.exe --help or .\bin\xchplot2.exe devices.
Add the extracted bin directory to PATH to run the CLI elsewhere.

x86_64 requirements:
  Windows 10 22H2, Windows 11, or Windows Server 2022/2025, x86_64.
  CPU with AES, SSSE3, and SSE4.1 instructions.
  NVIDIA: Maxwell or newer; driver 576.57+ recommended for CUDA 12.9.1.
  AMD: a Windows HIP-capable GPU and an Adrenalin driver with HIP runtime 6.
  Intel: a GPU and graphics driver supporting Level Zero.
  An ACL-capable filesystem (NTFS/ReFS) for ordinary plots, manifests, and spill.
  Grouped plotting and growth require local NTFS with ordinary files.

The x86_64 ZIP includes all three GPU backends, AdaptiveCpp, LLVM, CUDA runtime,
HIP runtime compiler libraries and device bitcode, the Intel Level Zero
loader and SPIR-V translator, and Microsoft Visual C++ runtime DLLs.
No CUDA, HIP/ROCm, or oneAPI SDK installation is needed. Install your normal
graphics driver and keep the complete bin/ tree together.
AMD's graphics driver supplies the HIP runtime (amdhip64_6.dll).

The ARM64 ZIP requires Windows 11 ARM64 with crypto extensions. It includes
CUDA 13.4.2 (Turing and newer), Level Zero, and OpenCL, with native ARM64
runtime libraries. Each backend needs a compatible ARM64 graphics driver.
OpenCL requires SPIR-V, Intel USM or fine-grained system SVM, and 64-bit
atomics. There is no native Windows ARM64 HIP runtime in AMD's SDK.

Windows support is experimental, including GPU plotting. Grouped construction
rejects sparse, compressed, encrypted and network-backed storage. The x86_64
background worker can stop after rapid service restarts; check compute service
status and compute status after restarting. See the command reference below.
Dependency notices are in licenses/.

First Ctrl-C or Ctrl-Break finishes the current plot; a second aborts.
Use --resume with the saved manifest to continue an interrupted job.
The default config is %APPDATA%\xchplot2\config.toml.
Use --temp-dir to select a disk with enough free space for spill files.
See the installation guide for driver requirements and known limitations.

Usage: https://github.com/Jsewill/xchplot2
Installation: https://github.com/Jsewill/xchplot2/blob/main/INSTALL.md
Commands and shared compute: https://github.com/Jsewill/xchplot2/blob/main/REFERENCE.md
Report issues: https://github.com/Jsewill/xchplot2/issues
