xchplot2 — Windows SYCL/AdaptiveCpp binary

Run .\bin\xchplot2.exe --help or .\bin\xchplot2.exe devices.
Add the extracted bin directory to PATH to run the CLI elsewhere.

Requirements:
  Windows 10 22H2, Windows 11, or Windows Server 2022/2025, x86_64.
  CPU with AES, SSSE3, and SSE4.1 instructions.
  NVIDIA: Maxwell or newer; driver 576.57+ recommended for CUDA 12.9.1.
  AMD: a Windows HIP-capable GPU and an Adrenalin driver with HIP runtime 6.
  Intel: a GPU and graphics driver supporting Level Zero.
  An ACL-capable filesystem (NTFS/ReFS) for plots, manifests, and spill files.

This ZIP includes all three GPU backends, AdaptiveCpp, LLVM, CUDA runtime,
HIP runtime compiler libraries and device bitcode, the Intel Level Zero
loader and SPIR-V translator, and Microsoft Visual C++ runtime DLLs.
No CUDA, HIP/ROCm, or oneAPI SDK installation is needed. Install your normal
graphics driver and keep the complete bin/ tree together.
AMD's graphics driver supplies the HIP runtime (amdhip64_6.dll).

Windows SYCL is experimental. Hosted Windows checks cover CPU plotting,
proofs, recovery, and offline AMD/Intel kernel compilation. GPU plotting
requires separate qualification on Windows hardware, including RX 6700 XT.
BUILDINFO.txt records the source and toolchain revisions;
licenses/ contains dependency notices.

First Ctrl-C or Ctrl-Break finishes the current plot; a second aborts.
Use --resume with the saved manifest to continue an interrupted job.
The default config is %APPDATA%\xchplot2\config.toml.
Use --temp-dir to select a disk with enough free space for spill files.
Consult the release notes for tested hardware and driver versions.

Usage: https://github.com/Jsewill/xchplot2
Installation: https://github.com/Jsewill/xchplot2/blob/main/INSTALL.md
Report issues: https://github.com/Jsewill/xchplot2/issues
