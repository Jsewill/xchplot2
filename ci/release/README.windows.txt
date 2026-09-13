xchplot2 — Windows SYCL/AdaptiveCpp NVIDIA binary

Run .\bin\xchplot2.exe --help or .\bin\xchplot2.exe devices.
Add the extracted bin directory to PATH to run the CLI elsewhere.

Requirements:
  Windows 10 22H2, Windows 11, or Windows Server 2022/2025, x86_64.
  CPU with AES, SSSE3, and SSE4.1 instructions.
  NVIDIA Maxwell or newer GPU; driver 576.57+ recommended for CUDA 12.9.1.
  An ACL-capable filesystem (NTFS/ReFS) for plots, manifests, and spill files.

AdaptiveCpp, LLVM's JIT, and CUDA runtime DLLs are included under bin/.
Keep the complete bin/ tree together. No development toolkit is needed.
Install the Microsoft Visual C++ x64 Redistributable if it is absent:
https://aka.ms/vs/17/release/vc_redist.x64.exe

This Windows SYCL backend is experimental; AMD and Intel are not packaged.
Hosted Windows checks cover CPU plotting, proofs, and recovery. GPU plotting
requires separate qualification on Windows NVIDIA hardware.
BUILDINFO.txt records the source and toolchain revisions;
licenses/ contains dependency notices. GPU drivers are not bundled.

First Ctrl-C or Ctrl-Break finishes the current plot; a second aborts.
Use --resume with the saved manifest to continue an interrupted job.
The default config is %APPDATA%\xchplot2\config.toml.
Use --temp-dir to select a disk with enough free space for spill files.
Consult the release notes for tested hardware and driver versions.

Usage: https://github.com/Jsewill/xchplot2
Installation: https://github.com/Jsewill/xchplot2/blob/main/INSTALL.md
Report issues: https://github.com/Jsewill/xchplot2/issues
