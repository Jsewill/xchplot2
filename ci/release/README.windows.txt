xchplot2 — Windows SYCL/AdaptiveCpp binary

Run .\bin\xchplot2.exe --help or .\bin\xchplot2.exe devices.
Add the extracted bin directory to PATH to run the CLI elsewhere.

Requirements:
  Windows 10 22H2, Windows 11, or Windows Server 2022/2025, x86_64.
  CPU with AES, SSSE3, and SSE4.1 instructions.
  Download sycl-amd for AMD GPUs or sycl-nvidia for NVIDIA GPUs.
  NVIDIA: Maxwell or newer; driver 576.57+ recommended for CUDA 12.9.1.
  AMD: a Windows HIP-capable GPU and a current AMD Adrenalin driver.
  An ACL-capable filesystem (NTFS/ReFS) for plots, manifests, and spill files.

Run the dependency helper from the extracted directory:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\install-dependencies.ps1
It checks the GPU vendor and installs the Microsoft Visual C++ x64 runtime
if absent. For AMD it obtains HIP SDK 6.4.2 from AMD when needed; select SDK
Core and Runtime Compiler in AMD's installer. Extra math libraries are not
needed. Installers show their own UI/license and may request administrator
access. With an existing SDK elsewhere, pass -HipPath "C:\path\to\6.4".

AdaptiveCpp and LLVM's JIT are included under bin/. The NVIDIA archive also
includes CUDA runtime DLLs and needs no toolkit. The AMD helper copies the
required DLLs and device bitcode from your SDK for local use under AMD's
terms; see licenses/amd-runtime.txt before redistributing those added files.
Keep the complete bin/ tree together. GPU drivers are installed separately.

Windows SYCL is experimental; Intel is not packaged.
Hosted Windows checks cover CPU plotting, proofs, and recovery. GPU plotting
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
