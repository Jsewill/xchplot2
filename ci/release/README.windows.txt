xchplot2 — native Windows CUDA binary

Run .\bin\xchplot2.exe --help or .\bin\xchplot2.exe devices.
Add the extracted bin directory to PATH to run the CLI elsewhere.

Requirements:
  Windows 10 22H2, Windows 11, or Windows Server 2022/2025, x86_64.
  CPU with AES, SSSE3, and SSE4.1 instructions.
  NVIDIA Maxwell or newer GPU; driver 576.57+ recommended for CUDA 12.9.1.
  An ACL-capable filesystem (NTFS/ReFS) for plots, manifests, and spill files.

CUDA and the Microsoft C/C++ runtime are linked statically. No development
toolkit is needed. BUILDINFO.txt records the source and toolchain revisions;
licenses/ contains dependency notices. GPU drivers are not bundled.

First Ctrl-C or Ctrl-Break finishes the current plot; a second aborts.
Use --resume with the saved manifest to continue an interrupted job.
The default config is %APPDATA%\xchplot2\config.toml.
Use --temp-dir to select a disk with enough free space for spill files.
Consult the release notes for tested hardware and driver versions.

Usage: https://github.com/Jsewill/xchplot2/tree/cuda-only
Installation: https://github.com/Jsewill/xchplot2/blob/cuda-only/INSTALL.md
Report issues: https://github.com/Jsewill/xchplot2/issues
