// build.rs — drive the existing CMake build to produce the static libs
// that the Rust `[[bin]] xchplot2` then links against.
//
// The CMake build is the authoritative one (CUDA, separable compilation,
// pos2-chip FetchContent, the keygen-rs Rust shim). We just call it from
// here so a `cargo install` works end-to-end on a machine with the build
// dependencies listed in README.md (CMake ≥ 3.24, CUDA Toolkit, C++20
// compiler, and a Rust toolchain — the last one cargo provides).

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Ask `nvidia-smi` for the local GPU's compute capability and return it as
/// a CMake-style integer (e.g. "89" for an sm_89 RTX 4090, "120" for an
/// sm_120 RTX 5090). Returns None on any failure — no nvidia-smi, no GPU,
/// driver issue — so callers can fall back cleanly.
fn detect_cuda_arch() -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = std::str::from_utf8(&out.stdout).ok()?.trim();
    if s.is_empty() {
        return None;
    }
    // If multiple GPUs, just use the first; user can override with
    // $CUDA_ARCHITECTURES (which accepts CMake's `89;120` multi-arch syntax)
    // if they need a fat binary.
    let first = s.lines().next()?.trim();
    let cap: f32 = first.parse().ok()?;        // "8.9" -> 8.9
    let arch = (cap * 10.0).round() as u32;    // -> 89
    Some(arch.to_string())
}

/// Same probe as `detect_cuda_arch`, but filters out NVIDIA GPUs
/// below our README-documented minimum compute capability (sm_61,
/// Pascal / GTX 10-series). Below sm_53 the GPU also lacks native
/// FP16 intrinsics (`__hadd` / `__hsub` / `__hmul` / `__hdiv` /
/// `__hlt` / `__hle` / `__hgt` / `__hge`) that AdaptiveCpp's
/// `half.hpp` emits unconditionally in any nvcc device pass —
/// `cuda_fp16.h` guards those behind `__CUDA_ARCH__ >= 530`. Users
/// with an ancient secondary NVIDIA card (e.g. a GTX 750 Ti sitting
/// next to a real AMD / NVIDIA workhorse) otherwise get routed onto
/// the CUB fast path via vendor-precedence and fail to compile
/// SortCuda.cu with a cascade of "identifier `__hXXX` is undefined".
///
/// Returns Some(arch) only when nvidia-smi reports a card at or
/// above our minimum; emits a cargo:warning and returns None
/// otherwise so callers fall through to the AMD / Intel detection.
fn usable_nvidia_arch() -> Option<String> {
    let arch = detect_cuda_arch()?;
    let n: u32 = arch.parse().ok()?;
    if n < 61 {
        println!(
            "cargo:warning=xchplot2: nvidia-smi detected sm_{arch} — below our \
             minimum supported compute capability (sm_61 / Pascal). Ignoring \
             NVIDIA for default targeting; set CUDA_ARCHITECTURES={arch} + \
             XCHPLOT2_BUILD_CUDA=ON to force-build the CUB path anyway (not \
             recommended — AdaptiveCpp half.hpp references sm_53+ FP16 \
             intrinsics that your card's headers don't provide).");
        return None;
    }
    Some(arch)
}

/// Check whether nvcc is on $PATH and runnable. Used as the fall-back
/// signal for XCHPLOT2_BUILD_CUDA when no GPU is enumerable (headless
/// CI / container builds). Runs `nvcc --version` rather than a simple
/// PATH lookup so stale symlinks don't pass.
fn detect_nvcc() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Parse nvcc's major version from `nvcc --version` output.
/// The release line looks like:
///   "Cuda compilation tools, release 13.0, V13.0.48"
/// Returns None if nvcc isn't on PATH or the line can't be parsed —
/// callers treat that as "skip the version-vs-arch compat check"
/// rather than blocking the build.
fn detect_nvcc_major() -> Option<u32> {
    let out = Command::new("nvcc").arg("--version").output().ok()?;
    if !out.status.success() { return None; }
    let s = std::str::from_utf8(&out.stdout).ok()?;
    for line in s.lines() {
        let mut iter = line.split_whitespace();
        while let Some(w) = iter.next() {
            if w == "release" {
                let next = iter.next()?;                         // "13.0,"
                let major = next.trim_end_matches(',').split('.').next()?;
                return major.parse().ok();
            }
        }
    }
    None
}

/// Minimum integer arch from a CMake-style CUDA_ARCHITECTURES list
/// ("61", "61;86", "61;86;120"). Tolerates "sm_61" / "compute_61"
/// prefixes that Cargo users sometimes pass through. Returns None
/// when the list parses to nothing.
fn min_arch(arch_list: &str) -> Option<u32> {
    arch_list.split(';')
        .filter_map(|s| {
            let s = s.trim()
                .trim_start_matches("sm_")
                .trim_start_matches("compute_");
            s.parse().ok()
        })
        .min()
}

/// Probe /sys/class/drm for a display-class PCI device with Intel's
/// vendor ID (0x8086). Used as a heuristic to default
/// XCHPLOT2_BUILD_CUDA=OFF on Intel hosts, mirroring what rocminfo
/// already does for AMD. Returns false on non-Linux or when the sysfs
/// path isn't accessible — callers fall back to the next signal.
fn detect_intel_gpu() -> bool {
    let entries = match std::fs::read_dir("/sys/class/drm") {
        Ok(d) => d,
        Err(_) => return false,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Skip connector nodes like card0-DP-1; we only want the card itself.
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let vendor = entry.path().join("device/vendor");
        if let Ok(v) = std::fs::read_to_string(&vendor) {
            if v.trim() == "0x8086" {
                return true;
            }
        }
    }
    false
}

/// Ask `rocminfo` for the first AMD GPU's architecture, e.g. "gfx1100" for
/// an RX 7900 XTX. Returns None when rocminfo is missing or there's no AMD
/// GPU. Used to set ACPP_TARGETS=hip:gfxXXXX so AdaptiveCpp can AOT-compile
/// the kernels for the actual hardware.
fn detect_amd_gfx() -> Option<String> {
    let out = Command::new("rocminfo").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = std::str::from_utf8(&out.stdout).ok()?;
    for line in s.lines() {
        if let Some(rest) = line.trim().strip_prefix("Name:") {
            let name = rest.trim();
            if name.starts_with("gfx") {
                // RDNA1 workaround: gfx1010/1011/1012 aren't direct
                // AdaptiveCpp HIP targets. Community-tested (Radeon Pro
                // W5700) that gfx1013 is ISA-close enough to run on
                // gfx1010 silicon. Not parity-validated — flagged via
                // cargo:warning so users know they're on the workaround
                // path.
                let spoofed = match name {
                    "gfx1010" | "gfx1011" | "gfx1012" => {
                        println!(
                            "cargo:warning=xchplot2: RDNA1 {name} detected — \
                             building for gfx1013 (community workaround, \
                             not parity-validated; verify plots with \
                             `xchplot2 verify` before farming)");
                        "gfx1013".to_string()
                    }
                    other => other.to_string(),
                };
                return Some(spoofed);
            }
        }
    }
    None
}

/// Probe whether `cmd` is on PATH and runnable. Used by preflight()
/// to detect missing toolchain pieces before cmake gets to fail with
/// a cryptic message.
fn command_runs(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Locate `ld.lld` either on PATH or in the conventional LLVM-{16..20}
/// install prefixes. Mirrors the find_program HINTS list in
/// CMakeLists.txt's FetchContent block. AdaptiveCpp's CMake aborts
/// with "Cannot find ld.lld" without it.
fn ld_lld_findable() -> bool {
    if command_runs("ld.lld") { return true; }
    for p in &[
        "/usr/lib/llvm-20/bin/ld.lld", "/usr/lib/llvm-19/bin/ld.lld",
        "/usr/lib/llvm-18/bin/ld.lld", "/usr/lib/llvm-17/bin/ld.lld",
        "/usr/lib/llvm-16/bin/ld.lld",
        "/usr/lib/llvm20/bin/ld.lld",  "/usr/lib/llvm19/bin/ld.lld",
        "/usr/lib/llvm18/bin/ld.lld",
        "/usr/lib64/llvm20/bin/ld.lld", "/usr/lib64/llvm19/bin/ld.lld",
        "/usr/lib64/llvm18/bin/ld.lld",
        "/opt/llvm-20/bin/ld.lld", "/opt/llvm-19/bin/ld.lld",
        "/opt/llvm-18/bin/ld.lld",
    ] {
        if std::path::Path::new(p).exists() { return true; }
    }
    false
}

/// True when AdaptiveCpp is already installed — at $ACPP_PREFIX if
/// set, otherwise the install-deps.sh default of /opt/adaptivecpp.
/// When this is true the FetchContent fallback won't fire and
/// AdaptiveCpp's own build-time deps (notably ld.lld) aren't needed
/// for our build.
fn adaptivecpp_installed() -> bool {
    let prefix = env::var("ACPP_PREFIX")
        .unwrap_or_else(|_| "/opt/adaptivecpp".to_string());
    std::path::Path::new(&format!(
        "{prefix}/lib/cmake/AdaptiveCpp/AdaptiveCppConfig.cmake"
    )).exists()
}

/// Walk critical build-time prerequisites and return human-readable
/// names of anything missing. Cargo install users in particular don't
/// read the Build section of README.md (and don't expect to need to),
/// so a friendly preflight is much better than letting CMake or
/// AdaptiveCpp fail with cryptic errors deep into a build.
fn preflight(build_cuda_on: bool) -> Vec<String> {
    let mut missing: Vec<String> = vec![];
    if !command_runs("cmake") {
        missing.push("cmake (3.24+) — apt install cmake / dnf install cmake / pacman -S cmake".into());
    }
    if !command_runs("c++") && !command_runs("g++") && !command_runs("clang++") {
        missing.push("C++20 compiler (g++ ≥ 13 or clang++ ≥ 18) — apt install build-essential, dnf install gcc-c++, or pacman -S base-devel".into());
    }
    // ld.lld is only required when FetchContent will rebuild
    // AdaptiveCpp; a pre-installed AdaptiveCpp linked against ld.lld
    // at its own install time, so consumers don't need it again.
    if !adaptivecpp_installed() && !ld_lld_findable() {
        missing.push("ld.lld (apt: lld-18, dnf/pacman: lld) — required by AdaptiveCpp's FetchContent build".into());
    }
    if build_cuda_on && !detect_nvcc() {
        missing.push("nvcc (CUDA Toolkit 12+) — XCHPLOT2_BUILD_CUDA=ON requested but no nvcc on PATH".into());
    }
    missing
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir      = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cmake_build  = out_dir.join("cmake-build");
    std::fs::create_dir_all(&cmake_build).expect("create cmake-build dir");

    // Architecture precedence:
    //   1. $CUDA_ARCHITECTURES if set (lets the user pick or list multiple).
    //   2. nvidia-smi probe of the build machine's local GPU.
    //   3. 89 (sm_89, RTX 4090 / Ada Lovelace) as a sensible default for
    //      machines without nvidia-smi (e.g. CI, headless package builds).
    let (cuda_arch, source) = match env::var("CUDA_ARCHITECTURES") {
        Ok(v) => (v, "$CUDA_ARCHITECTURES"),
        Err(_) => match detect_cuda_arch() {
            Some(v) => (v, "nvidia-smi probe"),
            None    => ("89".to_string(), "fallback (no nvidia-smi)"),
        },
    };
    println!("cargo:warning=xchplot2: building for CUDA arch {cuda_arch} ({source})");

    // AdaptiveCpp target precedence:
    //   1. $ACPP_TARGETS if set.
    //   2. NVIDIA: "generic" (LLVM SSCP). Empirically a few percent
    //      faster than cuda:sm_<arch> on our kernels.
    //   3. AMD:    hip:gfx<...> via rocminfo. SSCP's HIP path is less
    //      mature, so AOT-compile for the gfx target.
    //   4. generic (LLVM SSCP, JITs on first use).
    let (acpp_targets, acpp_source) = match env::var("ACPP_TARGETS") {
        // Treat an empty env var the same as unset — Containerfile build
        // args propagate as `ACPP_TARGETS=` when the user doesn't override
        // them, and acpp rejects an empty target string.
        Ok(v) if !v.is_empty() => (v, "$ACPP_TARGETS"),
        Ok(_) | Err(_) => {
            // Prefer a USABLE NVIDIA GPU (sm_61+) over AMD, otherwise fall
            // through to AMD / fallback. `detect_cuda_arch` alone would
            // trigger on an ancient secondary NVIDIA card even when AMD is
            // the real plotting target (see usable_nvidia_arch).
            if usable_nvidia_arch().is_some() {
                ("generic".to_string(), "NVIDIA detected — using SSCP")
            } else if let Some(gfx) = detect_amd_gfx() {
                (format!("hip:{gfx}"), "rocminfo probe")
            } else {
                ("generic".to_string(), "fallback (LLVM SSCP)")
            }
        }
    };
    println!("cargo:warning=xchplot2: ACPP_TARGETS={acpp_targets} ({acpp_source})");

    // XCHPLOT2_BUILD_CUDA toggles whether the CUB sort + nvcc-compiled
    // CUDA TUs (AesGpu.cu, SortCuda.cu, AesGpuBitsliced.cu) are built.
    // Autodetect prefers actual GPU vendor over toolchain availability:
    // dual-toolchain hosts (AMD / Intel GPU, CUDA Toolkit also installed)
    // would otherwise try to compile SortCuda.cu through nvcc + AdaptiveCpp
    // — which has triggered upstream `half.hpp` compile errors for at
    // least one Radeon Pro W5700 user. Priority order:
    //   NVIDIA GPU → ON      (CUB is the fast path)
    //   AMD GPU    → OFF     (SYCL/HIP path; CUB unused anyway)
    //   Intel GPU  → OFF     (SYCL/L0 path)
    //   no GPU, nvcc present → ON  (CI / container build)
    //   no GPU, no nvcc      → OFF
    let (build_cuda, bc_source) = match env::var("XCHPLOT2_BUILD_CUDA") {
        Ok(v) if !v.is_empty() => (v, "$XCHPLOT2_BUILD_CUDA"),
        _ => {
            // Same usable-arch gate as the ACPP_TARGETS block: an
            // ancient secondary NVIDIA card (e.g. sm_52 alongside an
            // AMD W5700) must NOT claim the CUB path, because
            // AdaptiveCpp half.hpp references sm_53+ FP16 intrinsics
            // that the old card's cuda_fp16.h guards out.
            let nvidia_gpu = usable_nvidia_arch().is_some();
            let amd_gpu    = detect_amd_gfx().is_some();
            let intel_gpu  = detect_intel_gpu();
            if nvidia_gpu {
                ("ON".to_string(), "NVIDIA GPU detected")
            } else if amd_gpu {
                ("OFF".to_string(), "AMD GPU detected — skipping CUDA TUs")
            } else if intel_gpu {
                ("OFF".to_string(), "Intel GPU detected — skipping CUDA TUs")
            } else if detect_nvcc() {
                ("ON".to_string(), "no GPU probe, nvcc present — assuming CI/container")
            } else {
                ("OFF".to_string(), "no GPU, no nvcc — skipping CUDA TUs")
            }
        },
    };
    println!("cargo:warning=xchplot2: XCHPLOT2_BUILD_CUDA={build_cuda} ({bc_source})");

    // Preflight critical system deps BEFORE invoking cmake. Cargo
    // install users land here without reading README.md's Build
    // section; without preflight, missing deps surface as cryptic
    // CMake / AdaptiveCpp errors deep in the configure / build.
    let missing = preflight(build_cuda == "ON");
    if !missing.is_empty() {
        let bullets = missing.iter()
            .map(|m| format!("  - {m}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "\nxchplot2: build prerequisites missing:\n{bullets}\n\n\
             Recommended fix: run scripts/install-deps.sh from a \
             repo checkout — auto-detects vendor, installs the \
             toolchain + AdaptiveCpp. Headless / CI builds need \
             --gpu nvidia. The Containerfile is another option \
             (see README's Build section, or scripts/build-container.sh).\n\n\
             If you already ran install-deps.sh and still see this, \
             check its tail output — it names the missing package \
             before exiting.\n"
        );
    }

    // CUDA 13.0 dropped codegen for sm_50/52/53/60/61/62/70/72 entirely
    // — its nvcc fails the CMake TryCompile probe with "Unsupported gpu
    // architecture 'compute_61'" on Pascal, "compute_70" on Volta, etc.
    // Catch that mismatch HERE so the failure surfaces with a clear fix
    // path, not buried in a CMakeError.log 40 lines into a TryCompile.
    // Skipped when nvcc version or arch list can't be parsed (treat as
    // "preflight not actionable, let cmake try" — preserves prior
    // behaviour for unusual setups).
    if build_cuda == "ON" {
        if let (Some(nvcc_major), Some(min)) = (detect_nvcc_major(), min_arch(&cuda_arch)) {
            if nvcc_major >= 13 && min < 75 {
                // Container detection: Docker writes /.dockerenv, Podman writes
                // /run/.containerenv. Either presence means the host-side fixes
                // (apt install cuda-toolkit, set CUDA_PATH) are not actionable
                // from inside this build — the user needs to rebuild the image
                // with a different BASE_DEVEL.
                let in_container = std::path::Path::new("/.dockerenv").exists()
                    || std::path::Path::new("/run/.containerenv").exists();
                let fix_block = if in_container {
                    format!(
                        "You're building inside a container — the toolkit comes from the\n\
                         base image, not the host. Rebuild the image with a CUDA 12.x base:\n  \
                           - Recommended: rerun scripts/build-container.sh on the host;\n    \
                             it auto-pins nvidia/cuda:12.9.1 when CUDA_ARCH < 75.\n  \
                           - Or pass --build-arg explicitly:\n      \
                               podman build -t xchplot2:cuda \\\n        \
                                 --build-arg BASE_DEVEL=docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 \\\n        \
                                 --build-arg BASE_RUNTIME=docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 \\\n        \
                                 --build-arg CUDA_ARCH={min} \\\n        \
                                 .\n  \
                           - Or via compose with env vars:\n      \
                               CUDA_ARCH={min} \\\n        \
                                 BASE_DEVEL=docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 \\\n        \
                                 BASE_RUNTIME=docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 \\\n        \
                                 podman compose build cuda\n"
                    )
                } else {
                    "Fix one of:\n  \
                       - Install CUDA 12.9 (last toolkit with Pascal/Volta support):\n      \
                           Ubuntu/Debian:  sudo apt install cuda-toolkit-12-9\n      \
                           Arch:           pacman -S cuda  (or pin to a 12.x channel)\n    \
                         then point the build at it:\n      \
                           CUDA_PATH=/usr/local/cuda-12.9 cargo install \\\n      \
                             --git https://github.com/Jsewill/xchplot2 --force\n  \
                       - Or override the arch (only valid if you actually have a Turing+ card):\n      \
                           CUDA_ARCHITECTURES=75 cargo install \\\n      \
                             --git https://github.com/Jsewill/xchplot2 --force\n  \
                       - Or use the container path — scripts/build-container.sh auto-pins\n    \
                         the 12.9 base image when it detects a pre-Turing GPU.\n".to_string()
                };
                panic!(
                    "\nxchplot2: CUDA Toolkit {nvcc_major}.x dropped codegen for sm_{min} \
                     (Pascal / Volta / pre-Turing).\n\
                     \n\
                     Detected:\n  \
                       nvcc {nvcc_major}.x\n  \
                       target arch: sm_{min} (from CUDA_ARCHITECTURES={cuda_arch})\n\
                     \n\
                     {fix_block}"
                );
            }
        }
    }

    // ---- configure ----
    let status = Command::new("cmake")
        .args([
            "-S", manifest_dir.to_str().unwrap(),
            "-B", cmake_build.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
        ])
        .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"))
        .arg(format!("-DACPP_TARGETS={acpp_targets}"))
        .arg(format!("-DXCHPLOT2_BUILD_CUDA={build_cuda}"))
        .status()
        .expect("failed to invoke cmake — is it installed?");
    if !status.success() {
        panic!("cmake configure failed");
    }

    // ---- build only the static libs we need; skip the cmake-built
    // executable (we're producing our own via cargo) and the parity tests.
    let status = Command::new("cmake")
        .args([
            "--build", cmake_build.to_str().unwrap(),
            "--target", "xchplot2_cli",
            "--parallel",
        ])
        .status()
        .expect("failed to invoke cmake --build");
    if !status.success() {
        panic!("cmake build of xchplot2_cli failed");
    }

    // ---- tell rustc where each static lib lives ----
    let cb = cmake_build.display();
    println!("cargo:rustc-link-search=native={cb}");
    println!("cargo:rustc-link-search=native={cb}/fse");
    println!("cargo:rustc-link-search=native={cb}/keygen-rs-target/release");

    // Order matters: xchplot2_cli depends on pos2_gpu_host depends on pos2_gpu.
    // Wrap in --start-group/--end-group so the static linker resolves any
    // remaining cross-archive references without us having to pin order.
    //
    // --allow-multiple-definition: pos2_keygen.a is a Rust staticlib, so it
    // bundles its own copy of libstd (rust_eh_personality, ARGV_INIT_ARRAY,
    // EMPTY_PANIC). The host xchplot2 binary also brings in libstd. Both
    // copies come from the same toolchain and are bit-identical, so letting
    // the linker pick the first is safe. The clean alternative is to make
    // keygen-rs a Rust workspace member with crate-type = ["rlib"], but
    // that breaks the standalone CMake-only build path which expects a
    // staticlib for the cmake-built executable.
    println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    println!("cargo:rustc-link-lib=static=xchplot2_cli");
    println!("cargo:rustc-link-lib=static=pos2_gpu_host");
    println!("cargo:rustc-link-lib=static=pos2_gpu");
    println!("cargo:rustc-link-lib=static=pos2_keygen");
    println!("cargo:rustc-link-lib=static=fse");
    println!("cargo:rustc-link-arg=-Wl,--end-group");

    // ---- AdaptiveCpp runtime ----
    // The static archives produced by CMake reference hipsycl::rt::* symbols
    // that live in libacpp-rt + libacpp-common (shared). CMake writes the
    // exact lib directory to $cmake_build/acpp-prefix.txt during configure;
    // honour that, then $ACPP_PREFIX / standard locations as fallbacks.
    let acpp_lib_dir = std::fs::read_to_string(cmake_build.join("acpp-prefix.txt"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| env::var("ACPP_PREFIX").ok().map(|p| format!("{p}/lib")))
        .or_else(|| env::var("AdaptiveCpp_ROOT").ok().map(|p| format!("{p}/lib")))
        .unwrap_or_else(|| {
            for guess in ["/opt/adaptivecpp/lib", "/usr/local/lib",
                          "/usr/lib/x86_64-linux-gnu", "/usr/lib"] {
                if std::path::Path::new(&format!("{guess}/libacpp-rt.so")).exists() {
                    return guess.to_string();
                }
            }
            "/opt/adaptivecpp/lib".to_string()
        });
    println!("cargo:rustc-link-search=native={acpp_lib_dir}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{acpp_lib_dir}");
    println!("cargo:rustc-link-lib=acpp-rt");
    println!("cargo:rustc-link-lib=acpp-common");

    // ---- CUDA runtime ----
    // Only needed when XCHPLOT2_BUILD_CUDA=ON — then the nvcc-compiled
    // TUs (SortCuda, AesGpu, AesGpuBitsliced) pull in cudart / cudadevrt.
    // On the AMD/Intel OFF path there's no CUDA Toolkit on the image and
    // nothing in the static archives references cudart, so emitting
    // `-lcudart` would make rust-lld fail with "unable to find library".
    if build_cuda == "ON" {
        // Honour $CUDA_PATH / $CUDA_HOME if set, else fall back to
        // /opt/cuda (Arch / CachyOS) then /usr/local/cuda (Debian-ish).
        let cuda_root = env::var("CUDA_PATH")
            .or_else(|_| env::var("CUDA_HOME"))
            .unwrap_or_else(|_| {
                for guess in ["/opt/cuda", "/usr/local/cuda"] {
                    if std::path::Path::new(guess).exists() { return guess.to_string(); }
                }
                "/opt/cuda".to_string()
            });
        println!("cargo:rustc-link-search=native={cuda_root}/lib64");
        println!("cargo:rustc-link-search=native={cuda_root}/lib");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cudadevrt");
    }

    // ---- HIP runtime ----
    // When ACPP_TARGETS is "hip:gfxXXXX", AdaptiveCpp's HIP backend
    // compiles SYCL kernels into HIP fat binaries whose host-side
    // launcher stubs reference __hipPushCallConfiguration /
    // __hipRegisterFatBinary / hipLaunchKernel from libamdhip64. Without
    // -lamdhip64 rust-lld fails with "undefined symbol: __hip*".
    // Honour $ROCM_PATH if set, else fall back to /opt/rocm (standard
    // bare-metal + all official ROCm container images).
    if acpp_targets.starts_with("hip:") {
        let rocm_root = env::var("ROCM_PATH")
            .unwrap_or_else(|_| "/opt/rocm".to_string());
        println!("cargo:rustc-link-search=native={rocm_root}/lib");
        println!("cargo:rustc-link-search=native={rocm_root}/hip/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{rocm_root}/lib");
        println!("cargo:rustc-link-lib=amdhip64");
    }

    // C++ stdlib + POSIX bits the static libs (Rust std + pthread inside
    // pos2_keygen, std::async + std::thread in pos2_gpu_host) reach for.
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=rt");

    // ---- rebuild triggers ----
    for p in &[
        "src", "tools", "keygen-rs/src", "keygen-rs/Cargo.toml",
        "keygen-rs/Cargo.lock", "CMakeLists.txt", "build.rs",
    ] {
        println!("cargo:rerun-if-changed={p}");
    }
    println!("cargo:rerun-if-env-changed=CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
