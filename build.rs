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
                let next = iter.next()?;
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

/// Walk critical build-time prerequisites and return human-readable
/// names of anything missing. Cargo install users in particular don't
/// read the Build section of README.md (and don't expect to need to),
/// so a friendly preflight is much better than letting CMake fail
/// with cryptic errors deep into a build.
///
/// cuda-only's dep list is intentionally short: no AdaptiveCpp / SYCL
/// / LLVM / lld plumbing — just cmake, a C++20 compiler, and nvcc.
fn preflight() -> Vec<String> {
    let mut missing: Vec<String> = vec![];
    if !command_runs("cmake") {
        missing.push("cmake (3.24+) — apt install cmake / dnf install cmake / pacman -S cmake".into());
    }
    if !command_runs("c++") && !command_runs("g++") && !command_runs("clang++") {
        missing.push("C++20 compiler (g++ ≥ 13 or clang++ ≥ 18) — apt install build-essential, dnf install gcc-c++, or pacman -S base-devel".into());
    }
    // cuda-only is by definition NVIDIA — nvcc is always required.
    if !command_runs("nvcc") {
        missing.push("nvcc (CUDA Toolkit 12+) — install from developer.nvidia.com/cuda-downloads (or the apt cuda-toolkit-12-X package)".into());
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
    //   3. A sensible default for machines without nvidia-smi (e.g. CI,
    //      headless package builds). x86_64 defaults to sm_89 (Ada / RTX
    //      4090); aarch64 defaults to sm_87 (Jetson Orin — Ada doesn't
    //      exist on ARM). Cross-vendor targets should set
    //      $CUDA_ARCHITECTURES explicitly.
    let fallback_arch = if cfg!(target_arch = "aarch64") { "87" } else { "89" };
    let (cuda_arch, source) = match env::var("CUDA_ARCHITECTURES") {
        Ok(v) => (v, "$CUDA_ARCHITECTURES"),
        Err(_) => match detect_cuda_arch() {
            Some(v) => (v, "nvidia-smi probe"),
            None    => (fallback_arch.to_string(), "fallback (no nvidia-smi)"),
        },
    };
    println!("cargo:warning=xchplot2: building for CUDA arch {cuda_arch} ({source})");

    // Preflight critical system deps BEFORE invoking cmake. Cargo
    // install users land here without reading the Build section;
    // missing deps would otherwise surface as a cryptic CMake error
    // deep into the configure step.
    let missing = preflight();
    if !missing.is_empty() {
        let bullets = missing.iter()
            .map(|m| format!("  - {m}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "\nxchplot2 (cuda-only): build prerequisites missing:\n{bullets}\n\n\
             Recommended fix: install the CUDA Toolkit 12+ from \
             developer.nvidia.com/cuda-downloads (or your distro's \
             cuda-toolkit-12-X package), plus a C++20 compiler and \
             cmake — the cuda-only branch deliberately has no other \
             dependencies. The main branch's scripts/install-deps.sh \
             does NOT exist on this branch — install manually.\n"
        );
    }

    // CUDA 13.0 dropped codegen for sm_50/52/53/60/61/62/70/72 entirely
    // — its nvcc fails the CMake TryCompile probe with "Unsupported gpu
    // architecture 'compute_61'" on Pascal, "compute_70" on Volta, etc.
    // Catch that mismatch HERE so the failure surfaces with a clear fix
    // path, not buried in a CMakeError.log 40 lines into a TryCompile.
    // Skipped silently when nvcc version or arch list can't be parsed
    // (treat as "preflight not actionable, let cmake try" — preserves
    // prior behaviour for unusual setups).
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
                           podman build -t xchplot2:cuda-only \\\n        \
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
                         --git https://github.com/Jsewill/xchplot2 --branch cuda-only --force\n  \
                   - Or override the arch (only valid if you actually have a Turing+ card):\n      \
                       CUDA_ARCHITECTURES=75 cargo install \\\n      \
                         --git https://github.com/Jsewill/xchplot2 --branch cuda-only --force\n  \
                   - Or use the container path — scripts/build-container.sh auto-pins\n    \
                     the 12.9 base image when it detects a pre-Turing GPU.\n".to_string()
            };
            panic!(
                "\nxchplot2 (cuda-only): CUDA Toolkit {nvcc_major}.x dropped codegen for sm_{min} \
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

    // ---- configure ----
    let status = Command::new("cmake")
        .args([
            "-S", manifest_dir.to_str().unwrap(),
            "-B", cmake_build.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
        ])
        .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"))
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
    // pos2_gpu used to be a STATIC archive containing the CUDA .o
    // files; it's now an INTERFACE lib (no .a produced), and the .o
    // files live exclusively in xchplot2_cli to satisfy the nvlink
    // device-link's "exactly one definition" rule. So we drop the
    // -lpos2_gpu line — there's nothing to link.
    println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    println!("cargo:rustc-link-lib=static=xchplot2_cli");
    println!("cargo:rustc-link-lib=static=pos2_gpu_host");
    println!("cargo:rustc-link-lib=static=pos2_keygen");
    println!("cargo:rustc-link-lib=static=fse");
    println!("cargo:rustc-link-arg=-Wl,--end-group");

    // ---- CUDA runtime ----
    // Honour $CUDA_PATH / $CUDA_HOME if set, else fall back to /opt/cuda
    // (Arch / CachyOS) then /usr/local/cuda (Debian-ish).
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
    // NVIDIA ARM platforms ship CUDA under targets/<triple>/lib:
    //   Jetson (JetPack / L4T):  targets/aarch64-linux/lib
    //   GH200 / SBSA servers:    targets/sbsa-linux/lib
    // Harmless on x86_64 — just silences "no such dir" search misses.
    println!("cargo:rustc-link-search=native={cuda_root}/targets/aarch64-linux/lib");
    println!("cargo:rustc-link-search=native={cuda_root}/targets/sbsa-linux/lib");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cudadevrt");

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
