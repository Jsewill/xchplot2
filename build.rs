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

/// Does the host have any NVIDIA GPU? Sysfs PCI vendor-ID probe
/// (0x10de) — independent of `nvidia-smi`, which can fail on older
/// drivers, partial enumeration, container / chroot / sudo invocation
/// where the binary isn't on PATH, etc. Used to differentiate "no
/// NVIDIA card on this host" (CI / cross-compile) from "NVIDIA card
/// present but nvidia-smi probe failed" (use the default arch with a
/// pointed warning so the user knows they should set $CUDA_ARCHITECTURES
/// if their card differs from the default).
fn nvidia_gpu_present() -> bool {
    let entries = match std::fs::read_dir("/sys/class/drm") {
        Ok(d) => d,
        Err(_) => return false,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let vendor = entry.path().join("device/vendor");
        if let Ok(v) = std::fs::read_to_string(&vendor) {
            if v.trim() == "0x10de" {
                return true;
            }
        }
    }
    false
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

/// Detect a container engine on PATH, preferring podman (matches
/// scripts/build-container.sh's default). Used to phrase the preflight
/// panic differently when the user already has tooling that lets them
/// skip the host-side install entirely.
fn detect_container_engine() -> Option<&'static str> {
    if command_runs("podman") { return Some("podman"); }
    if command_runs("docker") { return Some("docker"); }
    None
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
            None    => {
                // nvidia-smi probe failed. Distinguish two sub-cases via
                // sysfs so the warning tells the user what's actually
                // happening on their host:
                //
                //   sysfs sees an NVIDIA card → nvidia-smi is broken /
                //     missing / on a different PATH; we still target this
                //     host, just with the default arch. User should set
                //     $CUDA_ARCHITECTURES if it isn't sm_${fallback_arch}.
                //
                //   sysfs sees no NVIDIA card → assume CI / headless /
                //     cross-compile. Build for the default arch; the user
                //     who actually has a card on a different host can
                //     override.
                if nvidia_gpu_present() {
                    (fallback_arch.to_string(),
                     "fallback (NVIDIA in sysfs, nvidia-smi probe failed)")
                } else {
                    (fallback_arch.to_string(),
                     "fallback (no NVIDIA detected — CI / cross-compile)")
                }
            }
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
        // Surface the container path proactively when we can already
        // see podman/docker — for many users that's the smoothest fix
        // because the toolchain stays bundled in the image.
        let next_steps = match detect_container_engine() {
            Some(engine) => format!(
                "Two ways forward, pick whichever fits:\n\n  \
                   - Install those packages on the host (the cuda-only branch keeps\n    \
                     the dep list intentionally short — no AdaptiveCpp / LLVM / lld):\n      \
                       # apt example (Ubuntu/Debian):\n      \
                       sudo apt install cmake build-essential cuda-toolkit-12-9\n\n  \
                   - Or, since you have {engine} installed, build inside a container —\n    \
                     toolchain stays in the image, no host changes needed:\n      \
                       ./scripts/build-container.sh\n      \
                       {engine} compose run --rm cuda plot ...\n\n\
                 (cuda-only deliberately has no scripts/install-deps.sh — its small\n\
                 dep set is meant to be installed manually or via the container.)"
            ),
            None => format!(
                "Two ways forward, pick whichever fits:\n\n  \
                   - Install those packages on the host (the cuda-only branch keeps\n    \
                     the dep list intentionally short — no AdaptiveCpp / LLVM / lld):\n      \
                       # apt example (Ubuntu/Debian):\n      \
                       sudo apt install cmake build-essential cuda-toolkit-12-9\n\n  \
                   - Or build inside a container (no host toolchain needed beyond\n    \
                     podman or docker — install whichever you prefer first):\n      \
                       ./scripts/build-container.sh\n\n\
                 (cuda-only deliberately has no scripts/install-deps.sh — its small\n\
                 dep set is meant to be installed manually or via the container.)"
            ),
        };
        panic!("\nxchplot2 (cuda-only): build prerequisites missing:\n{bullets}\n\n{next_steps}\n");
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
    //
    // Order matters: the *first* libcudart_static.a the linker finds on
    // `-L` paths is what gets statically linked. If the user has more
    // than one toolkit installed (an old CUDA 11.x leftover plus the
    // CUDA 12.x they actually compiled with), trusting `/usr/local/cuda`
    // or `$CUDA_PATH` can pick the wrong one — exactly the failure
    // mode behind the 0.7.2 user report (linker found a CUDA 11
    // libcudart_static.a, the static archive lacked
    // `cudaGetDeviceProperties_v2`, link died).
    //
    // The reliable source of truth is the nvcc that CMake actually
    // invoked to compile the .o files — its sibling lib dirs hold the
    // matching libcudart_static.a. We canonicalize `which nvcc` to
    // resolve the `/usr/local/cuda` → `cuda-12.x` symlink chain and
    // put *that* toolkit's lib dir first on the search list. The
    // legacy /opt/cuda + /usr/local/cuda + /usr/lib/* fallbacks stay
    // as later entries so plain setups still work without nvcc on PATH.
    let nvcc_toolkit_root = nvcc_canonical_toolkit_root();
    let cuda_root = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .ok()
        .or_else(|| nvcc_toolkit_root.clone())
        .unwrap_or_else(|| {
            for guess in ["/opt/cuda", "/usr/local/cuda"] {
                if std::path::Path::new(guess).exists() { return guess.to_string(); }
            }
            "/opt/cuda".to_string()
        });

    // Emit nvcc's own toolkit dirs FIRST (when distinct from cuda_root),
    // so the linker resolves libcudart_static.a from there ahead of
    // any stale lookalikes under /usr/local/cuda or /opt/cuda.
    if let Some(ref root) = nvcc_toolkit_root {
        if root != &cuda_root {
            println!("cargo:rustc-link-search=native={root}/targets/x86_64-linux/lib");
            println!("cargo:rustc-link-search=native={root}/lib64");
            println!("cargo:rustc-link-search=native={root}/lib");
        }
    }
    println!("cargo:rustc-link-search=native={cuda_root}/lib64");
    println!("cargo:rustc-link-search=native={cuda_root}/lib");
    // Per-host-triple library layout used by recent NVIDIA toolkits
    // (apt repo cuda-toolkit-12-5 and newer reorganised on x86_64 too,
    // not just ARM). Also covers Jetson JetPack/L4T (aarch64-linux)
    // and GH200/SBSA servers. Harmless when the dir doesn't exist.
    println!("cargo:rustc-link-search=native={cuda_root}/targets/x86_64-linux/lib");
    println!("cargo:rustc-link-search=native={cuda_root}/targets/aarch64-linux/lib");
    println!("cargo:rustc-link-search=native={cuda_root}/targets/sbsa-linux/lib");
    // Distro-packaged CUDA fallbacks. Debian/Ubuntu's
    // `apt install nvidia-cuda-toolkit` ships libcudart_static.a /
    // libcudadevrt.a at the multi-arch path /usr/lib/x86_64-linux-gnu,
    // not the /usr/local/cuda layout the NVIDIA apt repo / runfile
    // installer uses. Fedora/RHEL parks them at /usr/lib64. Emit both
    // as additional search paths so cargo install works on stock
    // distro packages too. Gated on dir existence so we don't pollute
    // the search list on non-Linux hosts.
    for extra in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64"] {
        if std::path::Path::new(extra).is_dir() {
            println!("cargo:rustc-link-search=native={extra}");
        }
    }
    // Static-link the CUDA runtime so we don't depend on whatever
    // libcudart.so happens to be earliest on the user's link path.
    // Reported failure was `undefined symbol: cudaGetDeviceProperties_v2`
    // — that symbol was added in CUDA 12.0; users with a stale pre-12
    // libcudart.so somewhere on the linker path (mixed installs, post-
    // upgrade leftovers, certain WSL setups) saw the linker resolve
    // against the old lib even though nvcc compiled against 12-era
    // headers. libcudart_static.a is the toolkit's own runtime, so it
    // always matches our headers and there's nothing to mismatch
    // against. Costs ~600 KB of binary size; eliminates a whole class
    // of distro-install bugs.
    //
    // cudart_static drags in libculibos (CUDA's internal OS shim) plus
    // pthread/dl/rt (already linked below). cudadevrt is .a-only (no
    // .so exists) — separable-compilation device-code linker, always
    // static.
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=static=culibos");
    println!("cargo:rustc-link-lib=static=cudadevrt");

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

/// Locate nvcc on PATH (or under $CUDA_PATH/bin, $CUDA_HOME/bin) and
/// return the canonical (symlink-resolved) parent-of-bin directory.
/// That dir is the toolkit root whose lib subdirs hold the
/// libcudart_static.a that matches what nvcc compiled the .o against.
///
/// Returns None if no nvcc is reachable — caller falls back to the
/// legacy /opt/cuda / /usr/local/cuda probe.
fn nvcc_canonical_toolkit_root() -> Option<String> {
    // Candidate nvcc locations in priority order:
    //   1. $CUDA_PATH/bin/nvcc, $CUDA_HOME/bin/nvcc (explicit user pick)
    //   2. PATH (mirrors CMake's default CUDA-compiler search)
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    for var in &["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(p) = env::var(var) {
            candidates.push(std::path::PathBuf::from(p).join("bin").join("nvcc"));
        }
    }
    if let Ok(path) = env::var("PATH") {
        for dir in env::split_paths(&path) {
            candidates.push(dir.join("nvcc"));
        }
    }
    for cand in candidates {
        if !cand.is_file() { continue; }
        // canonicalize() resolves symlinks — that's exactly what we
        // want: /usr/local/cuda/bin/nvcc → /usr/local/cuda-12.9/bin/nvcc
        // collapses the wrong-symlink scenario the user hit.
        let real = match std::fs::canonicalize(&cand) {
            Ok(p) => p,
            Err(_) => continue,
        };
        // Toolkit root = parent of bin = parent of nvcc's parent.
        if let Some(toolkit) = real.parent().and_then(|bin| bin.parent()) {
            return toolkit.to_str().map(String::from);
        }
    }
    None
}
