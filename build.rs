// build.rs — drive the existing CMake build to produce the static libs
// that the Rust `[[bin]] xchplot2` then links against.
//
// The CMake build is the authoritative one (CUDA, separable compilation,
// pos2-chip FetchContent, the keygen-rs Rust shim). We just call it from
// here so a `cargo install` works end-to-end on a machine with the build
// dependencies listed in README.md (CMake ≥ 3.24, CUDA Toolkit, C++20
// compiler, and a Rust toolchain — the last one cargo provides).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

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

/// Canonical CUDA Toolkit install prefixes — the same roots CMakeLists.txt's
/// find_program() already hints:
///   /opt/cuda        Arch / CachyOS / Manjaro (pacman `cuda`)
///   /usr/local/cuda  NVIDIA's .run and .deb installers, NGC / RunPod images,
///                    `cuda-toolkit-X-Y` — usually a symlink to a versioned
///                    /usr/local/cuda-X.Y sibling, which we also scan (newest
///                    first) in case the symlink is absent.
fn cuda_prefixes() -> Vec<PathBuf> {
    let mut prefixes = vec![
        PathBuf::from("/opt/cuda"),
        PathBuf::from("/usr/local/cuda"),
    ];
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        let mut versioned: Vec<(u32, u32, PathBuf)> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter_map(|p| {
                let name = p.file_name()?.to_str()?.to_string();
                let ver = name.strip_prefix("cuda-")?;
                let mut parts = ver.split('.');
                let major: u32 = parts.next()?.parse().ok()?;
                let minor: u32 = parts.next().and_then(|m| m.parse().ok()).unwrap_or(0);
                Some((major, minor, p))
            })
            .collect();
        // Sort on the parsed version, newest first. Sorting the strings would
        // rank cuda-9.2 above cuda-12.8, and since this list is the last-resort
        // probe that would silently select an ancient toolkit on a host with no
        // /usr/local/cuda symlink — trading a clean "no nvcc" error for a
        // cryptic C++20 failure deep inside nvcc.
        versioned.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
        prefixes.extend(versioned.into_iter().map(|(_, _, path)| path));
    }
    prefixes
}

/// True when this path is an nvcc that actually executes. Runs
/// `nvcc --version` rather than testing the exec bit so a stale symlink
/// or a wrong-arch binary doesn't pass.
fn nvcc_runs(nvcc: &Path) -> bool {
    nvcc.is_file()
        && Command::new(nvcc)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
}

/// Resolve nvcc to an absolute, runnable path.
///
/// Probe order: $CUDAToolkit_ROOT / $CUDA_PATH / $CUDA_HOME (an explicit
/// override wins) → $PATH → the canonical install prefixes. The env vars
/// deliberately outrank PATH: that is CMake's own CUDAToolkit precedence,
/// and it is how a user picks between several installed toolkits when a
/// stale nvcc also sits on PATH.
///
/// The last step is the one that earns its keep. Distro packages do not all
/// leave nvcc on a default PATH: Arch's `cuda` installs it to /opt/cuda/bin
/// and front-loads PATH from /etc/profile.d/cuda.sh — which calls
/// append_path(), a helper defined in /etc/profile, so it only fires in a
/// *login* shell. A PATH-only probe therefore reports "no CUDA Toolkit" on a
/// box that demonstrably has one, and the preflight below tells the user to
/// go download the toolkit they already installed.
fn find_nvcc() -> Option<&'static Path> {
    static NVCC: OnceLock<Option<PathBuf>> = OnceLock::new();
    NVCC.get_or_init(|| {
        let mut candidates: Vec<PathBuf> = Vec::new();
        for var in ["CUDAToolkit_ROOT", "CUDA_PATH", "CUDA_HOME"] {
            if let Ok(root) = env::var(var) {
                if !root.is_empty() {
                    candidates.push(PathBuf::from(root).join("bin").join("nvcc"));
                }
            }
        }
        if let Ok(path) = env::var("PATH") {
            candidates.extend(env::split_paths(&path).map(|dir| dir.join("nvcc")));
        }
        candidates.extend(cuda_prefixes().iter().map(|p| p.join("bin").join("nvcc")));
        candidates.into_iter().find(|c| nvcc_runs(c))
    })
    .as_deref()
}

/// Parse nvcc's major version from `nvcc --version` output.
/// The release line looks like:
///   "Cuda compilation tools, release 13.0, V13.0.48"
/// Returns None if no nvcc is reachable or the line can't be parsed —
/// callers treat that as "skip the version-vs-arch compat check"
/// rather than blocking the build.
fn detect_nvcc_major() -> Option<u32> {
    let out = Command::new(find_nvcc()?).arg("--version").output().ok()?;
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

/// The virtual GPU architectures the installed nvcc can actually codegen,
/// via `nvcc --list-gpu-arch` (CUDA 11.5+). Output is one `compute_XX`
/// per line — and NOT sorted (CUDA 13.x prints
/// `compute_100 compute_110 compute_103 compute_120 compute_121`), so the
/// caller must take the numeric max, not the last line. Returns the parsed
/// arches (e.g. [75,80,86,89,90]); None when nvcc is missing, predates the
/// flag, or the output can't be parsed — callers then skip the ceiling
/// check and let cmake try, preserving prior behaviour.
fn nvcc_supported_arches() -> Option<Vec<u32>> {
    let out = Command::new(find_nvcc()?).arg("--list-gpu-arch").output().ok()?;
    if !out.status.success() { return None; }
    let s = std::str::from_utf8(&out.stdout).ok()?;
    let v: Vec<u32> = s.lines()
        .filter_map(|l| l.trim().strip_prefix("compute_"))
        // `compute_90a` and friends -> drop the arch-feature suffix
        .filter_map(|n| n.split(|c: char| !c.is_ascii_digit()).next())
        .filter_map(|n| n.parse().ok())
        .collect();
    if v.is_empty() { None } else { Some(v) }
}

/// Parse one CMake CUDA_ARCHITECTURES token to its integer arch,
/// tolerating the `sm_`/`compute_` prefixes Cargo users pass through and
/// the `-real`/`-virtual` suffixes CMake accepts ("sm_90" / "compute_90"
/// / "90-virtual" -> 90). None for non-numeric tokens ("native", "all").
fn arch_num(tok: &str) -> Option<u32> {
    let t = tok.trim()
        .trim_start_matches("sm_")
        .trim_start_matches("compute_");
    t.split('-').next().unwrap_or(t).parse().ok()
}

/// Minimum integer arch from a CMake-style CUDA_ARCHITECTURES list
/// ("61", "61;86", "61;86;120"). None when nothing parses.
fn min_arch(arch_list: &str) -> Option<u32> {
    arch_list.split(';').filter_map(arch_num).min()
}

/// Maximum integer arch from a CMake-style CUDA_ARCHITECTURES list.
/// Used to catch a target arch NEWER than the installed nvcc can
/// codegen (e.g. sm_120 Blackwell on a CUDA 12.4 toolkit).
fn max_arch(arch_list: &str) -> Option<u32> {
    arch_list.split(';').filter_map(arch_num).max()
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
    if find_nvcc().is_none() {
        missing.push(
            "nvcc (CUDA Toolkit 12+) — not on $PATH, under $CUDA_PATH / $CUDA_HOME, \
             or in /opt/cuda or /usr/local/cuda*.\n    \
             Install it (developer.nvidia.com/cuda-downloads, apt cuda-toolkit-12-X, \
             pacman -S cuda), or\n    \
             if it IS installed somewhere else, point us at it:\n      \
             export CUDA_PATH=/path/to/cuda    # the dir holding bin/nvcc"
                .into(),
        );
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
    let (mut cuda_arch, source) = match env::var("CUDA_ARCHITECTURES") {
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

    // Symmetric ceiling check: a target arch NEWER than this nvcc can
    // codegen. A Blackwell GeForce RTX 50-series reports compute_cap 12.0,
    // so autodetect targets sm_120 — but only CUDA 12.8+ added compute_120
    // codegen. On an older toolkit nvcc dies with
    //   "nvcc fatal: Unsupported gpu architecture 'compute_120'"
    // deep inside a CMake TryCompile (exactly the cargo-install failure
    // Blackwell users hit on CUDA 12.4). PTX is forward-compatible, so
    // rather than fail we emit PTX for the highest arch THIS nvcc supports;
    // the driver JIT-compiles it to the real GPU at runtime (the driver
    // clearly knows the GPU — nvidia-smi just read its compute_cap). Native
    // SASS for any already-supported arch in the list is preserved. We
    // loudly recommend a 12.8+ toolkit for native codegen.
    if let (Some(supported), Some(want)) =
        (nvcc_supported_arches(), max_arch(&cuda_arch))
    {
        let max_supported = supported.iter().copied().max().unwrap();
        if want > max_supported {
            // Keep every requested arch the toolkit CAN target as-is
            // (real SASS); replace the too-new ones with one PTX entry
            // from the highest supported arch.
            let mut kept: Vec<String> = cuda_arch
                .split(';')
                .filter(|tok| arch_num(tok).map_or(true, |n| n <= max_supported))
                .map(|tok| tok.trim().to_string())
                .collect();
            let ptx = format!("{max_supported}-virtual");
            if !kept.iter().any(|k| *k == ptx) {
                kept.push(ptx);
            }
            let new_arch = kept.join(";");
            println!(
                "cargo:warning=xchplot2 (cuda-only): target sm_{want} is newer than this \
                 CUDA Toolkit can codegen (nvcc tops out at compute_{max_supported}; \
                 sm_100/sm_120 Blackwell need CUDA 12.8+). Falling back to \
                 CUDA_ARCHITECTURES={new_arch} — PTX the driver JIT-compiles to your GPU \
                 at runtime. For native SASS install CUDA 12.8+ and rebuild with \
                 CUDA_PATH=/usr/local/cuda-12.8 (or set CUDA_ARCHITECTURES explicitly)."
            );
            cuda_arch = new_arch;
        }
    }

    // ---- configure ----
    let mut configure = Command::new("cmake");
    configure
        .args([
            "-S", manifest_dir.to_str().unwrap(),
            "-B", cmake_build.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
        ])
        .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"));

    // Hand CMake the exact nvcc preflight just validated. project(LANGUAGES
    // ... CUDA) otherwise runs CMake's own toolkit search, which covers
    // $CUDAToolkit_ROOT, $CUDA_PATH, $PATH and /usr/local/cuda — but NOT
    // /opt/cuda, where Arch puts it. Without this, a host whose toolkit we
    // version-checked seconds earlier still fails configure with "Failed to
    // find nvcc. Please set the CUDAToolkit_ROOT variable." Passing it also
    // removes any skew between the nvcc we checked (arch guards) and the one
    // CMake would have picked.
    if let Some(nvcc) = find_nvcc() {
        configure.arg(format!("-DCMAKE_CUDA_COMPILER={}", nvcc.display()));
        if let Some(root) = nvcc.parent().and_then(|bin| bin.parent()) {
            configure.arg(format!("-DCUDAToolkit_ROOT={}", root.display()));
        }
    }

    let status = configure
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

    // WSL defensive rpath. The statically-linked libcudart code calls
    // dlopen("libcuda.so.1") at first GPU touch — that driver lib lives
    // at /usr/lib/wsl/lib on WSL2 (injected from the Windows side, not
    // owned by the Linux distro). WSL distros are *supposed* to register
    // that dir via /etc/ld.so.conf.d/ld.wsl.conf, but on non-wslg setups,
    // custom WSL images, or pre-injection-era distros the entry can be
    // missing — then the binary installs fine but fails at first GPU
    // call with `libcuda.so.1: cannot open shared object file`.
    //
    // Bake /usr/lib/wsl/lib into the binary's runtime search path so
    // we don't depend on the loader's system-wide config being right.
    // --disable-new-dtags is the meaningful bit: it emits DT_RPATH
    // (legacy) instead of DT_RUNPATH. DT_RUNPATH only helps DT_NEEDED
    // entries declared by *us* — we don't declare libcuda directly,
    // libcudart_static does it via dlopen, so RUNPATH wouldn't apply.
    // DT_RPATH propagates to dlopen calls made from within the same
    // module (our binary, into which libcudart was linked).
    //
    // No cost on non-WSL: the loader hits the missing dir, skips it,
    // moves on through the usual search.
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/wsl/lib");
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");

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
    // find_nvcc() applies the priority order this used to hand-roll
    // ($CUDA_PATH / $CUDA_HOME first, then PATH) and additionally knows the
    // canonical install prefixes. Sharing it also guarantees the toolkit we
    // link against is the one CMake compiled the .o files with — build.rs
    // passes the very same nvcc as -DCMAKE_CUDA_COMPILER.
    //
    // canonicalize() resolves symlinks — exactly what we want:
    // /usr/local/cuda/bin/nvcc → /usr/local/cuda-12.9/bin/nvcc collapses the
    // wrong-symlink scenario the user hit. Toolkit root = parent of bin.
    let real = std::fs::canonicalize(find_nvcc()?).ok()?;
    let toolkit = real.parent().and_then(|bin| bin.parent())?;
    toolkit.to_str().map(String::from)
}
