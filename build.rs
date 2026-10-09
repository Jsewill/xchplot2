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

/// Same probe as `detect_cuda_arch`, but filters out NVIDIA GPUs
/// below our README-documented minimum compute capability (sm_50,
/// Maxwell first-gen / GTX 750-class). The floor used to be sm_61 on
/// the assumption that AdaptiveCpp's `half.hpp` referenced FP16
/// intrinsics (`__hadd` / `__hsub` / `__hmul` / `__hdiv` / `__hlt` /
/// `__hgt`) only available on sm_53+ — but those intrinsics are
/// *implemented* in `cuda_fp16.hpp` via `NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53, …)`
/// with a fp32 emulation fallback for pre-sm_53 cards. CUDA 12.x
/// toolkits compile cleanly for sm_50/52/53. The real floor is the
/// toolkit's own codegen support: CUDA 12.x supports sm_50-90+,
/// CUDA 13.x dropped sm_50-72 (CMakeLists' nvcc-vs-arch preflight
/// catches that pairing with a FATAL_ERROR + fix block).
///
/// Returns Some(arch) only when nvidia-smi reports a card at or
/// above our minimum; emits a cargo:warning and returns None
/// otherwise so callers fall through to the AMD / Intel detection.
///
/// Memoized: three call sites want this answer (ACPP_TARGETS, the
/// XCHPLOT2_BUILD_CUDA selector, and the preflight failure message), and
/// the uncached version forked nvidia-smi once per call AND re-emitted its
/// cargo:warnings each time — a host with a sub-sm_50 card printed the same
/// "below our minimum" paragraph twice.
fn usable_nvidia_arch() -> Option<String> {
    static ARCH: OnceLock<Option<String>> = OnceLock::new();
    ARCH.get_or_init(usable_nvidia_arch_probe).clone()
}

fn usable_nvidia_arch_probe() -> Option<String> {
    let arch = match detect_cuda_arch() {
        Some(a) => a,
        None => {
            // nvidia-smi missing or its `--query-gpu=compute_cap` query
            // failed. Fall back to a sysfs PCI probe so hosts with old
            // drivers or partial enumeration still get NVIDIA-aware
            // build flags. We can't know the real compute_cap from
            // sysfs, so honor $CUDA_ARCHITECTURES if set; otherwise
            // default to sm_75 (Turing — works on every CUDA toolkit
            // 12.x or 13.x without the Maxwell/Pascal/Volta drop).
            if !nvidia_gpu_present() {
                return None;
            }
            let (fallback_arch, source) = match env::var("CUDA_ARCHITECTURES")
                .ok()
                .and_then(|s| min_arch(&s))
            {
                Some(a) => (a.to_string(), "$CUDA_ARCHITECTURES"),
                None => ("75".to_string(), "default (Turing)"),
            };
            println!(
                "cargo:warning=xchplot2: nvidia-smi --query-gpu=compute_cap \
                 failed, but /sys/class/drm reports an NVIDIA GPU (vendor \
                 0x10de). Falling back to sm_{fallback_arch} ({source}). If \
                 your card is older or newer, set $CUDA_ARCHITECTURES \
                 explicitly (e.g. CUDA_ARCHITECTURES=89 for an RTX 4090) \
                 — autodetect can't read the compute_cap from sysfs alone.");
            fallback_arch
        }
    };
    let n: u32 = arch.parse().ok()?;
    if n < 50 {
        println!(
            "cargo:warning=xchplot2: nvidia-smi detected sm_{arch} — below our \
             minimum supported compute capability (sm_50 / Maxwell). CUDA 11.x \
             was the last toolkit to compile for Kepler (sm_30-37); we don't \
             support that path. Ignoring NVIDIA for default targeting; if \
             this card is your only GPU, force the build with \
             CUDA_ARCHITECTURES={arch} + XCHPLOT2_BUILD_CUDA=ON and an \
             appropriately-old CUDA toolkit, or fall back to \
             ACPP_TARGETS=omp for AdaptiveCpp's CPU OpenMP backend.");
        return None;
    }
    if n < 75 && detect_nvcc_major().map(|m| m >= 13).unwrap_or(false) {
        println!(
            "cargo:warning=xchplot2: nvidia-smi detected sm_{arch} (Maxwell / \
             Pascal / Volta) but nvcc is CUDA 13.x, which dropped codegen \
             for sm_50-72. Ignoring NVIDIA for default targeting; install \
             CUDA 12.9 (last toolkit with Maxwell-Volta support) and re-run, \
             or use scripts/build-container.sh which auto-pins the right \
             base image. CMakeLists' preflight will FATAL_ERROR with the \
             exact remediation if you force-build anyway.");
        return None;
    }
    Some(arch)
}

/// Canonical CUDA Toolkit install prefixes — the same roots that
/// scripts/install-deps.sh and CMakeLists.txt already probe:
///   /opt/cuda        Arch / CachyOS / Manjaro (pacman `cuda`)
///   /usr/local/cuda  NVIDIA's .run and .deb installers, NGC / RunPod
///                    images, `cuda-toolkit-X-Y` — usually a symlink to a
///                    versioned /usr/local/cuda-X.Y sibling, which we also
///                    scan (newest first) in case the symlink is absent.
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
/// *login* shell. The shell that just ran install-deps.sh therefore cannot
/// see nvcc, and a PATH-only probe concludes "no CUDA Toolkit" on a box that
/// demonstrably has one — failing the build with an error telling the user
/// to install what they just installed.
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

/// Whether a usable nvcc exists anywhere find_nvcc() looks. Also the
/// fall-back signal for XCHPLOT2_BUILD_CUDA when no GPU is enumerable
/// (headless CI / container builds).
fn detect_nvcc() -> bool {
    find_nvcc().is_some()
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
                let next = iter.next()?;                         // "13.0,"
                let major = next.trim_end_matches(',').split('.').next()?;
                return major.parse().ok();
            }
        }
    }
    None
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

/// Does the host have any NVIDIA GPU? Sysfs PCI vendor-ID probe (0x10de)
/// — same fallback shape as `amd_gpu_present()`. Used by
/// `usable_nvidia_arch()` to recover when `nvidia-smi --query-gpu=
/// compute_cap` fails (older driver, partial enumeration, container
/// missing nvidia-smi binary, etc.) but the host clearly has an NVIDIA
/// card. Doesn't tell us the compute_cap; callers fall back to
/// `$CUDA_ARCHITECTURES` or a sensible default if this returns true.
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

/// Does the host have any AMD GPU detectable by rocminfo? Independent
/// of which ACPP_TARGETS string we pick (including RDNA1 generic SSCP).
/// The GPU is still present and BUILD_CUDA detection
/// should still see it as "AMD host, skip CUDA TUs".
///
/// Falls back to /sys/class/drm vendor-ID probe (0x1002) when rocminfo
/// isn't on $PATH at build time. That happens reliably when users
/// install ROCm via /opt/rocm/bin without sourcing /etc/profile.d/rocm.sh
/// in the shell that runs `cargo install`, or run `cargo install` under
/// systemd / sudo / chroot where the parent shell's PATH is stripped.
/// Without the fallback the BUILD_CUDA selector falls through to the
/// `nvcc present → ON, "CI fallback"` arm, the build links CUB, and the
/// streaming pipeline dies on first sort dispatch against the AMD card.
fn amd_gpu_present() -> bool {
    if let Ok(out) = Command::new("rocminfo").output() {
        if out.status.success() {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if s.lines().any(|l| {
                    l.trim().strip_prefix("Name:")
                        .map(|rest| rest.trim().starts_with("gfx"))
                        .unwrap_or(false)
                }) {
                    return true;
                }
            }
        }
    }
    // PCI fallback — same pattern as detect_intel_gpu(). Doesn't need any
    // user-space tools, only readable sysfs (true on every Linux host
    // with the amdgpu / radeon kernel module loaded).
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
            if v.trim() == "0x1002" {
                return true;
            }
        }
    }
    false
}

/// Ask rocminfo for the first AMD GPU architecture. TargetSelection.cmake
/// applies RDNA1 defaults and the legacy opt-in overrides.
fn detect_amd_gfx() -> Option<String> {
    let out = Command::new("rocminfo").output().ok()?;
    if !out.status.success() { return None; }
    let text = std::str::from_utf8(&out.stdout).ok()?;
    text.lines().filter_map(|line| line.trim().strip_prefix("Name:"))
        .map(str::trim).find(|name| name.starts_with("gfx")).map(String::from)
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
/// set, otherwise the install-deps.sh default of /opt/adaptivecpp; also
/// recognize the per-user ~/.local install made by the CMake fallback.
/// When this is true the FetchContent fallback won't fire and
/// AdaptiveCpp's own build-time deps (notably ld.lld) aren't needed
/// for our build.
fn adaptivecpp_installed() -> bool {
    let prefix = env::var("ACPP_PREFIX")
        .unwrap_or_else(|_| "/opt/adaptivecpp".to_string());
    let local = format!("{}/.local", env::var("HOME").unwrap_or_default());
    [prefix, local].iter().any(|root| std::path::Path::new(&format!(
        "{root}/lib/cmake/AdaptiveCpp/adaptivecpp-config.cmake"
    )).exists())
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
        missing.push(
            "nvcc (CUDA Toolkit 12+) — XCHPLOT2_BUILD_CUDA=ON requested, but no runnable \
             nvcc on $PATH, under $CUDA_PATH / $CUDA_HOME, or in /opt/cuda or /usr/local/cuda*.\n    \
             If the toolkit IS installed somewhere else, point us at it:\n      \
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
    //   3. 89 (sm_89, RTX 4090 / Ada Lovelace) as a sensible default for
    //      machines without nvidia-smi (e.g. CI, headless package builds).
    //
    // Reported below rather than here: every CMAKE_CUDA_ARCHITECTURES
    // consumer in CMakeLists.txt sits inside an `if(XCHPLOT2_BUILD_CUDA)`,
    // so on an AMD / Intel host this value is inert. Announcing
    // "building for CUDA arch 89 (fallback (no nvidia-smi))" before we've
    // decided BUILD_CUDA made a missing nvidia-smi look like the cause of
    // any later failure, and sent non-NVIDIA users hunting for a driver
    // they don't need.
    let (cuda_arch, arch_source) = match env::var("CUDA_ARCHITECTURES") {
        Ok(v) => (v, "$CUDA_ARCHITECTURES"),
        Err(_) => match detect_cuda_arch() {
            Some(v) => (v, "nvidia-smi probe"),
            None    => ("89".to_string(), "fallback (no nvidia-smi)"),
        },
    };

    // Keep platform probes and prerequisite diagnostics here; CMake owns
    // the target policy shared with standalone builds.
    let nvidia_gpu = usable_nvidia_arch().is_some();
    let amd_gpu = amd_gpu_present();
    let intel_gpu = detect_intel_gpu();
    let selection = cmake_build.join("target-selection.txt");
    if !command_runs("cmake") {
        let missing = preflight(false).join("\n  - ");
        panic!("xchplot2: build prerequisites missing:\n  - {missing}\nInstall with {} or build with {}",
            manifest_dir.join("scripts/install-deps.sh").display(),
            manifest_dir.join("scripts/build-container.sh").display());
    }
    let mut select = Command::new("cmake");
    select.arg(format!("-DX2_HAVE_NVIDIA={nvidia_gpu}"))
        .arg(format!("-DX2_HAVE_AMD={amd_gpu}"))
        .arg(format!("-DX2_HAVE_INTEL={intel_gpu}"))
        .arg(format!("-DX2_HAVE_NVCC={}", detect_nvcc()))
        .arg(format!("-DX2_AMD_GFX={}", detect_amd_gfx().unwrap_or_default()))
        .arg(format!("-DX2_SELECTION_OUTPUT={}", selection.display()));
    for var in ["ACPP_TARGETS", "XCHPLOT2_BUILD_CUDA"] {
        if let Ok(value) = env::var(var) {
            select.arg(format!("-D{var}={value}"));
        }
    }
    let status = select.arg("-P").arg(manifest_dir.join("cmake/TargetSelection.cmake"))
        .status().expect("failed to invoke cmake — is it installed?");
    assert!(status.success(), "CMake target selection failed");
    let selected = std::fs::read_to_string(selection).expect("read CMake target selection");
    let mut values = selected.lines();
    let acpp_targets = values.next().expect("AdaptiveCpp target selection").to_string();
    let build_cuda = values.next().expect("CUDA build selection").to_string();
    println!("cargo:warning=xchplot2: ACPP_TARGETS={acpp_targets}");
    println!("cargo:warning=xchplot2: XCHPLOT2_BUILD_CUDA={build_cuda}");

    // (prose label, matching service in compose.yaml). None = nothing
    // enumerable — headless CI, or a container without /sys/class/drm.
    let vendor: Option<(&str, &str)> = if nvidia_gpu {
        Some(("NVIDIA", "cuda"))
    } else if amd_gpu {
        Some(("AMD", "rocm"))
    } else if intel_gpu {
        Some(("Intel", "intel"))
    } else {
        None
    };

    // Deferred from the arch block above: only meaningful once we know the
    // CUDA TUs are actually being compiled.
    if build_cuda == "ON" {
        println!("cargo:warning=xchplot2: building for CUDA arch {cuda_arch} ({arch_source})");
    }

    // Preflight critical system deps BEFORE configuring dependencies. Cargo
    // install users land here without reading README.md's Build
    // section; without preflight, missing deps surface as cryptic
    // CMake / AdaptiveCpp errors deep in the configure / build.
    let missing = preflight(build_cuda == "ON");
    if !missing.is_empty() {
        let bullets = missing.iter()
            .map(|m| format!("  - {m}"))
            .collect::<Vec<_>>()
            .join("\n");
        // Absolute paths, not `./scripts/...`. The audience for this panic
        // is `cargo install --git` users, whose shell is in some unrelated
        // cwd while the sources sit under
        // ~/.cargo/git/checkouts/xchplot2-<hash>/<rev>/ — a relative path
        // is not runnable for exactly the people it's addressed to.
        let scripts         = manifest_dir.join("scripts");
        let install_deps    = scripts.join("install-deps.sh");
        let build_container = scripts.join("build-container.sh");
        let install_deps    = install_deps.display();
        let build_container = build_container.display();

        // install-deps.sh auto-detects the vendor when --gpu is omitted,
        // using the same PCI precedence we do for all three vendors.
        let host_install = format!(
            "- Install those packages on the host — it auto-detects your GPU\n    \
               vendor and builds AdaptiveCpp:\n      \
                 {install_deps}"
        );

        // Say plainly that this isn't an NVIDIA problem when we've already
        // routed the build away from CUDA. Without it, the arch line
        // further up ("fallback (no nvidia-smi)") reads as the cause and
        // AMD / Intel users go install a driver that changes nothing.
        let vendor_note = match vendor {
            Some((label, _)) if label != "NVIDIA" => format!(
                "None of the above is NVIDIA-specific. This build already selected the \
                 {label}\npath (XCHPLOT2_BUILD_CUDA=OFF), so neither nvidia-smi nor nvcc is \
                 needed —\nthey are host toolchain packages, missing on any fresh machine.\n\n"
            ),
            _ => String::new(),
        };

        // Lead the container example with the service matching the GPU we
        // found, so it's copy-pasteable rather than aspirational.
        let service = vendor.map(|(_, svc)| svc).unwrap_or("cpu");

        // Surface the container path proactively when we can already
        // see podman/docker — for many users that's the smoothest fix
        // because the toolchain stays bundled in the image.
        let next_steps = match detect_container_engine() {
            Some(engine) => format!(
                "Two ways forward, pick whichever fits:\n\n  \
                   {host_install}\n\n  \
                   - Or, since you have {engine} installed, build inside a container —\n    \
                     toolchain stays in the image, no host changes needed:\n      \
                       {build_container}\n      \
                       {engine} compose run --rm {service} plot ...\n\n\
                 If install-deps.sh just ran and you're still seeing this, check\n\
                 its tail output — it names the failed package before exiting."
            ),
            None => format!(
                "Two ways forward, pick whichever fits:\n\n  \
                   {host_install}\n\n  \
                   - Or build inside a container (no host toolchain needed beyond\n    \
                     podman or docker — install whichever you prefer first):\n      \
                       {build_container}\n\n\
                 If install-deps.sh just ran and you're still seeing this, check\n\
                 its tail output — it names the failed package before exiting."
            ),
        };
        panic!("\nxchplot2: build prerequisites missing:\n{bullets}\n\n{vendor_note}{next_steps}\n");
    }

    // CMake performs nvcc floor/ceiling compatibility checks for both builds.

    // Cargo's job budget applies to CMake and its dependency bootstrap too.
    let jobs = env::var("CMAKE_BUILD_PARALLEL_LEVEL").ok().filter(|v| !v.is_empty())
        .or_else(|| env::var("NUM_JOBS").ok()).unwrap_or_else(|| "1".to_string());

    // ---- configure ----
    let mut configure = Command::new("cmake");
    configure
        .env("CMAKE_BUILD_PARALLEL_LEVEL", &jobs)
        .args([
            "-S", manifest_dir.to_str().unwrap(),
            "-B", cmake_build.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
        ])
        .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"))
        .arg(format!("-DACPP_TARGETS={acpp_targets}"))
        .arg(format!("-DXCHPLOT2_BUILD_CUDA={build_cuda}"));

    // Hand CMake the exact nvcc preflight just validated. enable_language(CUDA)
    // otherwise runs CMake's own toolkit search, which covers $CUDAToolkit_ROOT,
    // $CUDA_PATH, $PATH and /usr/local/cuda — but NOT /opt/cuda, where Arch puts
    // it. Without this, a host whose toolkit we version-checked seconds earlier
    // still fails configure with "Failed to find nvcc. Please set the
    // CUDAToolkit_ROOT variable." Passing it explicitly also removes any skew
    // between the nvcc we checked and the one CMake would have picked.
    if build_cuda == "ON" {
        if let Some(nvcc) = find_nvcc() {
            configure.arg(format!("-DCMAKE_CUDA_COMPILER={}", nvcc.display()));
            if let Some(root) = nvcc.parent().and_then(|bin| bin.parent()) {
                configure.arg(format!("-DCUDAToolkit_ROOT={}", root.display()));
            }
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
        .arg(&jobs)
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
    println!("cargo:rustc-link-lib=static=pos2_sha256");
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

    // ---- LLVM OpenMP runtime (SYCL→OMP backend) ----
    // AdaptiveCpp's OMP backend lowers SYCL nd_range kernels to OpenMP
    // parallel loops. The compiled .o files reference libomp's runtime
    // symbols (__kmpc_fork_call, __kmpc_global_thread_num, __kmpc_barrier,
    // __kmpc_for_static_init_8u / _fini). cc / rust-lld don't auto-link
    // libomp — pos2_gpu's SYCL TUs would then fail to link with
    //
    //   rust-lld: error: undefined symbol: __kmpc_fork_call
    //
    // Only fire on builds where ACPP_TARGETS includes "omp"; HIP and
    // SSCP-with-CUDA backends translate to their own runtimes and don't
    // need libomp at link time.
    //
    // Locations:
    //   Ubuntu/Debian (apt libomp-18-dev): /usr/lib/llvm-18/lib/libomp.so
    //   Arch (pacman openmp):              /usr/lib/libomp.so
    //   AdaptiveCpp install (bundled):     $ACPP_PREFIX/lib/libomp.so
    if acpp_targets.split(';').any(|t| t.trim() == "omp") {
        for guess in ["/usr/lib/llvm-18/lib", "/usr/lib/llvm-19/lib",
                      "/usr/lib/llvm-20/lib", "/usr/lib"] {
            if std::path::Path::new(&format!("{guess}/libomp.so")).exists()
                || std::path::Path::new(&format!("{guess}/libomp.so.5")).exists() {
                println!("cargo:rustc-link-search=native={guess}");
                println!("cargo:rustc-link-arg=-Wl,-rpath,{guess}");
                break;
            }
        }
        println!("cargo:rustc-link-lib=omp");
    }

    // ---- CUDA runtime ----
    // Only needed when XCHPLOT2_BUILD_CUDA=ON — then the nvcc-compiled
    // TUs (SortCuda, AesGpu, AesGpuBitsliced) pull in cudart / cudadevrt.
    // On the AMD/Intel OFF path there's no CUDA Toolkit on the image and
    // nothing in the static archives references cudart, so emitting
    // `-lcudart` would make rust-lld fail with "unable to find library".
    if build_cuda == "ON" {
        // Order matters: the *first* libcudart_static.a the linker
        // finds wins. If the user has multiple toolkits installed and
        // /usr/local/cuda symlinks to a stale CUDA 11.x leftover, we'd
        // statically link the wrong runtime and fail on the v2 ABI.
        // The reliable source of truth is the nvcc that CMake actually
        // invoked — its sibling lib dirs always hold a matching
        // libcudart_static.a. Canonicalize `which nvcc` to resolve
        // the `/usr/local/cuda` symlink chain and put that toolkit's
        // lib dirs first on the search list. See cuda-only branch
        // commit history for the user-bug-report context.
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
        // nvcc's own toolkit dirs FIRST (when distinct from cuda_root),
        // so the linker resolves libcudart_static.a from there ahead
        // of any stale lookalikes under /usr/local/cuda or /opt/cuda.
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
        // (apt repo cuda-toolkit-12-5+ reorganised x86_64 too, not just
        // ARM). Also covers Jetson JetPack/L4T (aarch64-linux) and
        // GH200/SBSA servers. Harmless when the dir doesn't exist.
        println!("cargo:rustc-link-search=native={cuda_root}/targets/x86_64-linux/lib");
        println!("cargo:rustc-link-search=native={cuda_root}/targets/aarch64-linux/lib");
        println!("cargo:rustc-link-search=native={cuda_root}/targets/sbsa-linux/lib");
        // Distro-packaged CUDA fallbacks. Debian/Ubuntu's
        // `apt install nvidia-cuda-toolkit` ships libcudart_static.a /
        // libcudadevrt.a at the multi-arch path /usr/lib/x86_64-linux-gnu,
        // not the /usr/local/cuda layout the NVIDIA apt repo / runfile
        // installer uses. Fedora/RHEL parks them at /usr/lib64. Emit
        // both as additional search paths so cargo install works on
        // stock distro packages too. Gated on dir existence so we don't
        // pollute the search list on non-Linux hosts.
        for extra in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64"] {
            if std::path::Path::new(extra).is_dir() {
                println!("cargo:rustc-link-search=native={extra}");
            }
        }
        // Static-link the CUDA runtime so we don't depend on whatever
        // libcudart.so happens to be earliest on the user's link path.
        // Reported failure was `undefined symbol: cudaGetDeviceProperties_v2`
        // — that symbol was added in CUDA 12.0; users with a stale
        // pre-12 libcudart.so somewhere on the linker path (mixed
        // installs, post-upgrade leftovers, certain WSL setups) saw
        // the linker resolve against the old lib even though nvcc
        // compiled against 12-era headers. libcudart_static.a is the
        // toolkit's own runtime, so it always matches our headers and
        // there's nothing to mismatch against. Costs ~600 KB of binary
        // size; eliminates a whole class of distro-install bugs.
        //
        // cudart_static drags in libculibos (CUDA's internal OS shim)
        // plus pthread/dl/rt (already linked below). cudadevrt is
        // .a-only (no .so) — separable-compilation device-code linker,
        // always static.
        println!("cargo:rustc-link-lib=static=cudart_static");
        println!("cargo:rustc-link-lib=static=culibos");
        println!("cargo:rustc-link-lib=static=cudadevrt");

        // WSL defensive rpath. libcudart_static's dlopen("libcuda.so.1")
        // needs /usr/lib/wsl/lib on the runtime loader path. WSL distros
        // usually set that up via /etc/ld.so.conf.d/ld.wsl.conf, but
        // non-wslg / custom images can be missing the entry — then the
        // binary installs fine but fails at first GPU call. Bake
        // /usr/lib/wsl/lib into the binary's runtime search path.
        //
        // --disable-new-dtags emits DT_RPATH (legacy) instead of
        // DT_RUNPATH. We need DT_RPATH because it propagates to dlopen
        // calls made from libraries linked into this binary (libcudart);
        // DT_RUNPATH only helps DT_NEEDED resolution we declare directly.
        //
        // No cost on non-WSL: loader hits the missing dir, skips it.
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/wsl/lib");
        println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    }

    // ---- HIP runtime ----
    // When ACPP_TARGETS is "hip:gfxXXXX", AdaptiveCpp's HIP backend
    // compiles SYCL kernels into HIP fat binaries whose host-side
    // launcher stubs reference __hipPushCallConfiguration /
    // __hipRegisterFatBinary / hipLaunchKernel from libamdhip64. Without
    // -lamdhip64 rust-lld fails with "undefined symbol: __hip*".
    // Honour $ROCM_PATH if set, else fall back to /opt/rocm (standard
    // bare-metal + all official ROCm container images).
    // Link libamdhip64 whenever ROCm is reachable, not just when
    // ACPP_TARGETS is hip-prefixed. ACPP_TARGETS=generic (SSCP JIT) on
    // an AMD host still needs the HIP runtime at load time —
    // librt-backend-hip.so dlopens libamdhip64, but glibc doesn't walk
    // the binary's RUNPATH for transitive backend deps. By making
    // libamdhip64 a direct dependency of the binary, the loader pulls
    // it in at startup via RUNPATH, and AdaptiveCpp's runtime dlopen
    // finds the already-loaded handle. Without this, an AMD-host
    // build with the new RDNA1 default (generic instead of the
    // gfx1013 spoof) fails at first queue construction with
    // "No matching device" because HIP can't initialise.
    //
    // We pass the full .so path (rather than `cargo:rustc-link-lib=amdhip64`
    // which becomes `-lamdhip64`) because the SSCP path emits no host-
    // side HIP symbol references, and the linker's default --as-needed
    // would drop a name-only -l flag from NEEDED. A positional path
    // argument bypasses --as-needed and keeps the library in the link.
    // Same approach as CMakeLists.txt's `link_libraries(.../libamdhip64.so)`.
    let rocm_root = env::var("ROCM_PATH")
        .unwrap_or_else(|_| "/opt/rocm".to_string());
    // <root>/lib is AMD's own installer layout. Multilib distros package the
    // runtime elsewhere — Fedora's rocm-hip ships /usr/lib64/libamdhip64.so and
    // has no /opt/rocm at all — so probing only <root>/lib missed it entirely
    // there. On a hip:gfx* target that merely cost us the positional-path trick
    // below (the -l fallback still resolves from the default search path), but
    // on ACPP_TARGETS=generic the whole block was skipped, and per the note
    // above that means libamdhip64 never reaches DT_NEEDED, the HIP backend
    // never initialises, and the plotter runs on the OpenMP host device while
    // reporting success. That is the RDNA1 default path, so it was reachable.
    let amdhip_lib = [format!("{rocm_root}/lib"), format!("{rocm_root}/lib64"),
                      "/usr/lib/x86_64-linux-gnu".to_string(),
                      "/usr/lib64".to_string(), "/usr/lib".to_string()]
        .into_iter()
        .map(|dir| format!("{dir}/libamdhip64.so"))
        .find(|path| std::path::Path::new(path).exists());
    if acpp_targets.starts_with("hip:") || amdhip_lib.is_some() {
        println!("cargo:rustc-link-search=native={rocm_root}/lib");
        println!("cargo:rustc-link-search=native={rocm_root}/hip/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{rocm_root}/lib");
        if let Some(ref amdhip_lib) = amdhip_lib {
            let libdir = std::path::Path::new(amdhip_lib)
                .parent()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| format!("{rocm_root}/lib"));
            println!("cargo:rustc-link-search=native={libdir}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
            // Wrap with --no-as-needed/--as-needed: even a positional
            // .so path gets dropped from NEEDED by ld's --as-needed
            // when no symbol references it (true for the SSCP path
            // that has zero host-side HIP symbol refs). The library
            // itself must end up in DT_NEEDED so AdaptiveCpp's runtime
            // dlopen finds it already loaded; otherwise HIP backend
            // never initialises and we throw "No matching device".
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg={amdhip_lib}");
            println!("cargo:rustc-link-arg=-Wl,--as-needed");
        } else {
            // Fallback: the runtime wasn't in any probed directory, but
            // the user set ACPP_TARGETS=hip:* explicitly — so trust them
            // and let the linker's own search path find it. AOT HIP fat
            // binaries reference HIP symbols directly, so --as-needed
            // keeps -lamdhip64 in NEEDED on that path.
            println!("cargo:rustc-link-lib=amdhip64");
        }
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
        "keygen-rs/Cargo.lock", "CMakeLists.txt", "cmake", "build.rs",
    ] {
        println!("cargo:rerun-if-changed={p}");
    }
    // Every env var read anywhere in this script must appear here, or a
    // user flipping it (e.g. XCHPLOT2_BUILD_CUDA=OFF after a failed
    // build) silently keeps the stale configuration.
    for var in &[
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "CUDA_ARCHITECTURES",
        "CUDAToolkit_ROOT",
        "PATH",
        "CUDA_PATH",
        "CUDA_HOME",
        "ACPP_TARGETS",
        "ACPP_PREFIX",
        "AdaptiveCpp_ROOT",
        "ROCM_PATH",
        "XCHPLOT2_BUILD_CUDA",
        "XCHPLOT2_FORCE_GFX_SPOOF",
        "XCHPLOT2_NO_GFX_SPOOF",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
}

/// The canonical (symlink-resolved) toolkit root of the nvcc find_nvcc()
/// resolved — i.e. the parent of its bin/ dir. That root's lib subdirs hold
/// the libcudart_static.a matching what nvcc compiled the .o against.
///
/// Returns None if no nvcc is reachable — caller falls back to the
/// legacy /opt/cuda / /usr/local/cuda probe.
fn nvcc_canonical_toolkit_root() -> Option<String> {
    let real = std::fs::canonicalize(find_nvcc()?).ok()?;
    let toolkit = real.parent().and_then(|bin| bin.parent())?;
    toolkit.to_str().map(String::from)
}
