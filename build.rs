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
                return Some(name.to_string());
            }
        }
    }
    None
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
            if source != "fallback (no nvidia-smi)" {
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
    // Default ON keeps the existing NVIDIA fast path; AMD/Intel container
    // builds set XCHPLOT2_BUILD_CUDA=OFF to skip nvcc.
    let build_cuda = env::var("XCHPLOT2_BUILD_CUDA").unwrap_or_else(|_| "ON".into());

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
