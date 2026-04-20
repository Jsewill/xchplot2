# GPU portability sketch: porting `compute_bucket_offsets` to SYCL and Vulkan

This document ports one representative kernel from `src/gpu/T1Kernel.cu` —
`compute_bucket_offsets` — to two cross-vendor GPU technologies, so the
relative cost of each path can be compared concretely on real plotter code.

`compute_bucket_offsets` is a good probe: it is small, has no AES /
shared-memory dependency, uses one global atomic-free pattern (one thread per
bucket runs a binary search over a sorted stream), and exercises every
mechanism the rest of the pipeline needs — restrict pointers, struct-of-arrays
loads, sentinel writes, and a 1-D launch.

Source (CUDA, current code, [`src/gpu/T1Kernel.cu:58`](../src/gpu/T1Kernel.cu)):

```cuda
__global__ void compute_bucket_offsets(
    XsCandidateGpu const* __restrict__ sorted,
    uint64_t total,
    int num_match_target_bits,
    uint32_t num_buckets,
    uint64_t* __restrict__ offsets)
{
    uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b > num_buckets) return;
    if (b == num_buckets) { offsets[num_buckets] = total; return; }

    uint32_t bucket_shift = static_cast<uint32_t>(num_match_target_bits);
    uint64_t lo = 0, hi = total;
    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) >> 1);
        uint32_t bucket_mid = sorted[mid].match_info >> bucket_shift;
        if (bucket_mid < b) lo = mid + 1;
        else                hi = mid;
    }
    offsets[b] = lo;
}
```

Launch (host side):

```cpp
uint32_t threads = 256;
uint32_t blocks  = (num_buckets + 1 + threads - 1) / threads;
compute_bucket_offsets<<<blocks, threads, 0, stream>>>(
    d_sorted, total, p.num_match_target_bits, num_buckets, d_offsets);
```

---

## 1. SYCL — single source, three vendors

SYCL is single-source C++ where kernels are submitted as lambdas. With
AdaptiveCpp (formerly hipSYCL) one binary can target NVIDIA (CUDA backend),
AMD (HIP backend), and Intel (Level Zero / OpenCL backend). The kernel body
is a near-mechanical port; what changes is the launch boilerplate and the
mental model around buffers/USM.

```cpp
#include <sycl/sycl.hpp>

void compute_bucket_offsets(
    sycl::queue& q,
    XsCandidateGpu const* sorted, // USM device pointer
    uint64_t total,
    int num_match_target_bits,
    uint32_t num_buckets,
    uint64_t* offsets)
{
    constexpr size_t threads = 256;
    size_t blocks = (num_buckets + 1 + threads - 1) / threads;
    sycl::nd_range<1> rng{ blocks * threads, threads };

    q.parallel_for(rng, [=](sycl::nd_item<1> it) {
        uint32_t b = it.get_global_id(0);
        if (b > num_buckets) return;
        if (b == num_buckets) { offsets[num_buckets] = total; return; }

        uint32_t bucket_shift = static_cast<uint32_t>(num_match_target_bits);
        uint64_t lo = 0, hi = total;
        while (lo < hi) {
            uint64_t mid = lo + ((hi - lo) >> 1);
            uint32_t bucket_mid = sorted[mid].match_info >> bucket_shift;
            if (bucket_mid < b) lo = mid + 1;
            else                hi = mid;
        }
        offsets[b] = lo;
    });
}
```

**What changes for the rest of the pipeline:**

- `__shared__` becomes a `sycl::local_accessor<uint32_t, 1>` captured by the
  lambda — `load_aes_tables_smem` translates 1:1.
- `__syncthreads()` → `it.barrier(sycl::access::fence_space::local_space)`.
- `atomicAdd` (used in `match_all_buckets` for the output cursor) →
  `sycl::atomic_ref<unsigned long long, memory_order::relaxed,
  memory_scope::device>`.
- `cub::DeviceRadixSort` has no in-tree SYCL equivalent. Options: oneDPL's
  `sort_by_key` (Intel-blessed, runs on all three vendors via SYCL but slower
  on NVIDIA than CUB), or keep CUB on NVIDIA and ship a backend-specific sort
  (rocPRIM on AMD, oneDPL on Intel) selected at compile time.
- Streams → `sycl::queue`s; in-order queues give CUDA-stream-like semantics.
- Constant memory has no direct SYCL equivalent — the AES T-tables stay in
  global memory and rely on the L1/L2 cache, or get loaded into local memory
  per workgroup like the existing `load_aes_tables_smem` already does.

**Net cost:** moderate — a week or two to port the kernel surface, plus
ongoing work to deal with three sort backends. The reward is one source tree
covering all three vendors.

---

## 2. Vulkan compute — most universal, heaviest rewrite

Vulkan compute kernels are GLSL (or HLSL) compiled to SPIR-V; the host code
manages descriptor sets, pipelines, command buffers, and memory by hand.
Nothing in the existing C++ kernel body survives literally — it must be
re-expressed in GLSL.

`compute_bucket_offsets.comp`:

```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256) in;

struct XsCandidateGpu { uint match_info; uint x; };

layout(std430, binding = 0) readonly buffer SortedBuf { XsCandidateGpu sorted[]; };
layout(std430, binding = 1) writeonly buffer OffsetsBuf { uint64_t offsets[]; };

layout(push_constant) uniform Params {
    uint64_t total;
    uint     num_match_target_bits;
    uint     num_buckets;
} pc;

void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b > pc.num_buckets) return;
    if (b == pc.num_buckets) { offsets[pc.num_buckets] = pc.total; return; }

    uint bucket_shift = pc.num_match_target_bits;
    uint64_t lo = 0ul, hi = pc.total;
    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) >> 1);
        uint     bucket_mid = sorted[uint(mid)].match_info >> bucket_shift;
        if (bucket_mid < b) lo = mid + 1ul;
        else                hi = mid;
    }
    offsets[b] = lo;
}
```

Host side (sketched, real code is ~150 lines for one dispatch):

```cpp
// 1. Compile compute_bucket_offsets.comp → SPIR-V via glslangValidator.
// 2. Create VkShaderModule, VkDescriptorSetLayout (2 storage buffers),
//    VkPipelineLayout (with push-constant range), VkComputePipeline.
// 3. Allocate VkBuffer+VkDeviceMemory for `sorted` and `offsets`
//    (DEVICE_LOCAL), map staging buffers for H2D/D2H.
// 4. Per dispatch:
//    vkCmdBindPipeline(cb, COMPUTE, pipe);
//    vkCmdBindDescriptorSets(cb, COMPUTE, layout, 0, 1, &set, 0, nullptr);
//    vkCmdPushConstants(cb, layout, COMPUTE, 0, sizeof(pc), &pc);
//    vkCmdDispatch(cb, (num_buckets + 1 + 255) / 256, 1, 1);
// 5. vkQueueSubmit + VkFence (or timeline semaphore) for stream-like ordering.
```

**What changes for the rest of the pipeline:**

- No CUB, no rocPRIM, no oneDPL. The radix sort in `XsKernel.cu` has to be
  reimplemented as compute shaders or replaced with a third-party Vulkan
  sort library (e.g. FidelityFX Parallel Sort, vk_radix_sort). This is the
  single biggest hidden cost of the Vulkan path.
- `__shared__` → `shared` qualifier in GLSL, sized by `local_size_x`.
- `__syncthreads()` → `barrier()` + `memoryBarrierShared()`.
- `atomicAdd` on `unsigned long long` → `atomicAdd` on a `uint64_t` SSBO
  member (requires `GL_EXT_shader_atomic_int64` and matching device feature
  `shaderBufferInt64Atomics`).
- Streams → command buffers + timeline semaphores. The existing
  double-buffered D2H pipeline (`GpuBufferPool`) maps reasonably well to
  two command buffers ping-ponging on a single queue, but the `cudaMemcpy`
  / `cudaMemcpyAsync` calls all become explicit staging-buffer copies with
  pipeline barriers.
- Constant memory → push constants (≤128 B typical) for small params, UBO
  for the AES T-tables (1 KB, fits comfortably).
- `cudaMemGetInfo` for the streaming-vs-pool VRAM dispatch →
  `vkGetPhysicalDeviceMemoryProperties` + budget extension.

**Net cost:** by far the largest. Plan on weeks for the kernel ports, plus
significant time on the sort replacement, plus a one-time Vulkan-runtime
scaffolding investment (instance/device/queue/descriptor pool boilerplate)
that the CUDA build never had to write. The payoff is the only path that
runs on a stock driver with no ROCm/Level Zero/oneAPI runtime install on
the user's machine.

---

## Summary table

| Path   | Kernel-body change | Sort path                        | Runtime install on user's box     | Targets                                    | Effort    |
|--------|--------------------|----------------------------------|-----------------------------------|--------------------------------------------|-----------|
| SYCL   | small lambda wrap  | oneDPL or per-backend sort       | SYCL runtime + vendor backend     | NVIDIA + AMD + Intel Arc                   | 1–2 weeks |
| Vulkan | full GLSL rewrite  | Reimplement or 3rd-party library | None beyond the GPU driver        | NVIDIA + AMD + Intel Arc + ARM/Adreno/etc. | Weeks     |

## Recommendation

**Go straight to SYCL, with AdaptiveCpp as the implementation.** AdaptiveCpp
on NVIDIA emits CUDA/PTX (no perf loss vs. the current nvcc path), and on
AMD it lowers through HIP/ROCm — so a SYCL build *is* a HIP build with a
different frontend. Maintaining a separate hand-written HIP tree alongside
CUDA would be ongoing cost — every algorithm change and bugfix landing in N
places — for no permanent benefit once the parity tests in `tools/parity/`
are passing on AMD via SYCL. For ~1100 lines of kernel code covered by
byte-identity tests, the single-source-tree win dominates.

What about HIP for debugging? The argument that a raw-HIP companion helps
bisect "SYCL frontend bug vs. ROCm backend bug" doesn't survive contact with
the actual workflow: `tools/parity/` already detects divergence from CPU
ground truth (which is what matters), and `rocgdb` / `rocprof` work directly
on the SYCL-compiled binary because AdaptiveCpp lowers to HIP for AMD. The
teams shipping cross-vendor compute via SYCL (PyTorch's SYCL path, GROMACS,
etc.) don't keep shadow HIP companions; we don't need to either.

Vulkan stays a separate, optional project — only worth it if a driver-only
deployment story (no ROCm / Level Zero install) becomes a hard requirement.

---

## Distribution: how SYCL slots into the existing Rust crate

The current Rust crate distribution flow is well-defined in
[`build.rs`](../build.rs) and [`README.md`](../README.md):

1. `cargo install --git ...` triggers `build.rs`.
2. `detect_cuda_arch()` shells out to `nvidia-smi --query-gpu=compute_cap` —
   produces `"89"` on a 4090, `"120"` on a 5090.
3. Precedence: `$CUDA_ARCHITECTURES` env override → nvidia-smi probe →
   `"89"` fallback (CI / containers without a GPU).
4. CMake is invoked with `-DCMAKE_CUDA_ARCHITECTURES=...`; produces the
   `xchplot2_cli` static lib.
5. `build.rs` emits `rustc-link-search=native=$CUDA_PATH/lib64` plus
   `rustc-link-lib=cudart,cudadevrt` (probes `/opt/cuda`, `/usr/local/cuda`
   if env unset).
6. `cargo:rerun-if-env-changed` on `CUDA_ARCHITECTURES`, `CUDA_PATH`,
   `CUDA_HOME`.

Every piece of that has a clean SYCL/AdaptiveCpp equivalent. The mapping:

| Concern                          | CUDA today                                                     | SYCL via AdaptiveCpp                                                                |
|----------------------------------|----------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Build-time toolchain             | `nvcc` (CMake `enable_language(CUDA)`)                         | `acpp` driver (CMake `find_package(AdaptiveCpp)` + `add_sycl_to_target`)            |
| Per-vendor probe                 | `nvidia-smi --query-gpu=compute_cap`                           | + `rocminfo` for AMD `gfx*`; SPIR-V `generic` covers Intel without a probe          |
| Arch override env                | `$CUDA_ARCHITECTURES`                                          | `$XCHPLOT2_GPU_TARGETS="cuda:sm_89;hip:gfx1100;generic"` (passed to `--acpp-targets`) |
| Default when no GPU at build     | `sm_89`                                                        | `generic` (SSCP — one SPIR-V, JIT on first launch, needs no SDK at build time)      |
| `build.rs` link libs             | `cudart`, `cudadevrt`                                          | `acpp-rt` only                                                                      |
| SDK path probe                   | `$CUDA_PATH` → `/opt/cuda` → `/usr/local/cuda`                 | `$ACPP_INSTALL_DIR` → CMake `AdaptiveCppConfig.cmake` discovery                     |
| Backend SDKs at user runtime     | CUDA driver (always linked)                                    | `dlopen`'d on first use: `libcuda.so` / `libamdhip64.so` / `libze_loader.so`        |

The single genuine improvement from this change is the last row: **the
backend libraries become runtime dependencies, not link-time ones**. CUDA
today forces every build host to have the CUDA Toolkit installed even if it
has no GPU (because `cudart` is a hard link-time dep). Under AdaptiveCpp,
`build.rs` only needs `acpp` itself; backends are discovered at first
launch on the user's box. That means a single `cargo install` on a CI box
with no GPU produces a binary that runs on whichever vendor card is in the
user's machine — assuming the user has the matching vendor runtime.

User-facing runtime install burden, by vendor:

- **NVIDIA:** unchanged — same `libcuda.so` from the proprietary driver.
- **Intel Arc:** `intel-compute-runtime` + `intel-level-zero-gpu`, packaged
  in most modern distros (`apt install intel-opencl-icd intel-level-zero-gpu`).
- **AMD:** ROCm runtime. Not in most distro repos — users add AMD's apt/dnf
  repo or build from source. Worse, ROCm's official support matrix excludes
  many consumer Radeon cards (RX 6700 XT etc.); affected users typically
  need `HSA_OVERRIDE_GFX_VERSION=10.3.0` or similar. There is no shipping
  around this short of going Vulkan; it's the cost of touching AMD compute
  via ROCm.

---

## `build.rs` rewrite sketch

Here is the concrete shape of the changes to `build.rs`. It preserves the
"probe local hardware, build for it, fall back cleanly" pattern but
generalises it across the three vendors and adds the always-on `generic`
JIT target so a binary always runs *somewhere*.

```rust
// build.rs — SYCL/AdaptiveCpp variant.
//
// Drives CMake (which uses find_package(AdaptiveCpp) + add_sycl_to_target
// to feed source files through `acpp`) and links the resulting static libs
// into the Rust [[bin]] xchplot2.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// One AdaptiveCpp target string, e.g. "cuda:sm_89", "hip:gfx1100", "generic".
type Target = String;

/// Ask `nvidia-smi` for the local NVIDIA GPU's compute capability and return
/// the AdaptiveCpp CUDA target string. None on any failure.
fn detect_nvidia_target() -> Option<Target> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output().ok()?;
    if !out.status.success() { return None; }
    let s = std::str::from_utf8(&out.stdout).ok()?.trim().to_string();
    let first = s.lines().next()?.trim();
    let cap: f32 = first.parse().ok()?;          // "8.9" -> 8.9
    let arch = (cap * 10.0).round() as u32;      // -> 89
    Some(format!("cuda:sm_{arch}"))
}

/// Ask `rocminfo` for the local AMD GPU's gfx ISA name. None on any failure.
/// rocminfo prints "  Name:                    gfx1100" for each agent.
fn detect_amd_target() -> Option<Target> {
    let out = Command::new("rocminfo").output().ok()?;
    if !out.status.success() { return None; }
    let s = std::str::from_utf8(&out.stdout).ok()?;
    for line in s.lines() {
        if let Some(rest) = line.trim().strip_prefix("Name:") {
            let name = rest.trim();
            if name.starts_with("gfx") {
                return Some(format!("hip:{name}"));
            }
        }
    }
    None
}

/// Probe the build host for any locally-attached supported GPUs and return
/// the corresponding AdaptiveCpp target list. Always appends "generic" so
/// the binary runs *somewhere* even on hosts whose hardware we can't see.
fn detect_targets() -> Vec<Target> {
    let mut targets: Vec<Target> = Vec::new();
    if let Some(t) = detect_nvidia_target() { targets.push(t); }
    if let Some(t) = detect_amd_target()    { targets.push(t); }
    // Intel Arc: SPIR-V + Level Zero JIT, covered by `generic` below.
    targets.push("generic".to_string());
    targets
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir      = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cmake_build  = out_dir.join("cmake-build");
    std::fs::create_dir_all(&cmake_build).expect("create cmake-build dir");

    // Target precedence:
    //   1. $XCHPLOT2_GPU_TARGETS, raw acpp-targets string (e.g. "cuda:sm_89;generic")
    //   2. probe local hardware (nvidia-smi + rocminfo) and append "generic"
    //   3. "generic" only — JIT path, works on any vendor with a SYCL backend
    let (targets, source) = match env::var("XCHPLOT2_GPU_TARGETS") {
        Ok(v) => (v, "$XCHPLOT2_GPU_TARGETS"),
        Err(_) => {
            let detected = detect_targets();
            let any_aot = detected.iter().any(|t| t != "generic");
            let source = if any_aot { "hardware probe" }
                         else       { "fallback (no GPU detected)" };
            (detected.join(";"), source)
        }
    };
    println!("cargo:warning=xchplot2: building for SYCL targets [{targets}] ({source})");

    // ---- configure ----
    let status = Command::new("cmake")
        .args([
            "-S", manifest_dir.to_str().unwrap(),
            "-B", cmake_build.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
        ])
        .arg(format!("-DACPP_TARGETS={targets}"))
        .status()
        .expect("failed to invoke cmake — is it installed?");
    if !status.success() { panic!("cmake configure failed"); }

    let status = Command::new("cmake")
        .args(["--build", cmake_build.to_str().unwrap(),
               "--target", "xchplot2_cli", "--parallel"])
        .status().expect("cmake --build failed");
    if !status.success() { panic!("cmake build failed"); }

    // ---- link ----
    let lib_dir = cmake_build.join("src");          // wherever the static libs land
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    println!("cargo:rustc-link-lib=static=xchplot2_cli");
    println!("cargo:rustc-link-lib=static=pos2_gpu_host");
    println!("cargo:rustc-link-lib=static=pos2_gpu");
    println!("cargo:rustc-link-lib=static=pos2_keygen");
    println!("cargo:rustc-link-lib=static=fse");
    println!("cargo:rustc-link-arg=-Wl,--end-group");

    // ---- AdaptiveCpp runtime ----
    // Replaces the libcudart / libcudadevrt block. acpp-rt dlopen's the
    // per-vendor backend libraries (libcuda, libamdhip64, libze_loader)
    // on first device discovery — they are NOT link-time deps, which is
    // why `cargo install` works on a build host with no GPU at all.
    let acpp_root = env::var("ACPP_INSTALL_DIR")
        .unwrap_or_else(|_| {
            for guess in ["/opt/adaptivecpp", "/usr/local", "/usr"] {
                let p = std::path::Path::new(guess).join("lib/libacpp-rt.so");
                if p.exists() { return guess.to_string(); }
            }
            "/usr/local".to_string()
        });
    println!("cargo:rustc-link-search=native={acpp_root}/lib");
    println!("cargo:rustc-link-lib=acpp-rt");

    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=rt");

    for p in &["src", "tools", "keygen-rs/src", "keygen-rs/Cargo.toml",
               "keygen-rs/Cargo.lock", "CMakeLists.txt", "build.rs"] {
        println!("cargo:rerun-if-changed={p}");
    }
    println!("cargo:rerun-if-env-changed=XCHPLOT2_GPU_TARGETS");
    println!("cargo:rerun-if-env-changed=ACPP_INSTALL_DIR");
}
```

### Behavioural mapping vs. current `build.rs`

- `detect_cuda_arch()` → `detect_nvidia_target()`. Same `nvidia-smi`
  invocation; just wraps the result in `cuda:sm_NN` instead of returning the
  bare integer.
- `detect_amd_target()` is structurally identical to the NVIDIA probe — one
  process, parse one line, return `Option<String>`. Cleanly returns `None` on
  build hosts without ROCm installed (most of them), so AMD users opt in by
  installing ROCm; everyone else falls through to `generic`.
- The `89` fallback becomes `generic` — semantically the same idea ("a target
  that always works without inspecting hardware") but now it runs on *any*
  vendor at slight first-launch JIT cost, instead of running fast on Ada and
  not at all on Ampere.
- The `$CUDA_ARCHITECTURES` env var becomes `$XCHPLOT2_GPU_TARGETS`, which
  takes a raw `acpp-targets` semicolon list. Migration guide for the README:
  `CUDA_ARCHITECTURES=89` → `XCHPLOT2_GPU_TARGETS="cuda:sm_89;generic"`,
  `CUDA_ARCHITECTURES="89;120"` → `XCHPLOT2_GPU_TARGETS="cuda:sm_89;cuda:sm_120;generic"`.
- The `$CUDA_PATH` / `$CUDA_HOME` / `/opt/cuda` / `/usr/local/cuda` discovery
  block reduces to a single `$ACPP_INSTALL_DIR` probe — `acpp` knows where
  its own backends live.

### One wrinkle worth flagging in the README

AOT for `hip:gfxXXXX` requires AdaptiveCpp itself to have been built against
ROCm at the user's `cargo install` time. If the user installs AdaptiveCpp
from a generic distro package that wasn't compiled with ROCm support, the
`hip:` target will silently be unavailable and `acpp` will error out. The
`build.rs` warning line above (`cargo:warning=xchplot2: building for SYCL
targets [...]`) is the right hook to detect this — print a hint pointing at
the AdaptiveCpp build flags when an AMD GPU is detected but the user's
AdaptiveCpp isn't ROCm-enabled. Same shape as today's `nvidia-smi probe vs.
fallback` warning, just with an extra failure mode.
