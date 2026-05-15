cmake_minimum_required(VERSION 3.24)

project(pos2-gpu VERSION 0.6.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Every static library here is linked into both the standalone xchplot2
# executable and the top-level Rust crate's PIE binary (via build.rs +
# cargo install). rust-lld (the default linker on some distros) rejects
# non-PIC objects in a PIE output — seen in the wild as "relocation
# R_X86_64_32 cannot be used against local symbol; recompile with
# -fPIC" on Cancel.cpp, BatchPlotter.cpp, etc. Setting this globally
# ensures pos2_gpu, pos2_gpu_host, fse, and any other transitively-
# compiled object is built with -fPIC, so the linker choice doesn't
# matter. The per-target POSITION_INDEPENDENT_CODE ON below stay as
# explicit markers for the public-interface static libraries.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA toolchain is conditional in slice 15. The CUDA path provides:
#   - SortCuda.cu (CUB radix sort — best perf on NVIDIA)
#   - AesGpu.cu (T-tables in __constant__ memory + cudaMemcpyToSymbol init)
#   - AesGpuBitsliced.cu (bench-only bitsliced AES; needs nvcc)
#   - The cuda-flavoured parity tests in tools/parity/
# The non-CUDA path uses SortSycl.cpp + AesStub.cpp — runs on AMD/Intel via
# AdaptiveCpp's HIP / Level Zero backends. Default ON to preserve the
# existing NVIDIA workflow.
#
# CAVEAT: with XCHPLOT2_BUILD_CUDA=OFF the build still needs the CUDA
# Toolkit *headers* on the include path (the SYCL TUs reference cudaError_t
# / cudaStream_t / cuda_fp16.h via the kernel-wrapper headers). Lifting
# those CUDA-type dependencies out of the public SYCL API is a follow-up
# refactor (see slice 17 in docs/gpu-portability-sketch.md). nvcc itself is
# NOT required when XCHPLOT2_BUILD_CUDA=OFF — only the headers.
option(XCHPLOT2_BUILD_CUDA "Compile CUDA-only TUs (CUB sort, __constant__ AES init, bench tests)" ON)

# On dual-toolchain hosts (CUDA Toolkit + ROCm both installed), the SYCL
# TUs pull in CUDA's <cuda_runtime.h> via CudaHalfShim.hpp AND ROCm's
# <hip/hip_runtime.h> via AdaptiveCpp's HIP backend. Their vector_types
# headers declare conflicting typedefs for char1 / int2 / etc., which
# breaks the compile. CudaHalfShim respects XCHPLOT2_SKIP_CUDA_RUNTIME /
# _FP16 — turn them on when we're (a) NOT building CUDA TUs and (b) ROCm
# is present, so the shim falls back to its opaque stubs instead.
if(NOT XCHPLOT2_BUILD_CUDA)
    find_path(XCHPLOT2_HIP_RUNTIME_H hip/hip_runtime.h
              PATHS /opt/rocm/include /usr/include /usr/local/include
              NO_DEFAULT_PATH)
    if(XCHPLOT2_HIP_RUNTIME_H)
        add_compile_definitions(
            XCHPLOT2_SKIP_CUDA_RUNTIME
            XCHPLOT2_SKIP_CUDA_FP16)
        message(STATUS "xchplot2: ROCm at ${XCHPLOT2_HIP_RUNTIME_H} — "
                       "skipping CUDA runtime/fp16 includes (CudaHalfShim stubs)")
    endif()
endif()

if(XCHPLOT2_BUILD_CUDA)
    # Default arch: `native` — CMake 3.24+ probes the locally-visible
    # GPUs and emits SASS only for those compute capabilities. Falls
    # back to sm_89 (RTX 4090) when no GPU is visible at configure
    # time (e.g., container builds, headless CI). Override either via
    # -DCMAKE_CUDA_ARCHITECTURES=89;86 (multi-arch fatbin) or a
    # specific value when cross-compiling for hardware that's not
    # plugged into the build host.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Probe whether `native` resolves to anything; if no GPU is
        # visible CMake will error out cryptically. Fall back to sm_89
        # in that case.
        execute_process(
            COMMAND nvidia-smi -L
            OUTPUT_VARIABLE _xchplot2_nvsmi_out
            ERROR_QUIET
            RESULT_VARIABLE _xchplot2_nvsmi_rc)
        if(_xchplot2_nvsmi_rc EQUAL 0 AND _xchplot2_nvsmi_out MATCHES "GPU")
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 89)
        endif()
    endif()

    # Preflight nvcc-vs-arch compatibility BEFORE enable_language(CUDA),
    # which is what triggers the cryptic "Unsupported gpu architecture
    # 'compute_61'" TryCompile failure when Pascal/Volta meets CUDA 13.x.
    # CUDA 13.0 dropped codegen for sm_50/52/53/60/61/62/70/72 entirely.
    # Skip the check if nvcc isn't findable yet — enable_language(CUDA)
    # below will surface its own missing-toolchain message in that case.
    find_program(_xchplot2_nvcc nvcc
        HINTS ENV CUDA_PATH ENV CUDA_HOME /opt/cuda /usr/local/cuda
        PATH_SUFFIXES bin
        DOC "nvcc for arch-compat preflight")
    if(_xchplot2_nvcc)
        execute_process(
            COMMAND "${_xchplot2_nvcc}" --version
            OUTPUT_VARIABLE _nvcc_version_out
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        # Parse "Cuda compilation tools, release 13.0, V13.0.48" → 13
        if(_nvcc_version_out MATCHES "release ([0-9]+)")
            set(_nvcc_major "${CMAKE_MATCH_1}")
            set(_min_arch 9999)
            foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
                # Strip sm_ / compute_ prefixes some users pass through
                string(REGEX REPLACE "^(sm_|compute_)" "" _a "${_a}")
                if(_a MATCHES "^[0-9]+$" AND _a LESS _min_arch)
                    set(_min_arch ${_a})
                endif()
            endforeach()
            if(_nvcc_major GREATER_EQUAL 13 AND _min_arch LESS 75)
                # Container detection: Docker writes /.dockerenv, Podman writes
                # /run/.containerenv. Either presence means the host-side fixes
                # don't apply — the user needs to rebuild the image with a
                # different BASE_DEVEL.
                if(EXISTS "/.dockerenv" OR EXISTS "/run/.containerenv")
                    set(_fix_block
                        "You're building inside a container — the toolkit comes from\n"
                        "the base image, not the host. Rebuild with a CUDA 12.x base:\n"
                        "  - Recommended: rerun scripts/build-container.sh on the host;\n"
                        "    it auto-pins nvidia/cuda:12.9.1 when CUDA_ARCH < 75.\n"
                        "  - Or pass --build-arg explicitly:\n"
                        "      podman build -t xchplot2:cuda \\\n"
                        "        --build-arg BASE_DEVEL=docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 \\\n"
                        "        --build-arg BASE_RUNTIME=docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 \\\n"
                        "        --build-arg CUDA_ARCH=${_min_arch} \\\n"
                        "        .\n")
                else()
                    set(_fix_block
                        "Fix one of:\n"
                        "  - Install CUDA 12.9 (last toolkit with Pascal/Volta support) and re-run cmake:\n"
                        "      sudo apt install cuda-toolkit-12-9     (Ubuntu/Debian)\n"
                        "    Then point cmake at it:\n"
                        "      cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -B build -S . [...]\n"
                        "  - Or override the target arch (only valid if you actually have a Turing+ card):\n"
                        "      cmake -DCMAKE_CUDA_ARCHITECTURES=75 -B build -S . [...]\n"
                        "  - Or use the container path — scripts/build-container.sh auto-pins\n"
                        "    the 12.9 base image when it detects a pre-Turing GPU.\n")
                endif()
                message(FATAL_ERROR
                    "xchplot2: CUDA Toolkit ${_nvcc_major}.x dropped codegen for "
                    "sm_${_min_arch} (Pascal / Volta / pre-Turing).\n"
                    "\n"
                    "Detected:\n"
                    "  nvcc ${_nvcc_major}.x at ${_xchplot2_nvcc}\n"
                    "  target arch: sm_${_min_arch} (from CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})\n"
                    "\n"
                    ${_fix_block})
            endif()
        endif()
    endif()
    unset(_xchplot2_nvcc CACHE)

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# Optional: compile in clock64 instrumentation for T3 match_all_buckets.
# Prints a per-region cycle breakdown to stderr on every launch_t3_match
# call. Off by default — enable with -DXCHPLOT2_INSTRUMENT_MATCH=ON.
option(XCHPLOT2_INSTRUMENT_MATCH "Instrument T3 match_all_buckets with clock64 breakdown" OFF)

# SYCL kernels via AdaptiveCpp are the only backend; the previous
# XCHPLOT2_BACKEND={cuda,sycl} toggle was retired in slice 9 once the
# CUDA-native wrapper TUs (T*OffsetsCuda.cu, PipelineKernelsCuda.cu)
# were deleted. AdaptiveCpp is now a hard build dependency.

# AdaptiveCpp target autodetect — must run BEFORE find_package(AdaptiveCpp)
# so the package config sees a non-empty target list. acpp errors on an
# empty -DACPP_TARGETS= (which we'd otherwise pass through unchanged from
# the Containerfile's default build-arg).
#   1. NVIDIA: stay on "generic" (LLVM SSCP). Empirically a few percent
#      faster than cuda:sm_XX on our kernels at k=28 — SSCP's runtime
#      specialization beats the CUDA-AOT path for this workload.
#   2. AMD:    rocminfo Name: gfxXXXX → hip:gfxXXXX. SSCP's HIP path is
#      less mature, so AOT-compiling for the actual gfx target is the
#      safer pick on AMD.
#   3. Fallback: generic (works everywhere; JITs on first use).
# Override with -DACPP_TARGETS=... on the cmake command line.
if(NOT ACPP_TARGETS)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
        OUTPUT_VARIABLE _xchplot2_cuda_cap
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _xchplot2_nvsmi_rc
        ERROR_QUIET)
    if(_xchplot2_nvsmi_rc EQUAL 0 AND _xchplot2_cuda_cap)
        set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp target list" FORCE)
        message(STATUS "xchplot2: NVIDIA GPU detected; using ACPP_TARGETS=generic (SSCP)")
    else()
        execute_process(
            COMMAND rocminfo
            OUTPUT_VARIABLE _xchplot2_rocm_out
            RESULT_VARIABLE _xchplot2_rocminfo_rc
            ERROR_QUIET)
        if(_xchplot2_rocminfo_rc EQUAL 0)
            string(REGEX MATCH "Name:[ \t]+gfx[0-9a-f]+" _xchplot2_gfx_match "${_xchplot2_rocm_out}")
            string(REGEX REPLACE "Name:[ \t]+" "" _xchplot2_gfx "${_xchplot2_gfx_match}")
            if(_xchplot2_gfx)
                set(ACPP_TARGETS "hip:${_xchplot2_gfx}" CACHE STRING "AdaptiveCpp target list" FORCE)
                message(STATUS "xchplot2: ACPP_TARGETS auto-detected via rocminfo: ${ACPP_TARGETS}")
            endif()
        endif()
    endif()
    if(NOT ACPP_TARGETS)
        set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp target list" FORCE)
        message(STATUS "xchplot2: ACPP_TARGETS fell back to generic (no nvidia-smi/rocminfo)")
    endif()
endif()
message(STATUS "xchplot2: ACPP_TARGETS=${ACPP_TARGETS}")

# Lookup precedence:
#   1. find_package(AdaptiveCpp) — system or local install (e.g. /opt/adaptivecpp).
#      This is what scripts/install-deps.sh and the Containerfile produce.
#   2. FetchContent fallback — clones AdaptiveCpp at v25.10.0 and adds it as
#      a CMake subproject. Slow first build (LLVM compilation, ~15-30 min) but
#      removes the manual install step. Opt out with -DXCHPLOT2_FETCH_ADAPTIVECPP=OFF.
option(XCHPLOT2_FETCH_ADAPTIVECPP "Fall back to FetchContent if AdaptiveCpp not found" ON)

# HINTS /opt/adaptivecpp matches scripts/install-deps.sh's default install
# prefix, and ENV ACPP_PREFIX honours users who installed to a custom
# location with `ACPP_PREFIX=/elsewhere ./scripts/install-deps.sh`. Without
# these, find_package wouldn't search /opt (not a standard CMake path), the
# user would have to remember to `export CMAKE_PREFIX_PATH=/opt/adaptivecpp`
# between running install-deps.sh and the build (the script can't set env
# vars in the parent shell), and FetchContent would fire pointlessly.
find_package(AdaptiveCpp QUIET HINTS /opt/adaptivecpp ENV ACPP_PREFIX)
if(NOT AdaptiveCpp_FOUND)
    if(XCHPLOT2_FETCH_ADAPTIVECPP)
        message(STATUS "xchplot2: AdaptiveCpp not found — fetching v25.10.0 via FetchContent")
        message(STATUS "xchplot2:   first build will take ~15-30 min while AdaptiveCpp compiles")
        message(STATUS "xchplot2:   pre-install via scripts/install-deps.sh to skip this")

        # AdaptiveCpp's compiler/CMakeLists requires ld.lld at configure
        # time and aborts with "Cannot find ld.lld. Please provide path
        # via -DACPP_LLD_PATH=…" otherwise. Auto-probe the conventional
        # LLVM-{16..20} prefixes and pass the path through so users on a
        # FetchContent build don't have to know that detail. If the
        # binary isn't installed at all, fail loud with a copy-paste
        # install command — far less confusing than AdaptiveCpp's own
        # message.
        find_program(_xchplot2_ld_lld
            NAMES ld.lld
            HINTS
                /usr/lib/llvm-20/bin /usr/lib/llvm-19/bin /usr/lib/llvm-18/bin
                /usr/lib/llvm-17/bin /usr/lib/llvm-16/bin
                /usr/lib/llvm20/bin  /usr/lib/llvm19/bin  /usr/lib/llvm18/bin
                /usr/lib64/llvm20/bin /usr/lib64/llvm19/bin /usr/lib64/llvm18/bin
                /opt/llvm-20/bin /opt/llvm-19/bin /opt/llvm-18/bin
                /opt/llvm20/bin  /opt/llvm19/bin  /opt/llvm18/bin
            DOC "ld.lld required by AdaptiveCpp's compiler/CMakeLists")
        if(_xchplot2_ld_lld)
            set(ACPP_LLD_PATH "${_xchplot2_ld_lld}" CACHE FILEPATH
                "Path to ld.lld for AdaptiveCpp's compiler/CMakeLists" FORCE)
            message(STATUS "xchplot2:   auto-probed ld.lld at ${_xchplot2_ld_lld}")
        else()
            message(FATAL_ERROR
                "xchplot2: AdaptiveCpp's FetchContent build needs ld.lld "
                "but it isn't installed at any of the standard LLVM-16..20 "
                "prefixes. Install it:\n"
                "  Ubuntu/Debian:  sudo apt install lld-18\n"
                "  Fedora/RHEL:    sudo dnf install lld\n"
                "  Arch/CachyOS:   sudo pacman -S lld\n"
                "Or pre-install AdaptiveCpp via scripts/install-deps.sh "
                "(also installs ld.lld and builds AdaptiveCpp at "
                "/opt/adaptivecpp). Override the probe with "
                "-DACPP_LLD_PATH=/path/to/ld.lld.")
        endif()

        include(FetchContent)
        FetchContent_Declare(
            adaptivecpp
            GIT_REPOSITORY https://github.com/AdaptiveCpp/AdaptiveCpp.git
            GIT_TAG        v25.10.0
        )
        FetchContent_MakeAvailable(adaptivecpp)
        if(NOT COMMAND add_sycl_to_target)
            message(FATAL_ERROR
                "xchplot2: FetchContent built AdaptiveCpp but add_sycl_to_target "
                "wasn't exported. Install AdaptiveCpp via scripts/install-deps.sh "
                "or use the Containerfile.")
        endif()
    else()
        message(FATAL_ERROR
            "xchplot2: AdaptiveCpp not found. Install it via scripts/install-deps.sh, "
            "use the Containerfile, or re-run with -DXCHPLOT2_FETCH_ADAPTIVECPP=ON.")
    endif()
endif()

# Export the AdaptiveCpp lib directory to a file so build.rs knows where
# to add -L for libacpp-rt / libacpp-common at link time. Without this,
# the Rust binary fails to link on machines where AdaptiveCpp lives
# anywhere other than /opt/adaptivecpp or /usr/local (and on FetchContent
# builds, which leave the artifacts in CMake's _deps/ build tree).
set(_xchplot2_acpp_lib_dir "")
if(TARGET acpp-rt)
    # FetchContent-built target: ask CMake where it'll land.
    set(_xchplot2_acpp_lib_dir "$<TARGET_FILE_DIR:acpp-rt>")
elseif(AdaptiveCpp_DIR)
    # Installed AdaptiveCpp: AdaptiveCpp_DIR is <prefix>/lib/cmake/AdaptiveCpp,
    # so two parent dirs up gives <prefix>/lib.
    get_filename_component(_xchplot2_acpp_cmake_root "${AdaptiveCpp_DIR}" DIRECTORY)
    get_filename_component(_xchplot2_acpp_lib_dir    "${_xchplot2_acpp_cmake_root}" DIRECTORY)
endif()
if(_xchplot2_acpp_lib_dir)
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/acpp-prefix.txt"
                  CONTENT "${_xchplot2_acpp_lib_dir}\n")
    message(STATUS "xchplot2: AdaptiveCpp lib dir = ${_xchplot2_acpp_lib_dir}")
endif()

# Embed runtime library paths so binaries built via plain `cmake` (parity
# tests, dev rebuilds, anything not invoked through cargo+build.rs) can
# locate AdaptiveCpp's runtime lib + ROCm's libamdhip64.so without an
# external LD_LIBRARY_PATH. build.rs sets the same rpaths via
# rustc-link-arg for the cargo path, so this is idempotent for the
# production binary. Without this, a fresh `cmake -B build && cmake
# --build build --target sycl_t1_parity` produces a binary that throws
# "No matching device" at SYCL queue construction because
# librt-backend-hip.so can't dynamically link libamdhip64.so.
#
# The FetchContent path leaves _xchplot2_acpp_lib_dir as a generator
# expression ("$<TARGET_FILE_DIR:acpp-rt>") which can't go into the
# RPATH variables at config time — CMake's BUILD_WITH_INSTALL_RPATH=OFF
# default already handles in-tree targets in that case.
if(_xchplot2_acpp_lib_dir AND NOT _xchplot2_acpp_lib_dir MATCHES "\\$<")
    list(APPEND CMAKE_BUILD_RPATH   "${_xchplot2_acpp_lib_dir}")
    list(APPEND CMAKE_INSTALL_RPATH "${_xchplot2_acpp_lib_dir}")
endif()
if(XCHPLOT2_HIP_RUNTIME_H)
    get_filename_component(_xchplot2_rocm_root "${XCHPLOT2_HIP_RUNTIME_H}/.." ABSOLUTE)
    list(APPEND CMAKE_BUILD_RPATH   "${_xchplot2_rocm_root}/lib")
    list(APPEND CMAKE_INSTALL_RPATH "${_xchplot2_rocm_root}/lib")
    message(STATUS "xchplot2: embedded rpath includes ${_xchplot2_rocm_root}/lib")

    # Direct-link libamdhip64 so AdaptiveCpp's runtime-dlopen'd HIP
    # backend (librt-backend-hip.so) finds the library already loaded
    # in the process address space. dlopen of a backend's transitive
    # deps doesn't consult the calling binary's RUNPATH on glibc —
    # without this explicit link, ROCm silently fails to initialise
    # and AdaptiveCpp's default selector falls through to its OpenMP
    # host device. The fall-through makes hellosycl / sycl_t1_parity
    # report "ALL OK" while having executed entirely on CPU. Mirrors
    # build.rs:631 (cargo:rustc-link-lib=amdhip64) for the cargo
    # build path.
    if(EXISTS "${_xchplot2_rocm_root}/lib/libamdhip64.so")
        link_libraries("${_xchplot2_rocm_root}/lib/libamdhip64.so")
        message(STATUS "xchplot2: link_libraries(libamdhip64.so) — "
                       "AdaptiveCpp HIP backend will find ROCm at runtime")
    endif()
endif()

# pos2-chip dependency.
#
# Default behavior: FetchContent auto-clones Chia-Network/pos2-chip into
# third_party/pos2-chip at configure time, pinned to a specific commit.
# We use FetchContent rather than `git submodule` because pos2-gpu was
# created with SHA-256 object format and the upstream pos2-chip repo
# uses SHA-1 — git refuses to mix the two within a submodule.
#
# Devs hacking on pos2-chip locally can override the default by passing
# -DPOS2_CHIP_DIR=/abs/path/to/pos2-chip on the cmake command line; the
# fetch is then skipped entirely.
# In-tree default. Override on the command line with
# -DPOS2_CHIP_GIT_TAG=<sha> (the cache entry created by -D sticks across
# reconfigures). We deliberately do NOT use `set(... CACHE STRING)` here:
# CACHE STRING is no-op when the cache already has a value, so editing this
# line and reconfiguring an existing build/ dir would silently keep the OLD
# pin. The `if(NOT DEFINED ...)` guard lets a plain edit take effect on
# reconfigure while still honouring user -D overrides.
if(NOT DEFINED POS2_CHIP_GIT_TAG)
    set(POS2_CHIP_GIT_TAG "b0da7aa7bec3974833d651173a6d9953c21eb808")
endif()

if(NOT DEFINED POS2_CHIP_DIR)
    include(FetchContent)
    FetchContent_Declare(
        pos2_chip
        GIT_REPOSITORY https://github.com/Chia-Network/pos2-chip
        GIT_TAG        ${POS2_CHIP_GIT_TAG}
        SOURCE_DIR     "${CMAKE_SOURCE_DIR}/third_party/pos2-chip"
        SOURCE_SUBDIR  ".no-cmakelists-here"   # suppress auto add_subdirectory
    )
    FetchContent_MakeAvailable(pos2_chip)
    set(POS2_CHIP_DIR "${pos2_chip_SOURCE_DIR}")
endif()

if(NOT EXISTS "${POS2_CHIP_DIR}/src/pos/aes/AesHash.hpp")
    message(FATAL_ERROR
        "pos2-chip not found at ${POS2_CHIP_DIR}. "
        "Either remove -DPOS2_CHIP_DIR to let CMake fetch it automatically, "
        "or pass -DPOS2_CHIP_DIR=/abs/path/to/pos2-chip.")
endif()
message(STATUS "pos2-chip: ${POS2_CHIP_DIR}")

# pos2-chip vendors FSE under lib/fse with its own CMakeLists.txt.
# Bring it in so we can link against the same static lib pos2-chip uses.
add_subdirectory("${POS2_CHIP_DIR}/lib/fse" "${CMAKE_BINARY_DIR}/fse" EXCLUDE_FROM_ALL)

# Header-only pos2-chip include surface
add_library(pos2_chip_headers INTERFACE)
target_include_directories(pos2_chip_headers INTERFACE
    "${POS2_CHIP_DIR}/src"
)
target_link_libraries(pos2_chip_headers INTERFACE fse)
# Match pos2-chip's CPU AES build flags so its AesHash compiles in CPU TUs.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(pos2_chip_headers INTERFACE -maes -msse4.1 -mssse3 -msse2)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(pos2_chip_headers INTERFACE -march=armv8-a+crypto)
endif()

# Shared GPU support library (kernels). AesGpu.cu MUST come first — it
# owns the constant-memory T-tables that all later kernels reference.
# All backend-dispatched wrapper TUs (T*OffsetsSycl.cpp, PipelineKernelsSycl.cpp)
# go through AdaptiveCpp via add_sycl_to_target below.
set(POS2_GPU_SYCL_SRC
    src/gpu/SyclBackend.cpp
    src/gpu/T1OffsetsSycl.cpp
    src/gpu/T2OffsetsSycl.cpp
    src/gpu/T3OffsetsSycl.cpp
    src/gpu/PipelineKernelsSycl.cpp
    src/gpu/TwoLevelSortSycl.cpp
    src/gpu/XsKernel.cpp
    src/gpu/XsKernelsSycl.cpp
    src/gpu/T1Kernel.cpp
    src/gpu/T2Kernel.cpp
    src/gpu/T3Kernel.cpp
    src/host/GpuBufferPool.cpp
    src/host/GpuPipeline.cpp
    src/host/HostPinnedPool.cpp)

# Sort path: SortSycl.cpp (hand-rolled LSD radix in pure SYCL) is now
# always compiled — it's the runtime fallback for non-CUDA backends on
# dual-toolchain builds, and the only path on AMD-only / Intel-only /
# CPU builds. SortDispatch.cpp picks at runtime based on the queue's
# device backend (sycl::backend::cuda → _cub variant; everything else →
# _sycl variant). When BUILD_CUDA=OFF, the dispatcher's CUB branch is
# compiled out and reduces to a single tail call into SortSycl.cpp.
list(APPEND POS2_GPU_SYCL_SRC
    src/gpu/SortSycl.cpp
    src/gpu/SortDispatch.cpp
    src/gpu/SortDistributed.cpp
    src/gpu/SortDistributedPeer.cpp
    src/gpu/SyclDeviceList.cpp)

if(XCHPLOT2_BUILD_CUDA)
    set(POS2_GPU_CUDA_SRC
        src/gpu/AesGpu.cu
        src/gpu/AesGpuBitsliced.cu
        src/gpu/SortCuda.cu)
    # SortSyclCub.cpp is the SYCL-typed adapter that bridges
    # sycl::queue → CUB. SortCuda.cu used to provide the SYCL-typed
    # entry points itself, but mixing nvcc + <sycl/sycl.hpp> in one
    # TU drags AdaptiveCpp's libkernel half.hpp into the legacy CUDA
    # arm of __acpp_backend_switch — a path AdaptiveCpp doesn't
    # support. Splitting the SYCL surface into this acpp-compiled
    # adapter (does q.wait()) and a pure-CUDA cub_sort_* in
    # SortCuda.cu (does the work + cudaStreamSync) keeps each
    # compiler in its lane.
    list(APPEND POS2_GPU_SYCL_SRC
        src/gpu/SortSyclCub.cpp)
else()
    # AesStub.cpp: no-op initialize_aes_tables on builds without the
    # CUDA AOT path. AesGpu.cu provides the real implementation when
    # BUILD_CUDA=ON; SYCL workers ignore initialize_aes_tables anyway
    # (they upload AES T-tables lazily via SyclBackend.hpp's
    # aes_tables_device(q)).
    set(POS2_GPU_CUDA_SRC)
    list(APPEND POS2_GPU_SYCL_SRC
        src/gpu/AesStub.cpp)
endif()

# CUDA OBJECT library: compiled once, referenced via $<TARGET_OBJECTS:>
# from each consuming target EXACTLY ONCE. The earlier design tried to
# put the .o files in BOTH pos2_gpu (STATIC) AND xchplot2_cli for hash
# matching, but nvlink's device-link step at xchplot2_cli archive
# creation refuses the duplicate kAesT0..3 / kernel definitions:
#
#   nvlink error : Multiple definition of '_ZN7pos2gpu6kAesT0E' in
#       'libpos2_gpu.a:AesGpu.cu.o', first defined in
#       'CMakeFiles/pos2_gpu_cuda_obj.dir/src/gpu/AesGpu.cu.o'
#
# (--allow-multiple-definition is a host-linker flag — nvlink doesn't
# honour it.) So the .o files now live exclusively in xchplot2_cli for
# the cargo install path, and each parity test adds them explicitly
# below — pos2_gpu STATIC carries only the SYCL .cpp sources.
if(XCHPLOT2_BUILD_CUDA)
    add_library(pos2_gpu_cuda_obj OBJECT ${POS2_GPU_CUDA_SRC})
    target_include_directories(pos2_gpu_cuda_obj PRIVATE src)
    target_link_libraries(pos2_gpu_cuda_obj PRIVATE pos2_chip_headers)
    target_compile_features(pos2_gpu_cuda_obj PRIVATE cxx_std_20)
    set_target_properties(pos2_gpu_cuda_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(XCHPLOT2_INSTRUMENT_MATCH)
        target_compile_definitions(pos2_gpu_cuda_obj PRIVATE XCHPLOT2_INSTRUMENT_MATCH=1)
    endif()
endif()

add_library(pos2_gpu STATIC
    ${POS2_GPU_SYCL_SRC}
)
target_include_directories(pos2_gpu PUBLIC
    src
)
target_link_libraries(pos2_gpu PUBLIC pos2_chip_headers)
target_compile_features(pos2_gpu PUBLIC cxx_std_20)
if(XCHPLOT2_INSTRUMENT_MATCH)
    target_compile_definitions(pos2_gpu PUBLIC XCHPLOT2_INSTRUMENT_MATCH=1)
endif()
# Marker for SortDispatch.cpp: gates whether the runtime backend
# dispatcher includes the CUB branch. Defined when SortSyclCub.cpp +
# SortCuda.cu are linked (BUILD_CUDA=ON); undefined on AMD-only /
# Intel-only / CPU builds, in which case the dispatcher reduces to a
# single tail call into SortSycl.cpp.
if(XCHPLOT2_BUILD_CUDA)
    target_compile_definitions(pos2_gpu PUBLIC XCHPLOT2_HAVE_CUB=1)
endif()
add_sycl_to_target(TARGET pos2_gpu SOURCES ${POS2_GPU_SYCL_SRC})

# AdaptiveCpp's acpp driver doesn't auto-propagate CMake's standard
# CMAKE_CXX_FLAGS_RELEASE (-O3 -DNDEBUG) into the SYCL compile step.
# Without an explicit -O flag, acpp warns "No optimization flag was
# given, optimizations are disabled by default" and the AES-heavy SYCL
# kernels (Xs gen, T*match) compile at -O0, which is dramatically
# slower on amdgcn (Xs gen alone was 200 ms / ~25% of wall on RX 6700
# XT before this fix).
#
# An earlier attempt at -O3 was reverted because parity tests appeared
# to fail with it — but that diagnosis was confounded by an unrelated
# build-time bug (compose.yaml's silent ACPP_GFX default to gfx1100
# made every "broken" rebuild produce kernels for the wrong amdgcn
# ISA, which executed as no-ops regardless of opt level). With
# ACPP_GFX now enforced via ${VAR:?} in compose.yaml, -O3 should be
# testable cleanly. Drop to -O2 here if it actually does fail at -O3
# under correct gfx targeting.
target_compile_options(pos2_gpu PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-O2>
    $<$<CONFIG:MinSizeRel>:-Os>)
# The SYCL TUs include CUDA headers (cuda_fp16.h, transitively cuda_runtime.h
# from the kernel-wrapper headers) on both the CUDA and non-CUDA paths
# (slice 17 will lift the CUDA-type dependencies out of the public API).
# On the CUDA build we already have CMAKE_CUDA_COMPILER. On the non-CUDA
# build we need to locate the CUDA Toolkit headers via find_package
# (CUDAToolkit) — which does NOT require enable_language(CUDA).
if(XCHPLOT2_BUILD_CUDA)
    get_filename_component(_xchplot2_cuda_bin ${CMAKE_CUDA_COMPILER} DIRECTORY)
    get_filename_component(_xchplot2_cuda_root ${_xchplot2_cuda_bin} DIRECTORY)
    set(_xchplot2_cuda_include "${_xchplot2_cuda_root}/include")
else()
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_INCLUDE_DIRS)
        set(_xchplot2_cuda_include ${CUDAToolkit_INCLUDE_DIRS})
    else()
        # Last-resort guess; matches Arch / CachyOS layout.
        set(_xchplot2_cuda_include "/opt/cuda/include")
    endif()
endif()
target_include_directories(pos2_gpu PRIVATE ${_xchplot2_cuda_include})
if(XCHPLOT2_BUILD_CUDA)
    # OBJECT lib doesn't inherit pos2_gpu's PUBLIC includes via
    # $<TARGET_OBJECTS:> (only the .o files travel), so propagate the
    # CUDA include path explicitly. Mirrors the line above for pos2_gpu.
    target_include_directories(pos2_gpu_cuda_obj PRIVATE ${_xchplot2_cuda_include})
endif()

# Slice 17 removed the last SYCL-TU reference to a cudart *function* — only
# cuda* types survive (used for API compatibility), and types don't require
# a link against libcudart.so. On the NVIDIA build path the nvcc-compiled
# TUs (AesGpu.cu, SortCuda.cu, AesGpuBitsliced.cu) bring in cudart
# automatically. On non-NVIDIA builds cudart isn't needed at all.
# Now that the kernel-wrapper headers (T*Offsets.cuh, PipelineKernels.cuh,
# T*Kernel.cuh, XsKernel.cuh) take sycl::queue&, every TU that includes them
# needs sycl/sycl.hpp on its include path — including the parity tests
# compiled by nvcc. Make AdaptiveCpp's include dir PUBLIC so it propagates.
get_filename_component(_xchplot2_acpp_cmake_dir
    "${AdaptiveCpp_DIR}" DIRECTORY)  # /opt/adaptivecpp/lib/cmake/AdaptiveCpp/.. = /opt/adaptivecpp/lib/cmake
get_filename_component(_xchplot2_acpp_lib_dir
    "${_xchplot2_acpp_cmake_dir}" DIRECTORY)  # /opt/adaptivecpp/lib
get_filename_component(_xchplot2_acpp_root
    "${_xchplot2_acpp_lib_dir}" DIRECTORY)  # /opt/adaptivecpp
target_include_directories(pos2_gpu PUBLIC
    ${_xchplot2_acpp_root}/include
    ${_xchplot2_acpp_root}/include/AdaptiveCpp)
if(XCHPLOT2_BUILD_CUDA)
    # Same reasoning as the CUDA include above — propagate AdaptiveCpp's
    # include dir to the OBJECT lib explicitly so its .cu TUs see the
    # kernel-wrapper headers (T*Offsets.cuh / PipelineKernels.cuh / ...)
    # that pull in sycl/sycl.hpp.
    target_include_directories(pos2_gpu_cuda_obj PRIVATE
        ${_xchplot2_acpp_root}/include
        ${_xchplot2_acpp_root}/include/AdaptiveCpp)
endif()

set_target_properties(pos2_gpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    # No CUDA .o files in this archive (they live in pos2_gpu_cuda_obj
    # OBJECT lib and are added explicitly to each leaf consumer), so
    # device-symbol resolution doesn't apply here. CUDA_RESOLVE_DEVICE_SYMBOLS
    # is left explicitly OFF for clarity and to defend against any future
    # CUDA TU getting added to pos2_gpu's source list.
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# Host orchestration (no CUDA code, just thin wrappers).
# PlotFileWriterParallel is the SOLE TU pulling in pos2-chip's plot/PlotFile.hpp
# + pos/ProofParams.hpp chain. Keeping it in one .cpp avoids multiple-definition
# link errors caused by non-inline soft_aesenc/soft_aesdec in pos2-chip headers.
add_library(pos2_gpu_host STATIC
    src/host/GpuPlotter.cpp
    src/host/PlotFileWriterParallel.cpp
    src/host/BatchPlotter.cpp
    src/host/CpuPlotter.cpp
    src/host/Cancel.cpp
    src/host/MultiGpuPlotPipeline.cpp
    src/host/MultiGpuPipelineParallel.cpp
    src/host/TempFile.cpp
)
target_include_directories(pos2_gpu_host PUBLIC src)
target_link_libraries(pos2_gpu_host PUBLIC pos2_chip_headers pos2_gpu)

# ----------------------------------------------------------------------------
# pos2_keygen — Rust staticlib that wraps chia-rs (BLS + compute_plot_id_v2)
# so the standalone plotter can derive v2 plot_ids + memos without the chia
# Python toolchain.
#
# Built via cargo in keygen-rs/. We shell out to cargo from a custom_command
# rather than taking the Corrosion dependency, to keep the build system
# self-contained.
# ----------------------------------------------------------------------------
set(POS2_KEYGEN_CRATE_DIR "${CMAKE_SOURCE_DIR}/keygen-rs")
set(POS2_KEYGEN_TARGET_DIR "${CMAKE_BINARY_DIR}/keygen-rs-target")
set(POS2_KEYGEN_STATIC_LIB "${POS2_KEYGEN_TARGET_DIR}/release/libpos2_keygen.a")

# Resolve `cargo` once, with HINTS for the rustup default install
# location ($HOME/.cargo/bin). rustup writes its PATH entry to
# ~/.bashrc / ~/.profile / ~/.cargo/env at install time, but a non-
# login `ssh host "cmd"` shell doesn't source any of those — so a
# CI-style invocation that ran `install-deps.sh` and then immediately
# tried `cmake --build` from the same ssh session would otherwise
# fail to find cargo on PATH. Using find_program with HINTS makes
# the build pick up rustup's cargo regardless of shell login state.
find_program(POS2_CARGO_EXECUTABLE cargo
    HINTS "$ENV{HOME}/.cargo/bin"
          "$ENV{CARGO_HOME}/bin"
          "/root/.cargo/bin"
    DOC "Path to the rustup-installed cargo")
if(NOT POS2_CARGO_EXECUTABLE)
    message(FATAL_ERROR
        "xchplot2: cargo not found. Install Rust via rustup "
        "(scripts/install-deps.sh handles this) and re-run cmake. "
        "rustup's default install drops cargo at ~/.cargo/bin/cargo.")
endif()

# Collect the sources so CMake knows when to rebuild.
file(GLOB_RECURSE POS2_KEYGEN_SRCS
    "${POS2_KEYGEN_CRATE_DIR}/src/*.rs"
    "${POS2_KEYGEN_CRATE_DIR}/Cargo.toml"
    "${POS2_KEYGEN_CRATE_DIR}/Cargo.lock"
)

add_custom_command(
    OUTPUT "${POS2_KEYGEN_STATIC_LIB}"
    COMMAND ${CMAKE_COMMAND} -E env
            CARGO_TARGET_DIR=${POS2_KEYGEN_TARGET_DIR}
            ${POS2_CARGO_EXECUTABLE} build --release --manifest-path=${POS2_KEYGEN_CRATE_DIR}/Cargo.toml
    DEPENDS ${POS2_KEYGEN_SRCS}
    WORKING_DIRECTORY "${POS2_KEYGEN_CRATE_DIR}"
    COMMENT "cargo build --release (pos2_keygen Rust shim)"
    VERBATIM
)
add_custom_target(pos2_keygen_build DEPENDS "${POS2_KEYGEN_STATIC_LIB}")

add_library(pos2_keygen STATIC IMPORTED GLOBAL)
set_target_properties(pos2_keygen PROPERTIES
    IMPORTED_LOCATION "${POS2_KEYGEN_STATIC_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${POS2_KEYGEN_CRATE_DIR}/include"
)
add_dependencies(pos2_keygen pos2_keygen_build)

# Rust staticlibs link in std, which needs these system libs at final link.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(pos2_keygen INTERFACE pthread dl m)
endif()

# xchplot2_cli: the CLI dispatch logic as a static library, with the
# entrypoint exported as `extern "C" int xchplot2_main(int, char**)`.
# Two consumers:
#   - tools/xchplot2/main.cpp builds the standalone executable below.
#   - the top-level Cargo crate (build.rs + src/main.rs) wraps this in a
#     Rust `[[bin]]` so `cargo install --git ...` produces the same binary.
#
# CUDA_RESOLVE_DEVICE_SYMBOLS ON makes this static lib self-contained at
# the device-symbol level, so a non-CMake linker (rustc) doesn't need to
# do its own `nvcc -dlink` step. We set it on this aggregate lib rather
# than on `pos2_gpu` so the parity tests (which reference device symbols
# externally) keep working unchanged.
add_library(xchplot2_cli STATIC tools/xchplot2/cli.cpp)
target_include_directories(xchplot2_cli PUBLIC tools/xchplot2)
target_link_libraries(xchplot2_cli PUBLIC pos2_gpu_host pos2_keygen)
# CUDA_RESOLVE_DEVICE_SYMBOLS=ON triggers an nvcc --device-link step at
# archive creation, producing a host-side dlink.o that defines the
# `__cudaRegisterLinkedBinary_*` symbols every `__sti____cudaRegisterAll()`
# constructor references. cli_devlink.cu is the marker that flips
# xchplot2_cli to a CUDA-language target so the device-link actually
# fires (it's a silent no-op on pure-C++ targets — see cli_devlink.cu).
#
# Just adding cli_devlink.cu isn't enough: the dlink.o it produces only
# resolves symbols for .cu objects directly compiled into xchplot2_cli.
# Pulling pos2_gpu's CUDA .o files in via $<TARGET_OBJECTS:pos2_gpu_cuda_obj>
# brings them into xchplot2_cli's archive-time device-link scope so the
# resulting dlink.o covers them too. See the pos2_gpu_cuda_obj OBJECT-lib
# comment above for why we share the .o files instead of recompiling.
if(XCHPLOT2_BUILD_CUDA)
    target_sources(xchplot2_cli PRIVATE
        tools/xchplot2/cli_devlink.cu
        $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
endif()
set_target_properties(xchplot2_cli PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CLI: xchplot2  (the standalone plotter binary, formerly gpu_plotter)
#
# LINK_GROUP RESCAN wraps xchplot2_cli + pos2_gpu_host so the linker
# rescans them as a unit. xchplot2_cli holds the CUDA OBJECT files
# (initialize_aes_tables, cub_sort_*); pos2_gpu_host's BatchPlotter.cpp
# and SortSyclCub.cpp reference those symbols. With single-pass static-
# archive scanning the references would land after xchplot2_cli was
# already processed — rescan resolves the back-edge.
add_executable(xchplot2 tools/xchplot2/main.cpp)
target_link_libraries(xchplot2 PRIVATE
    "$<LINK_GROUP:RESCAN,xchplot2_cli,pos2_gpu_host>")
# pos2-chip headers define non-inline soft_aesenc/soft_aesdec, which now
# end up in two TUs (PlotFileWriterParallel.cpp and CpuPlotter.cpp) inside
# pos2_gpu_host. Tolerate the duplicates at host link.
target_link_options(xchplot2 PRIVATE LINKER:--allow-multiple-definition)

# Parity tests are nvcc-compiled (.cu) and reference __global__ kernels
# from the bench-specific bitsliced AES path. They build only on the CUDA
# target. The two SYCL-native parity tests below (sycl_*_parity) stay
# unconditional so AMD/Intel builds still have correctness coverage.
#
# Each test gets $<TARGET_OBJECTS:pos2_gpu_cuda_obj> explicitly:
# pos2_gpu (STATIC) doesn't carry the CUDA .o files anymore — putting
# them in both pos2_gpu and xchplot2_cli triggered nvlink's "Multiple
# definition" error at xchplot2_cli's archive-time device-link, which
# host-only --allow-multiple-definition can't suppress. So leaf
# executables that need kernel symbols (kAesT0..3, host-side
# kernel-wrapper functions in pos2_gpu_host) pull them in directly,
# making the .o files appear exactly once in each link line.
if(XCHPLOT2_BUILD_CUDA)
    foreach(t IN ITEMS aes_parity aes_bs_parity aes_bs_bench aes_tezcan_bench
                       xs_parity xs_bench t1_parity t1_debug t2_parity t3_parity)
        add_executable(${t} tools/parity/${t}.cu $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
        target_link_libraries(${t} PRIVATE pos2_gpu_host)
        set_target_properties(${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")
    endforeach()

    message(STATUS "pos2-gpu configured for CUDA arch(es): ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# plot_file_parity is a pure .cpp harness — reads a .plot file via
# pos2_gpu_host's file-format code and checks the header / table offsets.
# Builds on all backends (CUDA, HIP, SYCL-only). On the CUDA build it
# transitively needs pos2_gpu_host's kernel-wrapper symbols, which now
# live in the OBJECT lib rather than pos2_gpu.a — pull them in here.
if(XCHPLOT2_BUILD_CUDA)
    add_executable(plot_file_parity tools/parity/plot_file_parity.cpp
        $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
else()
    add_executable(plot_file_parity tools/parity/plot_file_parity.cpp)
endif()
target_link_libraries(plot_file_parity PRIVATE pos2_gpu_host)
set_target_properties(plot_file_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Group binaries under build/tools/...
set_target_properties(xchplot2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/xchplot2")

# Slice-1 standalone SYCL parity test: exercises compute_bucket_offsets in
# isolation against a CPU reference on synthetic input — orthogonal to the
# t1_parity full-pipeline test, useful for narrowing any divergence to the
# SYCL kernel itself.
add_executable(sycl_bucket_offsets_parity tools/parity/sycl_bucket_offsets_parity.cpp)
add_sycl_to_target(TARGET sycl_bucket_offsets_parity
                   SOURCES tools/parity/sycl_bucket_offsets_parity.cpp)

add_executable(host_pinned_pool_test tools/parity/host_pinned_pool_test.cpp)
add_sycl_to_target(TARGET host_pinned_pool_test
                   SOURCES tools/parity/host_pinned_pool_test.cpp)
target_link_libraries(host_pinned_pool_test PRIVATE pos2_gpu_host)

add_executable(temp_file_test tools/parity/temp_file_test.cpp)
target_link_libraries(temp_file_test PRIVATE pos2_gpu_host)
target_compile_features(temp_file_test PRIVATE cxx_std_20)
target_compile_features(sycl_bucket_offsets_parity PRIVATE cxx_std_20)
set_target_properties(sycl_bucket_offsets_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Slice-4 standalone: validates the SYCL-compiled AES g_x_smem against the
# same function run on the host. Pulls the AES headers (now portable behind
# PortableAttrs.hpp) directly, so a host-vs-device divergence in the AES
# math isolates here without t1_parity scaffolding.
add_executable(sycl_g_x_parity tools/parity/sycl_g_x_parity.cpp)
add_sycl_to_target(TARGET sycl_g_x_parity
                   SOURCES tools/parity/sycl_g_x_parity.cpp)
target_include_directories(sycl_g_x_parity PRIVATE src)
target_compile_features(sycl_g_x_parity PRIVATE cxx_std_20)
set_target_properties(sycl_g_x_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Slice-18 standalone: exercises launch_sort_pairs_u32_u32 and
# launch_sort_keys_u64 against a std::sort reference. Built always — runs
# the CUB-backed wrappers when XCHPLOT2_BUILD_CUDA=ON, the hand-rolled
# SYCL radix when OFF. Lets the SYCL sort path be validated on NVIDIA
# hardware without needing AMD/Intel access.
add_executable(sycl_sort_parity tools/parity/sycl_sort_parity.cpp)
add_sycl_to_target(TARGET sycl_sort_parity
                   SOURCES tools/parity/sycl_sort_parity.cpp)
target_link_libraries(sycl_sort_parity PRIVATE pos2_gpu)
# On the CUDA build path, pos2_gpu's SortSyclCub.cpp (the SYCL→CUB
# adapter) calls cub_sort_* defined in SortCuda.cu — now in
# pos2_gpu_cuda_obj OBJECT lib instead of pos2_gpu STATIC. Pull the
# OBJECT lib's .o files in directly so the CUB symbols resolve.
# AMD/Intel builds use SortSycl.cpp (pure SYCL) instead and don't
# need this.
if(XCHPLOT2_BUILD_CUDA)
    target_sources(sycl_sort_parity PRIVATE $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
endif()
# cuda_fp16.h transitively required by SyclBackend.hpp → sycl/sycl.hpp
# (AdaptiveCpp's half.hpp uses cuda_fp16 intrinsics on the CUDA backend).
target_include_directories(sycl_sort_parity PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_sort_parity PRIVATE cxx_std_20)
set_target_properties(sycl_sort_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Phase 0 of the StreamingPinned/StreamingDisk tier plan
# (project_streaming_pinned_disk_spec). Exercises the new scatter
# family (launch_compute_inverse_u32 + launch_scatter_{u64,u32} +
# launch_permute_t2_scatter) against the existing gather family.
# Pure SYCL — same wiring as sycl_sort_parity (CUDA-obj pull-in is
# only needed because pos2_gpu's SortSyclCub.cpp transitively calls
# CUB symbols, not because scatter itself needs CUB).
add_executable(sycl_scatter_parity tools/parity/sycl_scatter_parity.cpp)
add_sycl_to_target(TARGET sycl_scatter_parity
                   SOURCES tools/parity/sycl_scatter_parity.cpp)
target_link_libraries(sycl_scatter_parity PRIVATE pos2_gpu)
if(XCHPLOT2_BUILD_CUDA)
    target_sources(sycl_scatter_parity PRIVATE $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
endif()
target_include_directories(sycl_scatter_parity PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_scatter_parity PRIVATE cxx_std_20)
set_target_properties(sycl_scatter_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Phase 1 of the StreamingPinned/StreamingDisk tier plan. Wraps
# launch_sort_pairs_u32_u32 in a top-bits partition pass so the
# output is segmented into 2^num_top_bits contiguous buckets. The
# parity test compares byte-identical output to a single-level
# sort over the same inputs, plus structural checks on the
# d_bucket_starts output array.
add_executable(sycl_two_level_sort_parity tools/parity/sycl_two_level_sort_parity.cpp)
add_sycl_to_target(TARGET sycl_two_level_sort_parity
                   SOURCES tools/parity/sycl_two_level_sort_parity.cpp)
target_link_libraries(sycl_two_level_sort_parity PRIVATE pos2_gpu)
if(XCHPLOT2_BUILD_CUDA)
    target_sources(sycl_two_level_sort_parity PRIVATE $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
endif()
target_include_directories(sycl_two_level_sort_parity PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_two_level_sort_parity PRIVATE cxx_std_20)
set_target_properties(sycl_two_level_sort_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Phase 2.2 — sharded Xs-phase parity test. Confirms
# MultiGpuPlotPipeline's run_xs_phase produces a union of per-shard
# packed XsCandidateGpu outputs that's byte-identical to single-GPU
# launch_xs_gen + sort + launch_xs_pack. Uses N=2 virtual shards on
# the same physical queue so it runs on the dev box without real
# multi-GPU hardware.
add_executable(sycl_xs_phase_sharded_parity
    tools/parity/sycl_xs_phase_sharded_parity.cpp)
add_sycl_to_target(TARGET sycl_xs_phase_sharded_parity
                   SOURCES tools/parity/sycl_xs_phase_sharded_parity.cpp)
target_link_libraries(sycl_xs_phase_sharded_parity
    PRIVATE pos2_gpu pos2_gpu_host)
if(XCHPLOT2_BUILD_CUDA)
    target_sources(sycl_xs_phase_sharded_parity
                   PRIVATE $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
endif()
target_include_directories(sycl_xs_phase_sharded_parity
                           PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_xs_phase_sharded_parity PRIVATE cxx_std_20)
set_target_properties(sycl_xs_phase_sharded_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Phase 2.3a — sharded T1-phase parity test. Confirms
# MultiGpuPlotPipeline's run_t1_phase produces a union of per-shard
# sorted (mi, meta) outputs that's byte-identical to single-GPU
# launch_t1_match + identity-index radix sort by mi + meta gather.
# Uses N=2 virtual shards on the same physical queue so it runs on
# the dev box without real multi-GPU hardware.
add_executable(sycl_t1_phase_sharded_parity
    tools/parity/sycl_t1_phase_sharded_parity.cpp)
add_sycl_to_target(TARGET sycl_t1_phase_sharded_parity
                   SOURCES tools/parity/sycl_t1_phase_sharded_parity.cpp)
target_link_libraries(sycl_t1_phase_sharded_parity
    PRIVATE pos2_gpu pos2_gpu_host)
if(XCHPLOT2_BUILD_CUDA)
    target_sources(sycl_t1_phase_sharded_parity
                   PRIVATE $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
endif()
target_include_directories(sycl_t1_phase_sharded_parity
                           PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_t1_phase_sharded_parity PRIVATE cxx_std_20)
set_target_properties(sycl_t1_phase_sharded_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Phase 2.3b — sharded T2-phase parity test. Confirms
# MultiGpuPlotPipeline's run_t2_phase produces a union of per-shard
# sorted (mi, meta, xbits) outputs that's a multiset-equivalent to
# single-GPU launch_t2_match + identity sort + launch_permute_t2.
# Uses N=2 virtual shards on the same physical queue.
add_executable(sycl_t2_phase_sharded_parity
    tools/parity/sycl_t2_phase_sharded_parity.cpp)
add_sycl_to_target(TARGET sycl_t2_phase_sharded_parity
                   SOURCES tools/parity/sycl_t2_phase_sharded_parity.cpp)
target_link_libraries(sycl_t2_phase_sharded_parity
    PRIVATE pos2_gpu pos2_gpu_host)
if(XCHPLOT2_BUILD_CUDA)
    target_sources(sycl_t2_phase_sharded_parity
                   PRIVATE $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
endif()
target_include_directories(sycl_t2_phase_sharded_parity
                           PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_t2_phase_sharded_parity PRIVATE cxx_std_20)
set_target_properties(sycl_t2_phase_sharded_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Phase 2.3c — sharded T3-phase parity test. Confirms
# MultiGpuPlotPipeline's run_t3_phase produces a union of per-shard
# sorted u64 proof_fragment outputs that's a multiset-equivalent to
# single-GPU launch_t3_match + launch_sort_keys_u64.
# Uses N=2 virtual shards on the same physical queue.
add_executable(sycl_t3_phase_sharded_parity
    tools/parity/sycl_t3_phase_sharded_parity.cpp)
add_sycl_to_target(TARGET sycl_t3_phase_sharded_parity
                   SOURCES tools/parity/sycl_t3_phase_sharded_parity.cpp)
target_link_libraries(sycl_t3_phase_sharded_parity
    PRIVATE pos2_gpu pos2_gpu_host)
if(XCHPLOT2_BUILD_CUDA)
    target_sources(sycl_t3_phase_sharded_parity
                   PRIVATE $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
endif()
target_include_directories(sycl_t3_phase_sharded_parity
                           PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_t3_phase_sharded_parity PRIVATE cxx_std_20)
set_target_properties(sycl_t3_phase_sharded_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Phase 2.3d — sharded fragment-phase parity test. Drives the full
# pipeline.run() (Xs through fragment phase) and verifies the host-
# bounced concatenated fragment span is multiset-equivalent to single-
# GPU. Validates the D2H + concat in run_fragment_phase on top of the
# T3 phase parity.
add_executable(sycl_fragment_phase_sharded_parity
    tools/parity/sycl_fragment_phase_sharded_parity.cpp)
add_sycl_to_target(TARGET sycl_fragment_phase_sharded_parity
                   SOURCES tools/parity/sycl_fragment_phase_sharded_parity.cpp)
target_link_libraries(sycl_fragment_phase_sharded_parity
    PRIVATE pos2_gpu pos2_gpu_host)
if(XCHPLOT2_BUILD_CUDA)
    target_sources(sycl_fragment_phase_sharded_parity
                   PRIVATE $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
endif()
target_include_directories(sycl_fragment_phase_sharded_parity
                           PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_fragment_phase_sharded_parity PRIVATE cxx_std_20)
set_target_properties(sycl_fragment_phase_sharded_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Phase 2.0b — distributed sort fast-path regression test. Confirms
# launch_sort_*_distributed with shards.size() == 1 produces byte-
# identical output to the single-shard sort, and that N>1 throws the
# expected "not yet implemented" message. See
# docs/multi-gpu-single-plot-alt-bucket-partition.md.
add_executable(sycl_distributed_sort_parity
    tools/parity/sycl_distributed_sort_parity.cpp)
add_sycl_to_target(TARGET sycl_distributed_sort_parity
                   SOURCES tools/parity/sycl_distributed_sort_parity.cpp)
target_link_libraries(sycl_distributed_sort_parity PRIVATE pos2_gpu)
if(XCHPLOT2_BUILD_CUDA)
    target_sources(sycl_distributed_sort_parity
                   PRIVATE $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
endif()
target_include_directories(sycl_distributed_sort_parity
                           PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_distributed_sort_parity PRIVATE cxx_std_20)
set_target_properties(sycl_distributed_sort_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# SYCL-native sibling of t1_parity.cu. The .cu version is nvcc-only, so on
# AMD/Intel hosts the T1 matcher had no end-to-end CPU-vs-GPU coverage —
# this binary closes that gap. Same comparison semantics as t1_parity.cu
# (sorted-set equality of T1Pairings against pos2-chip's Table1Constructor),
# but uses sycl::malloc_device + q.memcpy in place of cudaMalloc /
# cudaMemcpy so it builds on the SYCL-only path too.
if(XCHPLOT2_BUILD_CUDA)
    add_executable(sycl_t1_parity tools/parity/sycl_t1_parity.cpp
        $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
else()
    add_executable(sycl_t1_parity tools/parity/sycl_t1_parity.cpp)
endif()
add_sycl_to_target(TARGET sycl_t1_parity
                   SOURCES tools/parity/sycl_t1_parity.cpp)
target_link_libraries(sycl_t1_parity PRIVATE pos2_gpu_host)
target_include_directories(sycl_t1_parity PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_t1_parity PRIVATE cxx_std_20)
# pos2-chip's plot/PlotLayout.hpp + plot/TableConstructorGeneric.hpp pull
# in non-inline soft_aesenc/soft_aesdec, which already exist in pos2_gpu_host
# via PlotFileWriterParallel.cpp + CpuPlotter.cpp. Same mitigation as the
# xchplot2 CLI link line — see the --allow-multiple-definition note above.
target_link_options(sycl_t1_parity PRIVATE LINKER:--allow-multiple-definition)
set_target_properties(sycl_t1_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Phase 2.1 — pipeline-parallel split parity. Drives the
# stop_after_t2_sort + start_at_t3_match entry points on the same
# physical queue and confirms the resulting plot fragments match a
# single-call run. Validates the T2-sort boundary handoff with the
# distinct-h_t2_meta buffer-reuse fix.
if(XCHPLOT2_BUILD_CUDA)
    add_executable(sycl_phase_split_parity tools/parity/sycl_phase_split_parity.cpp
        $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
else()
    add_executable(sycl_phase_split_parity tools/parity/sycl_phase_split_parity.cpp)
endif()
add_sycl_to_target(TARGET sycl_phase_split_parity
                   SOURCES tools/parity/sycl_phase_split_parity.cpp)
target_link_libraries(sycl_phase_split_parity PRIVATE pos2_gpu_host)
target_include_directories(sycl_phase_split_parity PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_phase_split_parity PRIVATE cxx_std_20)
target_link_options(sycl_phase_split_parity PRIVATE LINKER:--allow-multiple-definition)
set_target_properties(sycl_phase_split_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Phase 2.1c — pipeline-parallel orchestrator parity. Drives
# run_pipeline_parallel_split with both halves pointing at device 0
# (single-GPU virtual two-device test). Real two-GPU validation runs
# on the VPS via --dev-first 0 --dev-second 1.
if(XCHPLOT2_BUILD_CUDA)
    add_executable(sycl_pipeline_parallel_parity tools/parity/sycl_pipeline_parallel_parity.cpp
        $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
else()
    add_executable(sycl_pipeline_parallel_parity tools/parity/sycl_pipeline_parallel_parity.cpp)
endif()
add_sycl_to_target(TARGET sycl_pipeline_parallel_parity
                   SOURCES tools/parity/sycl_pipeline_parallel_parity.cpp)
target_link_libraries(sycl_pipeline_parallel_parity PRIVATE pos2_gpu_host)
target_include_directories(sycl_pipeline_parallel_parity PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_pipeline_parallel_parity PRIVATE cxx_std_20)
target_link_options(sycl_pipeline_parallel_parity PRIVATE LINKER:--allow-multiple-definition)
set_target_properties(sycl_pipeline_parallel_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Lowest-level diagnostic: a hello-world SYCL kernel that proves
# AdaptiveCpp's HIP / CUDA backend can dispatch *anything* on the
# detected device. No pos2_gpu / pos2_gpu_host link — purely the SYCL
# runtime + a 16-element parallel_for. Use it as the first step when
# sycl_t1_parity or the production CLI silently produces no output: if
# hellosycl FAILs, no xchplot2-level fix can recover and the issue is
# below our level (driver mismatch, JIT no-op stubs, etc.).
add_executable(hellosycl tools/sanity/hellosycl.cpp)
add_sycl_to_target(TARGET hellosycl SOURCES tools/sanity/hellosycl.cpp)
target_compile_features(hellosycl PRIVATE cxx_std_20)
set_target_properties(hellosycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/sanity")
