cmake_minimum_required(VERSION 3.24)

project(pos2-gpu VERSION 0.5.2 LANGUAGES C CXX)

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

if(XCHPLOT2_BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    # Default arch: sm_89 (RTX 4090). Override via -DCMAKE_CUDA_ARCHITECTURES=...
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
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
set(POS2_CHIP_GIT_TAG "0a69995b21a30ee201408300043336c8415e0b81"
    CACHE STRING "pos2-chip commit / tag to fetch")

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
    src/gpu/T1OffsetsSycl.cpp
    src/gpu/T2OffsetsSycl.cpp
    src/gpu/T3OffsetsSycl.cpp
    src/gpu/PipelineKernelsSycl.cpp
    src/gpu/XsKernel.cpp
    src/gpu/XsKernelsSycl.cpp
    src/gpu/T1Kernel.cpp
    src/gpu/T2Kernel.cpp
    src/gpu/T3Kernel.cpp
    src/host/GpuBufferPool.cpp
    src/host/GpuPipeline.cpp)

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
    # Non-CUDA path: SortSycl.cpp (hand-rolled LSD radix in pure SYCL) +
    # AesStub.cpp no-op for initialize_aes_tables. Both compiled by acpp
    # via add_sycl_to_target.
    set(POS2_GPU_CUDA_SRC)
    list(APPEND POS2_GPU_SYCL_SRC
        src/gpu/SortSycl.cpp
        src/gpu/AesStub.cpp)
endif()

add_library(pos2_gpu STATIC
    ${POS2_GPU_CUDA_SRC}
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

set_target_properties(pos2_gpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    # Do NOT pre-resolve device symbols — consumers (e.g. aes_parity.cu)
    # reference kAesT* directly and need them visible at final device link.
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
    src/host/Cancel.cpp
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
            cargo build --release --manifest-path=${POS2_KEYGEN_CRATE_DIR}/Cargo.toml
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
set_target_properties(xchplot2_cli PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CLI: xchplot2  (the standalone plotter binary, formerly gpu_plotter)
add_executable(xchplot2 tools/xchplot2/main.cpp)
target_link_libraries(xchplot2 PRIVATE xchplot2_cli)

# Parity tests are nvcc-compiled (.cu) and reference __global__ kernels
# from the bench-specific bitsliced AES path. They build only on the CUDA
# target. The two SYCL-native parity tests below (sycl_*_parity) stay
# unconditional so AMD/Intel builds still have correctness coverage.
if(XCHPLOT2_BUILD_CUDA)
    add_executable(aes_parity tools/parity/aes_parity.cu)
    target_link_libraries(aes_parity PRIVATE pos2_gpu_host)

    add_executable(aes_bs_parity tools/parity/aes_bs_parity.cu)
    target_link_libraries(aes_bs_parity PRIVATE pos2_gpu_host)

    add_executable(aes_bs_bench tools/parity/aes_bs_bench.cu)
    target_link_libraries(aes_bs_bench PRIVATE pos2_gpu_host)

    add_executable(aes_tezcan_bench tools/parity/aes_tezcan_bench.cu)
    target_link_libraries(aes_tezcan_bench PRIVATE pos2_gpu_host)

    add_executable(xs_parity tools/parity/xs_parity.cu)
    target_link_libraries(xs_parity PRIVATE pos2_gpu_host)

    add_executable(xs_bench tools/parity/xs_bench.cu)
    target_link_libraries(xs_bench PRIVATE pos2_gpu_host)

    add_executable(t1_parity tools/parity/t1_parity.cu)
    target_link_libraries(t1_parity PRIVATE pos2_gpu_host)

    add_executable(t1_debug tools/parity/t1_debug.cu)
    target_link_libraries(t1_debug PRIVATE pos2_gpu_host)
    set_target_properties(t1_debug PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

    add_executable(t2_parity tools/parity/t2_parity.cu)
    target_link_libraries(t2_parity PRIVATE pos2_gpu_host)

    add_executable(t3_parity tools/parity/t3_parity.cu)
    target_link_libraries(t3_parity PRIVATE pos2_gpu_host)

    foreach(t aes_parity aes_bs_parity aes_bs_bench aes_tezcan_bench xs_parity xs_bench t1_parity t2_parity t3_parity)
        set_target_properties(${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")
    endforeach()

    message(STATUS "pos2-gpu configured for CUDA arch(es): ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# plot_file_parity is a pure .cpp harness — reads a .plot file via
# pos2_gpu_host's file-format code and checks the header / table offsets.
# No CUDA dependency, so it builds on all backends (CUDA, HIP, SYCL-only).
add_executable(plot_file_parity tools/parity/plot_file_parity.cpp)
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
# cuda_fp16.h transitively required by SyclBackend.hpp → sycl/sycl.hpp
# (AdaptiveCpp's half.hpp uses cuda_fp16 intrinsics on the CUDA backend).
target_include_directories(sycl_sort_parity PRIVATE ${_xchplot2_cuda_include})
target_compile_features(sycl_sort_parity PRIVATE cxx_std_20)
set_target_properties(sycl_sort_parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")
