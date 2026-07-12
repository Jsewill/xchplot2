cmake_minimum_required(VERSION 3.24)

# Preflight nvcc-vs-arch compatibility BEFORE project(LANGUAGES ... CUDA),
# which is what triggers the cryptic "Unsupported gpu architecture
# 'compute_61'" TryCompile failure when Pascal/Volta meets CUDA 13.x.
# CUDA 13.0 dropped codegen for sm_50/52/53/60/61/62/70/72 entirely.
# Skip the check if nvcc isn't findable yet — project() below will
# surface its own missing-toolchain message in that case.
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
    if(_nvcc_version_out MATCHES "release ([0-9]+)")
        set(_nvcc_major "${CMAKE_MATCH_1}")
        # Use the user-passed CMAKE_CUDA_ARCHITECTURES if any (cache
        # var is visible pre-project), else the same default 89 we'll
        # set below — keeps the check accurate even before project()
        # runs the default-fallback.
        if(DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(_arches "${CMAKE_CUDA_ARCHITECTURES}")
        else()
            set(_arches 89)
        endif()
        set(_min_arch 9999)
        foreach(_a IN LISTS _arches)
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
                    "      podman build -t xchplot2:cuda-only \\\n"
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
                "xchplot2 (cuda-only): CUDA Toolkit ${_nvcc_major}.x dropped codegen for "
                "sm_${_min_arch} (Pascal / Volta / pre-Turing).\n"
                "\n"
                "Detected:\n"
                "  nvcc ${_nvcc_major}.x at ${_xchplot2_nvcc}\n"
                "  target arch: sm_${_min_arch} (from CMAKE_CUDA_ARCHITECTURES=${_arches})\n"
                "\n"
                ${_fix_block})
        endif()
    endif()
endif()
# Symmetric ceiling check: a target arch NEWER than this nvcc can codegen
# — e.g. a Blackwell GeForce RTX 50-series (compute_cap 12.0 -> sm_120) on
# a CUDA < 12.8 toolkit, where nvcc dies with "Unsupported gpu architecture
# 'compute_120'". PTX is forward-compatible, so fall back to PTX from the
# highest arch this nvcc supports (queried authoritatively via
# `nvcc --list-gpu-arch`); the driver JIT-compiles it to the real GPU at
# runtime. Mirrors build.rs's cargo-install path. Must run BEFORE project()
# so enable_language(CUDA)'s TryCompile uses the corrected arch.
if(_xchplot2_nvcc)
    execute_process(
        COMMAND "${_xchplot2_nvcc}" --list-gpu-arch
        OUTPUT_VARIABLE _nvcc_arch_out
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    # Highest compute_XX nvcc can emit. The list is NOT sorted on CUDA 13.x
    # (compute_100, compute_110, compute_103, ...), so take the max.
    set(_max_supported 0)
    string(REGEX MATCHALL "compute_([0-9]+)" _arch_matches "${_nvcc_arch_out}")
    foreach(_m IN LISTS _arch_matches)
        string(REGEX REPLACE "compute_" "" _m "${_m}")
        if(_m GREATER _max_supported)
            set(_max_supported ${_m})
        endif()
    endforeach()
    # Target arch: user-passed list if any, else the same default 89 the
    # floor check used. Resolve `native`/`all` via the live GPU compute_cap.
    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(_ceil_arches "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(_ceil_arches 89)
    endif()
    set(_want 0)
    if(_ceil_arches MATCHES "native|all")
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
            OUTPUT_VARIABLE _cap_out
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_cap_out MATCHES "([0-9]+)\\.([0-9]+)")
            math(EXPR _want "${CMAKE_MATCH_1} * 10 + ${CMAKE_MATCH_2}")
        endif()
    else()
        foreach(_a IN LISTS _ceil_arches)
            string(REGEX REPLACE "^(sm_|compute_)" "" _a "${_a}")
            string(REGEX REPLACE "-(real|virtual)$" "" _a "${_a}")
            if(_a MATCHES "^[0-9]+$" AND _a GREATER _want)
                set(_want ${_a})
            endif()
        endforeach()
    endif()
    if(_max_supported GREATER 0 AND _want GREATER _max_supported)
        message(WARNING
            "xchplot2 (cuda-only): target sm_${_want} is newer than this CUDA Toolkit "
            "can codegen (nvcc tops out at compute_${_max_supported}; sm_100/sm_120 "
            "Blackwell need CUDA 12.8+). Falling back to compute_${_max_supported} PTX, "
            "which the driver JIT-compiles to your GPU at runtime. For native SASS "
            "install CUDA 12.8+ and point cmake at it with "
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc.")
        set(CMAKE_CUDA_ARCHITECTURES "${_max_supported}-virtual")
    endif()
endif()
# Hand project(LANGUAGES ... CUDA) the nvcc the find_program above located.
# CMake's built-in toolkit search covers CUDAToolkit_ROOT / CUDA_PATH / $PATH /
# /usr/local/cuda — but NOT /opt/cuda, where Arch's `cuda` package installs it,
# and which only reaches PATH via /etc/profile.d/cuda.sh (a login-shell-only
# hook). Without this, configure dies with "Failed to find nvcc. Please set the
# CUDAToolkit_ROOT variable" on a host whose toolkit we found, ran, and
# version-checked seconds earlier. Must precede project() — that is what
# enables the CUDA language.
if(_xchplot2_nvcc AND NOT DEFINED CMAKE_CUDA_COMPILER)
    get_filename_component(_xchplot2_nvcc_bin  "${_xchplot2_nvcc}" DIRECTORY)
    get_filename_component(_xchplot2_nvcc_root "${_xchplot2_nvcc_bin}" DIRECTORY)
    set(CMAKE_CUDA_COMPILER "${_xchplot2_nvcc}"
        CACHE FILEPATH "nvcc located by xchplot2")
    if(NOT DEFINED CUDAToolkit_ROOT)
        set(CUDAToolkit_ROOT "${_xchplot2_nvcc_root}"
            CACHE PATH "CUDA toolkit root located by xchplot2")
    endif()
    message(STATUS "xchplot2: nvcc ${_xchplot2_nvcc} "
                   "(CUDAToolkit_ROOT=${_xchplot2_nvcc_root})")
endif()
unset(_xchplot2_nvcc CACHE)

project(pos2-gpu VERSION 0.8.0 LANGUAGES C CXX CUDA)

# Default to Release: a plain `cmake -B build -S .` otherwise builds
# with EMPTY optimization flags — host C++ (FSE compression, writer,
# host-side merges) at -O0 silently runs several times slower while
# still producing valid plots. build.rs pins Release explicitly; this
# covers manual configures (e.g. the README's Windows snippet).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
                 STRINGS Release Debug RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Every static library here is linked into both the standalone xchplot2
# executable and the top-level Rust crate's PIE binary (via build.rs +
# cargo install). rust-lld (the default linker on some distros) rejects
# non-PIC objects in a PIE output — seen in the wild as "relocation
# R_X86_64_32 cannot be used against local symbol; recompile with
# -fPIC" on BatchPlotter.cpp, GpuPlotter.cpp, etc. Setting this globally
# ensures pos2_gpu, pos2_gpu_host, fse, and any other transitively-
# compiled object is built with -fPIC, so the linker choice doesn't
# matter.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default arch: sm_89 (RTX 4090). Override via -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# Optional: compile in clock64 instrumentation for T3 match_all_buckets.
# Prints a per-region cycle breakdown to stderr on every launch_t3_match
# call. Off by default — enable with -DXCHPLOT2_INSTRUMENT_MATCH=ON.
option(XCHPLOT2_INSTRUMENT_MATCH "Instrument T3 match_all_buckets with clock64 breakdown" OFF)

# Shared-memory AES round variant for the hot kernels (Xs/T1/T2/T3).
#   auto     — tezcan16 on sm_89 when a TU opts in via XCHPLOT2_AES_TEZCAN_KERNEL;
#              ttable4 elsewhere / without opt-in (safe default for fat builds)
#   ttable4  — force the 4-table 4 KB path everywhere
#   tezcan16 — force Tezcan 16x-replica 16 KB path everywhere (needs hardware
#              validation on non-Ada arches; community measurements wanted)
set(XCHPLOT2_AES_ROUND "auto" CACHE STRING
    "AES smem round: auto | ttable4 | tezcan16")
set_property(CACHE XCHPLOT2_AES_ROUND PROPERTY STRINGS auto ttable4 tezcan16)
if(XCHPLOT2_AES_ROUND STREQUAL "ttable4")
    set(_AES_ROUND_DEFINE XCHPLOT2_AES_FORCE_TTABLE4=1)
elseif(XCHPLOT2_AES_ROUND STREQUAL "tezcan16")
    set(_AES_ROUND_DEFINE XCHPLOT2_AES_FORCE_TEZCAN16=1)
elseif(XCHPLOT2_AES_ROUND STREQUAL "auto")
    set(_AES_ROUND_DEFINE "")
else()
    message(FATAL_ERROR
        "XCHPLOT2_AES_ROUND must be auto, ttable4, or tezcan16 (got '${XCHPLOT2_AES_ROUND}')")
endif()
message(STATUS "XCHPLOT2_AES_ROUND=${XCHPLOT2_AES_ROUND}")

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
#
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
#
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
# honour it.) So pos2_gpu is now an INTERFACE library that only
# propagates includes and link deps; the actual .o files are added
# explicitly to each leaf consumer (xchplot2_cli + each parity test
# below) so they appear EXACTLY once in any link line.
add_library(pos2_gpu_cuda_obj OBJECT
    src/gpu/AesGpu.cu
    src/gpu/AesGpuBitsliced.cu
    src/gpu/XsKernel.cu
    src/gpu/T1Kernel.cu
    src/gpu/T2Kernel.cu
    src/gpu/T3Kernel.cu
    src/gpu/StreamingPartition.cu
    src/gpu/CudaDeviceList.cu
    src/host/GpuBufferPool.cu
    src/host/GpuPipeline.cu
)
target_include_directories(pos2_gpu_cuda_obj PRIVATE src)
target_link_libraries(pos2_gpu_cuda_obj PRIVATE pos2_chip_headers)
target_compile_features(pos2_gpu_cuda_obj PRIVATE cxx_std_20)
set_target_properties(pos2_gpu_cuda_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(XCHPLOT2_INSTRUMENT_MATCH)
    target_compile_definitions(pos2_gpu_cuda_obj PRIVATE XCHPLOT2_INSTRUMENT_MATCH=1)
endif()
if(_AES_ROUND_DEFINE)
    target_compile_definitions(pos2_gpu_cuda_obj PRIVATE ${_AES_ROUND_DEFINE})
endif()

# pos2_gpu: INTERFACE lib. Only propagates the include path and the
# pos2_chip_headers / fse / AES compile flags. Carries no .o files —
# consumers add $<TARGET_OBJECTS:pos2_gpu_cuda_obj> directly.
add_library(pos2_gpu INTERFACE)
target_include_directories(pos2_gpu INTERFACE src)
target_link_libraries(pos2_gpu INTERFACE pos2_chip_headers)
target_compile_features(pos2_gpu INTERFACE cxx_std_20)
if(XCHPLOT2_INSTRUMENT_MATCH)
    target_compile_definitions(pos2_gpu INTERFACE XCHPLOT2_INSTRUMENT_MATCH=1)
endif()
if(_AES_ROUND_DEFINE)
    target_compile_definitions(pos2_gpu INTERFACE ${_AES_ROUND_DEFINE})
endif()

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
    src/host/ConfigFile.cpp
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
add_library(xchplot2_cli STATIC
    tools/xchplot2/cli.cpp
    tools/xchplot2/cli_devlink.cu
    $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
target_include_directories(xchplot2_cli PUBLIC tools/xchplot2)
target_link_libraries(xchplot2_cli PUBLIC pos2_gpu_host pos2_keygen)
set_target_properties(xchplot2_cli PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CLI: xchplot2  (the standalone plotter binary, formerly gpu_plotter)
add_executable(xchplot2 tools/xchplot2/main.cpp)
# Wrap the CUDA-bearing static libs in --start-group/--end-group so
# the linker can rescan to resolve cross-archive references. Since
# pos2_gpu became an INTERFACE library (the nvlink-dedup fix moved
# pos2_gpu_cuda_obj's .o files into xchplot2_cli only), pos2_gpu_host
# now references symbols (initialize_aes_tables, run_gpu_pipeline,
# bind_current_device, ...) that live in xchplot2_cli — its own
# consumer. CMake's default transitive link order has xchplot2_cli
# BEFORE pos2_gpu_host on the cc line, so single-pass ld scans past
# the resolutions before it sees the unresolved references and bails.
# Cargo install already passes --start-group via build.rs; the cmake
# exe needs the same treatment.
target_link_libraries(xchplot2 PRIVATE
    "$<LINK_GROUP:RESCAN,xchplot2_cli,pos2_gpu_host>")
# pos2-chip's soft_aesenc / soft_aesdec are defined (not just declared)
# in headers without `inline`, so any TU that includes the chain — both
# PlotFileWriterParallel.cpp and CpuPlotter.cpp do, transitively via
# PlotFile.hpp / Plotter.hpp — emits its own copy. Definitions are
# bit-identical (same source, same compile flags) so first-wins is
# correctness-safe. cargo install already passes the same flag in
# build.rs for an unrelated keygen-rs / libstd duplication; the cmake
# exe needs it too once CpuPlotter joined PlotFileWriterParallel as a
# pos2-chip-including TU.
target_link_options(xchplot2 PRIVATE LINKER:--allow-multiple-definition)

# Parity tests. Each test gets $<TARGET_OBJECTS:pos2_gpu_cuda_obj>
# explicitly: pos2_gpu (INTERFACE) doesn't carry the .o files, so
# leaf executables that need kernel symbols (kAesT0..3, host-side
# kernel-wrapper functions in pos2_gpu_host) must pull them in
# directly. Doing it here — rather than via pos2_gpu_host — keeps
# the .o files appearing exactly once in each link, which nvlink's
# device-link step requires (it doesn't honour
# --allow-multiple-definition).
foreach(t IN ITEMS aes_parity aes_bs_parity aes_bs_bench aes_tezcan_bench
                   xs_parity xs_bench t1_parity t1_debug t2_parity t3_parity)
    add_executable(${t} tools/parity/${t}.cu $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
    target_link_libraries(${t} PRIVATE pos2_gpu_host)
endforeach()

add_executable(plot_file_parity tools/parity/plot_file_parity.cpp
    $<TARGET_OBJECTS:pos2_gpu_cuda_obj>)
target_link_libraries(plot_file_parity PRIVATE pos2_gpu_host)
set_target_properties(plot_file_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Group binaries under build/tools/...
set_target_properties(xchplot2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/xchplot2")
foreach(t aes_parity aes_bs_parity aes_bs_bench aes_tezcan_bench xs_parity xs_bench t1_parity t1_debug t2_parity t3_parity)
    set_target_properties(${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")
endforeach()

message(STATUS "pos2-gpu configured for CUDA arch(es): ${CMAKE_CUDA_ARCHITECTURES}")
