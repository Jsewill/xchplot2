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
unset(_xchplot2_nvcc CACHE)

project(pos2-gpu VERSION 0.5.2 LANGUAGES C CXX CUDA)

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

# Host orchestration (no CUDA code, just thin wrappers).
# PlotFileWriterParallel is the SOLE TU pulling in pos2-chip's plot/PlotFile.hpp
# + pos/ProofParams.hpp chain. Keeping it in one .cpp avoids multiple-definition
# link errors caused by non-inline soft_aesenc/soft_aesdec in pos2-chip headers.
add_library(pos2_gpu_host STATIC
    src/host/GpuPlotter.cpp
    src/host/PlotFileWriterParallel.cpp
    src/host/BatchPlotter.cpp
    src/host/CpuPlotter.cpp
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
target_link_libraries(xchplot2 PRIVATE xchplot2_cli)

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
