cmake_minimum_required(VERSION 3.24)

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
add_library(pos2_gpu STATIC
    src/gpu/AesGpu.cu
    src/gpu/AesGpuBitsliced.cu
    src/gpu/XsKernel.cu
    src/gpu/T1Kernel.cu
    src/gpu/T2Kernel.cu
    src/gpu/T3Kernel.cu
    src/host/GpuBufferPool.cu
    src/host/GpuPipeline.cu
)
target_include_directories(pos2_gpu PUBLIC
    src
)
target_link_libraries(pos2_gpu PUBLIC pos2_chip_headers)
target_compile_features(pos2_gpu PUBLIC cxx_std_20)
if(XCHPLOT2_INSTRUMENT_MATCH)
    target_compile_definitions(pos2_gpu PUBLIC XCHPLOT2_INSTRUMENT_MATCH=1)
endif()
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
# cli_devlink.cu is a stub CUDA source whose only job is to make
# xchplot2_cli a CUDA-language target. CUDA_RESOLVE_DEVICE_SYMBOLS=ON
# is only honoured by CMake on targets that contain at least one CUDA
# source — for a pure-C++ static lib it's a silent no-op, leaving
# `__cudaRegisterLinkedBinary_*` unresolved when Rust's host linker
# (cargo install) puts the final binary together. See cli_devlink.cu
# for the long-form explanation.
add_library(xchplot2_cli STATIC
    tools/xchplot2/cli.cpp
    tools/xchplot2/cli_devlink.cu)
target_include_directories(xchplot2_cli PUBLIC tools/xchplot2)
target_link_libraries(xchplot2_cli PUBLIC pos2_gpu_host pos2_keygen)
set_target_properties(xchplot2_cli PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# CLI: xchplot2  (the standalone plotter binary, formerly gpu_plotter)
add_executable(xchplot2 tools/xchplot2/main.cpp)
target_link_libraries(xchplot2 PRIVATE xchplot2_cli)

# Parity tests
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

add_executable(plot_file_parity tools/parity/plot_file_parity.cpp)
target_link_libraries(plot_file_parity PRIVATE pos2_gpu_host)
set_target_properties(plot_file_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")

# Group binaries under build/tools/...
set_target_properties(xchplot2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/xchplot2")
foreach(t aes_parity aes_bs_parity aes_bs_bench aes_tezcan_bench xs_parity xs_bench t1_parity t2_parity t3_parity)
    set_target_properties(${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")
endforeach()

message(STATUS "pos2-gpu configured for CUDA arch(es): ${CMAKE_CUDA_ARCHITECTURES}")
