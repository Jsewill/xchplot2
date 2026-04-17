cmake_minimum_required(VERSION 3.24)

project(pos2-gpu LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Default arch: sm_89 (RTX 4090). Override via -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# Locate pos2-chip relative to this repo. Override with -DPOS2_CHIP_DIR=/path
if(NOT DEFINED POS2_CHIP_DIR)
    set(POS2_CHIP_DIR "${CMAKE_SOURCE_DIR}/../pos2-chip" CACHE PATH "pos2-chip checkout")
endif()
if(NOT EXISTS "${POS2_CHIP_DIR}/src/pos/aes/AesHash.hpp")
    message(FATAL_ERROR
        "pos2-chip not found at ${POS2_CHIP_DIR}. "
        "Pass -DPOS2_CHIP_DIR=/abs/path/to/pos2-chip.")
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
    src/gpu/XsKernel.cu
    src/gpu/T1Kernel.cu
    src/gpu/T2Kernel.cu
    src/gpu/T3Kernel.cu
    src/host/GpuBufferPool.cu
    src/host/GpuPipeline.cu
    src/host/BatchPlotter.cu
)
target_include_directories(pos2_gpu PUBLIC
    src
)
target_link_libraries(pos2_gpu PUBLIC pos2_chip_headers)
target_compile_features(pos2_gpu PUBLIC cxx_std_20)
set_target_properties(pos2_gpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    # Do NOT pre-resolve device symbols — consumers (e.g. aes_parity.cu)
    # reference kAesT* directly and need them visible at final device link.
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# Host orchestration (no CUDA code, just thin wrappers)
add_library(pos2_gpu_host STATIC
    src/host/GpuPlotter.cpp
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

# CLI: gpu_plotter
add_executable(gpu_plotter tools/gpu_plotter/main.cpp)
target_link_libraries(gpu_plotter PRIVATE pos2_gpu_host pos2_keygen)

# Parity tests
add_executable(aes_parity tools/parity/aes_parity.cu)
target_link_libraries(aes_parity PRIVATE pos2_gpu_host)

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

# Group binaries under build/tools/...
set_target_properties(gpu_plotter PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/gpu_plotter")
foreach(t aes_parity xs_parity xs_bench t1_parity t2_parity t3_parity)
    set_target_properties(${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")
endforeach()

message(STATUS "pos2-gpu configured for CUDA arch(es): ${CMAKE_CUDA_ARCHITECTURES}")
