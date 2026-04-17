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

# CLI: gpu_plotter
add_executable(gpu_plotter tools/gpu_plotter/main.cpp)
target_link_libraries(gpu_plotter PRIVATE pos2_gpu_host)

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
