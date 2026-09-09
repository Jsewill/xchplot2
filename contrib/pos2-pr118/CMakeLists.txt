cmake_minimum_required(VERSION 3.20)
project(pos2_pr118_compatibility LANGUAGES C CXX)

# Separate build: the plotter continues using its production dependency pin.
if(NOT DEFINED POS2_PR118_DIR)
    include(FetchContent)
    FetchContent_Declare(pos2_pr118
        GIT_REPOSITORY https://github.com/Chia-Network/pos2-chip
        GIT_TAG 2d737fa40dd579431c48fcf4b584b59365ad1791
        SOURCE_SUBDIR .no-cmakelists-here)
    FetchContent_MakeAvailable(pos2_pr118)
    set(POS2_PR118_DIR "${pos2_pr118_SOURCE_DIR}")
endif()

find_package(Git REQUIRED)
find_package(Threads REQUIRED)
add_subdirectory("${POS2_PR118_DIR}/lib/fse" fse EXCLUDE_FROM_ALL)

# Exercise the same solver correctness fix as the production build.
foreach(header Solver.hpp ParallelRadixSort.hpp ProofSolverTimings.hpp)
    configure_file("${POS2_PR118_DIR}/src/solve/${header}"
        "${CMAKE_CURRENT_BINARY_DIR}/pos2-solver/solve/${header}" COPYONLY)
endforeach()
set(solver_patch "${CMAKE_CURRENT_SOURCE_DIR}/../pos2-solver-candidates.patch")
execute_process(COMMAND "${GIT_EXECUTABLE}" apply --ignore-space-change "${solver_patch}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pos2-solver"
    COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${solver_patch}")

add_executable(pos2_pr118_check reference.cpp "${POS2_PR118_DIR}/src/pos/sha/sha256.c")
target_compile_features(pos2_pr118_check PRIVATE cxx_std_20)
target_include_directories(pos2_pr118_check PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/pos2-solver" "${POS2_PR118_DIR}/src")
target_link_libraries(pos2_pr118_check PRIVATE fse Threads::Threads)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
    target_compile_options(pos2_pr118_check PRIVATE -maes -msse4.1 -mssse3 -msse2)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(pos2_pr118_check PRIVATE -march=armv8-a+crypto)
endif()
