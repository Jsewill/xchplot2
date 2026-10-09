cmake_minimum_required(VERSION 3.20)
project(pos2_pr118_compatibility LANGUAGES C CXX)

# Separate interoperability/assembly build, pinned to the production PoS2 version.
if(NOT DEFINED POS2_PR118_DIR)
    include(FetchContent)
    FetchContent_Declare(pos2_pr118
        GIT_REPOSITORY https://github.com/Chia-Network/pos2-chip
        GIT_TAG ff89d3e61f4315311a7f975409a6433e67f699ff
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

# Reuse the production Rust shim; no draft chia-rs dependency is introduced.
set(keygen_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../keygen-rs")
set(keygen_target "${CMAKE_CURRENT_BINARY_DIR}/keygen-rs-target")
set(keygen_lib "${keygen_target}/release/${CMAKE_STATIC_LIBRARY_PREFIX}pos2_keygen${CMAKE_STATIC_LIBRARY_SUFFIX}")
find_program(CARGO_EXECUTABLE cargo HINTS "$ENV{HOME}/.cargo/bin" REQUIRED)
add_custom_command(OUTPUT "${keygen_lib}"
    COMMAND "${CMAKE_COMMAND}" -E env "CARGO_TARGET_DIR=${keygen_target}"
        "${CARGO_EXECUTABLE}" build --release --locked --manifest-path "${keygen_dir}/Cargo.toml"
    DEPENDS "${keygen_dir}/src/lib.rs" "${keygen_dir}/Cargo.toml" "${keygen_dir}/Cargo.lock"
    VERBATIM)
add_custom_target(pos2_pr118_keygen DEPENDS "${keygen_lib}")
add_executable(pos2_pr118_group group.cpp "${POS2_PR118_DIR}/src/pos/sha/sha256.c")
# Only the grouped-job tool reads the experimental persisted member-index range.
# Keep pos2_pr118_check on the unmodified upstream file format and reader.
foreach(header plot/PlotFile.hpp plot/ChunkCompression.hpp plot/PlotData.hpp plot/PlotIO.hpp prove/Prover.hpp)
    configure_file("${POS2_PR118_DIR}/src/${header}"
        "${CMAKE_CURRENT_BINARY_DIR}/pos2-group/${header}" COPYONLY)
endforeach()
set(group_patch "${CMAKE_CURRENT_SOURCE_DIR}/plot-index-base.patch")
execute_process(COMMAND "${GIT_EXECUTABLE}" apply --ignore-space-change "${group_patch}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pos2-group"
    COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${group_patch}")
# Record consumed source content as well as HEAD, including local edits and
# source archives without Git metadata. A resume must use the same reference.
file(GLOB_RECURSE upstream_identity_sources CONFIGURE_DEPENDS
    "${POS2_PR118_DIR}/src/*.hpp" "${POS2_PR118_DIR}/src/pos/sha/*.c"
    "${POS2_PR118_DIR}/src/*.h" "${POS2_PR118_DIR}/lib/fse/*.c"
    "${POS2_PR118_DIR}/lib/fse/*.h")
set(upstream_identity_text "")
foreach(source IN LISTS upstream_identity_sources)
    file(SHA256 "${source}" source_hash)
    file(RELATIVE_PATH source_name "${POS2_PR118_DIR}" "${source}")
    string(APPEND upstream_identity_text "${source_name}:${source_hash}\n")
endforeach()
file(SHA256 "${solver_patch}" solver_patch_hash)
string(APPEND upstream_identity_text "pos2-solver-candidates.patch:${solver_patch_hash}\n")
file(SHA256 "${group_patch}" group_patch_hash)
string(APPEND upstream_identity_text "plot-index-base.patch:${group_patch_hash}\n")
string(SHA256 upstream_content_id "${upstream_identity_text}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${upstream_identity_sources})
set(upstream_git_result 1)
if(EXISTS "${POS2_PR118_DIR}/.git")
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${POS2_PR118_DIR}" rev-parse HEAD
        RESULT_VARIABLE upstream_git_result OUTPUT_VARIABLE upstream_revision
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
set(upstream_identity "source-sha256:${upstream_content_id}")
if(upstream_git_result EQUAL 0)
    set(upstream_identity "${upstream_revision}+${upstream_identity}")
endif()
target_compile_definitions(pos2_pr118_group PRIVATE "POS2_PR118_IDENTITY=\"${upstream_identity}\"")
add_dependencies(pos2_pr118_group pos2_pr118_keygen)
target_compile_features(pos2_pr118_group PRIVATE cxx_std_20)
target_include_directories(pos2_pr118_group PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/pos2-group" "${CMAKE_CURRENT_BINARY_DIR}/pos2-solver"
    "${POS2_PR118_DIR}/src" "${keygen_dir}/include")
target_link_libraries(pos2_pr118_group PRIVATE "${keygen_lib}" fse Threads::Threads ${CMAKE_DL_LIBS})
if(WIN32)
    target_link_libraries(pos2_pr118_group PRIVATE ws2_32 userenv bcrypt ntdll advapi32)
elseif(UNIX)
    target_link_libraries(pos2_pr118_group PRIVATE m)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
    target_compile_options(pos2_pr118_group PRIVATE -maes -msse4.1 -mssse3 -msse2)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(pos2_pr118_group PRIVATE -march=armv8-a+crypto)
endif()
