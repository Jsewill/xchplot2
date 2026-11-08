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

get_filename_component(XCHPLOT2_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
include_directories("${XCHPLOT2_SOURCE_DIR}/src")

find_package(Git REQUIRED)
find_package(Threads REQUIRED)
include("${XCHPLOT2_SOURCE_DIR}/cmake/GroupPlotter.cmake")
option(XCHPLOT2_BUILD_GROUP_POOL "Build shared compute reference services and tests" OFF)
add_subdirectory("${POS2_PR118_DIR}/lib/fse" fse EXCLUDE_FROM_ALL)

# Exercise the same solver correctness fix as the production build.
foreach(header Solver.hpp ParallelRadixSort.hpp ProofSolverTimings.hpp)
    configure_file("${POS2_PR118_DIR}/src/solve/${header}"
        "${CMAKE_CURRENT_BINARY_DIR}/pos2-solver/solve/${header}" COPYONLY)
endforeach()
foreach(solver_patch_name pos2-solver-candidates.patch pos2-solver-thread-limit.patch)
    set(solver_patch "${CMAKE_CURRENT_SOURCE_DIR}/../${solver_patch_name}")
    # Keep Git from discovering the enclosing source repository and silently
    # skipping paths in these standalone copies when the build is inside it.
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=GIT_DIR --unset=GIT_WORK_TREE
        "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}"
        "${GIT_EXECUTABLE}" apply --ignore-space-change "${solver_patch}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pos2-solver"
        COMMAND_ERROR_IS_FATAL ANY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${solver_patch}")
endforeach()

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

# External-assembly prototype: deliberately use the unmodified upstream reader
# and format headers, without the experimental member-index extension.
if(UNIX)
    add_executable(pos2_pr118_stream stream.cpp "${POS2_PR118_DIR}/src/pos/sha/sha256.c")
    target_compile_features(pos2_pr118_stream PRIVATE cxx_std_20)
    target_include_directories(pos2_pr118_stream PRIVATE "${POS2_PR118_DIR}/src")
    target_link_libraries(pos2_pr118_stream PRIVATE fse Threads::Threads)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
        target_compile_options(pos2_pr118_stream PRIVATE -maes -msse4.1 -mssse3 -msse2)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(pos2_pr118_stream PRIVATE -march=armv8-a+crypto)
    endif()

    # Same-file construction uses the ordinary upstream v2 codec and reader.
    # Transfer the CPU plotter's final vector instead of retaining a duplicate
    # through return. This leaves room for a resident batch on smaller hosts.
    pos2_grow_prepare("${POS2_PR118_DIR}" grow_include)
    add_executable(pos2_pr118_grow "${XCHPLOT2_SOURCE_DIR}/src/host/GroupPlotter.cpp"
        "${XCHPLOT2_SOURCE_DIR}/tools/xchplot2/group_options.cpp" "${XCHPLOT2_SOURCE_DIR}/tools/xchplot2/group_reference.cpp" "${POS2_PR118_DIR}/src/pos/sha/sha256.c")
    target_compile_features(pos2_pr118_grow PRIVATE cxx_std_20)
    target_include_directories(pos2_pr118_grow PRIVATE
        "${grow_include}" "${POS2_PR118_DIR}/src"
        "${POS2_PR118_DIR}/src/plot")
    target_link_libraries(pos2_pr118_grow PRIVATE fse Threads::Threads)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
        target_compile_options(pos2_pr118_grow PRIVATE -maes -msse4.1 -mssse3 -msse2)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(pos2_pr118_grow PRIVATE -march=armv8-a+crypto)
    endif()

    include("${XCHPLOT2_SOURCE_DIR}/cmake/GroupTests.cmake")
    pos2_group_tests("${POS2_PR118_DIR}" "${grow_include}")
endif()

# Network workers are optional; ordinary reference builds need no TLS library.
if(XCHPLOT2_BUILD_GROUP_POOL)
    include("${XCHPLOT2_SOURCE_DIR}/cmake/ComputePool.cmake")
    pos2_pool_targets("${POS2_PR118_DIR}" "${grow_include}")
    target_compile_definitions(pos2_pr118_grow PRIVATE GROW_POOL_NETWORK=1)
    target_link_libraries(pos2_pr118_grow PRIVATE pos2_pr118_grow_worker_host)
    # Unchanged upstream soft_aesenc/soft_aesdec are defined by both reference
    # translation units, as in the production host executable.
    target_link_options(pos2_pr118_grow PRIVATE "LINKER:--allow-multiple-definition")
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
if(TARGET pos2_pr118_grow)
    foreach(target pos2_pr118_grow pos2_pr118_grow_codec_test pos2_pr118_grow_staging_test pos2_pr118_grow_merge_test pos2_pr118_grow_external_test pos2_pr118_grow_scratch_test pos2_pr118_grow_inventory_test
                   pos2_pr118_grow_checkpoint_fixture pos2_pr118_grow_checkpoint_compatibility_test)
        add_dependencies(${target} pos2_pr118_keygen)
        target_include_directories(${target} PRIVATE "${keygen_dir}/include")
        target_link_libraries(${target} PRIVATE "${keygen_lib}" ${CMAKE_DL_LIBS} m)
    endforeach()
endif()
if(TARGET pos2_pr118_grow_pool_owner_test)
    add_dependencies(pos2_pr118_grow_pool_owner_test pos2_pr118_keygen)
    target_link_libraries(pos2_pr118_grow_pool_owner_test PRIVATE "${keygen_lib}" ${CMAKE_DL_LIBS} m)
endif()
add_executable(pos2_pr118_group group.cpp "${POS2_PR118_DIR}/src/pos/sha/sha256.c")
# Only the grouped-job tool reads the experimental persisted member-index range.
# Keep pos2_pr118_check on the unmodified upstream file format and reader.
foreach(header plot/PlotFile.hpp plot/ChunkCompression.hpp plot/PlotData.hpp plot/PlotIO.hpp prove/Prover.hpp)
    configure_file("${POS2_PR118_DIR}/src/${header}"
        "${CMAKE_CURRENT_BINARY_DIR}/pos2-group/${header}" COPYONLY)
endforeach()
set(group_patch "${CMAKE_CURRENT_SOURCE_DIR}/plot-index-base.patch")
execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=GIT_DIR --unset=GIT_WORK_TREE
    "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}"
    "${GIT_EXECUTABLE}" apply --ignore-space-change "${group_patch}"
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
foreach(solver_patch_name pos2-solver-candidates.patch pos2-solver-thread-limit.patch)
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/../${solver_patch_name}" solver_patch_hash)
    string(APPEND upstream_identity_text "${solver_patch_name}:${solver_patch_hash}\n")
endforeach()
file(SHA256 "${group_patch}" group_patch_hash)
string(APPEND upstream_identity_text "plot-index-base.patch:${group_patch_hash}\n")
string(SHA256 upstream_content_id "${upstream_identity_text}")
if(TARGET pos2_pr118_stream)
    set(stream_identity "${upstream_identity_text}")
    foreach(source "${CMAKE_CURRENT_SOURCE_DIR}/stream.cpp"
                   "${XCHPLOT2_SOURCE_DIR}/src/host/GroupCheckpoint.hpp"
                   "${XCHPLOT2_SOURCE_DIR}/src/host/GroupConstructionIdentity.hpp"
                   "${XCHPLOT2_SOURCE_DIR}/src/host/GroupFormat.hpp")
        file(SHA256 "${source}" source_hash)
        file(RELATIVE_PATH source_name "${XCHPLOT2_SOURCE_DIR}" "${source}")
        string(APPEND stream_identity "${source_name}:${source_hash}\n")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
    endforeach()
    string(SHA256 stream_identity "${stream_identity}")
    target_compile_definitions(pos2_pr118_stream PRIVATE "STREAM_IDENTITY=\"${stream_identity}\"")
endif()
if(TARGET pos2_pr118_grow)
    pos2_grow_identity(pos2_pr118_grow "${POS2_PR118_DIR}")
endif()
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

include("${XCHPLOT2_SOURCE_DIR}/cmake/RegisterTests.cmake")
