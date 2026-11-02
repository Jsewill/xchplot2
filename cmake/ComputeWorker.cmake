# Worker reference fanout is scoped per task, and inherited by its children.
# The upstream checkout and ordinary standalone producer keep their behavior.
include_guard(GLOBAL)
function(pos2_worker_prepare upstream_dir output_include)
    set(worker_dir "${CMAKE_CURRENT_BINARY_DIR}/pos2-worker")
    get_filename_component(repo_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    # Relative AES includes need the complete platform-patched header tree,
    # matching the ordinary plotter's ARM overlay.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        file(COPY "${upstream_dir}/src/" DESTINATION "${worker_dir}")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=GIT_DIR --unset=GIT_WORK_TREE
            "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}"
            "${GIT_EXECUTABLE}" apply --ignore-space-change "${repo_dir}/contrib/pos2-arm-cuda.patch"
            WORKING_DIRECTORY "${worker_dir}" COMMAND_ERROR_IS_FATAL ANY)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${repo_dir}/contrib/pos2-arm-cuda.patch")
    endif()
    foreach(header plot/Plotter.hpp plot/PlotLayout.hpp plot/TableConstructorGeneric.hpp plot/RadixSort.hpp
                   common/ParallelForRange.hpp common/thread.hpp)
        configure_file("${upstream_dir}/src/${header}" "${worker_dir}/${header}" COPYONLY)
    endforeach()
    configure_file("${repo_dir}/src/host/ComputeThreads.hpp"
        "${worker_dir}/common/grow-worker-threads.hpp" COPYONLY)
    foreach(patch pos2-plotter-output-move.patch pos2-worker-thread-limit.patch)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=GIT_DIR --unset=GIT_WORK_TREE
            "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}"
            "${GIT_EXECUTABLE}" apply --ignore-space-change
            "${repo_dir}/contrib/${patch}"
            WORKING_DIRECTORY "${worker_dir}" COMMAND_ERROR_IS_FATAL ANY)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${repo_dir}/contrib/${patch}")
    endforeach()
    set(${output_include} "${worker_dir}" PARENT_SCOPE)
endfunction()

# Canonical task compatibility is intentionally independent of GPU build flags,
# target architectures and machine-selected tiers. Both builds hash the same
# generation/codec source content, including the GPU implementation sources.
function(pos2_worker_identity target upstream_dir)
    get_filename_component(repo_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    file(GLOB_RECURSE upstream_sources CONFIGURE_DEPENDS
        "${upstream_dir}/src/*.hpp" "${upstream_dir}/src/*.h"
        "${upstream_dir}/src/*.cpp" "${upstream_dir}/src/*.c"
        "${upstream_dir}/lib/fse/*.h" "${upstream_dir}/lib/fse/*.c")
    file(GLOB_RECURSE gpu_sources CONFIGURE_DEPENDS
        "${repo_dir}/src/*.hpp" "${repo_dir}/src/*.h" "${repo_dir}/src/*.cuh"
        "${repo_dir}/src/*.cpp" "${repo_dir}/src/*.cu" "${repo_dir}/src/*.c")
    set(local_sources
        "${repo_dir}/cmake/ComputeWorker.cmake"
        "${repo_dir}/contrib/pos2-worker-thread-limit.patch"
        "${repo_dir}/contrib/pos2-plotter-output-move.patch"
        "${repo_dir}/contrib/pos2-arm-cuda.patch")
    set(identity "ordinary-group-network-compute-v1\n")
    foreach(source IN LISTS upstream_sources)
        file(SHA256 "${source}" digest)
        file(RELATIVE_PATH name "${upstream_dir}" "${source}")
        string(APPEND identity "upstream/${name}:${digest}\n")
    endforeach()
    foreach(source IN LISTS gpu_sources local_sources)
        file(SHA256 "${source}" digest)
        file(RELATIVE_PATH name "${repo_dir}" "${source}")
        string(APPEND identity "${name}:${digest}\n")
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${upstream_sources} ${gpu_sources} ${local_sources})
    string(SHA256 identity "${identity}")
    target_compile_definitions(${target} PRIVATE "GROW_COMPUTE_COMPATIBILITY=\"${identity}\"")
endfunction()
