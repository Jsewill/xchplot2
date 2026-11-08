# Shared CPU ownership patch and source provenance for both grow builds.
# Construction and farming compatibility use their explicit saved identities;
# legacy checkpoints and plans retain their original exact-build guards.
include_guard(GLOBAL)

function(pos2_grow_prepare upstream_dir output_include)
    set(grow_dir "${CMAKE_CURRENT_BINARY_DIR}/pos2-grow")
    get_filename_component(repo_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    set(patch "${repo_dir}/contrib/pos2-plotter-output-move.patch")
    # Relative AES includes need the complete platform-patched header tree,
    # matching the ordinary plotter's ARM overlay.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        file(COPY "${upstream_dir}/src/" DESTINATION "${grow_dir}")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=GIT_DIR --unset=GIT_WORK_TREE
            "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}"
            "${GIT_EXECUTABLE}" apply --ignore-space-change "${repo_dir}/contrib/pos2-arm-cuda.patch"
            WORKING_DIRECTORY "${grow_dir}" COMMAND_ERROR_IS_FATAL ANY)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${repo_dir}/contrib/pos2-arm-cuda.patch")
    endif()
    configure_file("${upstream_dir}/src/plot/Plotter.hpp"
        "${grow_dir}/plot/Plotter.hpp" COPYONLY)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=GIT_DIR --unset=GIT_WORK_TREE
        "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}"
        "${GIT_EXECUTABLE}" apply --ignore-space-change "${patch}"
        WORKING_DIRECTORY "${grow_dir}" COMMAND_ERROR_IS_FATAL ANY)
    file(STRINGS "${grow_dir}/plot/Plotter.hpp" move_marker
        REGEX "^#define GROW_MOVE_PLOTTER_OUTPUT 1$")
    if(NOT move_marker)
        message(FATAL_ERROR "CPU output ownership patch was not applied; grow RAM model would be invalid")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${patch}")
    set(${output_include} "${grow_dir}" PARENT_SCOPE)
endfunction()

function(pos2_grow_identity target upstream_dir)
    cmake_parse_arguments(grow "GPU;NETWORK" "" "" ${ARGN})
    get_filename_component(repo_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    set(host_dir "${repo_dir}/src/host")
    # Only source roots are globbed. Build-local patched copies cannot enter
    # discovery recursively when a build directory lives inside the checkout.
    file(GLOB_RECURSE upstream_sources CONFIGURE_DEPENDS
        "${upstream_dir}/src/*.hpp" "${upstream_dir}/src/*.h"
        "${upstream_dir}/src/*.c" "${upstream_dir}/src/*.cpp"
        "${upstream_dir}/lib/fse/*.c" "${upstream_dir}/lib/fse/*.h")
    list(APPEND upstream_sources "${upstream_dir}/lib/fse/CMakeLists.txt")
    set(identity "ordinary-group-grow\n")
    foreach(source IN LISTS upstream_sources)
        file(SHA256 "${source}" digest)
        file(RELATIVE_PATH name "${upstream_dir}" "${source}")
        string(APPEND identity "upstream/${name}:${digest}\n")
    endforeach()
    set(local_sources)
    foreach(name GroupPlotter.cpp GroupPlotter.hpp GroupOptions.hpp GroupEngine.hpp GroupCodec.hpp GroupMembers.hpp GroupPacked.hpp
                 GroupStaging.hpp GroupStorage.hpp GroupMergeState.hpp GroupMerge.hpp GroupExternal.hpp
                 GroupScratch.hpp GroupWriter.hpp GroupPolicy.hpp GroupFarming.hpp GroupFarmState.hpp
                 GroupFarmPreview.hpp GroupFarmUpdate.hpp GroupFarmProgress.hpp GroupFarmControlState.hpp GroupFarmControl.hpp GroupFarmOwner.hpp GroupFarmMembership.hpp GroupFarmMetadata.hpp GroupFarmMigration.hpp GroupFarmRetirement.hpp GroupFarmFile.hpp GroupFarmInventory.hpp GroupFarmResources.hpp GroupFarmReplotResources.hpp GroupFarmAdapters.hpp GroupFarmAdapterProcess.hpp GroupHarvesterProfile.hpp GroupCalibrationProcess.hpp
                 GroupCheckpoint.hpp GroupConstructionIdentity.hpp GroupFormat.hpp GroupFile.hpp GroupNativeIdentity.hpp WindowsFile.hpp Random.hpp Json.hpp Utf8.hpp)
        list(APPEND local_sources "${host_dir}/${name}")
    endforeach()
    list(APPEND local_sources "${repo_dir}/tools/xchplot2/group_options.cpp"
        "${repo_dir}/tools/xchplot2/group_commands.cpp" "${repo_dir}/tools/xchplot2/group_commands.hpp"
        "${repo_dir}/tools/xchplot2/device_options.hpp"
        "${repo_dir}/contrib/pos2-plotter-output-move.patch"
        "${repo_dir}/cmake/GroupPlotter.cmake")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        list(APPEND local_sources "${repo_dir}/contrib/pos2-arm-cuda.patch")
    endif()
    if(grow_NETWORK OR XCHPLOT2_BUILD_GROUP_POOL)
        foreach(name ComputeNetwork.hpp ComputeNetwork.cpp ComputePool.hpp ComputePool.cpp
                     ComputeWorker.hpp ComputeWorker.cpp ComputeCompatibility.cpp ComputeThreads.hpp NumaTopology.hpp NumaTopology.cpp GroupRemoteCodec.hpp GroupFarmComputeResources.hpp)
            list(APPEND local_sources "${host_dir}/${name}")
        endforeach()
        list(APPEND local_sources "${repo_dir}/cmake/ComputeWorker.cmake"
            "${repo_dir}/contrib/pos2-worker-thread-limit.patch" "${repo_dir}/cmake/ComputePool.cmake")
        string(APPEND identity "network-compute:1\n")
    endif()
    file(GLOB_RECURSE keygen_sources CONFIGURE_DEPENDS "${repo_dir}/keygen-rs/src/*.rs")
    list(APPEND local_sources ${keygen_sources}
        "${repo_dir}/keygen-rs/Cargo.toml" "${repo_dir}/keygen-rs/Cargo.lock"
        "${repo_dir}/keygen-rs/include/pos2_keygen.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
    if(grow_GPU)
        list(APPEND local_sources "${host_dir}/GroupGpu.hpp" "${host_dir}/GroupGpu.cpp"
            "${host_dir}/GroupGpuPacked.hpp")
        file(GLOB_RECURSE gpu_sources CONFIGURE_DEPENDS
            "${repo_dir}/src/*.hpp" "${repo_dir}/src/*.h" "${repo_dir}/src/*.cuh"
            "${repo_dir}/src/*.cpp" "${repo_dir}/src/*.cu" "${repo_dir}/src/*.c"
            "${repo_dir}/cmake/*.cmake")
        list(APPEND local_sources ${gpu_sources}
            "${repo_dir}/contrib/pos2-solver-candidates.patch"
            "${repo_dir}/contrib/pos2-solver-thread-limit.patch")
        if(XCHPLOT2_BUILD_CUDA)
            list(APPEND local_sources "${repo_dir}/contrib/cccl-arch-list.patch")
        endif()
        string(APPEND identity
            "gpu:${ACPP_TARGETS}:${XCHPLOT2_BUILD_CUDA}:${CMAKE_CUDA_ARCHITECTURES}:${XCHPLOT2_INSTRUMENT_MATCH}\n")
    endif()
    foreach(source IN LISTS local_sources)
        file(SHA256 "${source}" digest)
        file(RELATIVE_PATH name "${repo_dir}" "${source}")
        string(APPEND identity "${name}:${digest}\n")
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${upstream_sources} ${local_sources})
    string(SHA256 identity "${identity}")
    target_compile_definitions(${target} PRIVATE "GROW_IDENTITY=\"${identity}\"")
endfunction()
