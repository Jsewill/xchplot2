# Network compute targets shared by the CPU reference and GPU builds.
include_guard(GLOBAL)

function(pos2_pool_targets upstream_dir grow_include)
    if(TARGET pos2_pr118_grow_network)
        return()
    endif()
    if(NOT CMAKE_SYSTEM_NAME MATCHES "^(Linux|Windows)$")
        message(FATAL_ERROR "Shared compute requires Linux or Windows")
    endif()
    find_package(OpenSSL 3 REQUIRED COMPONENTS SSL Crypto)
    find_package(Threads REQUIRED)
    get_filename_component(repo_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    set(host_dir "${repo_dir}/src/host")
    # DNS-SD for standalone archives, without requiring a separate daemon.
    # Match the pinned FetchContent/source override used for pos2-chip.
    if(NOT DEFINED XCHPLOT2_MDNS_DIR)
        include(FetchContent)
        FetchContent_Declare(xchplot2_mdns
            GIT_REPOSITORY https://github.com/mjansson/mdns
            GIT_TAG a569c4759bd47e0f2a7bfc4d4c19620445782806
            SOURCE_DIR "${repo_dir}/third_party/mdns"
            SOURCE_SUBDIR ".no-cmakelists-here")
        FetchContent_MakeAvailable(xchplot2_mdns)
        set(XCHPLOT2_MDNS_DIR "${xchplot2_mdns_SOURCE_DIR}")
    endif()
    set(XCHPLOT2_MDNS_DIR "${XCHPLOT2_MDNS_DIR}" PARENT_SCOPE)
    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ComputeWorker.cmake")
    pos2_worker_prepare("${upstream_dir}" worker_include)

    add_library(pos2_pr118_grow_network STATIC
        "${repo_dir}/src/host/ComputeNetwork.cpp" "${repo_dir}/src/host/ComputePool.cpp"
        "${repo_dir}/src/host/ComputeIdentity.cpp" "${repo_dir}/src/host/ComputeDiscovery.cpp")
    target_compile_features(pos2_pr118_grow_network PUBLIC cxx_std_20)
    target_include_directories(pos2_pr118_grow_network PUBLIC "${repo_dir}/src")
    target_include_directories(pos2_pr118_grow_network SYSTEM PRIVATE "${XCHPLOT2_MDNS_DIR}")
    target_link_libraries(pos2_pr118_grow_network PUBLIC OpenSSL::SSL OpenSSL::Crypto Threads::Threads)
    if(WIN32)
        target_link_libraries(pos2_pr118_grow_network PUBLIC
            ws2_32 iphlpapi advapi32 crypt32 bcrypt shell32 ole32 uuid gdi32 user32)
    endif()
    add_executable(pos2_pr118_pool "${repo_dir}/tools/xchplot2/compute_broker.cpp")
    target_link_libraries(pos2_pr118_pool PRIVATE pos2_pr118_grow_network)
    add_executable(pos2_pr118_grow_pool_test "${repo_dir}/tools/parity/compute_pool_test.cpp")
    target_link_libraries(pos2_pr118_grow_pool_test PRIVATE pos2_pr118_grow_network)
    add_executable(compute_identity_test "${repo_dir}/tools/parity/compute_identity_test.cpp")
    target_link_libraries(compute_identity_test PRIVATE pos2_pr118_grow_network)
    add_executable(compute_membership_test "${repo_dir}/tools/parity/compute_membership_test.cpp")
    target_link_libraries(compute_membership_test PRIVATE pos2_pr118_grow_network)
    add_executable(compute_discovery_test "${repo_dir}/tools/parity/compute_discovery_test.cpp")
    target_link_libraries(compute_discovery_test PRIVATE pos2_pr118_grow_network)

    add_library(pos2_pr118_grow_worker_host STATIC "${repo_dir}/src/host/ComputeWorker.cpp"
        "${upstream_dir}/src/pos/sha/sha256.c")
    add_executable(pos2_pr118_grow_worker "${repo_dir}/tools/xchplot2/compute_worker.cpp")
    target_link_libraries(pos2_pr118_grow_worker PRIVATE pos2_pr118_grow_worker_host)
    # This test includes worker.cpp to inspect private admission and execution,
    # following the existing construction tests. Do not link it a second time.
    add_executable(pos2_pr118_grow_worker_test "${repo_dir}/tools/parity/compute_worker_test.cpp"
        "${upstream_dir}/src/pos/sha/sha256.c")
    add_executable(pos2_pr118_grow_worker_client_test "${repo_dir}/tools/parity/compute_worker_client_test.cpp"
        "${repo_dir}/src/host/ComputeSession.cpp"
        "${upstream_dir}/src/pos/sha/sha256.c")
    # Keep the independent parity reference on the original upstream headers.
    add_library(pos2_pr118_grow_worker_reference OBJECT "${repo_dir}/tools/parity/compute_worker_reference_test.cpp")
    target_compile_features(pos2_pr118_grow_worker_reference PRIVATE cxx_std_20)
    set(reference_include "${upstream_dir}/src")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(reference_include "${CMAKE_CURRENT_BINARY_DIR}/pos2-worker-reference")
        file(COPY "${upstream_dir}/src/" DESTINATION "${reference_include}")
        execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=GIT_DIR --unset=GIT_WORK_TREE
            "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}"
            "${GIT_EXECUTABLE}" apply --ignore-space-change "${repo_dir}/contrib/pos2-arm-cuda.patch"
            WORKING_DIRECTORY "${reference_include}" COMMAND_ERROR_IS_FATAL ANY)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${repo_dir}/contrib/pos2-arm-cuda.patch")
    endif()
    target_include_directories(pos2_pr118_grow_worker_reference PRIVATE
        "${reference_include}" "${reference_include}/plot")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(pos2_pr118_grow_worker_reference PRIVATE -maes -msse4.1 -mssse3 -msse2)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        target_compile_options(pos2_pr118_grow_worker_reference PRIVATE -march=armv8-a+crypto)
    endif()
    target_sources(pos2_pr118_grow_worker_test PRIVATE $<TARGET_OBJECTS:pos2_pr118_grow_worker_reference>)
    set(multiple_definitions "LINKER:--allow-multiple-definition")
    if(WIN32 AND NOT MINGW)
        set(multiple_definitions "LINKER:/FORCE:MULTIPLE")
    endif()
    target_link_options(pos2_pr118_grow_worker_test PRIVATE ${multiple_definitions})
    foreach(target pos2_pr118_grow_worker_host pos2_pr118_grow_worker_test pos2_pr118_grow_worker_client_test)
        target_compile_features(${target} PRIVATE cxx_std_20)
        target_include_directories(${target} BEFORE PRIVATE "${worker_include}"
            "${upstream_dir}/src" "${upstream_dir}/src/plot" "${repo_dir}/keygen-rs/include")
        target_link_libraries(${target} PUBLIC pos2_pr118_grow_network fse Threads::Threads)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            target_compile_options(${target} PRIVATE -maes -msse4.1 -mssse3 -msse2)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
            target_compile_options(${target} PRIVATE -march=armv8-a+crypto)
        endif()
        pos2_worker_identity(${target} "${upstream_dir}")
    endforeach()
    add_executable(pos2_pr118_grow_pool_codec_test "${repo_dir}/tools/parity/compute_pool_codec_test.cpp")
    add_executable(pos2_pr118_grow_pool_owner_test "${repo_dir}/tools/parity/compute_pool_owner_test.cpp"
        $<TARGET_OBJECTS:pos2_pr118_grow_worker_reference>)
    target_compile_definitions(pos2_pr118_grow_pool_owner_test PRIVATE GROW_TEST_CONSTRUCTION=1 GROW_POOL_NETWORK=1)
    target_include_directories(pos2_pr118_grow_pool_owner_test PRIVATE "${upstream_dir}/src/plot")
    foreach(target pos2_pr118_grow_pool_codec_test pos2_pr118_grow_pool_owner_test)
        target_compile_features(${target} PRIVATE cxx_std_20)
        target_include_directories(${target} BEFORE PRIVATE "${grow_include}")
        target_include_directories(${target} PRIVATE "${upstream_dir}/src" "${repo_dir}/keygen-rs/include")
        target_link_libraries(${target} PRIVATE pos2_pr118_grow_worker_host)
        target_link_options(${target} PRIVATE ${multiple_definitions})
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            target_compile_options(${target} PRIVATE -maes -msse4.1 -mssse3 -msse2)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
            target_compile_options(${target} PRIVATE -march=armv8-a+crypto)
        endif()
    endforeach()
endfunction()
