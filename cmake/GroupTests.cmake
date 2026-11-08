# Construction fixtures shared by the installed build and independent reference.
include_guard(GLOBAL)
function(pos2_group_tests upstream_dir grow_include)
    get_filename_component(repo_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    # Crash fixtures use fork/waitpid; native Windows exercises the installed CLI
    # plus the portable file, scratch and admission fixtures below.
    if(NOT WIN32)
        add_executable(pos2_pr118_grow_codec_test "${repo_dir}/tools/parity/group_codec_test.cpp"
            "${upstream_dir}/src/pos/sha/sha256.c")
        add_executable(pos2_pr118_grow_staging_test "${repo_dir}/tools/parity/group_staging_test.cpp"
            "${upstream_dir}/src/pos/sha/sha256.c")
        add_executable(pos2_pr118_grow_external_test "${repo_dir}/tools/parity/group_external_test.cpp"
            "${upstream_dir}/src/pos/sha/sha256.c")
        add_executable(pos2_pr118_grow_merge_test "${repo_dir}/tools/parity/group_merge_test.cpp"
            "${upstream_dir}/src/pos/sha/sha256.c")
        target_compile_definitions(pos2_pr118_grow_merge_test PRIVATE GROW_TEST_CONSTRUCTION=1)
        target_include_directories(pos2_pr118_grow_merge_test PRIVATE "${upstream_dir}/src/plot")
        # Independent binaries exercise real interrupted jobs across different
        # source builds while retaining the same declared construction contract.
        foreach(variant fixture compatibility_test)
            add_executable(pos2_pr118_grow_checkpoint_${variant}
                "${repo_dir}/tools/parity/group_checkpoint_compatibility_test.cpp"
                "${upstream_dir}/src/pos/sha/sha256.c")
        endforeach()
        foreach(variant fixture recovery)
            add_executable(pos2_pr118_stream_checkpoint_${variant}
                "${repo_dir}/contrib/pos2-pr118/stream.cpp" "${upstream_dir}/src/pos/sha/sha256.c")
        endforeach()
        target_compile_definitions(pos2_pr118_grow_checkpoint_fixture PRIVATE
            GROW_IDENTITY="checkpoint-source-a" STREAM_IDENTITY="checkpoint-source-a")
        target_compile_definitions(pos2_pr118_grow_checkpoint_compatibility_test PRIVATE
            GROW_IDENTITY="checkpoint-source-b" STREAM_IDENTITY="checkpoint-source-b")
        target_compile_definitions(pos2_pr118_stream_checkpoint_fixture PRIVATE STREAM_IDENTITY="checkpoint-source-a")
        target_compile_definitions(pos2_pr118_stream_checkpoint_recovery PRIVATE STREAM_IDENTITY="checkpoint-source-b")
        add_dependencies(pos2_pr118_grow_checkpoint_compatibility_test pos2_pr118_grow_checkpoint_fixture
            pos2_pr118_stream_checkpoint_fixture pos2_pr118_stream_checkpoint_recovery)
    endif()
    add_executable(pos2_pr118_grow_scratch_test "${repo_dir}/tools/parity/group_scratch_test.cpp"
        "${upstream_dir}/src/pos/sha/sha256.c")
    add_executable(pos2_pr118_grow_inventory_test "${repo_dir}/tools/parity/group_inventory_test.cpp"
        "${upstream_dir}/src/pos/sha/sha256.c")
    add_executable(pos2_pr118_grow_retirement_test "${repo_dir}/tools/parity/group_retirement_test.cpp"
        "${upstream_dir}/src/pos/sha/sha256.c")
    add_executable(pos2_pr118_grow_replot_resources_test "${repo_dir}/tools/parity/group_replot_resources_test.cpp"
        "${upstream_dir}/src/pos/sha/sha256.c")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_definitions(pos2_pr118_grow_external_test PRIVATE GROW_TEST_WRAP_PWRITE=1)
        target_link_options(pos2_pr118_grow_external_test PRIVATE "-Wl,--wrap=pwrite" "-Wl,--wrap=pread"
            "-Wl,--wrap=POS2_FSE_compress_usingCTable" "-Wl,--wrap=fsetxattr" "-Wl,--wrap=renameat2" "-Wl,--wrap=fsync")
        target_compile_definitions(pos2_pr118_grow_merge_test PRIVATE GROW_TEST_WRAP_PWRITE=1)
        target_link_options(pos2_pr118_grow_merge_test PRIVATE "-Wl,--wrap=pwrite" "-Wl,--wrap=pread" "-Wl,--wrap=unlink" "-Wl,--wrap=lstat" "-Wl,--wrap=renameat2" "-Wl,--wrap=fsetxattr")
    endif()
    foreach(target pos2_pr118_grow_codec_test pos2_pr118_grow_staging_test pos2_pr118_grow_merge_test pos2_pr118_grow_external_test pos2_pr118_grow_scratch_test pos2_pr118_grow_inventory_test pos2_pr118_grow_retirement_test pos2_pr118_grow_replot_resources_test
                   pos2_pr118_grow_checkpoint_fixture pos2_pr118_grow_checkpoint_compatibility_test
                   pos2_pr118_stream_checkpoint_fixture pos2_pr118_stream_checkpoint_recovery)
        if(NOT TARGET ${target})
            continue()
        endif()
        target_compile_features(${target} PRIVATE cxx_std_20)
        target_include_directories(${target} BEFORE PRIVATE "${grow_include}")
        target_include_directories(${target} PRIVATE "${upstream_dir}/src")
        target_link_libraries(${target} PRIVATE fse Threads::Threads)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
            target_compile_options(${target} PRIVATE -maes -msse4.1 -mssse3 -msse2)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
            target_compile_options(${target} PRIVATE -march=armv8-a+crypto)
        endif()
    endforeach()
    add_executable(pos2_pr118_grow_gpu_output_test "${repo_dir}/tools/parity/group_gpu_output_test.cpp")
    target_compile_features(pos2_pr118_grow_gpu_output_test PRIVATE cxx_std_20)
    target_include_directories(pos2_pr118_grow_gpu_output_test PRIVATE "${repo_dir}/src")
    target_link_libraries(pos2_pr118_grow_gpu_output_test PRIVATE Threads::Threads)
    add_executable(pos2_pr118_grow_farming_test "${repo_dir}/tools/parity/group_farming_test.cpp")
    target_compile_features(pos2_pr118_grow_farming_test PRIVATE cxx_std_20)
    if(TARGET pos2_pr118_grow_merge_test)
        target_sources(pos2_pr118_grow_merge_test PRIVATE "${repo_dir}/tools/xchplot2/group_options.cpp")
    endif()
    foreach(target pos2_pr118_grow_codec_test pos2_pr118_grow_staging_test
                   pos2_pr118_grow_external_test pos2_pr118_grow_scratch_test pos2_pr118_grow_inventory_test pos2_pr118_grow_retirement_test pos2_pr118_grow_replot_resources_test
                   pos2_pr118_grow_merge_test pos2_pr118_grow_gpu_output_test pos2_pr118_grow_farming_test
                   pos2_pr118_grow_checkpoint_fixture pos2_pr118_grow_checkpoint_compatibility_test
                   pos2_pr118_stream_checkpoint_fixture pos2_pr118_stream_checkpoint_recovery)
        if(NOT TARGET ${target})
            continue()
        endif()
        if(WIN32)
            target_link_libraries(${target} PRIVATE advapi32 shell32 ole32 uuid)
        endif()
        target_include_directories(${target} PRIVATE "${repo_dir}/src" "${repo_dir}/keygen-rs/include")
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")
    endforeach()
endfunction()
