# Refresh source provenance at build time, including incremental builds after
# a commit. Write the header only when its contents change.
if(CMAKE_SCRIPT_MODE_FILE)
    cmake_minimum_required(VERSION 3.24)
    set(revision "unknown source")
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse --show-toplevel
            OUTPUT_VARIABLE root OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        file(REAL_PATH "${SOURCE_DIR}" source_root)
        if(root STREQUAL source_root)
            execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse --short=12 HEAD
                OUTPUT_VARIABLE commit OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
                RESULT_VARIABLE result)
            if(result EQUAL 0)
                set(revision "${commit}")
                execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" diff --quiet HEAD --
                    RESULT_VARIABLE dirty ERROR_QUIET)
                if(dirty EQUAL 1)
                    string(APPEND revision "-dirty")
                endif()
            endif()
        endif()
    endif()
    file(CONFIGURE OUTPUT "${OUTPUT}" CONTENT
        "#pragma once\n#define XCHPLOT2_VERSION \"@VERSION@\"\n#define XCHPLOT2_REVISION \"@revision@\"\n" @ONLY)
else()
    set(_xchplot2_version_dir "${CMAKE_CURRENT_BINARY_DIR}/xchplot2-version")
    add_custom_target(xchplot2_version
        COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DVERSION=${PROJECT_VERSION}" "-DOUTPUT=${_xchplot2_version_dir}/BuildVersion.hpp"
            -P "${CMAKE_CURRENT_LIST_FILE}"
        BYPRODUCTS "${_xchplot2_version_dir}/BuildVersion.hpp"
        VERBATIM)
    function(xchplot2_target_version target)
        add_dependencies(${target} xchplot2_version)
        target_include_directories(${target} PRIVATE "${_xchplot2_version_dir}")
    endfunction()
endif()
