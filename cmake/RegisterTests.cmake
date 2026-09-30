# Keep interactive discovery and CTest on the same executable set.
include(CTest)
get_property(_xchplot2_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(_target IN LISTS _xchplot2_targets)
    get_target_property(_type ${_target} TYPE)
    if(WIN32 AND _type STREQUAL "EXECUTABLE")
        target_sources(${_target} PRIVATE tools/xchplot2/windows.manifest)
        target_link_libraries(${_target} PRIVATE advapi32 bcrypt psapi)
    endif()
    # Plain C++ consumers need the same OpenMP runtime as the SYCL objects.
    # Let AdaptiveCpp choose it; FindOpenMP probes only CMAKE_CXX_COMPILER,
    # which can be GCC even when AdaptiveCpp compiles kernels with Clang.
    if(ACPP_TARGETS MATCHES "(^|;)omp")
        get_target_property(_links ${_target} LINK_LIBRARIES)
        get_target_property(_link_rule ${_target} RULE_LAUNCH_LINK)
        if(_links MATCHES "pos2_gpu" AND NOT _link_rule)
            set_property(TARGET ${_target} PROPERTY RULE_LAUNCH_LINK
                "${ACPP_COMPILER_LAUNCH_RULE}")
        endif()
    endif()
    if(_target MATCHES "(_parity|_test)$")
        set_target_properties(${_target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/parity")
        if(BUILD_TESTING)
            add_test(NAME ${_target} COMMAND $<TARGET_FILE:${_target}>)
        endif()
    endif()
endforeach()
