# Check the compiler used for the dependency build before fetching or building
# AdaptiveCpp. The installed-package path does not need these build tools.
function(xchplot2_check_adaptivecpp_dependencies llvm_root llvm_version)
    set(probe "${CMAKE_BINARY_DIR}/CMakeFiles/xchplot2-prerequisites")
    if(XCHPLOT2_PREREQUISITES_DIR)
        set(probe "${XCHPLOT2_PREREQUISITES_DIR}")
    endif()
    file(MAKE_DIRECTORY "${probe}")
    file(WRITE "${probe}/cxx.cpp"
        "#include <span>\n#include <string>\nint main(int n, char** p) { std::span<char*> args(p, n); return std::string(args[0]).empty(); }\n")
    file(WRITE "${probe}/openmp.cpp"
        "#include <omp.h>\nint main() { return omp_get_max_threads() < 1; }\n")
    file(WRITE "${probe}/level_zero.cpp"
        "#include <level_zero/ze_api.h>\nint main() { return zeInit(0); }\n")
    separate_arguments(compiler_flags NATIVE_COMMAND "$ENV{CXXFLAGS}")
    separate_arguments(linker_flags NATIVE_COMMAND "$ENV{LDFLAGS}")
    set(missing)
    set(tests cxx openmp)
    set(level_zero_available OFF)
    if(WITH_LEVEL_ZERO_BACKEND OR (NOT DEFINED WITH_LEVEL_ZERO_BACKEND AND ACPP_TARGETS MATCHES "(^|;)generic(:|;|$)"))
        list(APPEND tests level_zero)
    endif()
    foreach(test IN LISTS tests)
        set(flags ${compiler_flags} -std=c++20)
        set(libraries)
        if(test STREQUAL openmp)
            list(APPEND flags -fopenmp)
        elseif(test STREQUAL level_zero)
            list(APPEND libraries -lze_loader)
        endif()
        execute_process(COMMAND "${llvm_root}/bin/clang++" ${flags}
            "${probe}/${test}.cpp" -o "${probe}/${test}" ${linker_flags} ${libraries}
            RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
        file(WRITE "${probe}/${test}.log" "${output}${error}")
        if(test STREQUAL level_zero)
            if(result EQUAL 0)
                set(level_zero_available ON)
            endif()
        elseif(NOT result EQUAL 0)
            if(test STREQUAL cxx)
                list(APPEND missing "C++20 headers and libraries for ${llvm_root}/bin/clang++ (Arch: base-devel, Debian/Ubuntu: build-essential, Fedora: gcc-c++)")
            else()
                list(APPEND missing "LLVM OpenMP headers and runtime (Arch: openmp, Debian/Ubuntu: libomp-${llvm_version}-dev, Fedora: libomp${llvm_version}-devel)")
            endif()
        endif()
    endforeach()

    if(NOT DEFINED WITH_LEVEL_ZERO_BACKEND)
        # A loader may come from unrelated software. It is not a request to
        # build Intel support or to install another GPU's development stack.
        if(ACPP_TARGETS MATCHES "(^|;)generic(:|;|$)" AND
           (level_zero_available OR (X2_HAVE_INTEL AND NOT X2_HAVE_NVIDIA AND NOT X2_HAVE_AMD)))
            set(WITH_LEVEL_ZERO_BACKEND ON)
        else()
            set(WITH_LEVEL_ZERO_BACKEND OFF)
            if(ACPP_TARGETS MATCHES "(^|;)generic(:|;|$)")
                message(STATUS "xchplot2: optional Level Zero backend disabled (loader and headers are not both usable)")
            endif()
        endif()
    endif()
    if(WITH_LEVEL_ZERO_BACKEND AND NOT level_zero_available)
        list(APPEND missing "Level Zero loader and headers for the Intel backend (Arch: level-zero-loader level-zero-headers, Debian/Ubuntu: libze-dev, Fedora: oneapi-level-zero-devel)")
    endif()
    set(WITH_LEVEL_ZERO_BACKEND "${WITH_LEVEL_ZERO_BACKEND}" PARENT_SCOPE)
    if(missing)
        list(JOIN missing "\n  - " description)
        message(FATAL_ERROR
            "xchplot2: build prerequisites missing:\n  - ${description}\n"
            "Compiler diagnostics: ${probe}/*.log\n"
            "Install the native build dependencies with:\n"
            "  bash \"${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../scripts/install-deps.sh\"\n"
            "Then repeat the original Cargo or CMake command.")
    endif()
endfunction()

if(CMAKE_SCRIPT_MODE_FILE AND XCHPLOT2_LLVM_ROOT)
    xchplot2_check_adaptivecpp_dependencies("${XCHPLOT2_LLVM_ROOT}" "${XCHPLOT2_LLVM_VERSION}")
endif()
