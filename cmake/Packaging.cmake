if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    message(FATAL_ERROR "Binary packaging currently supports Linux x86_64")
endif()
if(NOT XCHPLOT2_PACKAGE_GPU MATCHES "^(nvidia|amd|intel)$")
    message(FATAL_ERROR "XCHPLOT2_PACKAGE_GPU must be nvidia, amd, or intel")
endif()
if(NOT EXISTS "${XCHPLOT2_RUNTIME_DIR}/libacpp-rt.so")
    message(FATAL_ERROR "Collect the AdaptiveCpp runtime first: scripts/build-release.sh")
endif()
if(NOT EXISTS "${XCHPLOT2_LICENSE_DIR}/rust.txt")
    message(FATAL_ERROR "Generate the release licenses first: scripts/build-release.sh")
endif()
if(DEFINED ENV{RELEASE_TAG} AND NOT "$ENV{RELEASE_TAG}" STREQUAL ""
        AND NOT "$ENV{RELEASE_TAG}" STREQUAL "v${PROJECT_VERSION}")
    message(FATAL_ERROR "Release tag must match v${PROJECT_VERSION}")
endif()

execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE _release_revision OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${GIT_EXECUTABLE}" diff --quiet HEAD --
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" RESULT_VARIABLE _release_dirty)
if(NOT _release_dirty EQUAL 0)
    string(APPEND _release_revision "-dirty")
endif()
execute_process(COMMAND rustc --version
    OUTPUT_VARIABLE _release_rust OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
file(READ "${CMAKE_BINARY_DIR}/runtime-info.txt" _release_runtime)
set(_release_cuda "")
if(XCHPLOT2_BUILD_CUDA)
    set(_release_cuda "CUDA: ${CMAKE_CUDA_COMPILER_VERSION}\nCUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}\n")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/BUILDINFO.txt"
    "xchplot2 ${PROJECT_VERSION}\n"
    "Backend: SYCL/AdaptiveCpp (${XCHPLOT2_PACKAGE_GPU})\n"
    "Source: ${_release_revision}\n"
    "pos2-chip: ${POS2_CHIP_GIT_TAG} (contrib/pos2-solver-candidates.patch applied)\n"
    "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}\n"
    "C++: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\n"
    "SYCL targets: ${ACPP_TARGETS}\n"
    "${_release_cuda}"
    "Rust: ${_release_rust}\n"
    "${_release_runtime}")

set_target_properties(xchplot2 PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib" INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS xchplot2 RUNTIME DESTINATION bin)
install(DIRECTORY "${XCHPLOT2_RUNTIME_DIR}/" DESTINATION lib USE_SOURCE_PERMISSIONS)
install(FILES "${CMAKE_BINARY_DIR}/BUILDINFO.txt" ci/release/README.txt
    DESTINATION .)
install(FILES LICENSE DESTINATION licenses)
install(DIRECTORY "${XCHPLOT2_LICENSE_DIR}/" DESTINATION licenses)
install(FILES "${POS2_CHIP_DIR}/LICENSE" DESTINATION licenses RENAME pos2-chip.txt)
install(FILES "${POS2_CHIP_DIR}/lib/fse/LICENSE" DESTINATION licenses RENAME fse.txt)
file(READ "${POS2_CHIP_DIR}/src/pos/aes/soft_aes.hpp" _aes_source)
string(REGEX MATCH "^/\\*([^*]|\\*+[^*/])*\\*/" _aes_license "${_aes_source}")
if(NOT _aes_license)
    message(FATAL_ERROR "Could not extract the pos2-chip AES license")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/aes-license.txt" "${_aes_license}\n")
install(FILES "${CMAKE_BINARY_DIR}/aes-license.txt" DESTINATION licenses RENAME aes.txt)
set(CPACK_GENERATOR TGZ)
set(CPACK_PACKAGE_NAME xchplot2)
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "xchplot2-${PROJECT_VERSION}-linux-x86_64-sycl-${XCHPLOT2_PACKAGE_GPU}")
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_STRIP_FILES ON)
set(CPACK_SOURCE_GENERATOR "")
include(CPack)
