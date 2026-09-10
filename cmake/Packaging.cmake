if(NOT CMAKE_SYSTEM_NAME MATCHES "^(Linux|Windows)$" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    message(FATAL_ERROR "Binary packaging supports Linux and Windows x86_64")
endif()
if(NOT EXISTS "${XCHPLOT2_LICENSE_DIR}/rust.txt")
    message(FATAL_ERROR "Generate the release licenses first with scripts/build-release.sh or scripts/build-release.ps1")
endif()
if(DEFINED ENV{RELEASE_TAG} AND NOT "$ENV{RELEASE_TAG}" STREQUAL ""
        AND NOT "$ENV{RELEASE_TAG}" STREQUAL "v${PROJECT_VERSION}-cuda-only")
    message(FATAL_ERROR "Release tag must match v${PROJECT_VERSION}-cuda-only")
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
file(WRITE "${CMAKE_BINARY_DIR}/BUILDINFO.txt"
    "xchplot2 ${PROJECT_VERSION}\n"
    "Backend: native CUDA\n"
    "Source: ${_release_revision}\n"
    "pos2-chip: ${POS2_CHIP_GIT_TAG} (contrib/pos2-solver-candidates.patch applied)\n"
    "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}\n"
    "C++: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\n"
    "CUDA: ${CMAKE_CUDA_COMPILER_VERSION}\n"
    "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}\n"
    "Rust: ${_release_rust}\n")

install(TARGETS xchplot2 RUNTIME DESTINATION bin)
install(FILES "${CMAKE_BINARY_DIR}/BUILDINFO.txt" DESTINATION .)
if(WIN32)
    install(FILES ci/release/README.windows.txt DESTINATION . RENAME README.txt)
    install(FILES ci/release/microsoft-runtime.txt DESTINATION licenses)
else()
    install(FILES ci/release/README.txt DESTINATION .)
endif()
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
if(WIN32)
    set(CPACK_GENERATOR ZIP)
    set(_release_system windows)
else()
    set(CPACK_GENERATOR TGZ)
    set(_release_system linux)
endif()
set(CPACK_PACKAGE_NAME xchplot2)
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "xchplot2-${PROJECT_VERSION}-${_release_system}-x86_64-cuda")
set(CPACK_PACKAGE_CHECKSUM SHA256)
if(NOT MSVC)
    set(CPACK_STRIP_FILES ON)
endif()
set(CPACK_SOURCE_GENERATOR "")
include(CPack)
