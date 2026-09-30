# CUDA compatibility policy shared by Cargo and standalone CMake.
# Default arch: `native` — CMake 3.24+ probes the locally-visible
# GPUs and emits SASS only for those compute capabilities. Falls
# back to sm_89 (RTX 4090) when no GPU is visible at configure
# time (e.g., container builds, headless CI). Override either via
# -DCMAKE_CUDA_ARCHITECTURES=89;86 (multi-arch fatbin) or a
# specific value when cross-compiling for hardware that's not
# plugged into the build host.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Probe whether `native` resolves to anything; if no GPU is
    # visible CMake will error out cryptically. Fall back to sm_89
    # in that case.
    execute_process(
        COMMAND nvidia-smi -L
        OUTPUT_VARIABLE _xchplot2_nvsmi_out
        ERROR_QUIET
        RESULT_VARIABLE _xchplot2_nvsmi_rc)
    if(_xchplot2_nvsmi_rc EQUAL 0 AND _xchplot2_nvsmi_out MATCHES "GPU")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
endif()

# Preflight nvcc-vs-arch compatibility BEFORE enable_language(CUDA),
# which is what triggers the cryptic "Unsupported gpu architecture
# 'compute_61'" TryCompile failure when Pascal/Volta meets CUDA 13.x.
# CUDA 13.0 dropped codegen for sm_50/52/53/60/61/62/70/72 entirely.
# Skip the check if nvcc isn't findable yet — enable_language(CUDA)
# below will surface its own missing-toolchain message in that case.
if(CMAKE_CUDA_COMPILER)
    set(_xchplot2_nvcc "${CMAKE_CUDA_COMPILER}")
else()
    find_program(_xchplot2_nvcc nvcc
        HINTS "${CUDAToolkit_ROOT}/bin" "$ENV{CUDAToolkit_ROOT}/bin"
              "$ENV{CUDA_PATH}/bin" "$ENV{CUDA_HOME}/bin" /opt/cuda/bin /usr/local/cuda/bin)
endif()
if(_xchplot2_nvcc)
    execute_process(
        COMMAND "${_xchplot2_nvcc}" --version
        OUTPUT_VARIABLE _nvcc_version_out
        RESULT_VARIABLE _nvcc_version_rc
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    # Parse "Cuda compilation tools, release 13.0, V13.0.48" → 13
    if(_nvcc_version_rc EQUAL 0 AND _nvcc_version_out MATCHES "release ([0-9]+)")
        set(_nvcc_major "${CMAKE_MATCH_1}")
        set(_min_arch 9999)
        foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
            # Strip sm_ / compute_ prefixes some users pass through
            string(REGEX REPLACE "^(sm_|compute_)" "" _a "${_a}")
            string(REGEX REPLACE "-(real|virtual)$" "" _a "${_a}")
            if(_a MATCHES "^[0-9]+$" AND _a LESS _min_arch)
                set(_min_arch ${_a})
            endif()
        endforeach()
        if(_nvcc_major GREATER_EQUAL 13 AND _min_arch LESS 75)
            # Container detection: Docker writes /.dockerenv, Podman writes
            # /run/.containerenv. Either presence means the host-side fixes
            # don't apply — the user needs to rebuild the image with a
            # different BASE_DEVEL.
            if(EXISTS "/.dockerenv" OR EXISTS "/run/.containerenv")
                set(_fix_block
                    "You're building inside a container — the toolkit comes from\n"
                    "the base image, not the host. Rebuild with a CUDA 12.x base:\n"
                    "  - Recommended: rerun scripts/build-container.sh on the host;\n"
                    "    it auto-pins nvidia/cuda:12.9.1 when CUDA_ARCH < 75.\n"
                    "  - Or pass --build-arg explicitly:\n"
                    "      podman build -t xchplot2:cuda \\\n"
                    "        --build-arg BASE_DEVEL=docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 \\\n"
                    "        --build-arg BASE_RUNTIME=docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 \\\n"
                    "        --build-arg CUDA_ARCH=${_min_arch} \\\n"
                    "        .\n")
            else()
                set(_fix_block
                    "Fix one of:\n"
                    "  - Install CUDA 12.9 (last toolkit with Pascal/Volta support) and re-run cmake:\n"
                    "      sudo apt install cuda-toolkit-12-9     (Ubuntu/Debian)\n"
                    "    Then point cmake at it:\n"
                    "      cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -B build -S . [...]\n"
                    "  - Or override the target arch (only valid if you actually have a Turing+ card):\n"
                    "      cmake -DCMAKE_CUDA_ARCHITECTURES=75 -B build -S . [...]\n"
                    "  - Or use the container path — scripts/build-container.sh auto-pins\n"
                    "    the 12.9 base image when it detects a pre-Turing GPU.\n")
            endif()
            string(APPEND _fix_block
                "Cargo equivalents (use the same install source/options as before):\n"
                "  CUDA_PATH=/usr/local/cuda-12.9 cargo install --path . --locked\n"
                "  CUDA_ARCHITECTURES=75 cargo install --path . --locked\n"
                "For cargo install --git, keep --git in place of --path .\n")
            message(FATAL_ERROR
                "xchplot2: CUDA Toolkit ${_nvcc_major}.x dropped codegen for "
                "sm_${_min_arch} (Pascal / Volta / pre-Turing).\n"
                "\n"
                "Detected:\n"
                "  nvcc ${_nvcc_major}.x at ${_xchplot2_nvcc}\n"
                "  target arch: sm_${_min_arch} (from CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})\n"
                "\n"
                ${_fix_block})
        endif()
    endif()
endif()
# Symmetric ceiling check: a target arch NEWER than this nvcc can
# codegen — e.g. a Blackwell GeForce RTX 50-series (compute_cap 12.0 ->
# sm_120) on a CUDA < 12.8 toolkit, where nvcc dies with "Unsupported
# gpu architecture 'compute_120'". PTX is forward-compatible, so fall
# back to PTX from the highest arch this nvcc supports (queried
# authoritatively via `nvcc --list-gpu-arch`); the driver JIT-compiles
# it to the real GPU at runtime. Applied to the Cargo install path too.
if(_xchplot2_nvcc)
    execute_process(
        COMMAND "${_xchplot2_nvcc}" --list-gpu-arch
        OUTPUT_VARIABLE _nvcc_arch_out
        RESULT_VARIABLE _nvcc_arch_rc
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _nvcc_arch_rc EQUAL 0)
        set(_nvcc_arch_out "")
    endif()
    # Highest compute_XX nvcc can emit. The list is NOT sorted on CUDA
    # 13.x (compute_100, compute_110, compute_103, ...), so take the max.
    set(_max_supported 0)
    string(REGEX MATCHALL "compute_([0-9]+)" _arches "${_nvcc_arch_out}")
    foreach(_m IN LISTS _arches)
        string(REGEX REPLACE "compute_" "" _m "${_m}")
        if(_m GREATER _max_supported)
            set(_max_supported ${_m})
        endif()
    endforeach()
    # Target arch: resolve `native`/`all` via the live GPU's compute_cap;
    # otherwise take the max of the numeric CMAKE_CUDA_ARCHITECTURES list.
    set(_want 0)
    if(CMAKE_CUDA_ARCHITECTURES MATCHES "native|all")
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
            OUTPUT_VARIABLE _cap_out
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_cap_out MATCHES "([0-9]+)\\.([0-9]+)")
            math(EXPR _want "${CMAKE_MATCH_1} * 10 + ${CMAKE_MATCH_2}")
        endif()
    else()
        foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "^(sm_|compute_)" "" _a "${_a}")
            string(REGEX REPLACE "-(real|virtual)$" "" _a "${_a}")
            if(_a MATCHES "^[0-9]+$" AND _a GREATER _want)
                set(_want ${_a})
            endif()
        endforeach()
    endif()
    if(_max_supported GREATER 0 AND _want GREATER _max_supported)
        message(WARNING
            "xchplot2: target sm_${_want} is newer than this CUDA Toolkit can "
            "codegen (nvcc tops out at compute_${_max_supported}; sm_100/sm_120 "
            "Blackwell need CUDA 12.8+). Falling back to compute_${_max_supported} "
            "PTX, which the driver JIT-compiles to your GPU at runtime. For native "
            "SASS install CUDA 12.8+ and point cmake at it with "
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc "
            "(Cargo: CUDA_PATH=/usr/local/cuda-12.8 cargo install with your usual source/options).")
        # Retain supported SASS entries in a requested fat binary.
        set(_kept)
        foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "^(sm_|compute_)" "" _n "${_a}")
            string(REGEX REPLACE "-(real|virtual)$" "" _n "${_n}")
            if(_n MATCHES "^[0-9]+$" AND _n LESS_EQUAL _max_supported)
                list(APPEND _kept "${_a}")
            endif()
        endforeach()
        list(APPEND _kept "${_max_supported}-virtual")
        list(REMOVE_DUPLICATES _kept)
        set(CMAKE_CUDA_ARCHITECTURES "${_kept}")
    endif()
endif()
# Hand enable_language(CUDA) the nvcc the find_program above located.
# CMake's built-in toolkit search covers CUDAToolkit_ROOT / CUDA_PATH /
# $PATH / /usr/local/cuda — but NOT /opt/cuda, where Arch's `cuda` package
# installs it, and which only reaches PATH via /etc/profile.d/cuda.sh (a
# login-shell-only hook). Without this, configure dies with "Failed to find
# nvcc. Please set the CUDAToolkit_ROOT variable" on a host whose toolkit we
# found, ran, and version-checked seconds earlier.
if(_xchplot2_nvcc AND NOT DEFINED CMAKE_CUDA_COMPILER)
    get_filename_component(_xchplot2_nvcc_bin  "${_xchplot2_nvcc}" DIRECTORY)
    get_filename_component(_xchplot2_nvcc_root "${_xchplot2_nvcc_bin}" DIRECTORY)
    set(CMAKE_CUDA_COMPILER "${_xchplot2_nvcc}"
        CACHE FILEPATH "nvcc located by xchplot2")
    if(NOT DEFINED CUDAToolkit_ROOT)
        set(CUDAToolkit_ROOT "${_xchplot2_nvcc_root}"
            CACHE PATH "CUDA toolkit root located by xchplot2")
    endif()
    message(STATUS "xchplot2: nvcc ${_xchplot2_nvcc} "
                   "(CUDAToolkit_ROOT=${_xchplot2_nvcc_root})")
endif()
unset(_xchplot2_nvcc CACHE)
