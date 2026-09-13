#requires -Version 7.3
# Run from a VS 2022 developer shell with clang-cl 20.1.8 and Ninja.
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-Location (Join-Path $PSScriptRoot '../..')
$root = Join-Path $PWD 'build/windows-toolchain'
$prefix = Join-Path $root 'install'
$acpp = Join-Path $root 'acpp'
if (-not (Test-Path "$root/llvm")) {
    git clone --depth 1 --branch llvmorg-20.1.8 https://github.com/llvm/llvm-project.git "$root/llvm"
}
if (-not (Test-Path $acpp)) {
    git clone --depth 1 --branch v25.10.0 https://github.com/AdaptiveCpp/AdaptiveCpp.git $acpp
}
$revision = git -C $acpp rev-parse HEAD
if ($revision -ne '9f842c701a599107cc6d117d3539f971036363a1') {
    throw 'AdaptiveCpp source does not match the release notices'
}
if (-not $env:CUDA_PATH) { throw 'Set CUDA_PATH to the CUDA 12.9.1 installation' }
if (-not $env:HIP_PATH) { throw 'Set HIP_PATH to the HIP SDK 6.4.2 installation' }
$hip = (Resolve-Path $env:HIP_PATH).Path.Replace('\', '/')
# Backport the upstream Windows device-IR fixes without changing the pinned release.
if (-not (Get-Content "$acpp/src/compiler/sscp/TargetSeparationPass.cpp" -Raw).Contains('removeLinkerOptionsByPrefixes')) {
    git -C $acpp apply "$PWD/contrib/adaptivecpp-windows-hip.patch"
}
if (-not (Get-Content "$acpp/src/runtime/CMakeLists.txt" -Raw).Contains('LEVEL_ZERO_LOADER')) {
    git -C $acpp apply "$PWD/contrib/adaptivecpp-windows-level-zero.patch"
}
$ze = Join-Path $root 'level-zero'
if (-not (Test-Path $ze)) {
    git clone --depth 1 --branch v1.33.1 https://github.com/oneapi-src/level-zero.git $ze
}
if ((git -C $ze rev-parse HEAD) -ne '5c863340cab6631a31234653191b904f0028b93b') {
    throw 'Level Zero source does not match the release notices'
}
cmake -S $ze -B "$root/level-zero-build" -G Ninja -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl `
    "-DCMAKE_INSTALL_PREFIX=$root/level-zero-install" -DBUILD_L0_LOADER_TESTS=OFF
cmake --build "$root/level-zero-build" --target install --parallel 2
$spirv = Join-Path $root 'llvm-spirv'
if (-not (Test-Path $spirv)) {
    git clone --depth 1 --branch llvm_release_200 https://github.com/AdaptiveCpp/SPIRV-LLVM-Translator.git $spirv
}
if ((git -C $spirv rev-parse HEAD) -ne 'f0ae76f12c62ede090e57ece8c986f4c3c971a71') {
    throw 'LLVM-SPIRV source does not match the release notices'
}
# Remove the separately licensed Stack Overflow formatter; the standard
# library already formats Win32 system errors. Keep the upstream BSD notice.
$loader = Join-Path $acpp 'src/common/dylib_loader.cpp'
$source = Get-Content $loader -Raw
if ($source.Contains('format_win32_error')) {
    $helper = '(?s)// Adapted from: https://stackoverflow.com/a/17387176.*?(?=#endif)'
    if ([regex]::Matches($source, $helper).Count -ne 1 -or
        [regex]::Matches($source, 'format_win32_error\(errorCode\)').Count -ne 2) {
        throw 'AdaptiveCpp Windows error formatter changed; review the replacement'
    }
    $source = [regex]::Replace($source, $helper, "#include <system_error>`n")
    $source.Replace('format_win32_error(errorCode)', 'std::system_category().message(errorCode)') |
        Set-Content $loader -NoNewline
}
# 25.10 does not propagate the common DLL name to its generated config on Windows.
# Set it explicitly so runtime discovery follows the DLL after deployment.
cmake -S "$root/llvm/llvm" -B "$root/build" -G Ninja `
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=$prefix" `
    -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl `
    '-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU' `
    '-DLLVM_ENABLE_PROJECTS=clang;openmp;lld;compiler-rt' `
    -DLLVM_PARALLEL_LINK_JOBS=1 '-DLLVM_EXTERNAL_PROJECTS=SPIRVTranslator;AdaptiveCpp' `
    "-DLLVM_EXTERNAL_SPIRVTRANSLATOR_SOURCE_DIR=$spirv" `
    "-DLLVM_EXTERNAL_ADAPTIVECPP_SOURCE_DIR=$acpp" `
    -DLLVM_ADAPTIVECPP_LINK_INTO_TOOLS=ON `
    -DHIPSYCL_COMMON_LIBRARY_OUTPUT_NAME=acpp-common `
    -DWITH_CUDA_BACKEND=ON -DWITH_ROCM_BACKEND=ON -DWITH_LEVEL_ZERO_BACKEND=ON `
    -DWITH_OPENCL_BACKEND=OFF "-DROCM_PATH:PATH=$hip" "-DHIPRTC_LIBRARY:FILEPATH=$hip/lib/hiprtc.lib" `
    "-DCMAKE_PREFIX_PATH=$root/level-zero-install" `
    -DLLVM_TOOL_BUGPOINT_BUILD=OFF -DOPENMP_ENABLE_LIBOMPTARGET=OFF `
    -DLLVM_INCLUDE_TESTS=OFF
cmake --build "$root/build" --target install --parallel 2
New-Item -ItemType Directory -Force "$prefix/bin/hipSYCL/ext/llvm-spirv/bin" | Out-Null
Copy-Item "$prefix/bin/llvm-spirv.exe" "$prefix/bin/hipSYCL/ext/llvm-spirv/bin"
Copy-Item "$root/level-zero-install/bin/ze_loader.dll" "$prefix/bin"
Copy-Item "$ze/LICENSE" "$prefix/level-zero-license.txt"
Copy-Item "$spirv/LICENSE.TXT" "$prefix/llvm-spirv-license.txt"
Copy-Item "$root/build/tools/SPIRVTranslator/SPIRV-Headers/LICENSE" "$prefix/spirv-headers-license.txt"
Copy-Item "$acpp/LICENSE" "$prefix/adaptivecpp-license.txt"
Copy-Item "$root/llvm/llvm/LICENSE.TXT" "$prefix/llvm-license.txt"
@"
AdaptiveCpp: $revision
Windows modification: src/common/dylib_loader.cpp uses the C++ standard
library to format system errors; the third-party format_win32_error helper
is removed. The build sets the common DLL name for relative runtime discovery.
Built by ci/release/build-adaptivecpp-windows.ps1.
"@ | Set-Content "$prefix/adaptivecpp-windows.txt"
@'
Windows HIP backports: contrib/adaptivecpp-windows-hip.patch
4d0fedff89df49bf30457f7dda15764cfaac5fbd (remove host wchar_size)
c69d230e497dd5cafcf9a7004f2c0aa3647acce4 (Windows AMDGPU short wchar)
256589708f6902c47bb8f80bb199012d0a32a321 (remove MSVC linker directives)
Windows Level Zero integration: contrib/adaptivecpp-windows-level-zero.patch
Level Zero: 1.33.1 (5c863340cab6631a31234653191b904f0028b93b)
LLVM-SPIRV: f0ae76f12c62ede090e57ece8c986f4c3c971a71
'@ | Add-Content "$prefix/adaptivecpp-windows.txt"
