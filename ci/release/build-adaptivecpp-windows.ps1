#requires -Version 7.3
# Run from a VS 2022 developer shell with clang-cl 20.1.8 and Ninja.
param([ValidateSet('nvidia', 'amd')][string]$Gpu = 'nvidia')
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
$backendOptions = @('-DLLVM_TARGETS_TO_BUILD=X86;NVPTX', '-DWITH_CUDA_BACKEND=ON', '-DWITH_ROCM_BACKEND=OFF')
if ($Gpu -eq 'amd') {
    if (-not $env:HIP_PATH) { throw 'Set HIP_PATH to the HIP SDK 6.4.2 installation' }
    $hip = (Resolve-Path $env:HIP_PATH).Path.Replace('\', '/')
    $backendOptions = @('-DLLVM_TARGETS_TO_BUILD=X86;AMDGPU', '-DWITH_CUDA_BACKEND=OFF', '-DWITH_ROCM_BACKEND=ON',
        "-DROCM_PATH:PATH=$hip", "-DHIPRTC_LIBRARY:FILEPATH=$hip/lib/hiprtc.lib")
    # Backport the upstream Windows device-IR fixes without changing the pinned release.
    if (-not (Get-Content "$acpp/src/compiler/sscp/TargetSeparationPass.cpp" -Raw).Contains('removeLinkerOptionsByPrefixes')) {
        git -C $acpp apply "$PWD/contrib/adaptivecpp-windows-hip.patch"
    }
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
    @backendOptions `
    '-DLLVM_ENABLE_PROJECTS=clang;openmp;lld;compiler-rt' `
    -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_EXTERNAL_PROJECTS=AdaptiveCpp `
    "-DLLVM_EXTERNAL_ADAPTIVECPP_SOURCE_DIR=$acpp" `
    -DLLVM_ADAPTIVECPP_LINK_INTO_TOOLS=ON `
    -DHIPSYCL_COMMON_LIBRARY_OUTPUT_NAME=acpp-common `
    -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF `
    -DLLVM_TOOL_BUGPOINT_BUILD=OFF -DOPENMP_ENABLE_LIBOMPTARGET=OFF `
    -DLLVM_INCLUDE_TESTS=OFF
cmake --build "$root/build" --target install --parallel 2
Copy-Item "$acpp/LICENSE" "$prefix/adaptivecpp-license.txt"
Copy-Item "$root/llvm/llvm/LICENSE.TXT" "$prefix/llvm-license.txt"
@"
AdaptiveCpp: $revision
Windows modification: src/common/dylib_loader.cpp uses the C++ standard
library to format system errors; the third-party format_win32_error helper
is removed. The build sets the common DLL name for relative runtime discovery.
Built by ci/release/build-adaptivecpp-windows.ps1.
"@ | Set-Content "$prefix/adaptivecpp-windows.txt"
if ($Gpu -eq 'amd') {
    @'
Windows HIP backports: contrib/adaptivecpp-windows-hip.patch
4d0fedff89df49bf30457f7dda15764cfaac5fbd (remove host wchar_size)
c69d230e497dd5cafcf9a7004f2c0aa3647acce4 (Windows AMDGPU short wchar)
256589708f6902c47bb8f80bb199012d0a32a321 (remove MSVC linker directives)
'@ | Add-Content "$prefix/adaptivecpp-windows.txt"
}
