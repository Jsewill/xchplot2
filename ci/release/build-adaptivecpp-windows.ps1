#requires -Version 7.3
# Run from a VS 2022 developer shell with clang-cl 20.1.8, Ninja and CUDA 12.9.1.
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
cmake -S "$root/llvm/llvm" -B "$root/build" -G Ninja `
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=$prefix" `
    -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl `
    '-DLLVM_TARGETS_TO_BUILD=X86;NVPTX' `
    '-DLLVM_ENABLE_PROJECTS=clang;openmp;lld;compiler-rt' `
    -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_EXTERNAL_PROJECTS=AdaptiveCpp `
    "-DLLVM_EXTERNAL_ADAPTIVECPP_SOURCE_DIR=$acpp" `
    -DLLVM_ADAPTIVECPP_LINK_INTO_TOOLS=ON `
    -DWITH_CUDA_BACKEND=ON -DWITH_ROCM_BACKEND=OFF `
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
is removed. Built by ci/release/build-adaptivecpp-windows.ps1.
"@ | Set-Content "$prefix/adaptivecpp-windows.txt"
