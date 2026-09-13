#requires -Version 7.3
# Build a Windows SYCL archive with the LLVM-integrated AdaptiveCpp toolchain.
param([string]$BuildDir = 'build/release-windows')
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-Location (Join-Path $PSScriptRoot '..')

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $visualStudio = & $vswhere -latest -version '[17.0,18.0)' -products '*' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $visualStudio) { throw 'Visual Studio 2022 C++ build tools are required' }
    $vcvars = Join-Path $visualStudio 'VC/Auxiliary/Build/vcvars64.bat'
    cmd /c "call `"$vcvars`" >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
    }
}
if (-not $env:CUDA_PATH) { throw 'Set CUDA_PATH to the CUDA 12.9.1 installation' }
if (-not $env:HIP_PATH) { throw 'Set HIP_PATH to the HIP SDK 6.4.2 installation' }
if (-not $env:ACPP_PREFIX) { throw 'Set ACPP_PREFIX to the LLVM-integrated AdaptiveCpp installation' }
$prefix = (Resolve-Path $env:ACPP_PREFIX).Path
$env:PATH = "$prefix/bin;$env:CUDA_PATH/bin;$env:HIP_PATH/bin;" + $env:PATH
$env:ACPP_VISIBILITY_MASK = 'omp'
New-Item -ItemType Directory -Force $BuildDir | Out-Null
$BuildDir = (Resolve-Path $BuildDir).Path
$licenses = Join-Path $BuildDir 'licenses'
$runtime = Join-Path $BuildDir 'runtime'
foreach ($directory in $licenses, $runtime) {
    if (Test-Path $directory) { Remove-Item -Recurse -Force $directory }
    New-Item -ItemType Directory -Force $directory | Out-Null
}
python "$prefix/bin/acpp" "--acpp-deploy=core,cuda,hip:$runtime"
# Integrated LLVM's OpenMP manifest can name an import library instead of its DLL.
Get-ChildItem $runtime -Filter '*.lib' | Remove-Item
Copy-Item "$prefix/bin/libomp.dll" $runtime
Copy-Item "$prefix/adaptivecpp-license.txt" (Join-Path $licenses 'adaptivecpp.txt')
Copy-Item "$prefix/adaptivecpp-windows.txt" $licenses
Copy-Item ci/release/adaptivecpp-third-party.txt, ci/release/llvm-third-party.txt $licenses
Copy-Item "$prefix/llvm-license.txt" (Join-Path $licenses 'llvm.txt')
'9f842c701a599107cc6d117d3539f971036363a1' | Set-Content (Join-Path $licenses 'adaptivecpp-revision.txt')
@"
AdaptiveCpp: 9f842c701a599107cc6d117d3539f971036363a1
LLVM: $(& "$prefix/bin/llvm-config.exe" --version)
AdaptiveCpp Windows modification: see licenses/adaptivecpp-windows.txt
"@ | Set-Content (Join-Path $BuildDir 'runtime-info.txt')
# The minimal Windows installer omits the license; use the matching runtime archive.
$cudart = Join-Path $BuildDir 'cuda-cudart.zip'
Invoke-WebRequest 'https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.9.79-archive.zip' -OutFile $cudart
if ((Get-FileHash $cudart -Algorithm SHA256).Hash -ne '179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8') {
    throw 'CUDA runtime archive checksum mismatch'
}
Expand-Archive $cudart -DestinationPath (Join-Path $BuildDir 'cudart') -Force
Copy-Item (Join-Path $BuildDir 'cudart/cuda_cudart-windows-x86_64-12.9.79-archive/LICENSE') (Join-Path $licenses 'cuda.txt')
Copy-Item (Join-Path $BuildDir 'cudart/cuda_cudart-windows-x86_64-12.9.79-archive/bin/cudart64_12.dll') $runtime
Invoke-WebRequest 'https://raw.githubusercontent.com/NVIDIA/cccl/v2.8.2/LICENSE' `
    -OutFile (Join-Path $licenses 'cuda-cccl.txt')
# AMD's graphics driver supplies amdhip64_6.dll. Bundle the compiler
# libraries and device bitcode so end users do not need the HIP SDK.
foreach ($name in 'hiprtc0604.dll', 'hiprtc-builtins0604.dll', 'amd_comgr0604.dll') {
    Copy-Item "$env:HIP_PATH/bin/$name" $runtime
}
Copy-Item ci/release/amd-runtime.txt $licenses
Invoke-WebRequest 'https://raw.githubusercontent.com/ROCm/HIP/rocm-6.4.2/LICENSE.txt' `
    -OutFile (Join-Path $licenses 'hip-headers.txt')
Invoke-WebRequest 'https://raw.githubusercontent.com/ROCm/clr/123eb5128769ba30fda8f84414421bc858e8c695/LICENSE.txt' `
    -OutFile (Join-Path $licenses 'hiprtc.txt')
Invoke-WebRequest 'https://raw.githubusercontent.com/ROCm/llvm-project/d366fa84f3fdcbd4b10847ebd5db572ae12a34fb/amd/comgr/LICENSE.txt' `
    -OutFile (Join-Path $licenses 'rocm-comgr.txt')
Invoke-WebRequest 'https://raw.githubusercontent.com/ROCm/llvm-project/d366fa84f3fdcbd4b10847ebd5db572ae12a34fb/amd/device-libs/LICENSE.TXT' `
    -OutFile (Join-Path $licenses 'rocm-device-libs.txt')
# AdaptiveCpp 25.10 has no Level Zero deployment component.
Copy-Item "$prefix/bin/hipSYCL/rt-backend-ze.dll" "$runtime/hipSYCL"
Copy-Item "$prefix/lib/hipSYCL/bitcode/libkernel-sscp-spirv-full.bc" "$runtime/hipSYCL/bitcode"
Copy-Item -Recurse "$prefix/bin/hipSYCL/ext/llvm-spirv" "$runtime/hipSYCL/ext"
Copy-Item "$prefix/bin/ze_loader.dll" $runtime
Copy-Item "$prefix/level-zero-license.txt", "$prefix/llvm-spirv-license.txt", "$prefix/spirv-headers-license.txt" $licenses
# Microsoft's documented app-local deployment avoids an installer at first run.
$crt = Join-Path $env:VCToolsRedistDir 'x64/Microsoft.VC143.CRT'
foreach ($directory in $runtime, "$runtime/hipSYCL/ext/llvm/bin", "$runtime/hipSYCL/ext/llvm-spirv/bin") {
    foreach ($name in 'msvcp140.dll', 'msvcp140_atomic_wait.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
        Copy-Item (Join-Path $crt $name) $directory
    }
}
@"
HIP compiler libraries: 6.4.2 (AMD graphics driver supplies HIP runtime 6)
Level Zero loader: 1.33.1
LLVM-SPIRV: f0ae76f12c62ede090e57ece8c986f4c3c971a71
Visual C++ runtime: $((Get-Item "$crt/vcruntime140.dll").VersionInfo.FileVersion)
"@ | Add-Content (Join-Path $BuildDir 'runtime-info.txt')
$rustDocs = Join-Path (rustc --print sysroot) 'share/doc/rust'
$rustLicenses = Join-Path $licenses 'rust-standard-library'
New-Item -ItemType Directory -Force $rustLicenses | Out-Null
Copy-Item (Join-Path $rustDocs 'COPYRIGHT-library.html') $rustLicenses
Copy-Item -Recurse -Force (Join-Path $rustDocs 'licenses') $rustLicenses
cargo about generate --locked --fail --manifest-path keygen-rs/Cargo.toml `
    --target x86_64-pc-windows-msvc --output-file (Join-Path $licenses 'rust.txt') ci/release/licenses.hbs
cmake -S . -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_C_COMPILER=$prefix/bin/clang.exe" "-DCMAKE_CXX_COMPILER=$prefix/bin/clang++.exe" `
    '-DACPP_TARGETS=generic;omp' -DXCHPLOT2_BUILD_CUDA=ON -DXCHPLOT2_PACKAGE_GPU=all `
    "-DXCHPLOT2_RUNTIME_DIR:PATH=$runtime" `
    '-DCMAKE_CUDA_ARCHITECTURES=50-real;52-real;60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120' `
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Static -DXCHPLOT2_PACKAGE=ON "-DXCHPLOT2_LICENSE_DIR:PATH=$licenses"
# Catch host regressions before compiling CUDA for every supported architecture.
$hostTests = 'bench_stats_test', 'numa_topology_test', 'temp_file_test', 'spill_engine_test', `
    'spill_coverage_test', 'host_guard_test', 'host_spill_policy_test', 'vram_budget_test', `
    'cli_host_test', 'solver_filter_parity', 'pipeline_control_test', 'vram_probe_test'
cmake --build $BuildDir --parallel 2 --target $hostTests
ctest --test-dir $BuildDir --output-on-failure --no-tests=error `
    -R ('^(' + ($hostTests -join '|') + ')$')
cmake --build $BuildDir --parallel 2
ctest --test-dir $BuildDir --output-on-failure --no-tests=error -R '^(plot_file_parity|sycl_twophase_budget_test)$'
cpack --config (Join-Path $BuildDir 'CPackConfig.cmake') -B (Join-Path $BuildDir 'dist')
