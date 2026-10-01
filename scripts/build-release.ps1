#requires -Version 7.3
# Build the native Windows CUDA archive with Visual Studio and CUDA.
param([string]$BuildDir = 'build/release-windows')
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-Location (Join-Path $PSScriptRoot '..')
$arm64 = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq 'Arm64'
$rustTarget = if ($arm64) { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $component = if ($arm64) { 'Microsoft.VisualStudio.Component.VC.Tools.ARM64' } else { 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' }
    $visualStudio = & $vswhere -latest -products '*' -requires $component -property installationPath
    if (-not $visualStudio) { throw 'Visual Studio C++ build tools are required' }
    $vcvars = Join-Path $visualStudio 'VC/Auxiliary/Build/vcvarsall.bat'
    $target = if ($arm64) { 'arm64' } else { 'x64' }
    cmd /c "call `"$vcvars`" $target >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
    }
}
if (-not $env:CUDA_PATH) { throw 'Set CUDA_PATH to the CUDA installation' }
$env:PATH = (Join-Path $env:CUDA_PATH 'bin') + ';' + $env:PATH
New-Item -ItemType Directory -Force $BuildDir | Out-Null
$BuildDir = (Resolve-Path $BuildDir).Path
$licenses = Join-Path $BuildDir 'licenses'
New-Item -ItemType Directory -Force $licenses | Out-Null
# The minimal Windows installer omits the license; use the matching runtime archive.
if ($arm64) {
    Copy-Item "$env:CUDA_PATH/licenses/cuda_*.txt" $licenses
    Copy-Item "$env:CUDA_PATH/licenses/libnvvm.txt" $licenses
    Copy-Item "$env:CUDA_PATH/licenses/cuda_cudart.txt" (Join-Path $licenses 'cuda.txt')
    Copy-Item "$env:CUDA_PATH/licenses/cccl.txt" (Join-Path $licenses 'cuda-cccl.txt')
} else {
    $cudart = Join-Path $BuildDir 'cuda-cudart.zip'
    Invoke-WebRequest 'https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.9.79-archive.zip' -OutFile $cudart
    if ((Get-FileHash $cudart -Algorithm SHA256).Hash -ne '179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8') {
        throw 'CUDA runtime archive checksum mismatch'
    }
    Expand-Archive $cudart -DestinationPath (Join-Path $BuildDir 'cudart') -Force
    Copy-Item (Join-Path $BuildDir 'cudart/cuda_cudart-windows-x86_64-12.9.79-archive/LICENSE') (Join-Path $licenses 'cuda.txt')
    Invoke-WebRequest 'https://raw.githubusercontent.com/NVIDIA/cccl/v2.8.2/LICENSE' `
        -OutFile (Join-Path $licenses 'cuda-cccl.txt')
}
$rustDocs = Join-Path (rustc --print sysroot) 'share/doc/rust'
$rustLicenses = Join-Path $licenses 'rust-standard-library'
New-Item -ItemType Directory -Force $rustLicenses | Out-Null
Copy-Item (Join-Path $rustDocs 'COPYRIGHT-library.html') $rustLicenses
Copy-Item -Recurse -Force (Join-Path $rustDocs 'licenses') $rustLicenses
cargo about generate --locked --fail --manifest-path keygen-rs/Cargo.toml `
    --target $rustTarget --output-file (Join-Path $licenses 'rust.txt') ci/release/licenses.hbs
$architectures = '50-real;52-real;60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120'
if ($arm64) {
    $supported = @(& nvcc --list-gpu-arch | Where-Object { $_ -match '^compute_[0-9]+$' } |
        ForEach-Object { [int]($_ -replace '^compute_', '') } | Sort-Object)
    if (-not $supported.Count) { throw 'CUDA reported no GPU architectures' }
    $architectures = (@($supported | ForEach-Object { "$_-real" }) + "$($supported[-1])-virtual") -join ';'
}
cmake -S . -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_CUDA_ARCHITECTURES=$architectures" `
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Static -DXCHPLOT2_PACKAGE=ON "-DXCHPLOT2_LICENSE_DIR:PATH=$licenses"
# Catch host regressions before compiling CUDA for every supported architecture.
$hostTests = 'bench_stats_test', 'numa_topology_test', 'temp_file_test', 'spill_engine_test', `
    'spill_coverage_test', 'host_guard_test', 'host_spill_policy_test', 'vram_budget_test', `
    'cli_host_test', 'solver_filter_parity'
cmake --build $BuildDir --parallel 2 --target $hostTests
ctest --test-dir $BuildDir --output-on-failure --no-tests=error `
    -R ('^(' + ($hostTests -join '|') + ')$')
cmake --build $BuildDir --parallel 2
ctest --test-dir $BuildDir --output-on-failure --no-tests=error -R '^plot_file_parity$'
cpack --config (Join-Path $BuildDir 'CPackConfig.cmake') -B (Join-Path $BuildDir 'dist')
