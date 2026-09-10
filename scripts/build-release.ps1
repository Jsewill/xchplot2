#requires -Version 7.3
# Build the native Windows CUDA archive with Visual Studio 2022 and CUDA 12.9.1.
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
$env:PATH = (Join-Path $env:CUDA_PATH 'bin') + ';' + $env:PATH
New-Item -ItemType Directory -Force $BuildDir | Out-Null
$BuildDir = (Resolve-Path $BuildDir).Path
$licenses = Join-Path $BuildDir 'licenses'
New-Item -ItemType Directory -Force $licenses | Out-Null
Copy-Item (Join-Path $env:CUDA_PATH 'EULA.txt') (Join-Path $licenses 'cuda.txt')
Invoke-WebRequest 'https://raw.githubusercontent.com/NVIDIA/cccl/v2.8.2/LICENSE' `
    -OutFile (Join-Path $licenses 'cuda-cccl.txt')
$rustDocs = Join-Path (rustc --print sysroot) 'share/doc/rust'
$rustLicenses = Join-Path $licenses 'rust-standard-library'
New-Item -ItemType Directory -Force $rustLicenses | Out-Null
Copy-Item (Join-Path $rustDocs 'COPYRIGHT-library.html') $rustLicenses
Copy-Item -Recurse -Force (Join-Path $rustDocs 'licenses') $rustLicenses
cargo about generate --locked --fail --manifest-path keygen-rs/Cargo.toml `
    --target x86_64-pc-windows-msvc --output-file (Join-Path $licenses 'rust.txt') ci/release/licenses.hbs
cmake -S . -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release `
    '-DCMAKE_CUDA_ARCHITECTURES=50-real;52-real;60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120' `
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Static -DXCHPLOT2_PACKAGE=ON "-DXCHPLOT2_LICENSE_DIR=$licenses"
cmake --build $BuildDir --parallel 2
ctest --test-dir $BuildDir --output-on-failure --no-tests=error `
    -R '^(bench_stats_test|numa_topology_test|temp_file_test|spill_engine_test|spill_coverage_test|host_guard_test|host_spill_policy_test|vram_budget_test|cli_host_test|plot_file_parity|solver_filter_parity)$'
cpack --config (Join-Path $BuildDir 'CPackConfig.cmake') -B (Join-Path $BuildDir 'dist')
