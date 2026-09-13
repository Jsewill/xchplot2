#requires -Version 5.1
# Run from the extracted Windows release. Installers display their own UI/license.
param([string]$HipPath = "$env:ProgramFiles\AMD\ROCm\6.4")
$ErrorActionPreference = 'Stop'
if (-not [Environment]::Is64BitProcess) { throw 'Run this x64 release helper in 64-bit PowerShell' }
$bin = Join-Path $PSScriptRoot 'bin'
$amd = Test-Path (Join-Path $bin 'hipSYCL/rt-backend-hip.dll')
$nvidia = Test-Path (Join-Path $bin 'hipSYCL/rt-backend-cuda.dll')
if (-not ($amd -or $nvidia)) { throw 'Keep this script beside the extracted bin directory' }
$gpus = @(Get-CimInstance Win32_VideoController)
$gpus | ForEach-Object { Write-Host "GPU: $($_.Name); driver: $($_.DriverVersion)" }
if ($nvidia -and ($gpus.PNPDeviceID -match 'VEN_1002') -and -not ($gpus.PNPDeviceID -match 'VEN_10DE')) {
    throw 'This is the NVIDIA archive. Download the sycl-amd ZIP for your AMD GPU; installing CUDA will not help.'
}
if ($amd -and ($gpus.PNPDeviceID -match 'VEN_10DE') -and -not ($gpus.PNPDeviceID -match 'VEN_1002')) {
    throw 'This is the AMD archive. Download the sycl-nvidia ZIP for your NVIDIA GPU.'
}

function Install-Dependency([string]$Url, [string]$Name, [string]$Sha256) {
    $directory = Join-Path ([IO.Path]::GetTempPath()) ('xchplot2-' + [guid]::NewGuid())
    New-Item -ItemType Directory $directory | Out-Null
    try {
        $installer = Join-Path $directory "$Name.exe"
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        Write-Host "Downloading $Name from $Url"
        Invoke-WebRequest -UseBasicParsing $Url -OutFile $installer
        if ($Sha256) {
            if ((Get-FileHash $installer -Algorithm SHA256).Hash -ne $Sha256) { throw "$Name checksum mismatch" }
        } else {
            $signature = Get-AuthenticodeSignature $installer
            if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation,') {
                throw 'The Visual C++ installer does not have a valid Microsoft signature'
            }
        }
        $process = Start-Process $installer -Wait -PassThru
        if ($process.ExitCode -notin 0, 3010, 1638) { throw "$Name installer exited with $($process.ExitCode)" }
        if ($process.ExitCode -eq 3010) { Write-Host 'The installer requests a Windows restart.' }
    } finally {
        Remove-Item -Recurse -Force $directory
    }
}

$missingCrt = @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_atomic_wait.dll' |
    Where-Object { -not (Test-Path (Join-Path "$env:SystemRoot/System32" $_)) })
if ($missingCrt) {
    Install-Dependency 'https://aka.ms/vs/17/release/vc_redist.x64.exe' 'Visual C++ x64 Runtime' ''
}
if ($amd) {
    $dlls = 'amdhip64_6.dll', 'hiprtc0604.dll', 'hiprtc-builtins0604.dll', 'amd_comgr0604.dll', 'amd_comgr_2.dll'
    $bitcode = Join-Path $bin 'hipSYCL/ext/bitcode/amdgcn'
    $missingRuntime = @($dlls | Where-Object { -not (Test-Path (Join-Path $bin $_)) })
    if ($missingRuntime -or -not (Test-Path (Join-Path $bitcode 'ockl.bc'))) {
        if (-not (Test-Path (Join-Path $HipPath 'bin/hiprtc0604.dll'))) {
            Write-Host 'Install HIP SDK 6.4.2 Core and Runtime Compiler components in the AMD installer.'
            Write-Host 'The extra math libraries are not needed. Keep your current GPU driver.'
            Install-Dependency 'https://download.amd.com/developer/eula/rocm-hub/AMD-Software-PRO-Edition-25.Q3-Win10-Win11-For-HIP.exe' `
                'AMD HIP SDK 6.4.2' 'db474a0436edecbc2234a8d38402f1bae1cbf5a50c7e9b93eb6c0a7e0eb9c81a'
        }
        foreach ($dll in $dlls) {
            if (-not (Test-Path (Join-Path "$HipPath/bin" $dll))) {
                throw "Missing $dll in $HipPath. Install HIP SDK 6.4.2 Core and Runtime Compiler, or pass -HipPath with its location."
            }
        }
        if (-not (Test-Path "$HipPath/amdgcn/bitcode/ockl.bc")) { throw "Missing device bitcode in $HipPath" }
        New-Item -ItemType Directory -Force $bitcode | Out-Null
        Copy-Item "$HipPath/amdgcn/bitcode/*.bc" $bitcode -Force
        foreach ($dll in $dlls) { Copy-Item (Join-Path "$HipPath/bin" $dll) $bin -Force }
        Write-Host 'AMD runtime components copied from your SDK for local use under AMD terms (see licenses/amd-runtime.txt).'
    }
    Write-Host 'GPU execution also needs an AMD Adrenalin driver: https://www.amd.com/en/support/download/drivers.html'
} else {
    Write-Host 'GPU execution needs an NVIDIA driver: https://www.nvidia.com/Download/index.aspx'
}
& (Join-Path $bin 'xchplot2.exe') devices --config NUL
if ($LASTEXITCODE -ne 0) { throw "Device check failed: $LASTEXITCODE" }
