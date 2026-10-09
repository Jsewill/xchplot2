#requires -Version 7.3
# CUDA 13.4 is the first toolkit with native Windows ARM64 development tools.
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$prefix = Join-Path $env:RUNNER_TEMP 'cuda-arm64'
$base = 'https://developer.download.nvidia.com/compute/cuda/redist'
$manifest = Invoke-RestMethod "$base/redistrib_13.4.2.json"
New-Item -ItemType Directory -Force "$prefix/licenses" | Out-Null
foreach ($component in 'cuda_nvcc', 'cuda_cudart', 'cuda_crt', 'libnvvm', 'cccl') {
    $package = $manifest.$component.'windows-arm64'
    $archive = Join-Path $env:RUNNER_TEMP "$component.zip"
    $unpacked = Join-Path $env:RUNNER_TEMP $component
    Invoke-WebRequest "$base/$($package.relative_path)" -OutFile $archive
    if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne $package.sha256) {
        throw "CUDA component checksum mismatch: $component"
    }
    Expand-Archive $archive -DestinationPath $unpacked -Force
    $root = (Get-ChildItem $unpacked -Directory).FullName
    Copy-Item "$root/LICENSE" "$prefix/licenses/$component.txt"
    Get-ChildItem $root -Directory | ForEach-Object {
        Copy-Item $_.FullName $prefix -Recurse -Force
    }
    Remove-Item $archive, $unpacked -Recurse -Force
}
"CUDA_PATH=$prefix" >> $env:GITHUB_ENV
