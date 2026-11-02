#requires -Version 7.3
# Build the shared-compute TLS dependency in the current VS developer shell.
param([string]$BuildDir = 'build/windows-openssl')
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
if (-not (Get-Command perl -ErrorAction SilentlyContinue) -or (& perl -e 'print $^O') -ne 'MSWin32') {
    throw 'OpenSSL requires native Windows Perl (for example, Strawberry Perl) on PATH'
}
$version = '3.5.9'
$sha256 = '603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a'
$arm64 = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq 'Arm64'
$platform = if ($arm64) { 'VC-WIN64-ARM' } else { 'VC-WIN64A' }
New-Item -ItemType Directory -Force $BuildDir | Out-Null
$root = (Resolve-Path $BuildDir).Path
$prefix = Join-Path $root 'install'
$archive = Join-Path $root "openssl-$version.tar.gz"
if (-not (Test-Path $archive)) {
    Invoke-WebRequest "https://github.com/openssl/openssl/releases/download/openssl-$version/openssl-$version.tar.gz" -OutFile $archive
}
if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sha256) {
    throw 'OpenSSL source archive checksum mismatch'
}
$source = Join-Path $root "openssl-$version"
if (-not (Test-Path $source)) { tar -xzf $archive -C $root }
Push-Location $source
try {
    # Static libraries avoid a second DLL deployment mechanism. Match the
    # application's dynamic MSVC runtime; no assembler dependency is required.
    perl Configure $platform no-shared no-asm no-tests no-makedepend /MD "--prefix=$prefix" --libdir=lib
    nmake /nologo
    nmake /nologo install_sw
    Copy-Item LICENSE.txt (Join-Path $prefix 'openssl-license.txt')
} finally { Pop-Location }
Write-Output "OpenSSL $version installed in $prefix"
