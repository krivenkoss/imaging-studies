<#
Builds a codec listed in third_party/coders.json from its pinned source archive
(../REFERENCE.md#studies, item 5). Windows PowerShell 5.1, CMake and Visual Studio Build Tools.

Usage:   powershell -NoProfile -File tools/build_coders.ps1 vtm
Archive: <coder_sources of lab_paths.json>/<folder>/<archive>; its SHA-256 must equal coders.json.
Build:   tools/build/<bin_dir>/ - unpacked sources and the CMake tree (ignored by git).
Result:  tools/bin/<bin_dir>/ - the executables, copies of the listed cfg files and MANIFEST.json
         (tag, archive SHA-256, sha256 of every file, version printed by the encoder, compiler,
         CMake). An existing result or build folder is refused by name, never overwritten.
#>
param([Parameter(Mandatory = $true)][string]$Codec)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

function Refuse([string]$message) {
    Write-Host "REFUSED: $message"
    exit 2
}

function Sha256([string]$path) {
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
}

$pathsFile = Join-Path $root 'lab_paths.json'
if (-not (Test-Path -LiteralPath $pathsFile)) { Refuse "$pathsFile not found: copy lab_paths.example.json and set coder_sources" }
$paths = Get-Content -LiteralPath $pathsFile -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $paths.coder_sources) { Refuse "lab_paths.json has no key 'coder_sources'" }
$coders = Get-Content -LiteralPath (Join-Path $root 'third_party/coders.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$spec = $coders.$Codec
if (-not $spec) { Refuse "codec '$Codec' is not listed in third_party/coders.json" }

$sources = $paths.coder_sources
if (-not [IO.Path]::IsPathRooted($sources)) { $sources = [IO.Path]::GetFullPath((Join-Path $root $sources)) }
$archive = Join-Path (Join-Path $sources $spec.folder) $spec.archive
if (-not (Test-Path -LiteralPath $archive)) { Refuse "archive not found: $archive (download: $($spec.url))" }
$archiveSha = Sha256 $archive
if ($archiveSha -ne $spec.sha256) { Refuse "archive SHA-256 $archiveSha differs from third_party/coders.json $($spec.sha256)" }

$bin = Join-Path $root "tools/bin/$($spec.bin_dir)"
$build = Join-Path $root "tools/build/$($spec.bin_dir)"
if (Test-Path -LiteralPath $bin) { Refuse "$bin already exists: a build is never overwritten" }
if (Test-Path -LiteralPath $build) { Refuse "$build already exists: remove it to rebuild from the archive" }

Write-Host "Unpacking $archive (SHA-256 checked) into $build"
Expand-Archive -LiteralPath $archive -DestinationPath $build
$src = Join-Path $build $spec.top_dir
$cmakeTree = Join-Path $src 'build'
$configure = @($spec.cmake_configure)
& cmake -S $src -B $cmakeTree @configure
if ($LASTEXITCODE -ne 0) { Refuse "cmake configure failed, exit code $LASTEXITCODE" }
& cmake --build $cmakeTree --config Release --parallel --target @($spec.cmake_targets)
if ($LASTEXITCODE -ne 0) { Refuse "cmake build failed, exit code $LASTEXITCODE" }

New-Item -ItemType Directory -Path $bin | Out-Null
$files = [ordered]@{}
foreach ($exe in @($spec.executables)) {
    $found = @(Get-ChildItem -LiteralPath (Join-Path $src 'bin') -Recurse -Filter $exe |
               Where-Object { $_.FullName -match '[\\/]release[\\/]' })
    if ($found.Count -ne 1) { Refuse "expected one Release $exe under $src/bin, found $($found.Count)" }
    Copy-Item -LiteralPath $found[0].FullName -Destination $bin
    $files[$exe] = Sha256 (Join-Path $bin $exe)
}
foreach ($rel in @($spec.copy)) {
    $name = Split-Path -Leaf $rel
    Copy-Item -LiteralPath (Join-Path $src $rel) -Destination $bin
    $files[$name] = Sha256 (Join-Path $bin $name)
}

$ErrorActionPreference = 'Continue'  # the encoder prints its help and exits non-zero
$banner = (& (Join-Path $bin $spec.executables[0]) --help) | Out-String
$ErrorActionPreference = 'Stop'
$version = [regex]::Match($banner, 'VTM Encoder Version[^\r\n]*').Value.Trim()
if (-not $version) { Refuse "no version line in the output of $($spec.executables[0])" }
$compilerFile = @(Get-ChildItem -LiteralPath (Join-Path $cmakeTree 'CMakeFiles') -Recurse -Filter 'CMakeCXXCompiler.cmake')[0]
$compilerText = Get-Content -LiteralPath $compilerFile.FullName -Raw
$compiler = ([regex]::Match($compilerText, 'CMAKE_CXX_COMPILER_ID "([^"]*)"').Groups[1].Value + ' ' +
             [regex]::Match($compilerText, 'CMAKE_CXX_COMPILER_VERSION "([^"]*)"').Groups[1].Value)

$manifest = [ordered]@{
    codec = $Codec; name = $spec.name; tag = $spec.tag; archive = $spec.archive; archive_sha256 = $archiveSha
    url = $spec.url; license = $spec.license; files_sha256 = $files; encoder_version = $version
    compiler = $compiler; cmake = (& cmake --version | Select-Object -First 1); cmake_configure = $configure
    cmake_build = "--config Release --target $(@($spec.cmake_targets) -join ' ')"
    built_utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'); built_by = 'tools/build_coders.ps1'
}
$json = $manifest | ConvertTo-Json -Depth 4
[IO.File]::WriteAllText((Join-Path $bin 'MANIFEST.json'), $json, (New-Object Text.UTF8Encoding $false))
Write-Host "Built $($spec.tag): $bin"
Write-Host $json
