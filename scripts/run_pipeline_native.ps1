<#
.SYNOPSIS
    Native (Docker-free) launcher: native Windows vLLM -> chandra page-worker pipeline.

.DESCRIPTION
    The Docker-free counterpart of run_pipeline.ps1. Instead of starting Docker
    Desktop + a vllm/vllm-openai container, it starts the NATIVE Windows vLLM via
    serve_vllm_native.ps1 (in its own window) if nothing is already serving on
    http://localhost:8000, waits for /v1/models, then runs the chandra CLI.

    The vLLM window is left running so a re-run skips the ~3-minute model load;
    close that window (or Ctrl-C it) to stop the server.

.PARAMETER BookDir   Folder of source PDFs/images. Default ~/Documents/Book.
.PARAMETER OutDir    Output root. Default = BookDir.
.PARAMETER Workers   chandra page-pool size. Default 8.
#>

[CmdletBinding()]
param(
    [int]$VllmReadyTimeoutSec = 600,
    [int]$Port                = 8000,
    [string]$BookDir          = (Join-Path $env:USERPROFILE 'Documents\Book'),
    [string]$OutDir           = (Join-Path $env:USERPROFILE 'Documents\Book'),
    [int]$Workers             = 8
)

$ErrorActionPreference = 'Stop'
$ChandraExe   = 'C:\dev\chandra\.venv\Scripts\chandra.exe'
$ServeScript  = Join-Path $PSScriptRoot 'serve_vllm_native.ps1'
$VllmHealth   = "http://localhost:$Port/v1/models"

function Write-Step ($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok   ($m) { Write-Host "    $m" -ForegroundColor Green }
function Write-Err  ($m) { Write-Host "!!! $m" -ForegroundColor Red }
function Test-VllmReady {
    try { return ((Invoke-WebRequest -Uri $VllmHealth -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) }
    catch { return $false }
}

foreach ($p in @($ChandraExe, $ServeScript)) {
    if (-not (Test-Path $p)) { Write-Err "missing required path: $p"; exit 2 }
}
if (-not (Test-Path $BookDir)) { Write-Err "BookDir does not exist: $BookDir"; exit 2 }

Write-Step 'Checking native vLLM server'
if (Test-VllmReady) {
    Write-Ok "$VllmHealth already serving"
} else {
    Write-Step 'Starting native vLLM (serve_vllm_native.ps1) in a new window'
    Start-Process -FilePath 'powershell.exe' `
        -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-File', $ServeScript, '-Port', "$Port") | Out-Null

    $deadline = (Get-Date).AddSeconds($VllmReadyTimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-VllmReady) { break }
        Start-Sleep -Seconds 5
        Write-Host '.' -NoNewline
    }
    Write-Host ''
    if (-not (Test-VllmReady)) {
        Write-Err "vLLM did not become ready within $VllmReadyTimeoutSec s (see the vLLM window)"
        exit 6
    }
    Write-Ok 'vLLM ready'
}

Write-Step 'Running chandra pipeline'
# Force UTF-8 so the chandra CLI's progress glyphs (checkmarks) don't crash the
# cp1252 Windows console mid-assembly.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Write-Host "    input  : $BookDir"
Write-Host "    output : $OutDir"
Write-Host "    workers: $Workers"
& $ChandraExe $BookDir $OutDir --recursive --workers $Workers --method vllm
$rc = $LASTEXITCODE

Write-Host ''
if ($rc -eq 0) { Write-Ok 'pipeline finished cleanly' } else { Write-Host "chandra exited with code $rc" -ForegroundColor Yellow }
Write-Host 'Native vLLM is still running in its own window; close it to stop the server.' -ForegroundColor DarkGray
exit $rc
