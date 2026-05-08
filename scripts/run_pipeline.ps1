<#
.SYNOPSIS
    Unified launcher: Docker Desktop -> chandra vLLM -> batch_ocr.py

.DESCRIPTION
    1. Starts Docker Desktop if it isn't running and waits for the daemon.
    2. Starts the chandra vLLM container (tuned for 16 GB mobile RTX 4090)
       unless one is already serving on http://localhost:8000.
    3. Waits for /v1/models to return 200.
    4. Invokes the existing batch_ocr.py from %TEMP%\chandra-smoke, which
       walks ~\Documents\Book and skips PDFs whose .md is already present.

    The vLLM container is left running on exit so a re-run can skip the
    ~3-minute model load. Stop it manually with `docker stop chandra-vllm`.

.NOTES
    Tuned config (see chandra-3qu / earlier session):
      --max-num-seqs 16 --max-num-batched-tokens 2048
      --gpu-memory-utilization .88 (free VRAM is ~14.4 GiB after Windows desktop)
#>

[CmdletBinding()]
param(
    [int]$DockerStartTimeoutSec = 180,
    [int]$VllmReadyTimeoutSec   = 600,
    [string]$ContainerName      = 'chandra-vllm',
    [string]$VllmImage          = 'vllm/vllm-openai:v0.17.0',
    [string]$Model              = 'datalab-to/chandra-ocr-2',
    [int]$Port                  = 8000
)

$ErrorActionPreference = 'Stop'

$DockerDesktopExe = 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
$BatchOcrPy       = Join-Path $PSScriptRoot 'batch_ocr.py'
$ChandraVenvPy    = 'C:\dev\chandra\.venv\Scripts\python.exe'
$VllmHealthUrl    = "http://localhost:$Port/v1/models"

function Write-Step  ($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok    ($m) { Write-Host "    $m" -ForegroundColor Green }
function Write-Warn2 ($m) { Write-Host "    $m" -ForegroundColor Yellow }
function Write-Err   ($m) { Write-Host "!!! $m" -ForegroundColor Red }

function Test-DockerReady {
    try {
        $null = & docker info 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Test-VllmReady {
    try {
        $r = Invoke-WebRequest -Uri $VllmHealthUrl -UseBasicParsing -TimeoutSec 3
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

# --- preflight checks ---------------------------------------------------
foreach ($p in @($BatchOcrPy, $ChandraVenvPy, $DockerDesktopExe)) {
    if (-not (Test-Path $p)) {
        Write-Err "missing required path: $p"
        exit 2
    }
}

# --- step 1: docker desktop ---------------------------------------------
Write-Step 'Checking Docker Desktop'
if (Test-DockerReady) {
    Write-Ok 'docker daemon already responsive'
} else {
    Write-Warn2 'docker daemon not responding; starting Docker Desktop'
    Start-Process -FilePath $DockerDesktopExe | Out-Null

    $deadline = (Get-Date).AddSeconds($DockerStartTimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-DockerReady) { break }
        Start-Sleep -Seconds 3
        Write-Host '.' -NoNewline
    }
    Write-Host ''
    if (-not (Test-DockerReady)) {
        Write-Err "Docker did not become ready within $DockerStartTimeoutSec s"
        exit 3
    }
    Write-Ok 'docker daemon ready'
}

# --- step 2: vLLM container ---------------------------------------------
Write-Step 'Checking vLLM server'
if (Test-VllmReady) {
    Write-Ok "$VllmHealthUrl already serving"
} else {
    # If a container with our name exists (running or stopped), remove it so
    # `docker run --name` doesn't conflict.
    $existing = (& docker ps -a --filter "name=^/$ContainerName$" --format '{{.ID}}') 2>$null
    if ($existing) {
        Write-Warn2 "removing stale container $ContainerName ($existing)"
        & docker rm -f $ContainerName | Out-Null
    }

    Write-Step "Starting $ContainerName"
    $hfCache = Join-Path $env:USERPROFILE '.cache/huggingface'
    $mmkw    = '{"min_pixels": 3136, "max_pixels": 6291456}'

    $dockerArgs = @(
        'run', '-d', '--rm',
        '--name', $ContainerName,
        '--gpus', 'device=0',
        '-v', "$hfCache`:/root/.cache/huggingface",
        '-p', "$Port`:8000",
        '--ipc=host',
        $VllmImage,
        '--model', $Model,
        '--no-enforce-eager',
        '--max-num-seqs', '16',
        '--max-num-batched-tokens', '2048',
        '--dtype', 'bfloat16',
        '--max-model-len', '18000',
        '--gpu-memory-utilization', '.88',
        '--enable-prefix-caching',
        '--mm-processor-kwargs', $mmkw,
        '--served-model-name', 'chandra'
    )

    $cid = & docker @dockerArgs
    if ($LASTEXITCODE -ne 0 -or -not $cid) {
        Write-Err 'docker run failed'
        exit 4
    }
    Write-Ok "container started: $($cid.Substring(0,12))"

    Write-Step "Waiting for $VllmHealthUrl (timeout ${VllmReadyTimeoutSec}s)"
    $deadline = (Get-Date).AddSeconds($VllmReadyTimeoutSec)
    $tick = 0
    while ((Get-Date) -lt $deadline) {
        if (Test-VllmReady) { break }

        # Surface early container exits so we don't wait the full timeout.
        $running = (& docker ps --filter "name=^/$ContainerName$" --format '{{.ID}}') 2>$null
        if (-not $running) {
            Write-Err 'vLLM container exited before serving; recent logs:'
            & docker logs --tail 60 $ContainerName 2>&1 | Write-Host
            exit 5
        }

        Start-Sleep -Seconds 5
        $tick++
        if ($tick % 6 -eq 0) {
            Write-Host "    still waiting ($([int]((Get-Date) - $deadline.AddSeconds(-$VllmReadyTimeoutSec)).TotalSeconds)s elapsed)..."
        } else {
            Write-Host '.' -NoNewline
        }
    }
    Write-Host ''
    if (-not (Test-VllmReady)) {
        Write-Err "vLLM did not become ready within $VllmReadyTimeoutSec s"
        Write-Host 'recent container logs:'
        & docker logs --tail 80 $ContainerName 2>&1 | Write-Host
        exit 6
    }
    Write-Ok 'vLLM ready'
}

# --- step 3: batch OCR --------------------------------------------------
Write-Step "Running batch_ocr.py"
& $ChandraVenvPy $BatchOcrPy
$rc = $LASTEXITCODE

Write-Host ''
if ($rc -eq 0) {
    Write-Ok 'batch finished cleanly'
} else {
    Write-Warn2 "batch_ocr.py exited with code $rc"
}

Write-Host ''
Write-Host "vLLM container '$ContainerName' is still running." -ForegroundColor DarkGray
Write-Host "Stop it with: docker stop $ContainerName" -ForegroundColor DarkGray

exit $rc
