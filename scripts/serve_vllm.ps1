<#
.SYNOPSIS
    Launch the native Windows vLLM server for Chandra OCR (no Docker, no pipeline).

.DESCRIPTION
    SERVE-ONLY launcher: brings up `datalab-to/chandra-ocr-2` as model `chandra`
    on http://localhost:<Port>/v1 using the native Windows vLLM build at
    C:\dev\vllm-windows. It does NOT run the chandra CLI pipeline — it only serves
    the OCR endpoint that consumers target:

        endpoint   http://localhost:8000/v1
        model      chandra   (--served-model-name)

    Consumers that just need this endpoint up:
      * LLM-Wiki's chandra ingest worker (probes /v1/models, drives chandra.exe)
      * the `chandra` CLI run manually
      * Book-Scan `ocr-folder --ocr chandra`

    Serves in the foreground; Ctrl-C to stop. First load from the HF cache is
    ~2-3 minutes. For an end-to-end batch (server + chandra CLI) use
    run_pipeline_native.ps1 (native) or run_pipeline.ps1 (Docker) instead.

    Windows specifics handled here:
      * VLLM_USE_FLASHINFER_SAMPLER=0  (FlashInfer's sampler JIT fails on Windows)
      * CUDA 13.3 bin on PATH
      * run from %TEMP% so `import vllm` resolves the installed package, not a
        source checkout (the source tree has no compiled vllm._C)

.NOTES
    VRAM (16 GB mobile RTX 4090): defaults --gpu-memory-utilization 0.80 /
    --max-model-len 18000 / --enforce-eager (validated ~15.4 GB; the desktop
    holds ~3 GB). If you OOM at load, lower -GpuMemUtil (e.g. 0.75) or
    -MaxModelLen; with headroom, raise them or pass -Cudagraphs.
#>

[CmdletBinding()]
param(
    [string]$Model            = 'datalab-to/chandra-ocr-2',
    [string]$ServedName       = 'chandra',
    [int]$Port                = 8000,
    [string]$VllmExe          = 'C:\dev\vllm-windows\.venv\Scripts\vllm.exe',
    [string]$CudaBin          = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin',
    [double]$GpuMemUtil       = 0.80,
    [int]$MaxModelLen         = 18000,
    [int]$MaxNumSeqs          = 4,
    [int]$MaxBatchedTokens    = 2048,
    [switch]$Cudagraphs       = $false
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $VllmExe)) {
    Write-Error "vllm.exe not found at $VllmExe (build/install vLLM for Windows first)"
    exit 2
}

$env:PATH = "$CudaBin;$env:PATH"
$env:VLLM_USE_FLASHINFER_SAMPLER = '0'
$env:PYTHONUTF8 = '1'
Set-Location $env:TEMP   # avoid source-shadowing of the installed vllm._C

$mmkw = '{"min_pixels": 3136, "max_pixels": 6291456}'
$vllmArgs = @(
    'serve', $Model,
    '--served-model-name', $ServedName,
    '--dtype', 'bfloat16',
    '--max-model-len', "$MaxModelLen",
    '--gpu-memory-utilization', "$GpuMemUtil",
    '--max-num-seqs', "$MaxNumSeqs",
    '--max-num-batched-tokens', "$MaxBatchedTokens",
    # enforce-eager by default keeps VRAM headroom on the 16 GB card; pass
    # -Cudagraphs to trade ~1 GB for faster decode if you have room.
    $(if ($Cudagraphs) { '--no-enforce-eager' } else { '--enforce-eager' }),
    '--enable-prefix-caching',
    '--mm-processor-kwargs', $mmkw,
    '--host', '127.0.0.1',
    '--port', "$Port"
)

Write-Host "==> serving $Model as '$ServedName' on http://127.0.0.1:$Port/v1 (native, no Docker)" -ForegroundColor Cyan
& $VllmExe @vllmArgs
exit $LASTEXITCODE
