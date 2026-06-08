<#
.SYNOPSIS
    Serve Chandra OCR on the NATIVE Windows vLLM build (no Docker Desktop).

.DESCRIPTION
    Drop-in replacement for the `docker run vllm/vllm-openai ...` launcher in
    `scripts/vllm.py` / `run_pipeline.ps1`. Exposes the exact same contract every
    Chandra consumer already targets:

        endpoint   http://localhost:8000/v1
        model      chandra   (--served-model-name)

    so the `chandra` CLI and Book-Scan (`--vllm-base`/`--model` defaults) work
    unchanged. Serves in the foreground; Ctrl-C to stop. First load from the HF
    cache is ~2-3 minutes.

    Windows specifics handled here:
      * VLLM_USE_FLASHINFER_SAMPLER=0  (FlashInfer's sampler JIT fails on Windows)
      * CUDA 13.3 bin on PATH
      * run from %TEMP% so `import vllm` resolves the installed package, not a
        source checkout (the source tree has no compiled vllm._C)

.NOTES
    VRAM (16 GB mobile RTX 4090): the Docker config uses .88 / max-model-len
    18000. Natively, the Windows desktop holds ~3 GB, so the safe default here is
    --gpu-memory-utilization 0.85 / --max-model-len 16000. If you hit an OOM at
    load, lower -GpuMemUtil (e.g. 0.80) or -MaxModelLen; if you have headroom,
    raise them toward the Docker values.
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
