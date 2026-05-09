# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Chandra OCR 2 — a vision-language OCR model (`datalab-to/chandra-ocr-2`) that converts images and PDFs into HTML/Markdown/JSON with layout information. This repo is the open-source CLI, inference wrappers, and Streamlit demo around the model weights; the model itself is hosted on HuggingFace.

## Git remotes — fork-only push policy

This checkout is a fork. Remotes are configured so all pushes land on the fork, never upstream:

- `origin` → `https://github.com/davidcforbes/chandra.git` (fork — fetch + push)
- `upstream` → `https://github.com/datalab-to/chandra.git` (read-only; push URL is set to `DISABLED` so `git push upstream` will fail loudly)

**All commits and PRs go to `origin` only.** Do not re-enable pushes to `upstream` or open PRs against `datalab-to/chandra` without explicit instruction — upstream contributions are decided case-by-case after changes have settled on the fork. To pull in upstream changes: `git fetch upstream && git merge upstream/master` (or rebase).

## Common commands

Environment setup uses `uv`:

```bash
uv sync --group dev          # install with dev + all extras (hf, app)
source .venv/bin/activate    # bash; .venv\Scripts\activate on Windows
```

Tests are split:

```bash
# Unit tests — no GPU/model needed, fast (~2s for 130+ tests).
uv run pytest tests/unit
uv run pytest tests/unit/test_input.py -v

# Integration tests — require GPU + HF model download.
uv run pytest tests/integration
TORCH_ATTN=sdpa uv run pytest tests/integration   # CI uses sdpa attention
```

Coverage spans `input`, `output`, `model.vllm`, `model.util`, `manifest`, `pipeline`, `scripts.cli`, `scripts.screenshot_app`, `scripts.vllm`, and `settings`. New behavior should ship with unit tests — most logic in this repo is testable without a GPU because the OpenAI client, PIL, and pypdfium2 are easy to mock, and `pipeline.run_pipeline` accepts an injected `model` so the worker pool itself unit-tests against a mocked `InferenceManager`.

CLI entry points (defined in `pyproject.toml [project.scripts]`):

- `chandra <input> <output>` — main OCR CLI (`chandra/scripts/cli.py`)
- `chandra_app` — Streamlit demo (`chandra/scripts/run_app.py`)
- `chandra_vllm` — launches vLLM server in Docker (`chandra/scripts/vllm.py`)
- `chandra_screenshot` — screenshot utility

Lint/format (pre-commit runs ruff):

```bash
uv run pre-commit run --all-files
```

## Architecture

The pipeline is **discover → enqueue pages → parallel OCR workers → per-book assembler**, with the model layer at the worker step. `chandra/pipeline.py` owns the orchestration; `chandra/model/__init__.py` (the `InferenceManager`) is just one step inside a worker.

### Page-worker pipeline (`chandra/pipeline.py`)

The CLI is a thin shell that runs `discover_books(input_path, output_root, recursive, page_range)` then `run_pipeline(books, model, n_workers, ...)`.

- **`BookSpec`** carries `source_path`, `stem`, `stem_dir`, `total_pages`, `expected_pages` (what we plan to OCR — full file or `--page-range` subset), and `pending_pages` (`expected_pages` minus what's already on disk from a prior interrupted run).
- **Producer** (single thread) renders only the pending pages from each book via `iter_file_pages` and pushes `(book, page_num, image)` onto a bounded queue (size = `max(2*n_workers, 16)`). Backpressure keeps memory bounded.
- **Workers** (default 8 for vllm, 1 for hf) pull from the queue, call `model.generate([item])` with `BatchInputItem.file_stem` and `.page_num` set so chunks come back with stable global IDs, save image crops directly to `<stem_dir>/<hash>_<idx>_img.webp`, then atomically write the per-page artifact to `<stem_dir>/.partial/pages/NNNN.json` (tmp + rename — readers never see partials). Per-page exceptions are caught and recorded as `error: True` in the artifact so a single bad page doesn't kill the book.
- **Assembler** (single thread) polls each book's `.partial/` and runs `manifest.assemble_book` when `read_partial_state(stem_dir) == set(expected_pages)`. Two-phase commit: write all canonical artifacts (`<stem>.md`, `<stem>.html`, `<stem>_metadata.json`, `chunks.jsonl`) to `.tmp` siblings via `os.replace`, then `shutil.rmtree(.partial/)`.

### State on disk

```
<output_root>/
  <stem>/
    .partial/                  ← present only while in-flight
      _state.json              ← {source: {path,size,mtime}, expected_pages: [...]}
      pages/
        0042.json              ← per-page artifact (atomic; readers never see partials)
    <stem>.md                  ← canonical merged outputs
    <stem>.html
    <stem>_metadata.json
    chunks.jsonl               ← one JSON per line, per chunk, with stable IDs
    <hash>_<idx>_img.webp      ← extracted images (hash deterministic)
```

Resume detection on startup: canonical `<stem>.md` present → skip; `.partial/` present + source PDF mtime+size unchanged → resume on remaining pages; `.partial/` present but source mismatched → purge and start fresh.

### Stable chunk IDs

`parse_chunks(html, image, file_stem=..., page_num=...)` (`chandra/output.py`) emits chunks with `chunk_id` of the form `"<safe-stem>/<page:04d>/<idx:03d>"` — stem characters outside `[A-Za-z0-9_-]` collapse to `-`. Each chunk also carries `page` and `image_ref` (the `.webp` filename for Image/Figure chunks). These IDs survive in URLs, filesystem paths, and graph databases unchanged. Legacy callers that omit `file_stem`/`page_num` get `"_/NNN"` IDs — still unique within a page, just not globally.

### Two inference backends, one interface

`InferenceManager(method="vllm" | "hf")` dispatches to either:

- `chandra/model/vllm.py` — calls a vLLM OpenAI-compatible server via the `openai` SDK; parallelizes requests with a `ThreadPoolExecutor`. Includes a retry loop (`_should_retry`) that detects repeat-token failures via `detect_repeat_token` and bumps temperature on retry.
- `chandra/model/hf.py` — local HuggingFace transformers inference; `load_model()` runs once at construction.

Both produce `GenerationResult(raw, token_count, error)`. The manager then runs the same output-parsing pipeline regardless of backend, so any change to parsing applies uniformly.

### The model output is HTML, not Markdown

The core contract is in `chandra/prompts.py`: the model is prompted to emit a single HTML document where each layout block is a `<div>` with `data-bbox` (normalized 0–1000, scale controlled by `settings.BBOX_SCALE`) and `data-label` (one of ~18 labels: `Text`, `Table`, `Equation-Block`, `Image`, `Figure`, `Form`, `Page-Header`, etc.). Allowed tags/attributes are an explicit allowlist (`ALLOWED_TAGS`, `ALLOWED_ATTRIBUTES`).

`chandra/output.py` then transforms that raw HTML:

- `sanitize_html` — **enforces** `ALLOWED_TAGS`/`ALLOWED_ATTRIBUTES` via bleach, decomposes `<script>`/`<style>`/`<iframe>`/etc. with their content (bleach's `strip=True` keeps inner text otherwise), runs CSS through a `_SAFE_CSS_PROPERTIES` allowlist, and gates URL protocols. Called at the entry of both `parse_html` and `parse_layout` so downstream renderers (Streamlit, Flask viewer, third-party consumers) get sanitized HTML for free.
- `parse_html` — filtered HTML (respects `include_headers_footers`, `include_images`)
- `parse_markdown` — uses `markdownify` to convert
- `parse_chunks` — extracts the structured per-block list (label + bbox + content) used downstream
- `extract_images` — crops `Image`/`Figure` regions out of the source PIL image using each chunk's bbox

When changing prompt labels, allowed tags, or bbox encoding, update both `prompts.py` and the corresponding parser in `output.py` — they're a coupled contract. The bleach allowlist in `sanitize_html` reads `ALLOWED_TAGS`/`ALLOWED_ATTRIBUTES` directly from `prompts.py`, so adding a tag to the prompt automatically threads through.

### Two prompt modes

`PROMPT_MAPPING` in `prompts.py` exposes `ocr_layout` (default; emits layout div blocks with bboxes) and `ocr` (plain HTML, no layout). `BatchInputItem.prompt_type` selects which; `BatchInputItem.prompt` overrides with a custom prompt.

### Input handling — streaming, 1-indexed CLI

`chandra/input.py` has two parallel APIs:

- **Streaming** (preferred for large docs): `iter_pdf_pages` and `iter_file_pages` yield rendered PIL pages one at a time so peak memory is bounded by the work-queue size, not the full PDF. `count_file_pages` reports the page count without rendering. `chandra/pipeline.py::_producer` consumes these directly.
- **Eager**: `load_pdf_images` / `load_file` are thin `list(...)` wrappers retained for backwards compatibility.

`parse_range_str` is the **1-indexed → 0-indexed** boundary. CLI users say `--page-range 1-5,7` (1-indexed, the natural way humans count pages); this function returns `[0, 1, 2, 3, 4, 6]` (0-indexed pdfium indices). It also validates: empty, malformed, reversed, ≤ 0 → `ValueError` with a clear message. Code that calls `iter_pdf_pages` directly (e.g., `app.py`, `screenshot_app.py`) bypasses the conversion and must pass 0-indexed indices itself.

### Settings & env file

All config lives in `chandra/settings.py` (pydantic-settings). `_resolve_env_file` looks for **`.env` first** (python-dotenv default), falling back to legacy `local.env`. Notable knobs: `MODEL_CHECKPOINT`, `MAX_OUTPUT_TOKENS`, `VLLM_API_BASE`, `VLLM_MODEL_NAME`, `VLLM_GPUS`, `VLLM_IMAGE_FORMAT` (default JPEG), `VLLM_IMAGE_QUALITY` (default 92), `BBOX_SCALE`, `IMAGE_DPI`, `MIN_PDF_IMAGE_DIM`, `MIN_IMAGE_DIM`.

### vLLM client + retry loop

`chandra/model/vllm.py::get_openai_client` is a **thread-safe memoized client cache** keyed on `(api_base, api_key, headers)`. Don't construct `OpenAI()` directly — that breaks connection pooling under the executor. `_classify_error` maps openai exception types into `(category, retryable)` so the retry loop distinguishes auth/rate-limit/timeout/connection/5xx/4xx. Image payloads default to JPEG q=92 (~10× faster encode than the previous PNG path) — switch via `VLLM_IMAGE_FORMAT`.

`detect_repeat_token` (`chandra/model/util.py`) does both the literal-tail and trimmed-tail repeat-detection scans in a single call. Don't call it twice per result — that was the old pattern.

### vLLM server launcher

`chandra/scripts/vllm.py` runs `docker run vllm/vllm-openai:...` with model and tuned flags. `docker_invocation()` probes `docker info` and **only prepends `sudo` when needed**, so it works on Docker Desktop, rootless docker, and Linux+sudo. Override the binary, image, runtime, and port via `--docker-bin`, `--image`, `--gpu-runtime`, `--port`. `get_gpu_settings(gpu)` scales `--max-num-batched-tokens` and `--max-num-seqs` from an H100 baseline using the hardcoded `GPU_VRAM_GB` table.

### Screenshot Flask app — upload-only

`chandra/scripts/screenshot_app.py` accepts files via `multipart/form-data` (field `file`) and never reads server-side `file_path` from the request. Binds **127.0.0.1** by default (override with `--host 0.0.0.0`). `get_model()` uses double-checked locking so concurrent first-requests don't construct two `InferenceManager`s.

## CLI surface

```bash
chandra <input> <output>              # single file, folder, or "folder/*.pdf" glob
chandra books/ out/ --recursive       # walk subfolders too
chandra books/ out/ --workers 8       # tune the page-worker pool
chandra book.pdf out/ --page-range 1-50
```

`--workers` defaults to **8 for vllm** and **1 for hf**. The pipeline processes pages individually so there is no application-level batching anymore. Re-running the same command resumes interrupted books and skips fully-assembled ones — no separate orchestrator script needed.

Deprecated flags (still accepted, warn once): `--batch-size`, `--max-workers` (aliased to `--workers`), `--fail-fast` (per-page errors are recorded in `metadata.json` instead of aborting). Active flags: `--method`, `--recursive`, `--workers`, `--page-range`, `--max-output-tokens`, `--max-retries`, `--include-images/--no-images`, `--include-headers-footers/--no-headers-footers`, `--save-html/--no-html`, `--paginate-output` (legacy `--paginate_output` alias), `--log-every`, `--log-level`.

Exit code: **0** on clean success; **1** if any pages still pending (producer failure) or any pages tagged with `error: True`; **130** on `KeyboardInterrupt` (partial state preserved on disk for next run).

Each input file gets its own subdirectory in the output containing `<name>.md`, `<name>.html`, `<name>_metadata.json`, `chunks.jsonl`, plus extracted images saved as `.webp` keyed by an MD5 hash of the raw HTML (see `get_image_name` in `output.py`). The `chunks.jsonl` file is one JSON per line — `{chunk_id, page, label, bbox, content, image_ref}` — and is the canonical artifact for downstream graph indexing.

### Launcher

`scripts/run_pipeline.ps1` brings up Docker Desktop + the chandra vLLM container (tuned for the 16 GB mobile RTX 4090 — `--max-num-seqs 16`, `--max-num-batched-tokens 2048`, `--gpu-memory-utilization .88`) and then invokes `chandra <BookDir> <OutDir> --recursive --workers 8`. Override `-BookDir`, `-OutDir`, `-Workers` via parameters. The vLLM container is left running between runs so the ~3-minute model load isn't repeated.

`scripts/gen_toc.py` is a standalone post-processing tool that rewrites the printed-TOC region of an assembled `.md` with a navigable bullet TOC. The fix in commit `fe20aa2` made it bail safely when no terminator H1 follows `# Table of Contents` (the previous bug destroyed the body of any document without an explicit `# Preface`).


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:3216161c -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
