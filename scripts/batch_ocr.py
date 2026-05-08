"""
Batch-OCR every PDF in a folder with chandra, then rewrite each output's
markdown TOC. Resumable: skips PDFs whose output .md already exists.

Per PDF, the script:
  1. Determines the expected output dir: <BOOK_DIR>/<pdf-stem>/
  2. If <pdf-stem>.md already exists in that dir (above MIN_MD_BYTES), skip.
  3. Otherwise run `chandra <pdf> <BOOK_DIR>` (chandra creates the subfolder).
  4. On success, run gen_toc.py --apply on the produced .md.
  5. Append a JSONL entry to the log with status, timing, counts, errors.

Usage:
  python batch_ocr.py                                # process all unconverted
  python batch_ocr.py --dry-run                      # show what would run
  python batch_ocr.py --limit 5                      # do at most 5 PDFs
  python batch_ocr.py --pdf "Some Book.pdf"          # do one specific PDF
  python batch_ocr.py --timeout-minutes 240          # per-PDF wall-time cap
  python batch_ocr.py --pattern "AI*"                # glob filter on filenames
  python batch_ocr.py --skip-toc                     # OCR only, don't rewrite TOC

Ctrl+C is honored between PDFs and after a chandra subprocess returns.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"C:\dev\chandra")
CHANDRA_EXE = REPO / ".venv" / "Scripts" / "chandra.exe"
PYTHON_EXE = REPO / ".venv" / "Scripts" / "python.exe"

BOOK_DIR = Path(r"C:\Users\david\Documents\Book")
GEN_TOC = Path(__file__).resolve().parent / "gen_toc.py"

# Log lives under %TEMP% so it stays out of the chandra repo. Existing runs
# already wrote here, so this preserves history.
LOG_DIR = Path(os.environ.get("TEMP", r"C:\Users\david\AppData\Local\Temp")) / "chandra-smoke"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOG_DIR / "batch_ocr.log.jsonl"

DEFAULT_TIMEOUT_MIN = 240   # 4h hard cap per PDF
MIN_MD_BYTES = 1024         # below this, the .md is treated as a failed run

VLLM_HEALTH_URL = "http://localhost:8000/v1/models"

# Module-level interrupt flag so we exit cleanly between PDFs even after the
# subprocess inherits the SIGINT and ignores it.
_interrupted = False


def _on_sigint(signum, frame):  # noqa: ARG001
    global _interrupted
    _interrupted = True
    print("\n[SIGINT] will stop after current PDF.", flush=True)


def jlog(**kw) -> None:
    """Append one JSONL record with iso-timestamp."""
    kw.setdefault("ts", dt.datetime.now().isoformat(timespec="seconds"))
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kw, ensure_ascii=False) + "\n")


def vllm_alive() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(VLLM_HEALTH_URL, timeout=5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def existing_md(pdf: Path) -> Path | None:
    """Path to the produced .md if it already exists and is non-trivial."""
    out_dir = BOOK_DIR / pdf.stem
    md = out_dir / f"{pdf.stem}.md"
    if md.exists() and md.stat().st_size >= MIN_MD_BYTES:
        return md
    return None


def run_chandra(pdf: Path, timeout_s: int) -> tuple[bool, str, int]:
    """Returns (ok, message, exit_code)."""
    cmd = [
        str(CHANDRA_EXE),
        str(pdf),
        str(BOOK_DIR),
        "--log-level", "WARNING",
    ]
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s}s", -1
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "")[-2000:]
        return False, f"exit={p.returncode}; tail={tail!r}", p.returncode
    return True, "ok", 0


def run_gen_toc(md: Path) -> tuple[bool, str]:
    cmd = [str(PYTHON_EXE), str(GEN_TOC), str(md), "--apply"]
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, "gen_toc timeout"
    if p.returncode != 0:
        # gen_toc exits 1 when there's no printed-TOC region to replace —
        # that's a soft failure, the OCR is still good.
        return False, (p.stdout + p.stderr).strip()[-1000:]
    return True, "ok"


def count_image_resolution(md: Path) -> tuple[int, int]:
    """Returns (refs, present_on_disk) — 51/51 means all images resolve."""
    text = md.read_text(encoding="utf-8", errors="replace")
    refs = set(re.findall(r"[0-9a-f]{32}_\d+_img\.webp", text))
    out_dir = md.parent
    present = sum(1 for r in refs if (out_dir / r).exists())
    return len(refs), present


def discover_pdfs(pattern: str | None) -> list[Path]:
    pdfs = sorted(BOOK_DIR.glob("*.pdf"))
    if pattern:
        pdfs = [p for p in pdfs if fnmatch.fnmatch(p.name, pattern)]
    return pdfs


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="List PDFs that would run; don't OCR")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N PDFs this run")
    ap.add_argument("--pdf", help="Process a single PDF by filename")
    ap.add_argument("--pattern", help="fnmatch glob on filename, e.g. 'AI*'")
    ap.add_argument("--timeout-minutes", type=int, default=DEFAULT_TIMEOUT_MIN,
                    help=f"Per-PDF timeout (default: {DEFAULT_TIMEOUT_MIN})")
    ap.add_argument("--skip-toc", action="store_true",
                    help="Don't run gen_toc after OCR")
    args = ap.parse_args()

    if not args.dry_run and not vllm_alive():
        print("vLLM server not reachable at " + VLLM_HEALTH_URL, file=sys.stderr)
        print("Start it with: chandra_vllm --gpu t4 --gpu-runtime \"\"", file=sys.stderr)
        return 2

    if args.pdf:
        pdfs = [BOOK_DIR / args.pdf]
        if not pdfs[0].exists():
            print(f"missing: {pdfs[0]}", file=sys.stderr)
            return 2
    else:
        pdfs = discover_pdfs(args.pattern)

    pending = []
    skipped = []
    for p in pdfs:
        if existing_md(p):
            skipped.append(p)
        else:
            pending.append(p)

    print(f"BOOK_DIR:       {BOOK_DIR}")
    print(f"discovered:     {len(pdfs)}")
    print(f"already done:   {len(skipped)}")
    print(f"pending:        {len(pending)}")
    if args.limit and len(pending) > args.limit:
        pending = pending[:args.limit]
        print(f"limited to:     {args.limit}")

    if args.dry_run:
        print("\n=== would run ===")
        for p in pending[:50]:
            print(f"  {p.name}  ({fmt_bytes(p.stat().st_size)})")
        if len(pending) > 50:
            print(f"  ... +{len(pending) - 50} more")
        return 0

    signal.signal(signal.SIGINT, _on_sigint)

    timeout_s = args.timeout_minutes * 60
    print(f"timeout/pdf:    {args.timeout_minutes} min")
    print(f"log:            {LOG}")
    print()
    jlog(event="batch_start", pending=len(pending), skipped=len(skipped),
         pattern=args.pattern, limit=args.limit)

    t_start = time.time()
    ok_n = fail_n = toc_skipped_n = 0
    for i, pdf in enumerate(pending, 1):
        if _interrupted:
            jlog(event="batch_interrupted", processed=i - 1)
            print("interrupted; exiting.")
            break

        size = pdf.stat().st_size
        print(f"[{i}/{len(pending)}] {pdf.name}  ({fmt_bytes(size)})", flush=True)
        t0 = time.time()
        ok, msg, code = run_chandra(pdf, timeout_s=timeout_s)
        elapsed = time.time() - t0
        if not ok:
            fail_n += 1
            jlog(event="ocr_failed", pdf=pdf.name, size=size, elapsed=elapsed,
                 message=msg, exit=code)
            print(f"  FAIL  {elapsed:.1f}s  {msg[:200]}", flush=True)
            continue

        md = existing_md(pdf)
        if not md:
            fail_n += 1
            jlog(event="ocr_postcheck_failed", pdf=pdf.name, size=size,
                 elapsed=elapsed, message="md not produced or below MIN_MD_BYTES")
            print(f"  FAIL  postcheck: md not present at expected path", flush=True)
            continue

        refs, present = count_image_resolution(md)

        toc_ok = None
        toc_msg = ""
        if not args.skip_toc:
            toc_ok, toc_msg = run_gen_toc(md)
            if not toc_ok:
                toc_skipped_n += 1

        ok_n += 1
        jlog(event="ocr_ok", pdf=pdf.name, size=size, elapsed=elapsed,
             md_path=str(md), md_bytes=md.stat().st_size,
             image_refs=refs, image_present=present,
             toc_applied=bool(toc_ok), toc_msg=toc_msg)

        toc_str = "toc=ok" if toc_ok else (
            "toc=skipped" if args.skip_toc else f"toc={toc_msg[:60]}"
        )
        print(f"  OK    {elapsed:.1f}s  imgs={present}/{refs}  {toc_str}",
              flush=True)

    total_elapsed = time.time() - t_start
    jlog(event="batch_end", ok=ok_n, fail=fail_n, toc_failed=toc_skipped_n,
         elapsed=total_elapsed)
    print(f"\ndone: ok={ok_n} fail={fail_n} toc_failed={toc_skipped_n} "
          f"in {total_elapsed/60:.1f} min")
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
