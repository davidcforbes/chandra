"""
Recover .md files destroyed by the old gen_toc.py bug (chandra-37a).

For every <dir>/<name>.md inside BOOK_DIR with a sibling <name>.md.bak that is
larger, restore the .bak (which holds the original full OCR output) and re-run
gen_toc.py with the fixed logic. The original tiny .md is overwritten; the .bak
is kept on disk so a second pass is a no-op.

Usage:
  python recover_md.py            # dry run, show what would change
  python recover_md.py --apply    # actually restore + re-run gen_toc
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

REPO = Path(r"C:\dev\chandra")
PYTHON_EXE = REPO / ".venv" / "Scripts" / "python.exe"

BOOK_DIR = Path(r"C:\Users\david\Documents\Book")
GEN_TOC = Path(__file__).resolve().parent / "gen_toc.py"


def fmt_kb(n: int) -> str:
    return f"{n / 1024:.1f}KB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Restore from .bak and re-run gen_toc; otherwise dry-run")
    ap.add_argument("--no-toc", action="store_true",
                    help="Restore .bak but skip the gen_toc rerun")
    args = ap.parse_args()

    candidates = []
    for sub in sorted(p for p in BOOK_DIR.iterdir() if p.is_dir()):
        md = sub / f"{sub.name}.md"
        bak = md.with_suffix(md.suffix + ".bak")
        if not (md.exists() and bak.exists()):
            continue
        md_sz = md.stat().st_size
        bak_sz = bak.stat().st_size
        if bak_sz > md_sz:
            candidates.append((md, bak, md_sz, bak_sz))

    print(f"BOOK_DIR:   {BOOK_DIR}")
    print(f"candidates: {len(candidates)}")
    if not candidates:
        return 0

    for md, bak, md_sz, bak_sz in candidates:
        print(f"  {md.parent.name}: md={fmt_kb(md_sz)}  bak={fmt_kb(bak_sz)}")

    if not args.apply:
        print("\n(dry run — pass --apply to restore from .bak)")
        return 0

    restored = 0
    toc_ok = 0
    toc_skipped = 0
    for md, bak, _, _ in candidates:
        shutil.copyfile(bak, md)
        restored += 1
        if args.no_toc:
            continue
        # Re-run gen_toc with the fixed logic. With the fix, this either rewrites
        # the printed-TOC region in place or returns 1 leaving the file intact.
        cmd = [str(PYTHON_EXE), str(GEN_TOC), str(md), "--apply"]
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", check=False)
        if p.returncode == 0:
            toc_ok += 1
        else:
            toc_skipped += 1

    print(f"\nrestored: {restored}  toc_rewritten: {toc_ok}  toc_skipped: {toc_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
