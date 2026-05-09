"""
Replace the printed-TOC region of a chandra-produced .md with a navigable
markdown TOC built from the real heading structure of the document.

The printed-TOC region (between '# Table of Contents' and the next H1) is
visual fragments from the source PDF's multi-column TOC layout — '## 1',
'### **Index 297**', etc. — not useful chapter titles. We discard that
range and emit a bullet TOC built from the H1/H2/H3 headings that follow it.

If '# Table of Contents' is not present or no H1 follows it, the script
returns 1 and writes nothing — destroying content when the terminator is
missing was the cause of chandra-37a, so we'd rather skip than guess.

Usage:
  python gen_toc.py <path-to-md>             # dry run
  python gen_toc.py <path-to-md> --apply     # rewrite in place (a .bak is left)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
H1_RE = re.compile(r"^#\s+(.+?)\s*$")
TOC_START_RE = re.compile(r"^#\s+Table of Contents\s*$", re.IGNORECASE)
EMPH_RE = re.compile(r"(\*\*|__|\*|_)")


def clean_heading_text(s: str) -> str:
    """Strip markdown emphasis markers used in the source headings."""
    return EMPH_RE.sub("", s).strip()


def slugify(s: str, used: dict[str, int]) -> str:
    """GitHub-style anchor: lowercase, non-alnum -> hyphen, dedupe with -N."""
    base = re.sub(r"[^\w\s-]", "", s.lower())
    base = re.sub(r"[\s_]+", "-", base).strip("-")
    if not base:
        base = "section"
    n = used.get(base, 0)
    used[base] = n + 1
    return base if n == 0 else f"{base}-{n}"


def find_region(lines: list[str]) -> tuple[int, int] | None:
    """Locate the printed TOC region [start_idx, end_idx_exclusive).

    Region ends at the next H1 after '# Table of Contents' — that's where
    the real document body begins. If no following H1 exists, return None
    rather than gobbling the rest of the file (the printed TOC is bounded;
    if we can't find its end, refuse to mutate)."""
    start = None
    for i, line in enumerate(lines):
        if TOC_START_RE.match(line):
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        if H1_RE.match(lines[j]):
            return start, j  # j points at the next H1 — keep it
    return None


def extract_headings(lines: list[str], from_idx: int) -> list[tuple[int, str]]:
    """All H1-H3 headings starting at from_idx, as (level, text)."""
    out: list[tuple[int, str]] = []
    in_code = False
    for i in range(from_idx, len(lines)):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        text = clean_heading_text(m.group(2))
        if not text:
            continue
        out.append((level, text))
    return out


def build_toc(headings: list[tuple[int, str]]) -> str:
    """Bullet-list TOC with GitHub-style anchors. Top level becomes h1 entries."""
    used: dict[str, int] = {}
    out_lines = ["# Table of Contents", ""]
    if not headings:
        out_lines.append("_(no headings found)_")
        return "\n".join(out_lines) + "\n"
    min_lvl = min(lvl for lvl, _ in headings)
    for lvl, text in headings:
        indent = "  " * (lvl - min_lvl)
        anchor = slugify(text, used)
        out_lines.append(f"{indent}- [{text}](#{anchor})")
    out_lines.append("")
    return "\n".join(out_lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("md_path")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite the file in place; leave a .bak alongside",
    )
    args = ap.parse_args()

    md = Path(args.md_path)
    if not md.exists():
        print(f"missing: {md}", file=sys.stderr)
        return 2
    text = md.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    region = find_region(lines)
    if region is None:
        print(
            "no printed-TOC region found "
            "(missing '# Table of Contents' or no following H1); "
            "nothing to replace."
        )
        return 1
    start, end = region
    print(
        f"printed-TOC region: lines {start + 1}..{end} "
        f"({end - start} lines, ~{sum(len(line) for line in lines[start:end]) / 1024:.1f} KB)"
    )

    headings = extract_headings(lines, end)
    print(f"real headings found after TOC: {len(headings)}")
    levels = {}
    for lvl, _ in headings:
        levels[lvl] = levels.get(lvl, 0) + 1
    print(f"  by level: {dict(sorted(levels.items()))}")

    new_toc = build_toc(headings)

    print("\n=== first 20 lines of generated TOC ===")
    print("\n".join(new_toc.splitlines()[:20]))

    if not args.apply:
        print(f"\n(dry run — pass --apply to rewrite {md})")
        return 0

    bak = md.with_suffix(md.suffix + ".bak")
    bak.write_bytes(md.read_bytes())
    print(f"backup written: {bak}")

    new_lines = lines[:start] + [new_toc] + lines[end:]
    md.write_text("".join(new_lines), encoding="utf-8")
    print(f"rewrote {md} ({sum(len(line) for line in new_lines)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
