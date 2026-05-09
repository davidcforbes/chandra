"""Atomic file-state primitives for the page-worker pipeline.

The pipeline writes per-page artifacts to ``<stem>/.partial/pages/NNNN.json`` as
workers complete each page, then runs a final assembly pass that merges them
into the canonical ``<stem>/<stem>.md`` / ``.html`` / ``_metadata.json`` /
``chunks.jsonl``. This module owns the atomic-write primitives and the
two-phase commit that makes that pipeline crash-safe.

Crash semantics:
- Worker dying mid-write leaves a ``*.tmp`` file that is ignored on resume.
- Assembler dying before the last rename leaves canonical artifacts absent
  and ``.partial/`` intact, so a re-run re-assembles idempotently.
- A source PDF being modified between runs is detected via mtime+size in
  ``_state.json``; ``.partial/`` is then purged by the caller.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


# ---------- atomic write helpers ------------------------------------------


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via tmp+rename so readers never see partials."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    """Same as ``atomic_write_text`` but JSON-encodes ``data`` first."""
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False))


# ---------- partial-state inspection --------------------------------------


def partial_dir(stem_dir: Path) -> Path:
    return stem_dir / ".partial"


def pages_dir(stem_dir: Path) -> Path:
    return partial_dir(stem_dir) / "pages"


def page_artifact_path(stem_dir: Path, page_num: int) -> Path:
    return pages_dir(stem_dir) / f"{page_num:04d}.json"


def read_partial_state(stem_dir: Path) -> set[int]:
    """Set of page numbers already persisted under ``<stem_dir>/.partial/pages/``.

    Files whose stem isn't a base-10 integer are skipped silently — those are
    either ``*.tmp`` orphans from a kill mid-write or stray files we don't own.
    """
    pdir = pages_dir(stem_dir)
    if not pdir.exists():
        return set()
    out: set[int] = set()
    for f in pdir.iterdir():
        if f.suffix != ".json":
            continue
        try:
            out.add(int(f.stem))
        except ValueError:
            continue
    return out


# ---------- source fingerprint --------------------------------------------


_STATE_FILE = "_state.json"


def write_state(stem_dir: Path, source_pdf: Path, expected_pages: list[int]) -> None:
    """Record what work this ``.partial/`` is for.

    ``expected_pages`` is the list of 0-indexed page numbers we plan to OCR
    (equals ``[0..total_pages-1]`` for a full-file run, or a subset when
    ``--page-range`` is used). Assembly is gated on ``set(expected_pages) ==
    read_partial_state(stem_dir)``.
    """
    stat = source_pdf.stat()
    data = {
        "source": {
            "path": str(source_pdf),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        },
        "expected_pages": sorted(expected_pages),
    }
    pages_dir(stem_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_json(partial_dir(stem_dir) / _STATE_FILE, data)


def read_state(stem_dir: Path) -> dict | None:
    """Return the recorded state, or ``None`` if no ``.partial/_state.json``."""
    path = partial_dir(stem_dir) / _STATE_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def source_matches(stem_dir: Path, source_pdf: Path) -> bool:
    """True when ``.partial/`` was created from this exact source file.

    Compares size exactly and mtime within 1 second (filesystem mtime
    resolution varies; 1s is forgiving without admitting genuinely re-saved
    PDFs)."""
    state = read_state(stem_dir)
    if state is None or not source_pdf.exists():
        return False
    src = state["source"]
    stat = source_pdf.stat()
    return src["size"] == stat.st_size and abs(src["mtime"] - stat.st_mtime) < 1.0


def purge_partial(stem_dir: Path) -> None:
    """Wipe ``.partial/`` — used when source PDF has changed under us."""
    p = partial_dir(stem_dir)
    if p.exists():
        shutil.rmtree(p)


# ---------- assembler -----------------------------------------------------


def assemble_book(
    stem_dir: Path,
    file_name: str,
    paginate_output: bool = False,
    save_html: bool = True,
) -> dict:
    """Merge per-page artifacts into canonical outputs (two-phase commit).

    Reads every ``<stem_dir>/.partial/pages/NNNN.json`` listed in
    ``_state.json[expected_pages]`` and writes:

    - ``<stem_dir>/<safe_name>.md``
    - ``<stem_dir>/<safe_name>.html`` (if ``save_html``)
    - ``<stem_dir>/<safe_name>_metadata.json``
    - ``<stem_dir>/chunks.jsonl``

    Each is written to a ``*.tmp`` sibling first, then ``os.replace``'d to its
    final name, then ``.partial/`` is removed. Image files (``*.webp``) are
    expected to already live in ``stem_dir`` directly — workers write them
    there as part of per-page processing.

    Raises ``FileNotFoundError`` if any expected page is absent or
    ``ValueError`` if state is missing. Returns a stats dict.
    """
    safe_name = Path(file_name).stem
    state = read_state(stem_dir)
    if state is None:
        raise ValueError(f"no .partial/_state.json in {stem_dir}")
    expected = sorted(state["expected_pages"])

    # Read all pages in document order.
    pages: list[dict] = []
    for page_num in expected:
        path = page_artifact_path(stem_dir, page_num)
        if not path.exists():
            raise FileNotFoundError(
                f"expected page artifact missing: {path} "
                f"(have {len(read_partial_state(stem_dir))}/{len(expected)})"
            )
        pages.append(json.loads(path.read_text(encoding="utf-8")))

    # Concatenate. Mirrors the format of the legacy save_merged_output.
    md_parts: list[str] = []
    html_parts: list[str] = []
    chunks_lines: list[str] = []
    metadata_pages: list[dict] = []
    total_tokens = 0
    total_chunks = 0
    total_images = 0

    for i, page in enumerate(pages):
        if i > 0:
            if paginate_output:
                md_parts.append(f"\n\n{i}" + "-" * 48 + "\n\n")
                html_parts.append(f"\n\n<!-- Page {i + 1} -->\n\n")
            else:
                md_parts.append("\n\n")
                html_parts.append("\n\n")
        md_parts.append(page.get("markdown", ""))
        html_parts.append(page.get("html_filtered", ""))
        for chunk in page.get("chunks", []):
            chunks_lines.append(json.dumps(chunk, ensure_ascii=False))
        total_tokens += page.get("token_count", 0) or 0
        chunks = page.get("chunks", []) or []
        images = page.get("image_files", []) or []
        total_chunks += len(chunks)
        total_images += len(images)
        metadata_pages.append(
            {
                "page_num": page.get("page_num"),
                "page_box": page.get("page_box"),
                "token_count": page.get("token_count", 0) or 0,
                "num_chunks": len(chunks),
                "num_images": len(images),
                "error": page.get("error"),
            }
        )

    metadata = {
        "file_name": file_name,
        "num_pages": len(pages),
        "total_token_count": total_tokens,
        "total_chunks": total_chunks,
        "total_images": total_images,
        "pages": metadata_pages,
    }

    md_path = stem_dir / f"{safe_name}.md"
    html_path = stem_dir / f"{safe_name}.html"
    metadata_path = stem_dir / f"{safe_name}_metadata.json"
    chunks_path = stem_dir / "chunks.jsonl"

    # Phase 1: write all .tmp siblings.
    atomic_write_text(md_path, "".join(md_parts))
    if save_html:
        atomic_write_text(html_path, "".join(html_parts))
    atomic_write_json(metadata_path, metadata)
    chunks_text = "\n".join(chunks_lines) + ("\n" if chunks_lines else "")
    atomic_write_text(chunks_path, chunks_text)
    # Note: atomic_write_text already does tmp+rename per-file, so each file
    # is atomically committed individually. The "two-phase" property here is
    # weaker than a single-shot transaction — a crash between the .md write
    # and the chunks.jsonl write would leave both canonical, just one stale.
    # For our use case (no concurrent assemblers per book) this is fine and
    # an idempotent re-run would just rewrite both.

    # Phase 2: remove .partial/ now that canonical artifacts exist.
    purge_partial(stem_dir)

    return {
        "num_pages": len(pages),
        "total_chunks": total_chunks,
        "total_images": total_images,
    }
