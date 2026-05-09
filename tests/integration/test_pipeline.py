"""Integration tests for the chandra page-worker pipeline.

These exercise the full path from PIL image through InferenceManager (HF
backend) through the worker pool and assembler — only the model itself is
real, the producer is the in-process pipeline. Fast variants use 1 page;
the resume variant uses 2 pages.

These import the HF-backend InferenceManager and therefore require GPU
access for non-trivial inputs. The simple_text_image fixture comes from
tests/conftest.py.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from chandra import manifest
from chandra.model import InferenceManager
from chandra.pipeline import discover_books, run_pipeline


def _make_pdf(path: Path, *images: Image.Image) -> Path:
    buf = io.BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=list(images[1:]))
    path.write_bytes(buf.getvalue())
    return path


def test_pipeline_end_to_end_on_synthetic_pdf(simple_text_image, tmp_path):
    """One-PDF full-pipeline run: discover → render → OCR → assemble."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    pdf = _make_pdf(in_dir / "book.pdf", simple_text_image)
    out_dir = tmp_path / "out"

    books = discover_books(pdf, out_dir, recursive=False)
    assert len(books) == 1
    assert books[0].pending_pages == [0]

    model = InferenceManager(method="hf")
    stats = run_pipeline(
        books,
        model=model,
        n_workers=1,
        generate_kwargs={"max_output_tokens": 128},
    )

    assert stats["books"] == 1
    assert stats["pages_pending"] == 0

    stem_dir = out_dir / "book"
    md = (stem_dir / "book.md").read_text(encoding="utf-8")
    assert "Hello, World!" in md
    # chunks.jsonl exists; at least one chunk has the expected ID format.
    chunks = (stem_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(chunks) >= 1
    assert "book/0000/" in chunks[0]
    # .partial gone after assembly.
    assert not manifest.partial_dir(stem_dir).exists()


def test_pipeline_resumes_from_partial_state(simple_text_image, tmp_path):
    """Pre-populate one page in .partial/, run pipeline, verify only the
    missing page was generated and the final assembled output covers both."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    pdf = _make_pdf(in_dir / "book.pdf", simple_text_image, simple_text_image)
    out_dir = tmp_path / "out"

    # First-pass discover — this writes .partial/_state.json for both pages.
    books = discover_books(pdf, out_dir, recursive=False)
    assert books[0].expected_pages == [0, 1]

    stem_dir = out_dir / "book"
    # Pre-populate page 0 as if a prior run completed it.
    manifest.atomic_write_json(
        manifest.page_artifact_path(stem_dir, 0),
        {
            "page_num": 0,
            "raw_html": "",
            "markdown": "# Pre-populated page 0",
            "html_filtered": "<h1>Pre-populated page 0</h1>",
            "page_box": [0, 0, 100, 100],
            "token_count": 5,
            "error": False,
            "error_message": None,
            "chunks": [],
            "image_files": [],
        },
    )

    # Re-discover — should now show only page 1 pending.
    books = discover_books(pdf, out_dir, recursive=False)
    assert books[0].pending_pages == [1]
    assert books[0].already_done == 1

    model = InferenceManager(method="hf")
    run_pipeline(
        books,
        model=model,
        n_workers=1,
        generate_kwargs={"max_output_tokens": 128},
    )

    md = (stem_dir / "book.md").read_text(encoding="utf-8")
    # Page 0's pre-populated content survived through assembly.
    assert "Pre-populated page 0" in md
    # Page 1 was OCR'd and contains the model's output.
    assert "Hello, World!" in md
    assert not manifest.partial_dir(stem_dir).exists()
