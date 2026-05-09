"""Unit tests for chandra.pipeline — discovery, workers, assembly under mocks.

Avoids touching real PDFs by mocking ``iter_file_pages``, ``count_file_pages``,
and the ``InferenceManager``. Tests focus on:

- ``discover_books`` resolving file/folder/glob/recursive inputs
- Resume detection (canonical present, partial present, source mismatch)
- Worker writing per-page artifacts atomically
- Assembler running on completion
- End-to-end pipeline with synthetic pages
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

import pytest
from PIL import Image

from chandra import manifest, pipeline
from chandra.pipeline import BookSpec, Progress, _process_page, discover_books


# ---------- _find_files / discover_books ---------------------------------


class TestFindFiles:
    def test_single_file(self, tmp_path: Path):
        f = tmp_path / "book.pdf"
        f.write_bytes(b"x")
        assert pipeline._find_files(f, recursive=False) == [f]

    def test_unsupported_file_raises(self, tmp_path: Path):
        f = tmp_path / "book.txt"
        f.write_bytes(b"x")
        with pytest.raises(ValueError, match="Unsupported"):
            pipeline._find_files(f, recursive=False)

    def test_folder_top_level_only(self, tmp_path: Path):
        (tmp_path / "a.pdf").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.pdf").write_bytes(b"x")  # should NOT appear without --recursive

        files = pipeline._find_files(tmp_path, recursive=False)
        names = sorted(f.name for f in files)
        assert names == ["a.pdf", "b.png"]

    def test_folder_recursive(self, tmp_path: Path):
        (tmp_path / "a.pdf").write_bytes(b"x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.pdf").write_bytes(b"x")
        deeper = sub / "deeper"
        deeper.mkdir()
        (deeper / "d.pdf").write_bytes(b"x")

        files = pipeline._find_files(tmp_path, recursive=True)
        names = sorted(f.name for f in files)
        assert names == ["a.pdf", "c.pdf", "d.pdf"]

    def test_glob_pattern(self, tmp_path: Path):
        (tmp_path / "alpha.pdf").write_bytes(b"x")
        (tmp_path / "beta.pdf").write_bytes(b"x")
        (tmp_path / "other.png").write_bytes(b"x")

        files = pipeline._find_files(tmp_path / "*.pdf", recursive=False)
        names = sorted(f.name for f in files)
        assert names == ["alpha.pdf", "beta.pdf"]

    def test_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            pipeline._find_files(tmp_path / "nope.pdf", recursive=False)

    def test_case_insensitive_extension(self, tmp_path: Path):
        # Windows is case-insensitive — make sure we match .PDF too.
        (tmp_path / "shouty.PDF").write_bytes(b"x")
        files = pipeline._find_files(tmp_path, recursive=False)
        assert len(files) == 1
        assert files[0].suffix == ".PDF"


class TestDiscoverBooks:
    def _patch_count(self, monkeypatch, n_pages: int):
        monkeypatch.setattr(pipeline, "count_file_pages", lambda *a, **k: n_pages)

    def test_fresh_book_yields_all_pages_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"x")
        out = tmp_path / "out"
        self._patch_count(monkeypatch, n_pages=10)

        books = discover_books(pdf, out, recursive=False)
        assert len(books) == 1
        b = books[0]
        assert b.stem == "book"
        assert b.total_pages == 10
        assert b.expected_pages == list(range(10))
        assert b.pending_pages == list(range(10))
        assert b.already_done == 0

    def test_already_done_book_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"x")
        out = tmp_path / "out"
        stem_dir = out / "book"
        stem_dir.mkdir(parents=True)
        (stem_dir / "book.md").write_text("existing assembled output")
        self._patch_count(monkeypatch, n_pages=10)

        books = discover_books(pdf, out, recursive=False)
        assert books == []

    def test_resume_skips_done_pages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"x")
        out = tmp_path / "out"
        stem_dir = out / "book"
        self._patch_count(monkeypatch, n_pages=5)

        # Pre-populate state + 3 of the 5 expected pages.
        manifest.write_state(stem_dir, pdf, expected_pages=list(range(5)))
        for n in (0, 1, 3):
            manifest.atomic_write_json(
                manifest.page_artifact_path(stem_dir, n), {"page_num": n}
            )

        books = discover_books(pdf, out, recursive=False)
        assert len(books) == 1
        assert books[0].pending_pages == [2, 4]
        assert books[0].already_done == 3

    def test_source_mismatch_purges_partial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"x" * 100)
        out = tmp_path / "out"
        stem_dir = out / "book"
        # State recorded with the small PDF.
        manifest.write_state(stem_dir, pdf, expected_pages=[0, 1])
        manifest.atomic_write_json(
            manifest.page_artifact_path(stem_dir, 0), {"page_num": 0}
        )

        # Mutate the PDF so size differs.
        pdf.write_bytes(b"y" * 200)
        self._patch_count(monkeypatch, n_pages=2)

        books = discover_books(pdf, out, recursive=False)
        # .partial got purged, so all pages are pending again.
        assert books[0].pending_pages == [0, 1]
        assert not (stem_dir / ".partial" / "pages" / "0000.json").exists()

    def test_page_range_subsets_expected_pages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"x")
        out = tmp_path / "out"
        self._patch_count(monkeypatch, n_pages=10)

        books = discover_books(pdf, out, recursive=False, page_range="1-3,5")
        assert len(books) == 1
        # parse_range_str converts 1-indexed to 0-indexed.
        assert books[0].expected_pages == [0, 1, 2, 4]


# ---------- Progress -----------------------------------------------------


class TestProgress:
    def test_register_and_increment(self, capsys: pytest.CaptureFixture):
        p = Progress()
        p.register_book("book", total=3, already_done=0)
        p.page_done("book", page_num=0, queue_size=2)
        p.page_done("book", page_num=1, queue_size=1)
        out = capsys.readouterr().out
        assert "1/3" in out
        assert "2/3" in out
        assert "p0000" in out
        assert "p0001" in out

    def test_log_every_throttles_output(self, capsys: pytest.CaptureFixture):
        p = Progress(log_every=5)
        p.register_book("book", total=12, already_done=0)
        for i in range(11):
            p.page_done("book", page_num=i)
        out = capsys.readouterr().out
        # Lines appear at completions 5 and 10 — both intermediate.
        # 11 doesn't trigger (not a multiple of 5, not the total).
        lines = [ln for ln in out.splitlines() if "p0" in ln]
        assert len(lines) == 2

    def test_book_assembled_prints_summary(self, capsys: pytest.CaptureFixture):
        p = Progress()
        p.book_assembled("book", num_pages=5, num_chunks=42)
        out = capsys.readouterr().out
        assert "assembled" in out
        assert "5 pages" in out
        assert "42 chunks" in out


# ---------- _process_page ------------------------------------------------


def _fake_batch_output(stem: str, page_num: int, n_chunks: int = 2):
    """Synthetic BatchOutputItem-shaped object."""
    output = MagicMock()
    output.markdown = f"# Page {page_num}\nbody"
    output.html = f"<h1>Page {page_num}</h1><p>body</p>"
    output.raw = f"<div>raw {page_num}</div>"
    output.page_box = [0, 0, 100, 100]
    output.token_count = 100 + page_num
    output.images = {}
    output.error = False
    output.chunks = [
        {
            "label": "Section-Header",
            "bbox": [0, 0, 100, 20],
            "content": f"Page {page_num}",
            "chunk_id": f"{stem}/{page_num:04d}/{i:03d}",
            "page": page_num,
            "image_ref": None,
        }
        for i in range(n_chunks)
    ]
    return output


class TestProcessPage:
    def _make_book(self, tmp_path: Path, stem: str = "b") -> BookSpec:
        pdf = tmp_path / f"{stem}.pdf"
        pdf.write_bytes(b"x")
        stem_dir = tmp_path / "out" / stem
        stem_dir.mkdir(parents=True)
        manifest.write_state(stem_dir, pdf, expected_pages=[0, 1, 2])
        return BookSpec(
            source_path=pdf,
            stem=stem,
            stem_dir=stem_dir,
            total_pages=3,
            expected_pages=[0, 1, 2],
            pending_pages=[0, 1, 2],
        )

    def test_writes_per_page_artifact_on_success(self, tmp_path: Path):
        book = self._make_book(tmp_path)
        model = MagicMock()
        model.generate.return_value = [_fake_batch_output("b", 1)]

        ok = _process_page(book, 1, Image.new("RGB", (50, 50)), model, {})
        assert ok is True

        artifact = json.loads(
            manifest.page_artifact_path(book.stem_dir, 1).read_text(encoding="utf-8")
        )
        assert artifact["page_num"] == 1
        assert artifact["error"] is False
        assert "<h1>Page 1</h1>" in artifact["html_filtered"]
        assert len(artifact["chunks"]) == 2

    def test_records_error_when_model_raises(self, tmp_path: Path):
        book = self._make_book(tmp_path)
        model = MagicMock()
        model.generate.side_effect = RuntimeError("boom")

        ok = _process_page(book, 0, Image.new("RGB", (50, 50)), model, {})
        assert ok is False

        artifact = json.loads(
            manifest.page_artifact_path(book.stem_dir, 0).read_text(encoding="utf-8")
        )
        assert artifact["error"] is True
        assert "boom" in artifact["error_message"]
        # Page is still committed so book can complete despite the failure.
        assert artifact["chunks"] == []

    def test_records_error_when_model_returns_error_result(self, tmp_path: Path):
        book = self._make_book(tmp_path)
        bad = _fake_batch_output("b", 0)
        bad.error = True
        model = MagicMock()
        model.generate.return_value = [bad]

        ok = _process_page(book, 0, Image.new("RGB", (50, 50)), model, {})
        assert ok is False

    def test_image_files_saved_to_canonical_location(self, tmp_path: Path):
        book = self._make_book(tmp_path)
        result = _fake_batch_output("b", 2)
        result.images = {"abc_2_img.webp": Image.new("RGB", (10, 10), "red")}
        model = MagicMock()
        model.generate.return_value = [result]

        _process_page(book, 2, Image.new("RGB", (50, 50)), model, {})

        expected = book.stem_dir / "abc_2_img.webp"
        assert expected.exists()
        artifact = json.loads(
            manifest.page_artifact_path(book.stem_dir, 2).read_text(encoding="utf-8")
        )
        assert "abc_2_img.webp" in artifact["image_files"]


# ---------- end-to-end run_pipeline --------------------------------------


def _yield_synthetic_pages(n: int) -> Iterator[Image.Image]:
    """Tiny synthetic PIL pages — what iter_file_pages would produce."""
    for _ in range(n):
        yield Image.new("RGB", (100, 100), "white")


class TestRunPipeline:
    def _setup(self, tmp_path: Path, n_pages: int = 4) -> BookSpec:
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"x")
        stem_dir = tmp_path / "out" / "book"
        stem_dir.mkdir(parents=True)
        manifest.write_state(stem_dir, pdf, expected_pages=list(range(n_pages)))
        return BookSpec(
            source_path=pdf,
            stem="book",
            stem_dir=stem_dir,
            total_pages=n_pages,
            expected_pages=list(range(n_pages)),
            pending_pages=list(range(n_pages)),
        )

    def test_single_book_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        book = self._setup(tmp_path, n_pages=4)

        # Mock iter_file_pages so the producer "renders" synthetic pages.
        monkeypatch.setattr(
            pipeline,
            "iter_file_pages",
            lambda *a, **k: _yield_synthetic_pages(4),
        )

        model = MagicMock()
        model.generate.side_effect = lambda batch, **_: [
            _fake_batch_output("book", batch[0].page_num) for _ in batch
        ]

        stats = pipeline.run_pipeline([book], model=model, n_workers=2, log_every=10)
        assert stats["books"] == 1
        assert stats["pages_processed"] == 4
        assert stats["pages_pending"] == 0

        # Canonical artifacts present, .partial gone.
        assert (book.stem_dir / "book.md").exists()
        assert (book.stem_dir / "book.html").exists()
        assert (book.stem_dir / "book_metadata.json").exists()
        assert (book.stem_dir / "chunks.jsonl").exists()
        assert not manifest.partial_dir(book.stem_dir).exists()

        # Markdown contains all pages in order.
        md = (book.stem_dir / "book.md").read_text(encoding="utf-8")
        for i in range(4):
            assert f"# Page {i}" in md
        assert md.find("# Page 0") < md.find("# Page 3")

    def test_multi_book_pool_completes_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        a = self._setup(tmp_path, n_pages=3)
        # Second book in a separate tmp_path-like dir.
        pdf_b = tmp_path / "other.pdf"
        pdf_b.write_bytes(b"y")
        stem_b = tmp_path / "out" / "other"
        stem_b.mkdir(parents=True)
        manifest.write_state(stem_b, pdf_b, expected_pages=[0, 1])
        b = BookSpec(
            source_path=pdf_b,
            stem="other",
            stem_dir=stem_b,
            total_pages=2,
            expected_pages=[0, 1],
            pending_pages=[0, 1],
        )

        # Each call to iter_file_pages yields the right count for its book.
        def fake_iter(path, *_, **__):
            n = 3 if "book.pdf" in str(path) else 2
            return _yield_synthetic_pages(n)

        monkeypatch.setattr(pipeline, "iter_file_pages", fake_iter)

        model = MagicMock()

        def gen(batch, **_):
            stem = batch[0].file_stem
            page = batch[0].page_num
            return [_fake_batch_output(stem, page)]

        model.generate.side_effect = gen

        pipeline.run_pipeline([a, b], model=model, n_workers=4, log_every=10)

        assert (a.stem_dir / "book.md").exists()
        assert (b.stem_dir / "other.md").exists()
        assert not manifest.partial_dir(a.stem_dir).exists()
        assert not manifest.partial_dir(b.stem_dir).exists()

    def test_resume_only_processes_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        book = self._setup(tmp_path, n_pages=4)
        # Pre-populate pages 0 and 2 as already done.
        for n in (0, 2):
            manifest.atomic_write_json(
                manifest.page_artifact_path(book.stem_dir, n),
                {
                    "page_num": n,
                    "raw_html": "",
                    "markdown": f"# Page {n}",
                    "html_filtered": f"<h1>Page {n}</h1>",
                    "page_box": [0, 0, 100, 100],
                    "token_count": 50,
                    "error": False,
                    "error_message": None,
                    "chunks": [],
                    "image_files": [],
                },
            )
        book.pending_pages = [1, 3]  # what discover_books would have computed

        # Producer should only render the 2 pending pages.
        monkeypatch.setattr(
            pipeline, "iter_file_pages", lambda *a, **k: _yield_synthetic_pages(2)
        )

        seen_pages: list[int] = []

        def gen(batch, **_):
            seen_pages.append(batch[0].page_num)
            return [_fake_batch_output("book", batch[0].page_num)]

        model = MagicMock()
        model.generate.side_effect = gen

        pipeline.run_pipeline([book], model=model, n_workers=2, log_every=10)

        # Only pending pages were OCRed.
        assert sorted(seen_pages) == [1, 3]
        # All four pages appear in the merged output.
        md = (book.stem_dir / "book.md").read_text(encoding="utf-8")
        for i in range(4):
            assert f"# Page {i}" in md

    def test_no_books_returns_empty_stats(self):
        stats = pipeline.run_pipeline([], model=MagicMock(), n_workers=2)
        assert stats["books"] == 0


# ---------- _assembler thread directly -----------------------------------


class TestAssembler:
    def test_assembles_when_pages_arrive(self, tmp_path: Path):
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"x")
        stem_dir = tmp_path / "out" / "book"
        stem_dir.mkdir(parents=True)
        manifest.write_state(stem_dir, pdf, expected_pages=[0, 1])
        book = BookSpec(
            source_path=pdf,
            stem="book",
            stem_dir=stem_dir,
            total_pages=2,
            expected_pages=[0, 1],
            pending_pages=[0, 1],
        )

        # Drop pages onto disk in a separate thread to simulate workers.
        def write_pages():
            time.sleep(0.05)
            for n in (0, 1):
                manifest.atomic_write_json(
                    manifest.page_artifact_path(stem_dir, n),
                    {
                        "page_num": n,
                        "markdown": f"page {n}",
                        "html_filtered": f"<p>page {n}</p>",
                        "page_box": [0, 0, 100, 100],
                        "token_count": 0,
                        "error": False,
                        "chunks": [],
                        "image_files": [],
                        "raw_html": "",
                    },
                )
                time.sleep(0.02)

        stop = threading.Event()
        progress = Progress(log_every=10)
        writer = threading.Thread(target=write_pages)
        assembler = threading.Thread(
            target=pipeline._assembler,
            args=([book], progress, False, True, stop, 0.05),
        )
        writer.start()
        assembler.start()
        writer.join(timeout=5)
        assembler.join(timeout=5)

        assert (stem_dir / "book.md").exists()
        assert not manifest.partial_dir(stem_dir).exists()
