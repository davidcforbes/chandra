"""Unit tests for chandra.manifest — atomic writes, partial state, assembly."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from chandra.manifest import (
    assemble_book,
    atomic_write_json,
    atomic_write_text,
    page_artifact_path,
    pages_dir,
    partial_dir,
    purge_partial,
    read_partial_state,
    read_state,
    source_matches,
    write_state,
)


# ---------- atomic_write -------------------------------------------------


class TestAtomicWrite:
    def test_writes_text_and_no_tmp_left(self, tmp_path: Path):
        target = tmp_path / "out.txt"
        atomic_write_text(target, "hello\nworld")
        assert target.read_text(encoding="utf-8") == "hello\nworld"
        assert not (target.with_suffix(target.suffix + ".tmp")).exists()

    def test_writes_json(self, tmp_path: Path):
        target = tmp_path / "out.json"
        atomic_write_json(target, {"a": 1, "b": [2, 3]})
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [2, 3]}

    def test_overwrite_atomically(self, tmp_path: Path):
        target = tmp_path / "out.txt"
        atomic_write_text(target, "first")
        atomic_write_text(target, "second")
        assert target.read_text(encoding="utf-8") == "second"

    def test_creates_parent_dirs(self, tmp_path: Path):
        target = tmp_path / "deep" / "nested" / "out.txt"
        atomic_write_text(target, "ok")
        assert target.read_text(encoding="utf-8") == "ok"

    def test_kill_mid_write_leaves_tmp_not_canonical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """If os.replace fails (simulating a crash), .tmp survives but the
        canonical path is never touched — readers see no partial."""
        target = tmp_path / "out.txt"

        def boom(*args, **kwargs):  # noqa: ARG001
            raise OSError("simulated kill")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError, match="simulated kill"):
            atomic_write_text(target, "partial")
        assert not target.exists()
        # The .tmp file remains; resume code treats *.tmp as orphaned and ignores.
        tmp = target.with_suffix(target.suffix + ".tmp")
        assert tmp.exists()
        assert tmp.read_text(encoding="utf-8") == "partial"


# ---------- read_partial_state ------------------------------------------


class TestReadPartialState:
    def test_empty_when_no_partial_dir(self, tmp_path: Path):
        assert read_partial_state(tmp_path) == set()

    def test_returns_int_page_numbers(self, tmp_path: Path):
        pdir = pages_dir(tmp_path)
        pdir.mkdir(parents=True)
        for n in (0, 5, 12, 42):
            (pdir / f"{n:04d}.json").write_text("{}", encoding="utf-8")
        assert read_partial_state(tmp_path) == {0, 5, 12, 42}

    def test_ignores_tmp_files(self, tmp_path: Path):
        pdir = pages_dir(tmp_path)
        pdir.mkdir(parents=True)
        (pdir / "0001.json").write_text("{}", encoding="utf-8")
        (pdir / "0002.json.tmp").write_text("{}", encoding="utf-8")
        assert read_partial_state(tmp_path) == {1}

    def test_ignores_non_int_filenames(self, tmp_path: Path):
        pdir = pages_dir(tmp_path)
        pdir.mkdir(parents=True)
        (pdir / "0001.json").write_text("{}", encoding="utf-8")
        (pdir / "_state.json").write_text("{}", encoding="utf-8")
        (pdir / "weird-name.json").write_text("{}", encoding="utf-8")
        assert read_partial_state(tmp_path) == {1}


# ---------- source fingerprint ------------------------------------------


class TestSourceFingerprint:
    def _make_pdf(
        self, tmp_path: Path, name: str = "test.pdf", size: int = 256
    ) -> Path:
        path = tmp_path / name
        path.write_bytes(b"x" * size)
        return path

    def test_write_and_read_state(self, tmp_path: Path):
        pdf = self._make_pdf(tmp_path)
        stem_dir = tmp_path / "stem"
        write_state(stem_dir, pdf, expected_pages=[0, 1, 2, 3])

        state = read_state(stem_dir)
        assert state is not None
        assert state["source"]["size"] == 256
        assert state["expected_pages"] == [0, 1, 2, 3]

    def test_source_matches_unchanged_pdf(self, tmp_path: Path):
        pdf = self._make_pdf(tmp_path)
        stem_dir = tmp_path / "stem"
        write_state(stem_dir, pdf, expected_pages=[0])
        assert source_matches(stem_dir, pdf) is True

    def test_source_mismatch_on_size_change(self, tmp_path: Path):
        pdf = self._make_pdf(tmp_path, size=100)
        stem_dir = tmp_path / "stem"
        write_state(stem_dir, pdf, expected_pages=[0])
        # Mutate PDF — different size.
        pdf.write_bytes(b"y" * 200)
        assert source_matches(stem_dir, pdf) is False

    def test_source_mismatch_on_mtime_change(self, tmp_path: Path):
        pdf = self._make_pdf(tmp_path)
        stem_dir = tmp_path / "stem"
        write_state(stem_dir, pdf, expected_pages=[0])
        # Push mtime forward by 10s without changing size.
        future = time.time() + 10
        os.utime(pdf, (future, future))
        assert source_matches(stem_dir, pdf) is False

    def test_returns_false_when_no_state(self, tmp_path: Path):
        pdf = self._make_pdf(tmp_path)
        stem_dir = tmp_path / "stem"
        # No write_state called.
        assert source_matches(stem_dir, pdf) is False

    def test_purge_partial_removes_dir(self, tmp_path: Path):
        pdf = self._make_pdf(tmp_path)
        stem_dir = tmp_path / "stem"
        write_state(stem_dir, pdf, expected_pages=[0])
        assert partial_dir(stem_dir).exists()
        purge_partial(stem_dir)
        assert not partial_dir(stem_dir).exists()

    def test_purge_partial_retries_on_transient_permission_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Regression for chandra-xqk: on Windows, freshly-created .partial/
        # dirs sometimes hit WinError 5 from rmtree because of antivirus or
        # Search Indexer briefly holding handles. We retry through it.
        pdf = self._make_pdf(tmp_path)
        stem_dir = tmp_path / "stem"
        write_state(stem_dir, pdf, expected_pages=[0])

        from chandra import manifest as mf

        real_rmtree = mf.shutil.rmtree
        calls = {"n": 0}

        def flaky_rmtree(*a, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError("[WinError 5] Access is denied")
            return real_rmtree(*a, **kw)

        # No real sleeping in the test.
        monkeypatch.setattr(mf.time, "sleep", lambda *_: None)
        monkeypatch.setattr(mf.shutil, "rmtree", flaky_rmtree)

        purge_partial(stem_dir)
        assert calls["n"] == 3
        assert not partial_dir(stem_dir).exists()

    def test_purge_partial_logs_and_returns_when_exhausted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ):
        # If retries don't clear, return without raising — assembly already
        # wrote the canonical artifacts; .partial/ can be cleaned up on the
        # next run.
        pdf = self._make_pdf(tmp_path)
        stem_dir = tmp_path / "stem"
        write_state(stem_dir, pdf, expected_pages=[0])

        from chandra import manifest as mf

        monkeypatch.setattr(mf.time, "sleep", lambda *_: None)
        monkeypatch.setattr(
            mf.shutil,
            "rmtree",
            lambda *a, **kw: (_ for _ in ()).throw(
                PermissionError("[WinError 5] Access is denied")
            ),
        )

        with caplog.at_level("WARNING", logger="chandra.manifest"):
            ok = purge_partial(stem_dir)
        assert ok is False
        assert any("gave up" in r.message for r in caplog.records)
        # Dir still on disk; that's the documented fallback.
        assert partial_dir(stem_dir).exists()

    def test_purge_partial_quiet_suppresses_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ):
        # The opportunistic-cleanup path at startup (quiet=True) should NOT
        # log a warning when the dir is still locked — these orphans are
        # expected on Windows and clear naturally on subsequent runs.
        pdf = self._make_pdf(tmp_path)
        stem_dir = tmp_path / "stem"
        write_state(stem_dir, pdf, expected_pages=[0])

        from chandra import manifest as mf

        monkeypatch.setattr(mf.time, "sleep", lambda *_: None)
        monkeypatch.setattr(
            mf.shutil,
            "rmtree",
            lambda *a, **kw: (_ for _ in ()).throw(
                PermissionError("[WinError 5] Access is denied")
            ),
        )

        with caplog.at_level("WARNING", logger="chandra.manifest"):
            ok = purge_partial(stem_dir, quiet=True)
        assert ok is False
        assert not any("gave up" in r.message for r in caplog.records)

    def test_purge_partial_returns_true_on_success(self, tmp_path: Path):
        pdf = self._make_pdf(tmp_path)
        stem_dir = tmp_path / "stem"
        write_state(stem_dir, pdf, expected_pages=[0])
        assert purge_partial(stem_dir) is True
        # Already gone — second call short-circuits to True.
        assert purge_partial(stem_dir) is True


# ---------- assemble_book ----------------------------------------------


def _write_page(stem_dir: Path, page_num: int, **overrides) -> None:
    """Helper: synthesize a per-page artifact at .partial/pages/NNNN.json."""
    data = {
        "page_num": page_num,
        "raw_html": f"<div>raw {page_num}</div>",
        "markdown": f"# Page {page_num}\n\nBody for page {page_num}.",
        "html_filtered": f"<h1>Page {page_num}</h1><p>Body for page {page_num}.</p>",
        "page_box": [0, 0, 1000, 1000],
        "token_count": 100 + page_num,
        "error": None,
        "chunks": [
            {
                "chunk_id": f"book/{page_num:04d}/000",
                "page": page_num,
                "label": "Section-Header",
                "bbox": [0, 0, 100, 20],
                "content": f"Page {page_num}",
                "image_ref": None,
            }
        ],
        "image_files": [],
    }
    data.update(overrides)
    path = page_artifact_path(stem_dir, page_num)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class TestAssembleBook:
    def _setup_book(
        self, tmp_path: Path, expected_pages: list[int], file_name: str = "book.pdf"
    ) -> Path:
        pdf = tmp_path / file_name
        pdf.write_bytes(b"x" * 256)
        stem_dir = tmp_path / Path(file_name).stem
        write_state(stem_dir, pdf, expected_pages)
        for n in expected_pages:
            _write_page(stem_dir, n)
        return stem_dir

    def test_basic_assembly_writes_canonical(self, tmp_path: Path):
        stem_dir = self._setup_book(tmp_path, [0, 1, 2])
        stats = assemble_book(stem_dir, "book.pdf")
        assert stats == {"num_pages": 3, "total_chunks": 3, "total_images": 0}

        md = (stem_dir / "book.md").read_text(encoding="utf-8")
        assert "# Page 0" in md
        assert "# Page 2" in md

        html = (stem_dir / "book.html").read_text(encoding="utf-8")
        assert "<h1>Page 0</h1>" in html

        meta = json.loads((stem_dir / "book_metadata.json").read_text(encoding="utf-8"))
        assert meta["num_pages"] == 3
        assert meta["total_token_count"] == 100 + 101 + 102

    def test_emits_chunks_jsonl(self, tmp_path: Path):
        stem_dir = self._setup_book(tmp_path, [0, 1, 2])
        assemble_book(stem_dir, "book.pdf")
        chunks_path = stem_dir / "chunks.jsonl"
        assert chunks_path.exists()
        lines = chunks_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first["chunk_id"] == "book/0000/000"
        assert first["page"] == 0
        assert first["label"] == "Section-Header"

    def test_partial_dir_removed_after_commit(self, tmp_path: Path):
        stem_dir = self._setup_book(tmp_path, [0, 1, 2])
        assert partial_dir(stem_dir).exists()
        assemble_book(stem_dir, "book.pdf")
        assert not partial_dir(stem_dir).exists()

    def test_no_tmp_files_left(self, tmp_path: Path):
        stem_dir = self._setup_book(tmp_path, [0, 1])
        assemble_book(stem_dir, "book.pdf")
        for f in stem_dir.iterdir():
            assert not f.name.endswith(".tmp"), f"leftover tmp: {f}"

    def test_save_html_false_skips_html(self, tmp_path: Path):
        stem_dir = self._setup_book(tmp_path, [0, 1])
        assemble_book(stem_dir, "book.pdf", save_html=False)
        assert not (stem_dir / "book.html").exists()
        assert (stem_dir / "book.md").exists()

    def test_paginate_output_inserts_separators(self, tmp_path: Path):
        stem_dir = self._setup_book(tmp_path, [0, 1])
        assemble_book(stem_dir, "book.pdf", paginate_output=True)
        md = (stem_dir / "book.md").read_text(encoding="utf-8")
        # Page-break sentinel uses dashes — see save_merged_output legacy.
        assert "-" * 48 in md

    def test_missing_page_raises(self, tmp_path: Path):
        # Expected pages 0, 1, 2 but only 0, 2 written.
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"x" * 256)
        stem_dir = tmp_path / "book"
        write_state(stem_dir, pdf, [0, 1, 2])
        _write_page(stem_dir, 0)
        _write_page(stem_dir, 2)

        with pytest.raises(FileNotFoundError, match="missing"):
            assemble_book(stem_dir, "book.pdf")

    def test_missing_state_raises(self, tmp_path: Path):
        stem_dir = tmp_path / "book"
        stem_dir.mkdir()
        with pytest.raises(ValueError, match="no .partial"):
            assemble_book(stem_dir, "book.pdf")

    def test_pages_assembled_in_document_order(self, tmp_path: Path):
        # Pages were written in scrambled order to the dir; assembly must
        # still produce them in numerical order.
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"x" * 256)
        stem_dir = tmp_path / "book"
        write_state(stem_dir, pdf, [0, 1, 2, 3])
        for n in [3, 0, 2, 1]:  # write in scrambled order
            _write_page(stem_dir, n)
        assemble_book(stem_dir, "book.pdf")
        md = (stem_dir / "book.md").read_text(encoding="utf-8")
        # Page 0 must appear before page 1 must appear before page 2 etc.
        assert (
            md.find("# Page 0")
            < md.find("# Page 1")
            < md.find("# Page 2")
            < md.find("# Page 3")
        )

    def test_idempotent_re_assemble(self, tmp_path: Path):
        # If we call assemble_book twice (e.g. crash then retry), the result
        # should be the same. After first call, .partial/ is gone, so second
        # call would fail — that's the wrong behavior for a crash mid-rename
        # scenario. But for our model, the Assembler thread checks for
        # canonical artifacts existing before re-assembling. So this test
        # documents the sharp edge: second call WITHOUT .partial/ fails.
        stem_dir = self._setup_book(tmp_path, [0, 1])
        assemble_book(stem_dir, "book.pdf")
        with pytest.raises(ValueError):
            assemble_book(stem_dir, "book.pdf")
