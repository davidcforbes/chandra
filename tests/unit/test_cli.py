"""Unit tests for chandra.scripts.cli — flag parsing, exit codes, integration.

The CLI now wraps chandra.pipeline.run_pipeline; per-PDF batching helpers
that lived here previously have moved into pipeline.py and manifest.py.
These tests focus on what the CLI itself owns: argument parsing, deprecation
warnings, end-to-end smoke under a mocked InferenceManager, and exit codes.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from chandra.scripts import cli


def _save_png(path: Path, size=(400, 400)) -> Path:
    Image.new("RGB", size, "white").save(path)
    return path


def _build_pdf(path: Path, num_pages: int = 2) -> Path:
    pages = [Image.new("RGB", (200, 200), "white") for _ in range(num_pages)]
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    path.write_bytes(buf.getvalue())
    return path


def _make_fake_result(_input):
    """One synthetic, error-free BatchOutputItem-shaped object."""
    result = MagicMock()
    result.markdown = "# fake"
    result.html = "<p>fake</p>"
    result.chunks = []
    result.images = {}
    result.raw = "<p>fake</p>"
    result.page_box = [0, 0, 100, 100]
    result.token_count = 10
    result.error = False
    return result


@pytest.fixture
def fake_model():
    """Patch chandra.scripts.cli.InferenceManager to a no-op generator."""
    instance = MagicMock()
    instance.generate.side_effect = lambda batch, **kwargs: [
        _make_fake_result(item) for item in batch
    ]
    with patch.object(cli, "InferenceManager", return_value=instance):
        yield instance


# ---------- --version, --help -------------------------------------------


class TestCliBasics:
    def test_version_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli.main, ["--version"])
        assert result.exit_code == 0
        assert "chandra" in result.output.lower()

    def test_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(cli.main, ["--help"])
        assert result.exit_code == 0
        assert "Inference method" in result.output


# ---------- success paths -----------------------------------------------


class TestCliRun:
    def test_image_run_assembles_canonical_outputs(self, tmp_path, fake_model):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "img.png")
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            cli.main, [str(in_dir), str(out_dir), "--method", "hf"]
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "img" / "img.md").exists()
        assert (out_dir / "img" / "chunks.jsonl").exists()
        # .partial removed after assembly.
        assert not (out_dir / "img" / ".partial").exists()

    def test_paginate_output_hyphen_form(self, tmp_path, fake_model):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "img.png")
        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            [str(in_dir), str(out_dir), "--method", "hf", "--paginate-output"],
        )
        assert result.exit_code == 0, result.output

    def test_paginate_output_underscore_alias_still_accepted(
        self, tmp_path, fake_model
    ):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "img.png")
        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            [str(in_dir), str(out_dir), "--method", "hf", "--paginate_output"],
        )
        assert result.exit_code == 0, result.output


# ---------- recursive + glob inputs -------------------------------------


class TestRecursiveAndGlob:
    def test_recursive_walks_subdirs(self, tmp_path, fake_model):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "a.png")
        sub = in_dir / "sub"
        sub.mkdir()
        _save_png(sub / "b.png")
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            [str(in_dir), str(out_dir), "--method", "hf", "--recursive"],
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "a" / "a.md").exists()
        assert (out_dir / "b" / "b.md").exists()

    def test_top_level_only_by_default(self, tmp_path, fake_model):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "a.png")
        sub = in_dir / "sub"
        sub.mkdir()
        _save_png(sub / "b.png")
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            cli.main, [str(in_dir), str(out_dir), "--method", "hf"]
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "a" / "a.md").exists()
        # Subdir file NOT processed without --recursive.
        assert not (out_dir / "b" / "b.md").exists()


# ---------- resume detection --------------------------------------------


class TestResume:
    def test_already_done_book_is_skipped_with_zero_generate_calls(
        self, tmp_path, fake_model
    ):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "img.png")
        out_dir = tmp_path / "out"
        # Pre-create a canonical output so discover_books skips it.
        stem_dir = out_dir / "img"
        stem_dir.mkdir(parents=True)
        (stem_dir / "img.md").write_text("already done")

        runner = CliRunner()
        result = runner.invoke(
            cli.main, [str(in_dir), str(out_dir), "--method", "hf"]
        )
        assert result.exit_code == 0, result.output
        assert "every supported file is already assembled" in result.output
        assert fake_model.generate.call_count == 0


# ---------- exit codes --------------------------------------------------


class TestCliExitCodes:
    def test_failure_returns_nonzero_exit(self, tmp_path):
        """When the model raises on every page, the CLI should exit non-zero
        even though pages are recorded with error=True (book still assembles).
        """
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "img.png")
        out_dir = tmp_path / "out"

        instance = MagicMock()
        instance.generate.side_effect = RuntimeError("boom")
        with patch.object(cli, "InferenceManager", return_value=instance):
            runner = CliRunner()
            result = runner.invoke(
                cli.main, [str(in_dir), str(out_dir), "--method", "hf"]
            )
        assert result.exit_code == 1, result.output
        assert "pages_with_errors" in result.output


# ---------- deprecated flags -------------------------------------------


class TestDeprecatedFlags:
    def test_batch_size_warns_but_runs(self, tmp_path, fake_model):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "img.png")
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            [str(in_dir), str(out_dir), "--method", "hf", "--batch-size", "4"],
        )
        assert result.exit_code == 0, result.output
        assert "--batch-size is deprecated" in result.output

    def test_fail_fast_warns_but_runs(self, tmp_path, fake_model):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "img.png")
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            [str(in_dir), str(out_dir), "--method", "hf", "--fail-fast"],
        )
        assert result.exit_code == 0, result.output
        assert "--fail-fast is deprecated" in result.output

    def test_max_workers_aliases_workers(self, tmp_path, fake_model):
        # --max-workers is the legacy name; --workers takes precedence if
        # both are passed.
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "img.png")
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            [str(in_dir), str(out_dir), "--method", "hf", "--max-workers", "2"],
        )
        assert result.exit_code == 0, result.output
        assert "workers: 2" in result.output


# ---------- pipeline integration through CLI ----------------------------


def test_pdf_pages_each_get_their_own_generate_call(tmp_path, fake_model):
    """Pipeline calls model.generate once per page (single-item batches)."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _build_pdf(in_dir / "doc.pdf", num_pages=4)
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        cli.main, [str(in_dir / "doc.pdf"), str(out_dir), "--method", "hf"]
    )
    assert result.exit_code == 0, result.output
    # Worker pool processes each page individually; one call per page.
    assert fake_model.generate.call_count == 4


def test_extracted_images_are_saved_to_output(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _save_png(in_dir / "img.png")
    out_dir = tmp_path / "out"

    instance = MagicMock()

    def _generate(batch, **_):
        result = MagicMock()
        result.markdown = "# fake"
        result.html = "<p>fake</p>"
        result.chunks = []
        result.images = {"figure_1.webp": Image.new("RGB", (10, 10), "red")}
        result.raw = "<p>fake</p>"
        result.page_box = [0, 0, 100, 100]
        result.token_count = 10
        result.error = False
        return [result]

    instance.generate.side_effect = _generate
    with patch.object(cli, "InferenceManager", return_value=instance):
        runner = CliRunner()
        result = runner.invoke(
            cli.main, [str(in_dir), str(out_dir), "--method", "hf"]
        )
    assert result.exit_code == 0, result.output
    assert (out_dir / "img" / "figure_1.webp").exists()


def test_pdf_page_range_is_one_indexed(tmp_path, fake_model):
    """--page-range 1 means the first page (1-indexed)."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _build_pdf(in_dir / "doc.pdf", num_pages=3)
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            str(in_dir / "doc.pdf"),
            str(out_dir),
            "--method",
            "hf",
            "--page-range",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    # One page in range = exactly one generate call.
    assert fake_model.generate.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
