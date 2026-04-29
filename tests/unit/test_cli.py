"""Unit tests for chandra.scripts.cli — flag parsing, exit codes, batching."""

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


@pytest.fixture
def fake_model_class():
    """Patch ``InferenceManager`` to a no-op that returns 1 successful result per page."""

    def _make_result(_input):
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

    instance = MagicMock()
    instance.generate.side_effect = lambda batch, **kwargs: [
        _make_result(item) for item in batch
    ]

    with patch.object(cli, "InferenceManager", return_value=instance) as cls:
        yield cls, instance


# ---------- _chunked helper ----------


class TestChunked:
    def test_yields_chunks_of_correct_size(self):
        out = list(cli._chunked(range(10), 3))
        assert out == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_empty_iterable(self):
        assert list(cli._chunked([], 3)) == []

    def test_chunk_size_larger_than_input(self):
        assert list(cli._chunked([1, 2], 10)) == [[1, 2]]


# ---------- get_supported_files ----------


class TestGetSupportedFiles:
    def test_single_image_file(self, tmp_path):
        path = _save_png(tmp_path / "x.png")
        assert cli.get_supported_files(path) == [path]

    def test_directory_returns_sorted_list(self, tmp_path):
        a = _save_png(tmp_path / "a.png")
        b = _save_png(tmp_path / "b.png")
        out = cli.get_supported_files(tmp_path)
        assert sorted(out) == sorted([a, b])

    def test_unsupported_file_raises(self, tmp_path):
        bad = tmp_path / "x.docx"
        bad.write_text("no")
        import click

        with pytest.raises(click.BadParameter):
            cli.get_supported_files(bad)

    def test_missing_path_raises(self, tmp_path):
        import click

        with pytest.raises(click.BadParameter):
            cli.get_supported_files(tmp_path / "ghost")


# ---------- CLI integration: --version, --help (chandra-3o2) ----------


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


# ---------- CLI integration: success path ----------


class TestCliRun:
    def test_successful_image_run(self, tmp_path, fake_model_class):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "img.png")
        out_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            [str(in_dir), str(out_dir), "--method", "hf"],
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "img" / "img.md").exists()

    def test_paginate_output_hyphen_form(self, tmp_path, fake_model_class):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "img.png")
        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            [
                str(in_dir),
                str(out_dir),
                "--method",
                "hf",
                "--paginate-output",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_paginate_output_underscore_alias_still_accepted(
        self, tmp_path, fake_model_class
    ):
        # Backwards-compat for the legacy --paginate_output spelling.
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "img.png")
        out_dir = tmp_path / "out"
        runner = CliRunner()
        result = runner.invoke(
            cli.main,
            [
                str(in_dir),
                str(out_dir),
                "--method",
                "hf",
                "--paginate_output",
            ],
        )
        assert result.exit_code == 0, result.output


# ---------- CLI integration: failure exit code (chandra-x1n) ----------


class TestCliExitCodes:
    def test_failure_returns_nonzero_exit(self, tmp_path):
        """When the model raises, the CLI should exit non-zero."""
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
        assert result.exit_code == 1
        assert "1 failed" in result.output

    def test_partial_failure_with_fail_fast(self, tmp_path):
        """--fail-fast aborts on the first failure but still exits non-zero."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _save_png(in_dir / "a.png")
        _save_png(in_dir / "b.png")
        out_dir = tmp_path / "out"

        instance = MagicMock()
        instance.generate.side_effect = RuntimeError("boom")
        with patch.object(cli, "InferenceManager", return_value=instance):
            runner = CliRunner()
            result = runner.invoke(
                cli.main,
                [str(in_dir), str(out_dir), "--method", "hf", "--fail-fast"],
            )
        assert result.exit_code == 1


# ---------- CLI integration: streaming + batching (chandra-2wx) ----------


def test_pdf_streamed_in_batches(tmp_path, fake_model_class):
    """Multi-page PDF should be batched without holding all pages at once."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    pdf = _build_pdf(in_dir / "doc.pdf", num_pages=4)
    out_dir = tmp_path / "out"

    cls, instance = fake_model_class
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            str(pdf),
            str(out_dir),
            "--method",
            "hf",
            "--batch-size",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    # generate() is called once per batch; batch_size=2 over 4 pages = 2 calls.
    assert instance.generate.call_count == 2


def test_extracted_images_are_saved_to_output(tmp_path):
    """When the model returns images, they should land in the output dir."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _save_png(in_dir / "img.png")
    out_dir = tmp_path / "out"

    instance = MagicMock()

    def _generate(_batch, **_):
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
            cli.main,
            [str(in_dir), str(out_dir), "--method", "hf"],
        )
    assert result.exit_code == 0, result.output
    assert (out_dir / "img" / "figure_1.webp").exists()


def test_no_images_flag_skips_image_save(tmp_path, fake_model_class):
    """--no-images should suppress image extraction in save_merged_output."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _save_png(in_dir / "img.png")
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [str(in_dir), str(out_dir), "--method", "hf", "--no-images"],
    )
    assert result.exit_code == 0, result.output


def test_pdf_page_range_is_one_indexed(tmp_path, fake_model_class):
    """Smoke test for chandra-1bj at the CLI layer: --page-range 1 → first page."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _build_pdf(in_dir / "doc.pdf", num_pages=3)
    out_dir = tmp_path / "out"

    cls, instance = fake_model_class
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
    # One page processed = one call to generate with one item.
    assert instance.generate.call_count == 1
    first_call_args, _ = instance.generate.call_args_list[0]
    assert len(first_call_args[0]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
