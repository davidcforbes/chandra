"""Unit tests for chandra.input — page-range parsing and PDF/image streaming."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from chandra.input import (
    count_file_pages,
    count_pdf_pages,
    iter_file_pages,
    iter_pdf_pages,
    load_file,
    load_image,
    load_pdf_images,
    parse_range_str,
)


# ---------- parse_range_str (chandra-1bj, chandra-erw) ----------


class TestParseRangeStr:
    def test_single_page_is_zero_indexed(self):
        # User says "page 1", we return 0-indexed [0].
        assert parse_range_str("1") == [0]

    def test_simple_range(self):
        assert parse_range_str("1-5") == [0, 1, 2, 3, 4]

    def test_compound_input(self):
        assert parse_range_str("1-3,7,9-10") == [0, 1, 2, 6, 8, 9]

    def test_dedupes_and_sorts(self):
        assert parse_range_str("3,1,2,1-2") == [0, 1, 2]

    def test_whitespace_tolerated(self):
        assert parse_range_str(" 1 - 3 , 5 ") == [0, 1, 2, 4]

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="empty"):
            parse_range_str("")

    def test_rejects_whitespace_only(self):
        with pytest.raises(ValueError, match="empty"):
            parse_range_str("   ")

    def test_rejects_none(self):
        with pytest.raises(ValueError):
            parse_range_str(None)  # type: ignore[arg-type]

    def test_rejects_trailing_comma(self):
        with pytest.raises(ValueError, match="empty page-range token"):
            parse_range_str("1,")

    def test_rejects_dangling_dash(self):
        with pytest.raises(ValueError, match="invalid page range"):
            parse_range_str("5-")

    def test_rejects_double_dash(self):
        with pytest.raises(ValueError, match="invalid page range"):
            parse_range_str("1--5")

    def test_rejects_non_integer(self):
        with pytest.raises(ValueError, match="non-integer page"):
            parse_range_str("abc")

    def test_rejects_non_integer_in_range(self):
        with pytest.raises(ValueError, match="non-integer page range"):
            parse_range_str("1-abc")

    def test_rejects_zero_page(self):
        with pytest.raises(ValueError, match=">= 1"):
            parse_range_str("0")

    def test_rejects_negative(self):
        # The leading minus reads as a missing-start range, not a negative int.
        with pytest.raises(ValueError):
            parse_range_str("-1")

    def test_rejects_reversed_range(self):
        with pytest.raises(ValueError, match="reversed"):
            parse_range_str("5-1")

    def test_rejects_zero_in_range(self):
        # Both ends must be >= 1; "0-3" should be rejected.
        with pytest.raises(ValueError, match=">= 1"):
            parse_range_str("0-3")


# ---------- load_image ----------


def _save_temp_png(path: Path, size: tuple[int, int], color: str = "white") -> Path:
    img = Image.new("RGB", size, color)
    img.save(path)
    return path


class TestLoadImage:
    def test_loads_image_at_natural_size(self, tmp_path):
        path = _save_temp_png(tmp_path / "x.png", (2000, 2000))
        img = load_image(str(path))
        assert img.size == (2000, 2000)

    def test_upscales_small_images_to_min_dim(self, tmp_path):
        path = _save_temp_png(tmp_path / "tiny.png", (200, 200))
        img = load_image(str(path), min_image_dim=800)
        assert min(img.size) >= 800


# ---------- iter_pdf_pages / count_pdf_pages ----------


def _build_pdf_in_memory(num_pages: int) -> bytes:
    """Build a tiny multi-page PDF using pillow's PDF backend."""
    if num_pages < 1:
        raise ValueError("num_pages must be >= 1")
    pages = [
        Image.new("RGB", (100 + i, 100 + i), color=(i * 30 % 255, 200, 50))
        for i in range(num_pages)
    ]
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()


@pytest.fixture
def three_page_pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(_build_pdf_in_memory(3))
    return path


class TestPdfStreaming:
    def test_count_pdf_pages(self, three_page_pdf):
        assert count_pdf_pages(str(three_page_pdf)) == 3

    def test_iter_pdf_pages_yields_all_when_no_range(self, three_page_pdf):
        pages = list(iter_pdf_pages(str(three_page_pdf)))
        assert len(pages) == 3
        for p in pages:
            assert isinstance(p, Image.Image)

    def test_iter_pdf_pages_filters_by_indices(self, three_page_pdf):
        pages = list(iter_pdf_pages(str(three_page_pdf), page_range=[0, 2]))
        assert len(pages) == 2

    def test_iter_pdf_pages_is_lazy(self, three_page_pdf):
        # Calling without consuming returns a generator (no rendering yet).
        gen = iter_pdf_pages(str(three_page_pdf))
        # ``next`` advances by one and pulls a page; the rest stay un-rendered.
        first = next(gen)
        assert isinstance(first, Image.Image)
        gen.close()

    def test_load_pdf_images_returns_list(self, three_page_pdf):
        pages = load_pdf_images(str(three_page_pdf))
        assert isinstance(pages, list)
        assert len(pages) == 3


# ---------- iter_file_pages / count_file_pages / load_file ----------


class TestFileDispatch:
    def test_iter_file_pages_pdf(self, three_page_pdf):
        pages = list(iter_file_pages(str(three_page_pdf), config={}))
        assert len(pages) == 3

    def test_iter_file_pages_pdf_with_page_range(self, three_page_pdf):
        # User says "page 1" in 1-indexed CLI; should yield page 0 (the first).
        pages = list(iter_file_pages(str(three_page_pdf), config={"page_range": "1"}))
        assert len(pages) == 1

    def test_iter_file_pages_image(self, tmp_path):
        path = _save_temp_png(tmp_path / "x.png", (300, 300))
        pages = list(iter_file_pages(str(path), config={}))
        assert len(pages) == 1
        assert isinstance(pages[0], Image.Image)

    def test_count_file_pages_pdf(self, three_page_pdf):
        assert count_file_pages(str(three_page_pdf), config={}) == 3

    def test_count_file_pages_pdf_with_range(self, three_page_pdf):
        # "1-2" → 0,1 → 2 pages
        assert count_file_pages(str(three_page_pdf), config={"page_range": "1-2"}) == 2

    def test_count_file_pages_pdf_with_out_of_bounds_range(self, three_page_pdf):
        # "1-10" → 0..9, but only 0,1,2 are valid → 3 pages.
        assert count_file_pages(str(three_page_pdf), config={"page_range": "1-10"}) == 3

    def test_count_file_pages_image(self, tmp_path):
        path = _save_temp_png(tmp_path / "x.png", (200, 200))
        assert count_file_pages(str(path), config={}) == 1

    def test_load_file_eager_list(self, three_page_pdf):
        out = load_file(str(three_page_pdf), config={})
        assert isinstance(out, list)
        assert len(out) == 3


# ---------- regression: 1-indexed CLI input is honoured (chandra-1bj) ----------


def test_page_range_1_loads_first_page(three_page_pdf):
    """Documents the off-by-one fix: --page-range 1 must include page 0."""
    pages = list(iter_file_pages(str(three_page_pdf), config={"page_range": "1"}))
    assert len(pages) == 1


def test_page_range_2_skips_first_page(three_page_pdf):
    pages = list(iter_file_pages(str(three_page_pdf), config={"page_range": "2"}))
    assert len(pages) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
