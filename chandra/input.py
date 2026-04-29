from __future__ import annotations

import logging
from typing import Iterable, Iterator, List, Optional

import filetype
from PIL import Image
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from chandra.settings import settings

logger = logging.getLogger(__name__)


def flatten(page, flag=pdfium_c.FLAT_NORMALDISPLAY):
    rc = pdfium_c.FPDFPage_Flatten(page, flag)
    if rc == pdfium_c.FLATTEN_FAIL:
        logger.warning("Failed to flatten annotations / form fields on page %s", page)


def load_image(
    filepath: str, min_image_dim: int = settings.MIN_IMAGE_DIM
) -> Image.Image:
    image = Image.open(filepath).convert("RGB")
    if image.width < min_image_dim or image.height < min_image_dim:
        scale = min_image_dim / min(image.width, image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def count_pdf_pages(filepath: str) -> int:
    """Return the total page count without rendering any pages."""
    doc = pdfium.PdfDocument(filepath)
    try:
        return len(doc)
    finally:
        doc.close()


def iter_pdf_pages(
    filepath: str,
    page_range: Optional[Iterable[int]] = None,
    image_dpi: int = settings.IMAGE_DPI,
    min_pdf_image_dim: int = settings.MIN_PDF_IMAGE_DIM,
) -> Iterator[Image.Image]:
    """Yield rendered PIL images one page at a time so peak memory is bounded.

    ``page_range`` is interpreted as a collection of **0-indexed** page indices
    (use :func:`parse_range_str` to convert from 1-indexed user input). When
    ``page_range`` is falsy, every page is yielded.
    """
    page_set = set(page_range) if page_range else None
    doc = pdfium.PdfDocument(filepath)
    try:
        doc.init_forms()
        for page_index in range(len(doc)):
            if page_set is not None and page_index not in page_set:
                continue
            page_obj = doc[page_index]
            min_page_dim = min(page_obj.get_width(), page_obj.get_height())
            scale_dpi = max((min_pdf_image_dim / min_page_dim) * 72, image_dpi)
            flatten(page_obj)
            yield page_obj.render(scale=scale_dpi / 72).to_pil().convert("RGB")
    finally:
        doc.close()


def load_pdf_images(
    filepath: str,
    page_range: Optional[Iterable[int]] = None,
    image_dpi: int = settings.IMAGE_DPI,
    min_pdf_image_dim: int = settings.MIN_PDF_IMAGE_DIM,
) -> List[Image.Image]:
    """Eager wrapper around :func:`iter_pdf_pages` for callers that need a list."""
    return list(
        iter_pdf_pages(
            filepath,
            page_range=page_range,
            image_dpi=image_dpi,
            min_pdf_image_dim=min_pdf_image_dim,
        )
    )


def parse_range_str(range_str: str) -> List[int]:
    """Parse a 1-indexed page-range string into sorted 0-indexed page indices.

    Accepts strings like ``"1-5,7,9-12"``. Inclusive ranges; user-facing pages
    are 1-indexed (the first page of a document is page 1).

    Returns the sorted, deduplicated list of **0-indexed** page indices, ready
    to feed into :func:`iter_pdf_pages` / :func:`load_pdf_images`.

    Raises ``ValueError`` (with a human-readable message) for empty input,
    non-integer tokens, reversed ranges, or page numbers ≤ 0.
    """
    if range_str is None or not range_str.strip():
        raise ValueError("page range is empty")

    pages: set[int] = set()
    for raw_token in range_str.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"empty page-range token in {range_str!r}")

        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"invalid page range {token!r}")
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise ValueError(f"non-integer page range {token!r}: {exc}") from exc
            if start < 1 or end < 1:
                raise ValueError(f"page numbers must be >= 1 (got {token!r})")
            if start > end:
                raise ValueError(f"reversed page range {token!r} (start > end)")
            pages.update(range(start, end + 1))
        else:
            try:
                page = int(token)
            except ValueError as exc:
                raise ValueError(f"non-integer page {token!r}: {exc}") from exc
            if page < 1:
                raise ValueError(f"page numbers must be >= 1 (got {page})")
            pages.add(page)

    # User-facing 1-indexed → pdfium 0-indexed.
    return sorted(p - 1 for p in pages)


def _is_pdf(filepath: str) -> bool:
    input_type = filetype.guess(filepath)
    return bool(input_type and input_type.extension == "pdf")


def iter_file_pages(filepath: str, config: dict) -> Iterator[Image.Image]:
    """Yield pages from ``filepath`` lazily.

    For PDFs this streams from :func:`iter_pdf_pages`; for image files it
    yields exactly one rendered image. ``config`` may contain ``page_range``
    (a 1-indexed string parsed by :func:`parse_range_str`).
    """
    page_range = config.get("page_range")
    page_indices = parse_range_str(page_range) if page_range else None

    if _is_pdf(filepath):
        yield from iter_pdf_pages(filepath, page_indices)
    else:
        yield load_image(filepath)


def count_file_pages(filepath: str, config: dict) -> int:
    """Return the number of pages that :func:`iter_file_pages` will yield.

    Counts without rendering, so it's safe to call before deciding how to
    batch / progress-report.
    """
    page_range = config.get("page_range")
    page_indices = parse_range_str(page_range) if page_range else None

    if _is_pdf(filepath):
        total = count_pdf_pages(filepath)
        if page_indices is None:
            return total
        return sum(1 for p in page_indices if 0 <= p < total)
    return 1


def load_file(filepath: str, config: dict) -> List[Image.Image]:
    """Eager wrapper around :func:`iter_file_pages` for backwards compatibility."""
    return list(iter_file_pages(filepath, config))
