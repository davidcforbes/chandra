from dataclasses import dataclass
from typing import List

from PIL import Image


@dataclass
class GenerationResult:
    raw: str
    token_count: int
    error: bool = False


@dataclass
class BatchInputItem:
    image: Image.Image
    prompt: str | None = None
    prompt_type: str | None = None
    # Set these when you want chunks to carry stable global chunk_ids of the
    # form "<stem>/<page:04d>/<idx:03d>" (used by the page-worker pipeline).
    # Legacy single-PDF callers leave them None and chunks fall back to the
    # page-local "_/NNN" form.
    file_stem: str | None = None
    page_num: int | None = None


@dataclass
class BatchOutputItem:
    markdown: str
    html: str
    chunks: dict
    raw: str
    page_box: List[int]
    token_count: int
    images: dict
    error: bool
