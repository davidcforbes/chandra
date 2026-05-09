import hashlib
import logging
import re
from dataclasses import dataclass, asdict

import bleach
from bleach.css_sanitizer import CSSSanitizer
from PIL import Image
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter, re_whitespace

from chandra.prompts import ALLOWED_ATTRIBUTES, ALLOWED_TAGS
from chandra.settings import settings

logger = logging.getLogger(__name__)

# Tags retained for the layout/structural pass. The model is prompted to use
# `<div>` as the layout-block wrapper; that tag is structural and must survive
# the allowlist even though the prompt lists only inline/document tags.
_SANITIZE_TAGS = frozenset(list(ALLOWED_TAGS) + ["div"])
# `src` isn't in the prompt allowlist (the prompt says "do not fill src"), but
# downstream parse_html injects a generated src on image blocks and downstream
# consumers expect it to survive. Allow it per-tag where it makes sense; bleach
# still enforces the protocol allowlist below.
_SANITIZE_ATTRIBUTES = {
    "*": list(ALLOWED_ATTRIBUTES),
    "img": list(ALLOWED_ATTRIBUTES) + ["src"],
    "a": list(ALLOWED_ATTRIBUTES) + ["src"],
}


# Tags whose entire content (not just the tag) must be removed so attacker
# payloads cannot survive as bare text after sanitization.
_DECOMPOSE_TAGS = ("script", "style", "iframe", "object", "embed", "noscript")

# Conservative CSS allowlist — text styling only, no `position`, `background`,
# `url(...)`, etc. that could exfiltrate data or break out of containers.
_SAFE_CSS_PROPERTIES = frozenset(
    {
        "color",
        "background-color",
        "font-size",
        "font-weight",
        "font-style",
        "font-family",
        "text-align",
        "text-decoration",
        "line-height",
        "padding",
        "margin",
        "border",
        "border-color",
        "border-style",
        "border-width",
        "white-space",
        "vertical-align",
    }
)
_CSS_SANITIZER = CSSSanitizer(allowed_css_properties=sorted(_SAFE_CSS_PROPERTIES))


def sanitize_html(html: str) -> str:
    """Strip tags/attributes outside the prompt allowlist.

    Defends viewers (Streamlit, Flask, downstream consumers) against a
    misbehaving model emitting `<script>`, `<iframe>`, or `on*` event handlers.
    """
    if not html:
        return html

    # Bleach's `strip=True` removes the tag but keeps inner text — so
    # `<script>alert(1)</script>` becomes the literal string `alert(1)`. For
    # script/style/iframe/etc. that residual text is itself a payload, so we
    # decompose those nodes (tag + content) before handing off to bleach.
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_DECOMPOSE_TAGS):
        tag.decompose()

    return bleach.clean(
        str(soup),
        tags=_SANITIZE_TAGS,
        attributes=_SANITIZE_ATTRIBUTES,
        css_sanitizer=_CSS_SANITIZER,
        protocols=["http", "https", "data", "mailto"],
        strip=True,
        strip_comments=True,
    )


def _hash_html(html: str) -> str:
    return hashlib.md5(html.encode("utf-8"), usedforsecurity=False).hexdigest()


def get_image_name(html: str, div_idx: int):
    html_hash = _hash_html(html)
    return f"{html_hash}_{div_idx}_img.webp"


def extract_images(html: str, chunks: dict, image: Image.Image):
    # Hash the same string parse_html does so disk filenames match md/HTML refs.
    html = sanitize_html(html)
    images = {}
    div_idx = 0
    for idx, chunk in enumerate(chunks):
        div_idx += 1
        if chunk["label"] in ["Image", "Figure"]:
            img = BeautifulSoup(chunk["content"], "html.parser").find("img")
            if not img:
                continue
            bbox = chunk["bbox"]
            try:
                block_image = image.crop(bbox)
            except ValueError:
                # Happens when bbox coordinates are invalid
                continue
            img_name = get_image_name(html, div_idx)
            images[img_name] = block_image
    return images


def parse_html(
    html: str, include_headers_footers: bool = False, include_images: bool = True
):
    html = sanitize_html(html)
    soup = BeautifulSoup(html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    out_html = ""
    image_idx = 0
    div_idx = 0
    for div in top_level_divs:
        div_idx += 1
        label = div.get("data-label")

        if label == "Blank-Page":
            continue

        # Skip headers and footers if not included
        if label and not include_headers_footers:
            if label in ["Page-Header", "Page-Footer"]:
                continue
        if label and not include_images:
            if label in ["Image", "Figure"]:
                continue

        if label in ["Image", "Figure"]:
            img = div.find("img")
            img_src = get_image_name(html, div_idx)

            # If no tag, add one in
            if img:
                img["src"] = img_src
                image_idx += 1
            else:
                img = BeautifulSoup(f"<img src='{img_src}'/>", "html.parser")
                div.append(img)

        # Strip img tags without src in non-image blocks (model hallucinations)
        if label not in ["Image", "Figure"]:
            for img_tag in div.find_all("img"):
                if not img_tag.get("src"):
                    img_tag.decompose()

        # Wrap text content in <p> tags if no inner HTML tags exist
        if label in ["Text"] and not re.search(
            "<.+>", str(div.decode_contents()).strip()
        ):
            # Add inner p tags if missing for text blocks
            text_content = str(div.decode_contents()).strip()
            text_content = f"<p>{text_content}</p>"
            div.clear()
            div.append(BeautifulSoup(text_content, "html.parser"))

        content = str(div.decode_contents())
        out_html += content
    return out_html


class Markdownify(MarkdownConverter):
    def __init__(
        self,
        inline_math_delimiters,
        block_math_delimiters,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inline_math_delimiters = inline_math_delimiters
        self.block_math_delimiters = block_math_delimiters

    def convert_math(self, el, text, parent_tags):
        block = el.has_attr("display") and el["display"] == "block"
        if block:
            return (
                "\n"
                + self.block_math_delimiters[0]
                + text.strip()
                + self.block_math_delimiters[1]
                + "\n"
            )
        else:
            return (
                " "
                + self.inline_math_delimiters[0]
                + text.strip()
                + self.inline_math_delimiters[1]
                + " "
            )

    def convert_table(self, el, text, parent_tags):
        return "\n\n" + str(el) + "\n\n"

    def convert_a(self, el, text, parent_tags):
        text = self.escape(text)
        # Escape brackets and parentheses in text
        text = re.sub(r"([\[\]()])", r"\\\1", text)
        return super().convert_a(el, text, parent_tags)

    def escape(self, text, parent_tags=None):
        text = super().escape(text, parent_tags)
        if self.options["escape_dollars"]:
            text = text.replace("$", r"\$")
        return text

    def process_text(self, el, parent_tags=None):
        text = str(el) or ""

        # normalize whitespace if we're not inside a preformatted element
        if not el.find_parent("pre"):
            text = re_whitespace.sub(" ", text)

        # escape special characters if we're not inside a preformatted or code element
        if not el.find_parent(["pre", "code", "kbd", "samp", "math"]):
            text = self.escape(text)

        # remove trailing whitespaces if any of the following condition is true:
        # - current text node is the last node in li
        # - current text node is followed by an embedded list
        if el.parent.name == "li" and (
            not el.next_sibling or el.next_sibling.name in ["ul", "ol"]
        ):
            text = text.rstrip()

        return text


def parse_markdown(
    html: str, include_headers_footers: bool = False, include_images: bool = True
):
    html = parse_html(html, include_headers_footers, include_images)

    md_cls = Markdownify(
        heading_style="ATX",
        bullets="-",
        escape_misc=False,
        escape_underscores=True,
        escape_asterisks=True,
        escape_dollars=True,
        sub_symbol="<sub>",
        sup_symbol="<sup>",
        inline_math_delimiters=("$", "$"),
        block_math_delimiters=("$$", "$$"),
    )
    try:
        markdown = md_cls.convert(html)
    except Exception:
        logger.exception("Error converting HTML to Markdown")
        markdown = ""
    return markdown.strip()


@dataclass
class LayoutBlock:
    bbox: list[int]
    label: str
    content: str


def parse_layout(html: str, image: Image.Image, bbox_scale=settings.BBOX_SCALE):
    html = sanitize_html(html)
    soup = BeautifulSoup(html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    width, height = image.size
    width_scaler = width / bbox_scale
    height_scaler = height / bbox_scale
    layout_blocks = []
    for div in top_level_divs:
        label = div.get("data-label")
        if label == "Blank-Page":
            continue

        raw_bbox = div.get("data-bbox")

        try:
            bbox_parts = raw_bbox.split(" ")
            bbox = list(map(int, bbox_parts))
            if len(bbox) != 4:
                raise ValueError(f"expected 4 ints, got {len(bbox)}")
        except (AttributeError, ValueError, TypeError) as exc:
            # Fall back to the full page so downstream cropping/drawing remains
            # meaningful instead of producing a 1×1-pixel artifact.
            logger.warning(
                "Invalid bbox %r (%s); falling back to full image", raw_bbox, exc
            )
            bbox = [0, 0, bbox_scale, bbox_scale]

        # Normalize bbox
        bbox = [
            max(0, int(bbox[0] * width_scaler)),
            max(0, int(bbox[1] * height_scaler)),
            min(int(bbox[2] * width_scaler), width),
            min(int(bbox[3] * height_scaler), height),
        ]
        if not label:
            label = "block"
        content = str(div.decode_contents())

        # Strip nested data-bbox attributes (not needed in open source)
        content_soup = BeautifulSoup(content, "html.parser")
        for tag in content_soup.find_all(attrs={"data-bbox": True}):
            del tag["data-bbox"]
        content = str(content_soup)

        layout_blocks.append(LayoutBlock(bbox=bbox, label=label, content=content))
    return layout_blocks


_CHUNK_ID_STEM_RE = re.compile(r"[^\w-]+")
_IMG_SRC_RE = re.compile(r'src="([^"]+\.webp)"')


def _format_chunk_id(file_stem: str | None, page_num: int | None, idx: int) -> str:
    """Stable chunk identifier of the form ``<stem>/<page:04d>/<idx:03d>``.

    Stem characters outside ``[A-Za-z0-9_-]`` are collapsed to ``-`` so the ID
    survives in URLs, filesystem paths, and graph databases unchanged. When
    ``file_stem`` or ``page_num`` is ``None`` (the case for callers that don't
    yet thread these through, e.g. the legacy app.py preview), fall back to a
    page-local ``_/NNN`` form — still unique within a page, just not globally.
    """
    if file_stem is None or page_num is None:
        return f"_/{idx:03d}"
    safe = _CHUNK_ID_STEM_RE.sub("-", file_stem).strip("-") or "_"
    return f"{safe}/{page_num:04d}/{idx:03d}"


def _extract_image_ref(content: str) -> str | None:
    """First ``*.webp`` src referenced inside a chunk's HTML, if any.

    parse_html injects ``src="<md5>_<idx>_img.webp"`` on Image/Figure divs.
    This pulls that filename back out so chunks can reference their image
    asset directly without a downstream consumer re-parsing HTML.
    """
    m = _IMG_SRC_RE.search(content)
    return m.group(1) if m else None


def parse_chunks(
    html: str,
    image: Image.Image,
    bbox_scale: int = settings.BBOX_SCALE,
    file_stem: str | None = None,
    page_num: int | None = None,
):
    """Per-block list with stable IDs and image references for graph indexing.

    Each chunk dict has: ``bbox``, ``label``, ``content`` (from LayoutBlock),
    plus ``chunk_id``, ``page``, and ``image_ref``. Pass ``file_stem`` and
    ``page_num`` to get globally-stable IDs; omit them for the legacy
    page-local form.
    """
    layout = parse_layout(html, image, bbox_scale=bbox_scale)
    chunks: list[dict] = []
    for idx, block in enumerate(layout):
        chunk = asdict(block)
        chunk["page"] = page_num
        chunk["chunk_id"] = _format_chunk_id(file_stem, page_num, idx)
        chunk["image_ref"] = _extract_image_ref(block.content)
        chunks.append(chunk)
    return chunks
