"""Unit tests for chandra.output — sanitization, parsing, and image extraction.

These tests use small synthetic HTML/PIL images and do not require a GPU or
the HF model. They exercise the parsing pipeline that runs after generation.
"""

from __future__ import annotations

import pytest
from PIL import Image

from chandra.output import (
    LayoutBlock,
    extract_images,
    parse_chunks,
    parse_html,
    parse_layout,
    parse_markdown,
    sanitize_html,
)


# ---------- sanitize_html (chandra-311) ----------


class TestSanitizeHtml:
    def test_strips_script_tag(self):
        assert "<script>" not in sanitize_html("<div><script>alert(1)</script>x</div>")

    def test_strips_iframe_tag(self):
        out = sanitize_html('<div><iframe src="evil"></iframe>hello</div>')
        assert "<iframe" not in out
        assert "hello" in out

    def test_strips_event_handlers(self):
        out = sanitize_html('<img src="x" onerror="alert(1)" alt="a"/>')
        assert "onerror" not in out
        assert 'src="x"' in out

    def test_strips_unknown_attributes(self):
        out = sanitize_html('<div data-label="Text" foo="bar">hi</div>')
        assert "foo=" not in out
        assert "data-label" in out

    def test_keeps_allowlisted_tags(self):
        html = (
            '<div data-label="Text" data-bbox="0 0 100 100">'
            "<p>hello <b>world</b> <math>x</math></p>"
            "</div>"
        )
        out = sanitize_html(html)
        assert "<div" in out
        assert "<p>" in out
        assert "<b>" in out
        assert "<math>" in out

    def test_strips_javascript_url(self):
        out = sanitize_html('<a href="javascript:alert(1)">click</a>')
        assert "javascript:" not in out

    def test_strips_html_comments(self):
        out = sanitize_html("<div>hello<!-- secret --></div>")
        assert "<!--" not in out

    def test_strips_dangerous_css_properties(self):
        # `position: fixed` and `background: url(...)` are CSS-injection
        # vectors. The CSS sanitizer should drop them but keep safe text-style
        # properties.
        html = (
            '<div style="color: red; position: fixed; '
            'background: url(http://evil/);">x</div>'
        )
        out = sanitize_html(html)
        assert "color" in out
        assert "position" not in out
        assert "url(" not in out

    def test_empty_input(self):
        assert sanitize_html("") == ""

    def test_plain_text_passthrough(self):
        assert sanitize_html("hello world") == "hello world"


# ---------- parse_html ----------


class TestParseHtml:
    def test_strips_blank_page(self):
        out = parse_html('<div data-label="Blank-Page">x</div>')
        assert out == ""

    def test_skips_headers_footers_by_default(self):
        html = (
            '<div data-label="Page-Header">H</div>'
            '<div data-label="Text"><p>body</p></div>'
            '<div data-label="Page-Footer">F</div>'
        )
        out = parse_html(html)
        assert "body" in out
        assert "H" not in out and "F" not in out

    def test_includes_headers_footers_when_requested(self):
        html = (
            '<div data-label="Page-Header">HEADER</div>'
            '<div data-label="Text"><p>body</p></div>'
        )
        out = parse_html(html, include_headers_footers=True)
        assert "HEADER" in out

    def test_skips_images_when_requested(self):
        html = '<div data-label="Image"><img alt="a"/></div>'
        out = parse_html(html, include_images=False)
        assert "<img" not in out

    def test_wraps_bare_text_blocks_in_p(self):
        html = '<div data-label="Text">just text</div>'
        out = parse_html(html)
        assert "<p>just text</p>" in out

    def test_strips_imgs_without_src_in_text_blocks(self):
        html = '<div data-label="Text"><p>hi</p><img alt="ghost"/></div>'
        out = parse_html(html)
        assert "<img" not in out
        assert "hi" in out

    def test_injects_src_for_image_block(self):
        html = '<div data-label="Image"><img alt="cat"/></div>'
        out = parse_html(html)
        assert "src=" in out

    def test_sanitizes_script_in_input(self):
        html = '<div data-label="Text"><p>ok</p><script>alert(1)</script></div>'
        out = parse_html(html)
        assert "<script" not in out
        assert "alert" not in out


# ---------- parse_layout (chandra-990) ----------


class TestParseLayout:
    def test_invalid_bbox_falls_back_to_full_page(self):
        image = Image.new("RGB", (1000, 800))
        html = '<div data-label="Text" data-bbox="not a bbox">hi</div>'
        blocks = parse_layout(html, image)
        assert len(blocks) == 1
        # Full image bbox = the image dimensions, not a 1x1 sliver.
        assert blocks[0].bbox == [0, 0, 1000, 800]

    def test_missing_bbox_falls_back_to_full_page(self):
        image = Image.new("RGB", (500, 500))
        html = '<div data-label="Text">hi</div>'
        blocks = parse_layout(html, image)
        assert blocks[0].bbox == [0, 0, 500, 500]

    def test_bbox_with_wrong_arity_falls_back(self):
        image = Image.new("RGB", (500, 500))
        html = '<div data-label="Text" data-bbox="0 0 100">hi</div>'
        blocks = parse_layout(html, image)
        assert blocks[0].bbox == [0, 0, 500, 500]

    def test_valid_bbox_is_scaled(self):
        image = Image.new("RGB", (1000, 1000))
        html = '<div data-label="Text" data-bbox="0 0 500 500">hi</div>'
        blocks = parse_layout(html, image, bbox_scale=1000)
        assert blocks[0].bbox == [0, 0, 500, 500]

    def test_strips_nested_data_bbox_attrs(self):
        image = Image.new("RGB", (100, 100))
        html = (
            '<div data-label="Text" data-bbox="0 0 100 100">'
            '<p data-bbox="should be removed">hi</p>'
            "</div>"
        )
        blocks = parse_layout(html, image)
        assert "data-bbox" not in blocks[0].content

    def test_skips_blank_page(self):
        image = Image.new("RGB", (100, 100))
        blocks = parse_layout('<div data-label="Blank-Page"></div>', image)
        assert blocks == []

    def test_sanitizes_input_html(self):
        image = Image.new("RGB", (100, 100))
        html = (
            '<div data-label="Text" data-bbox="0 0 100 100">'
            "<script>alert(1)</script>ok"
            "</div>"
        )
        blocks = parse_layout(html, image)
        assert "<script" not in blocks[0].content


# ---------- extract_images / parse_chunks ----------


class TestExtractImages:
    def test_extracts_image_blocks(self):
        image = Image.new("RGB", (200, 200), "white")
        html = '<div data-label="Image" data-bbox="0 0 1000 1000"><img alt="a"/></div>'
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)
        assert len(images) == 1
        # The single key should look like a filename.
        ((name, img),) = images.items()
        assert name.endswith(".webp")
        assert isinstance(img, Image.Image)

    def test_ignores_non_image_blocks(self):
        image = Image.new("RGB", (200, 200))
        html = '<div data-label="Text" data-bbox="0 0 1000 1000"><p>ok</p></div>'
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)
        assert images == {}

    def test_image_block_without_img_tag_is_skipped(self):
        # parse_html injects an img into Image-labeled divs so chunks always
        # carry one in practice, but extract_images should still no-op safely
        # when content has no <img>.
        image = Image.new("RGB", (200, 200))
        chunks = [
            {"label": "Image", "bbox": [0, 0, 200, 200], "content": "<p>no img</p>"}
        ]
        images = extract_images("<div></div>", chunks, image)
        assert images == {}

    def test_image_filenames_match_parse_html_refs(self):
        # Regression (chandra-3gu): when sanitize_html was added to parse_html,
        # the local html was reassigned before hashing. extract_images still
        # hashed the raw input, so md/HTML refs pointed at filenames that
        # never existed on disk. The script tag here gets decomposed by
        # sanitize_html, so raw vs sanitized hashes diverge.
        import re

        image = Image.new("RGB", (200, 200), "white")
        html = (
            '<div data-label="Image" data-bbox="0 0 1000 1000"><img alt="a"/></div>'
            "<script>alert(1)</script>"
        )
        rendered = parse_html(html)
        chunks = parse_chunks(html, image)
        images = extract_images(html, chunks, image)
        md_refs = set(re.findall(r'src="([^"]+\.webp)"', rendered))
        disk_names = set(images.keys())
        assert md_refs == disk_names


# ---------- parse_markdown ----------


class TestParseMarkdown:
    def test_renders_paragraph(self):
        out = parse_markdown('<div data-label="Text"><p>hello</p></div>')
        assert "hello" in out

    def test_renders_heading(self):
        out = parse_markdown(
            '<div data-label="Section-Header"><h1>Title</h1></div>',
        )
        assert "# Title" in out

    def test_strips_script_at_markdown_layer(self):
        out = parse_markdown(
            '<div data-label="Text"><p>safe</p><script>alert(1)</script></div>'
        )
        assert "alert" not in out


# ---------- LayoutBlock dataclass ----------


def test_layoutblock_is_serializable():
    block = LayoutBlock(bbox=[0, 0, 1, 1], label="Text", content="hi")
    assert block.bbox == [0, 0, 1, 1]
    assert block.label == "Text"


# Smoke test: importing the module sets up the logger without printing.
def test_module_does_not_print(capfd):
    import importlib

    import chandra.output as mod

    importlib.reload(mod)
    out, _err = capfd.readouterr()
    assert out == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
