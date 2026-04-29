"""Unit tests for chandra.scripts.screenshot_app — Flask routes and security."""

from __future__ import annotations

import io
import threading
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from chandra.scripts import screenshot_app


@pytest.fixture(autouse=True)
def reset_module_state():
    """Make each test see a fresh model singleton."""
    screenshot_app._model = None
    yield
    screenshot_app._model = None


@pytest.fixture
def fake_model():
    """Patch InferenceManager so /process never tries to load real weights."""
    instance = MagicMock()

    def _generate(batch, **_):
        result = MagicMock()
        result.markdown = "# fake"
        result.html = '<div data-label="Text"><p>fake</p></div>'
        result.images = {}
        result.raw = (
            '<div data-label="Text" data-bbox="0 0 1000 1000"><p>fake</p></div>'
        )
        return [result]

    instance.generate.side_effect = _generate
    with patch.object(screenshot_app, "InferenceManager", return_value=instance):
        yield instance


@pytest.fixture
def client():
    screenshot_app.app.config["TESTING"] = True
    return screenshot_app.app.test_client()


# ---------- chandra-6yy: input validation / no-arbitrary-file-read ----------


def test_process_rejects_missing_file(client):
    resp = client.post("/process", data={})
    assert resp.status_code == 400
    assert "missing file upload" in resp.get_json()["error"]


def test_process_rejects_unsupported_extension(client):
    data = {"file": (io.BytesIO(b"hello"), "test.docx")}
    resp = client.post("/process", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "unsupported file type" in resp.get_json()["error"]


def test_process_rejects_non_integer_page(client):
    img_bytes = io.BytesIO()
    Image.new("RGB", (50, 50)).save(img_bytes, "PNG")
    img_bytes.seek(0)
    data = {
        "file": (img_bytes, "test.png"),
        "page_number": "not-a-number",
    }
    resp = client.post("/process", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "integer" in resp.get_json()["error"]


def test_process_rejects_negative_page(client):
    img_bytes = io.BytesIO()
    Image.new("RGB", (50, 50)).save(img_bytes, "PNG")
    img_bytes.seek(0)
    data = {"file": (img_bytes, "test.png"), "page_number": "-1"}
    resp = client.post("/process", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_process_does_not_accept_file_path_field(client, fake_model):
    """The legacy ``file_path`` JSON field is gone; only multipart upload works."""
    resp = client.post("/process", json={"file_path": "/etc/passwd"})
    assert resp.status_code == 400


# ---------- happy path ----------


def test_process_image_upload_succeeds(client, fake_model):
    img_bytes = io.BytesIO()
    # Use 2000x1700 so neither dim trips MIN_IMAGE_DIM=1536 upscaling.
    Image.new("RGB", (2000, 1700), "white").save(img_bytes, "PNG")
    img_bytes.seek(0)
    data = {"file": (img_bytes, "test.png")}
    resp = client.post("/process", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200, resp.data
    body = resp.get_json()
    assert body["image_width"] == 2000
    assert body["image_height"] == 1700
    assert "html" in body
    assert "markdown" in body
    assert "blocks" in body


# ---------- chandra-2g1: get_model thread safety ----------


class TestGetModelThreadSafety:
    def test_concurrent_calls_create_one_instance(self):
        call_count = 0
        sentinel = MagicMock(name="sentinel-model")

        def fake_init(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            import time

            time.sleep(0.01)  # Widen the race window.
            return sentinel

        with patch.object(screenshot_app, "InferenceManager", side_effect=fake_init):
            results: list = []

            def worker():
                results.append(screenshot_app.get_model())

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All 8 workers see the same model and only one was constructed.
            assert call_count == 1
            assert all(r is sentinel for r in results)


# ---------- color palette ----------


def test_color_palette_has_default():
    palette = screenshot_app.get_color_palette()
    assert "default" in palette


# ---------- image embedding branch ----------


def test_process_embeds_extracted_images_with_alt(client):
    """Images returned by the model should be inlined as data URLs and the
    alt-text wrapper branch should be exercised."""
    instance = MagicMock()

    def _generate(_batch, **_):
        result = MagicMock()
        result.markdown = "# fake"
        # The HTML contains an image whose src matches a key in result.images;
        # screenshot_app should rewrite that src to a data URL.
        result.html = '<div class="x"><img src="figure_1.webp" alt="a chart"/></div>'
        result.images = {
            "figure_1.webp": Image.new("RGB", (10, 10), "blue"),
        }
        result.raw = '<div data-label="Image" data-bbox="0 0 1000 1000"></div>'
        return [result]

    instance.generate.side_effect = _generate
    with patch.object(screenshot_app, "InferenceManager", return_value=instance):
        img_bytes = io.BytesIO()
        Image.new("RGB", (2000, 1700), "white").save(img_bytes, "PNG")
        img_bytes.seek(0)
        resp = client.post(
            "/process",
            data={"file": (img_bytes, "test.png")},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200, resp.data
    body = resp.get_json()
    assert "data:image/png;base64," in body["html"]
    assert "image-wrapper" in body["html"]


def test_main_uses_argparse_defaults(monkeypatch):
    """Smoke test that screenshot_app.main() wires CLI args to app.run."""
    monkeypatch.setattr("sys.argv", ["screenshot_app"])
    with patch.object(screenshot_app.app, "run") as fake_run:
        screenshot_app.main()
    fake_run.assert_called_once()
    kwargs = fake_run.call_args.kwargs
    # Defaults should bind to localhost (chandra-6yy).
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8503


def test_main_respects_host_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["screenshot_app", "--host", "0.0.0.0", "--port", "9000"]
    )
    with patch.object(screenshot_app.app, "run") as fake_run:
        screenshot_app.main()
    kwargs = fake_run.call_args.kwargs
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 9000


def test_process_returns_500_when_model_raises(client):
    """Model exceptions surface as HTTP 500 with the exception text."""
    instance = MagicMock()
    instance.generate.side_effect = RuntimeError("model crashed")
    with patch.object(screenshot_app, "InferenceManager", return_value=instance):
        img_bytes = io.BytesIO()
        Image.new("RGB", (2000, 1700), "white").save(img_bytes, "PNG")
        img_bytes.seek(0)
        resp = client.post(
            "/process",
            data={"file": (img_bytes, "test.png")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 500
    assert "model crashed" in resp.get_json()["error"]


def test_process_pdf_with_invalid_page_returns_400(client, fake_model):
    """An invalid page on a PDF should surface as a 400 with the parse error."""
    # Build a minimal PDF.
    img = Image.new("RGB", (100, 100), "white")
    pdf_bytes = io.BytesIO()
    img.save(pdf_bytes, format="PDF")
    pdf_bytes.seek(0)
    resp = client.post(
        "/process",
        data={"file": (pdf_bytes, "test.pdf"), "page_number": "9999"},
        content_type="multipart/form-data",
    )
    # Page 10000 (1-indexed) doesn't exist; iter_file_pages yields nothing.
    assert resp.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
