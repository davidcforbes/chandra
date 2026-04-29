"""Flask app for generating screenshot-ready OCR visualizations.

Displays original image with layout overlays on the left and extracted
markdown on the right. Accepts uploaded files via multipart/form-data so the
server never reads arbitrary host paths supplied by the client.
"""

from __future__ import annotations

import argparse
import base64
import logging
import threading
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request
from PIL import Image

from chandra.input import iter_file_pages
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from chandra.output import parse_layout

logger = logging.getLogger(__name__)

app = Flask(__name__)

_model: InferenceManager | None = None
_model_lock = threading.Lock()
_method = "vllm"


def get_model() -> InferenceManager:
    """Return the singleton InferenceManager. Thread-safe lazy init."""
    global _model
    if _model is None:
        with _model_lock:
            # Double-checked locking: avoid re-instantiation on contention.
            if _model is None:
                logger.info("Loading inference manager (method=%s)", _method)
                _model = InferenceManager(method=_method)
    return _model


def pil_image_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL image to base64 data URL."""
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def get_color_palette() -> dict[str, str]:
    return {
        "Section-Header": "#4ECDC4",
        "Text": "#45B7D1",
        "List-Group": "#96CEB4",
        "Table": "#FFEAA7",
        "Figure": "#DDA15E",
        "Image": "#BC6C25",
        "Caption": "#C77DFF",
        "Equation": "#9D4EDD",
        "Page-Header": "#E0AFA0",
        "Page-Footer": "#D4A5A5",
        "Footnote": "#A8DADC",
        "Form": "#F4A261",
        "default": "#FF00FF",
    }


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".bmp",
}


@app.route("/")
def index():
    return render_template("screenshot.html")


@app.route("/process", methods=["POST"])
def process():
    """Run OCR on an uploaded file. Accepts multipart/form-data with field 'file'.

    Replaces the previous server-side ``file_path`` parameter, which let any
    network client read arbitrary host files.
    """
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "missing file upload (field name 'file')"}), 400

    suffix = Path(upload.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"unsupported file type {suffix!r}"}), 400

    try:
        page_number = int(request.form.get("page_number", 0))
    except ValueError:
        return jsonify({"error": "page_number must be an integer"}), 400
    if page_number < 0:
        return jsonify({"error": "page_number must be >= 0"}), 400

    # Persist the upload to a private temp file so iter_file_pages can use
    # filetype.guess + pypdfium2 paths. NamedTemporaryFile cleans up on close.
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        upload.save(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        config = {"page_range": str(page_number + 1)} if suffix == ".pdf" else {}
        try:
            img = next(iter_file_pages(str(tmp_path), config))
        except StopIteration:
            return jsonify({"error": "no pages found in file"}), 400
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        model = get_model()
        batch = BatchInputItem(image=img, prompt_type="ocr_layout")
        result = model.generate([batch])[0]
        layout_blocks = parse_layout(result.raw, img)
        html = result.html

        # Embed extracted images back into HTML as data URLs.
        soup = BeautifulSoup(html, "html.parser")
        for img_name, pil_img in result.images.items():
            img_base64 = pil_image_to_base64(pil_img, format="PNG")
            img_tags = soup.find_all("img", src=img_name)
            if not img_tags:
                logger.debug("No img tags found for %s", img_name)
            for img_tag in img_tags:
                img_tag["src"] = img_base64
                alt_text = img_tag.get("alt", "")
                if alt_text:
                    wrapper = soup.new_tag("div", **{"class": "image-wrapper"})
                    alt_div = soup.new_tag("div", **{"class": "image-alt-text"})
                    alt_div.string = alt_text
                    img_container = soup.new_tag(
                        "div", **{"class": "image-container-wrapper"}
                    )
                    img_tag_copy = img_tag
                    img_tag.replace_with(wrapper)
                    img_container.append(img_tag_copy)
                    wrapper.append(alt_div)
                    wrapper.append(img_container)

        html_with_images = str(soup)

        page_b64 = pil_image_to_base64(img, format="PNG")
        img_width, img_height = img.size
        color_palette = get_color_palette()

        blocks_data = [
            {
                "bbox": block.bbox,
                "label": block.label,
                "color": color_palette.get(block.label, color_palette["default"]),
            }
            for block in layout_blocks
        ]

        return jsonify(
            {
                "image_base64": page_b64,
                "image_width": img_width,
                "image_height": img_height,
                "blocks": blocks_data,
                "html": html_with_images,
                "markdown": result.markdown,
            }
        )

    except Exception as exc:  # noqa: BLE001 — surface as HTTP 500
        logger.exception("Error in /process")
        return jsonify({"error": str(exc)}), 500

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove temp upload %s", tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Chandra screenshot demo server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1; use 0.0.0.0 to expose on the network)",
    )
    parser.add_argument("--port", type=int, default=8503)
    parser.add_argument(
        "--method",
        choices=["hf", "vllm"],
        default="vllm",
        help="Inference backend to use",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    global _method
    _method = args.method

    logger.info(
        "Starting screenshot app on %s:%d (method=%s)",
        args.host,
        args.port,
        args.method,
    )
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
