from __future__ import annotations

import json
import logging
import sys
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, List

import click

from chandra.input import count_file_pages, iter_file_pages
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem

logger = logging.getLogger(__name__)


def _chunked(it: Iterable, size: int) -> Iterator[list]:
    """Yield successive ``size``-element chunks from ``it``."""
    iterator = iter(it)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def get_supported_files(input_path: Path) -> List[Path]:
    """Get list of supported image/PDF files from path."""
    supported_extensions = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".tiff",
        ".bmp",
    }

    if input_path.is_file():
        if input_path.suffix.lower() in supported_extensions:
            return [input_path]
        else:
            raise click.BadParameter(f"Unsupported file type: {input_path.suffix}")

    elif input_path.is_dir():
        # Use a set to dedupe — case-insensitive filesystems (Windows, macOS
        # default) match the lowercase and uppercase glob patterns to the same
        # underlying file.
        files: set[Path] = set()
        for ext in supported_extensions:
            files.update(input_path.glob(f"*{ext}"))
            files.update(input_path.glob(f"*{ext.upper()}"))
        return sorted(files)

    else:
        raise click.BadParameter(f"Path does not exist: {input_path}")


def save_merged_output(
    output_dir: Path,
    file_name: str,
    results: List,
    save_images: bool = True,
    save_html: bool = True,
    paginate_output: bool = False,
):
    """Save merged OCR results for all pages to output directory."""
    safe_name = Path(file_name).stem
    file_output_dir = output_dir / safe_name
    file_output_dir.mkdir(parents=True, exist_ok=True)

    all_markdown = []
    all_html = []
    all_metadata = []
    total_tokens = 0
    total_chunks = 0
    total_images = 0

    for page_num, result in enumerate(results):
        if page_num > 0 and paginate_output:
            all_markdown.append(f"\n\n{page_num}" + "-" * 48 + "\n\n")
            all_html.append(f"\n\n<!-- Page {page_num + 1} -->\n\n")

        all_markdown.append(result.markdown)
        all_html.append(result.html)

        if not paginate_output and page_num < len(results) - 1:
            all_markdown.append("\n\n")
            all_html.append("\n\n")

        total_tokens += result.token_count
        total_chunks += len(result.chunks)
        total_images += len(result.images)

        page_metadata = {
            "page_num": page_num,
            "page_box": result.page_box,
            "token_count": result.token_count,
            "num_chunks": len(result.chunks),
            "num_images": len(result.images),
        }
        all_metadata.append(page_metadata)

        if save_images and result.images:
            file_output_dir.mkdir(exist_ok=True)
            for img_name, pil_image in result.images.items():
                img_path = file_output_dir / img_name
                pil_image.save(img_path)

    markdown_path = file_output_dir / f"{safe_name}.md"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("".join(all_markdown))

    if save_html:
        html_path = file_output_dir / f"{safe_name}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("".join(all_html))

    metadata = {
        "file_name": file_name,
        "num_pages": len(results),
        "total_token_count": total_tokens,
        "total_chunks": total_chunks,
        "total_images": total_images,
        "pages": all_metadata,
    }
    metadata_path = file_output_dir / f"{safe_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"  Saved: {markdown_path} ({len(results)} page(s))")


def _process_single_file(
    file_path: Path,
    output_path: Path,
    model: InferenceManager,
    method: str,
    page_range: str | None,
    max_output_tokens: int | None,
    max_workers: int | None,
    max_retries: int | None,
    include_images: bool,
    include_headers_footers: bool,
    save_html: bool,
    batch_size: int,
    paginate_output: bool,
) -> bool:
    """Process one file. Returns True on success, False on failure."""
    config = {"page_range": page_range} if page_range else {}

    try:
        page_count = count_file_pages(str(file_path), config)
    except (ValueError, OSError) as exc:
        click.echo(f"  Error counting pages of {file_path.name}: {exc}", err=True)
        return False

    click.echo(f"  Found {page_count} page(s)")

    pages_iter = iter_file_pages(str(file_path), config)
    all_results = []
    pages_processed = 0

    try:
        for batch in _chunked(pages_iter, batch_size):
            batch_start = pages_processed + 1
            batch_end = pages_processed + len(batch)
            click.echo(f"  Processing pages {batch_start}-{batch_end}...")

            batch_items = [
                BatchInputItem(image=img, prompt_type="ocr_layout") for img in batch
            ]

            generate_kwargs: dict = {
                "include_images": include_images,
                "include_headers_footers": include_headers_footers,
            }
            if max_output_tokens is not None:
                generate_kwargs["max_output_tokens"] = max_output_tokens
            if method == "vllm":
                if max_workers is not None:
                    generate_kwargs["max_workers"] = max_workers
                if max_retries is not None:
                    generate_kwargs["max_retries"] = max_retries

            results = model.generate(batch_items, **generate_kwargs)
            all_results.extend(results)
            pages_processed = batch_end

        save_merged_output(
            output_path,
            file_path.name,
            all_results,
            save_images=include_images,
            save_html=save_html,
            paginate_output=paginate_output,
        )
        click.echo(f"  Completed: {file_path.name}")
        return True
    except Exception as exc:  # noqa: BLE001 — surface to user, continue with next file
        click.echo(f"  Error processing {file_path.name}: {exc}", err=True)
        logger.exception("Error processing %s", file_path.name)
        return False


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="chandra-ocr", prog_name="chandra")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--method",
    type=click.Choice(["hf", "vllm"], case_sensitive=False),
    default="vllm",
    help="Inference method: 'hf' for local model, 'vllm' for vLLM server.",
)
@click.option(
    "--page-range",
    type=str,
    default=None,
    help="1-indexed page range for PDFs (e.g., '1-5,7,9-12'). Only applicable to PDF files.",
)
@click.option(
    "--max-output-tokens",
    type=int,
    default=None,
    help="Maximum number of output tokens per page.",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Maximum number of parallel workers for vLLM inference.",
)
@click.option(
    "--max-retries",
    type=int,
    default=None,
    help="Maximum number of retries for vLLM inference (covers both repeat-token and transient-error paths).",
)
@click.option(
    "--include-images/--no-images",
    default=True,
    help="Include images in output.",
)
@click.option(
    "--include-headers-footers/--no-headers-footers",
    default=False,
    help="Include page headers and footers in output.",
)
@click.option(
    "--save-html/--no-html",
    default=True,
    help="Save HTML output files.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Number of pages to process in a batch.",
)
@click.option(
    "--paginate-output/--no-paginate-output",
    "paginate_output",
    default=False,
    help="Insert page separators in merged output.",
)
@click.option(
    "--paginate_output",
    "paginate_output_legacy",
    is_flag=True,
    default=False,
    hidden=True,
    help="Deprecated alias for --paginate-output.",
)
@click.option(
    "--log-level",
    default="WARNING",
    show_default=True,
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Python logging level for chandra modules.",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    default=False,
    help="Stop processing on the first file failure.",
)
def main(
    input_path: Path,
    output_path: Path,
    method: str,
    page_range: str,
    max_output_tokens: int,
    max_workers: int,
    max_retries: int,
    include_images: bool,
    include_headers_footers: bool,
    save_html: bool,
    batch_size: int,
    paginate_output: bool,
    paginate_output_legacy: bool,
    log_level: str,
    fail_fast: bool,
):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    paginate_output = paginate_output or paginate_output_legacy

    if method == "hf":
        click.echo(
            "When using '--method hf', ensure that the batch size is set correctly. "
            "We will default to batch size of 1."
        )
        if batch_size is None:
            batch_size = 1
    elif method == "vllm" and batch_size is None:
        batch_size = 28

    click.echo("Chandra CLI - Starting OCR processing")
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Method: {method}")

    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"\nLoading model with method '{method}'...")
    model = InferenceManager(method=method)
    click.echo("Model loaded successfully.")

    files_to_process = get_supported_files(input_path)
    click.echo(f"\nFound {len(files_to_process)} file(s) to process.")

    if not files_to_process:
        click.echo("No supported files found. Exiting.")
        return

    failures = 0
    for file_idx, file_path in enumerate(files_to_process, 1):
        click.echo(
            f"\n[{file_idx}/{len(files_to_process)}] Processing: {file_path.name}"
        )
        ok = _process_single_file(
            file_path=file_path,
            output_path=output_path,
            model=model,
            method=method,
            page_range=page_range,
            max_output_tokens=max_output_tokens,
            max_workers=max_workers,
            max_retries=max_retries,
            include_images=include_images,
            include_headers_footers=include_headers_footers,
            save_html=save_html,
            batch_size=batch_size,
            paginate_output=paginate_output,
        )
        if not ok:
            failures += 1
            if fail_fast:
                break

    summary = (
        f"\nProcessed {len(files_to_process)} file(s); "
        f"{failures} failed. Results saved to: {output_path}"
    )
    click.echo(summary)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
