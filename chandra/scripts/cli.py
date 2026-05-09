"""Chandra OCR CLI.

Entry point for the page-worker pipeline. Accepts a single file, a folder
(top-level by default; recursive with ``--recursive``), or a glob pattern.
Resume is automatic: re-running the same command skips already-assembled
books and resumes mid-PDF runs from where they left off.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from chandra import manifest
from chandra.model import InferenceManager
from chandra.pipeline import discover_books, run_pipeline

logger = logging.getLogger(__name__)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="chandra-ocr", prog_name="chandra")
@click.argument("input_path", type=click.Path(path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--method",
    type=click.Choice(["hf", "vllm"], case_sensitive=False),
    default="vllm",
    help="Inference method: 'hf' for local model, 'vllm' for vLLM server.",
)
@click.option(
    "--recursive/--no-recursive",
    default=False,
    help="When INPUT_PATH is a folder, walk subfolders too.",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Outer page-worker pool size (default: 8 for vllm, 1 for hf).",
)
@click.option(
    "--page-range",
    type=str,
    default=None,
    help="1-indexed page range for PDFs (e.g., '1-5,7,9-12'). Only applies "
    "to single-PDF runs; in folder mode it applies to every PDF.",
)
@click.option(
    "--max-output-tokens",
    type=int,
    default=None,
    help="Maximum number of output tokens per page.",
)
@click.option(
    "--max-retries",
    type=int,
    default=None,
    help="Maximum number of retries for vLLM inference (repeat-token + "
    "transient-error paths).",
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
    "--log-every",
    type=int,
    default=1,
    show_default=True,
    help="Print one progress line every N completed pages per book.",
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
# Deprecated flags kept as silent aliases so existing callers don't break.
@click.option(
    "--batch-size",
    type=int,
    default=None,
    hidden=True,
    help="Deprecated: pages now flow through a worker pool individually.",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    hidden=True,
    help="Deprecated alias for --workers.",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    default=False,
    hidden=True,
    help="Deprecated: per-page errors no longer abort the run.",
)
def main(
    input_path: Path,
    output_path: Path,
    method: str,
    recursive: bool,
    workers: int | None,
    page_range: str | None,
    max_output_tokens: int | None,
    max_retries: int | None,
    include_images: bool,
    include_headers_footers: bool,
    save_html: bool,
    paginate_output: bool,
    paginate_output_legacy: bool,
    log_every: int,
    log_level: str,
    batch_size: int | None,
    max_workers: int | None,
    fail_fast: bool,
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    paginate_output = paginate_output or paginate_output_legacy

    # Soft-deprecate flags that the new pipeline doesn't honor.
    if batch_size is not None:
        click.echo(
            "warning: --batch-size is deprecated; pages flow through a worker "
            "pool individually now.",
            err=True,
        )
    if fail_fast:
        click.echo(
            "warning: --fail-fast is deprecated; per-page errors are recorded "
            "in metadata.json and don't abort the run.",
            err=True,
        )
    if max_workers is not None and workers is None:
        workers = max_workers

    if workers is None:
        workers = 8 if method == "vllm" else 1

    click.echo(f"Chandra CLI — input: {input_path}  output: {output_path}")
    click.echo(f"  method: {method}  workers: {workers}  recursive: {recursive}")

    # Resolve the input into BookSpecs and detect resume state.
    try:
        books = discover_books(
            input_path, output_path, recursive=recursive, page_range=page_range
        )
    except ValueError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)

    if not books:
        click.echo("No work to do — every supported file is already assembled.")
        return

    pending = sum(len(b.pending_pages) for b in books)
    already_done = sum(b.already_done for b in books)
    click.echo(
        f"  books: {len(books)}  pages pending: {pending}  pages already on disk: {already_done}"
    )

    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"\nLoading model with method '{method}'...")
    model = InferenceManager(method=method)
    click.echo("Model loaded.\n")

    # Build kwargs that pass through to InferenceManager.generate().
    generate_kwargs: dict = {
        "include_images": include_images,
        "include_headers_footers": include_headers_footers,
    }
    if max_output_tokens is not None:
        generate_kwargs["max_output_tokens"] = max_output_tokens
    if method == "vllm":
        if max_retries is not None:
            generate_kwargs["max_retries"] = max_retries
        # Force the inner per-call pool to 1: we drive parallelism from the
        # outer page pool, and each model.generate() call carries one page.
        generate_kwargs["max_workers"] = 1

    try:
        stats = run_pipeline(
            books,
            model=model,
            n_workers=workers,
            paginate_output=paginate_output,
            save_html=save_html,
            generate_kwargs=generate_kwargs,
            log_every=log_every,
        )
    except KeyboardInterrupt:
        click.echo(
            "\ninterrupted; partial state preserved on disk for resume.", err=True
        )
        sys.exit(130)

    failed_pages = _count_failed_pages(books)
    click.echo(
        f"\nDone. books={stats['books']} pages_processed={stats['pages_processed']} "
        f"pages_pending={stats['pages_pending']} pages_with_errors={failed_pages}"
    )

    if stats["pages_pending"] > 0 or failed_pages > 0:
        sys.exit(1)


def _count_failed_pages(books) -> int:
    """After a pipeline run, count error-tagged pages by reading each book's
    metadata.json (post-assembly) or .partial pages (pre-assembly)."""
    import json

    total = 0
    for book in books:
        meta_path = book.stem_dir / f"{book.stem}_metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                total += sum(1 for p in meta.get("pages", []) if p.get("error"))
            except Exception:  # noqa: BLE001
                pass
            continue
        # Book wasn't assembled — count errors in surviving partial artifacts.
        for n in manifest.read_partial_state(book.stem_dir):
            try:
                page = json.loads(
                    manifest.page_artifact_path(book.stem_dir, n).read_text(
                        encoding="utf-8"
                    )
                )
                if page.get("error"):
                    total += 1
            except Exception:  # noqa: BLE001
                pass
    return total


if __name__ == "__main__":
    main()
