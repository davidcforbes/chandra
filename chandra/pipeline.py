"""Page-worker pipeline: multi-book queue + parallel workers + assembler.

This module is the heart of the unified architecture (chandra-f1k).
``run_pipeline`` accepts a list of ``BookSpec``s, spins up a single producer,
``n_workers`` workers, and one assembler thread, and runs them until every
book is fully assembled into canonical artifacts.

Key invariants:

- Pages are independent. Worker N never depends on worker M.
- Every per-page artifact is written via tmp+rename — readers (the assembler,
  resume-detection on the next run) never see partials.
- A book is "complete" when every ``expected_pages[i]`` has a corresponding
  ``<stem_dir>/.partial/pages/NNNN.json`` on disk. The assembler polls for
  this and runs ``manifest.assemble_book`` exactly once per book.
- Resume on the next run is automatic: ``discover_books`` checks for canonical
  artifacts (skip), then for ``.partial/`` (resume on remaining pages), then
  starts fresh.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

from chandra import manifest
from chandra.input import count_file_pages, iter_file_pages, parse_range_str
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem

logger = logging.getLogger(__name__)


SUPPORTED_EXTS = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".bmp"}
)


# ---------- BookSpec + discovery -----------------------------------------


@dataclasses.dataclass
class BookSpec:
    source_path: Path  # the PDF/image file we're OCRing
    stem: str  # source_path.stem
    stem_dir: Path  # output_root / stem
    total_pages: int  # full page count of the source
    expected_pages: list[int]  # 0-indexed page numbers we plan to OCR
    pending_pages: list[int]  # expected_pages minus what's already on disk

    @property
    def is_complete(self) -> bool:
        return not self.pending_pages

    @property
    def already_done(self) -> int:
        return len(self.expected_pages) - len(self.pending_pages)


def discover_books(
    input_path: Path,
    output_root: Path,
    recursive: bool = False,
    page_range: Optional[str] = None,
) -> list[BookSpec]:
    """Resolve ``input_path`` to a sorted list of ``BookSpec``s with resume
    detection applied. Skips files that are already fully assembled."""
    files = _find_files(input_path, recursive)
    books: list[BookSpec] = []
    for f in files:
        spec = _prepare_book(f, output_root, page_range)
        if spec is not None:
            books.append(spec)
    return books


def _is_chandra_output_dir(d: Path) -> bool:
    """True if ``d`` looks like a chandra-produced output subdirectory.

    A typical output dir is ``<output_root>/<stem>/`` containing one or more
    of: ``<stem>.md``, ``chunks.jsonl``, ``.partial/``. We must NEVER recurse
    into these — chandra's own extracted ``<hash>_<idx>_img.webp`` files
    live inside, and ``.webp`` is a valid OCR input format, so a naive
    recursive walk would treat every extracted image as a new "book" and
    create bogus output dirs (chandra-xqk).
    """
    if not d.is_dir():
        return False
    return (
        (d / f"{d.name}.md").is_file()
        or (d / "chunks.jsonl").is_file()
        or (d / ".partial").is_dir()
    )


def _walk_supported(root: Path, recursive: bool) -> list[Path]:
    """Walk ``root`` for supported source files, pruning chandra output dirs.

    Uses ``os.walk`` so we can edit ``dirnames`` in place to skip whole
    subtrees. Without this, a recursive walk over a corpus that's been OCR'd
    before would re-process every extracted image (chandra-xqk).
    """
    import os

    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        # Prune chandra output dirs so we don't descend into them.
        # Mutate dirnames in place — os.walk respects this when topdown=True.
        dirnames[:] = [sub for sub in dirnames if not _is_chandra_output_dir(d / sub)]
        for name in filenames:
            if Path(name).suffix.lower() in SUPPORTED_EXTS:
                out.append(d / name)
        if not recursive:
            break  # only the top level
    return sorted(out)


def _find_files(input_path: Path, recursive: bool) -> list[Path]:
    """Resolve a file, directory, or glob pattern to supported source files.

    Glob detection: if any path component contains ``*``, ``?``, or ``[``,
    treat the path as a pattern relative to its longest non-glob prefix.
    Folder mode (recursive or not) prunes chandra output subdirectories so
    extracted images aren't re-OCR'd.
    """
    s = str(input_path)
    has_glob = any(c in s for c in "*?[")

    if has_glob:
        parts = input_path.parts
        glob_idx = next(
            (i for i, part in enumerate(parts) if any(c in part for c in "*?[")),
            len(parts),
        )
        base = Path(*parts[:glob_idx]) if glob_idx > 0 else Path(".")
        pattern = str(Path(*parts[glob_idx:]))
        return sorted(
            p
            for p in base.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
        return [input_path]

    if input_path.is_dir():
        return _walk_supported(input_path, recursive=recursive)

    raise ValueError(f"Path does not exist: {input_path}")


def _prepare_book(
    source_path: Path, output_root: Path, page_range: Optional[str]
) -> Optional[BookSpec]:
    """Build a ``BookSpec``. Returns ``None`` if the book is already fully
    assembled and should be skipped."""
    stem = source_path.stem
    stem_dir = output_root / stem

    # Already done? Canonical .md present.
    canonical_md = stem_dir / f"{stem}.md"
    if canonical_md.exists() and canonical_md.stat().st_size > 0:
        return None

    stem_dir.mkdir(parents=True, exist_ok=True)

    # Source PDF changed under us? Purge stale .partial.
    if manifest.partial_dir(stem_dir).exists() and not manifest.source_matches(
        stem_dir, source_path
    ):
        logger.warning(
            "source for %s changed since last run; purging stale .partial/", stem
        )
        manifest.purge_partial(stem_dir)

    # Count pages and resolve expected_pages (full file or page-range subset).
    config = {"page_range": page_range} if page_range else {}
    try:
        total_pages = count_file_pages(str(source_path), config)
    except Exception as exc:  # noqa: BLE001 - surface to caller, skip this book
        logger.error("count_file_pages failed for %s: %s", source_path, exc)
        return None

    if page_range:
        parsed = parse_range_str(page_range)
        # Clamp to actual page count; parse_range_str doesn't know the cap.
        expected_pages = sorted(p for p in parsed if 0 <= p < total_pages)
    else:
        expected_pages = list(range(total_pages))

    # Record state so resume + source-fingerprinting work next time.
    if manifest.read_state(stem_dir) is None:
        manifest.write_state(stem_dir, source_path, expected_pages)

    done = manifest.read_partial_state(stem_dir)
    pending = [p for p in expected_pages if p not in done]

    return BookSpec(
        source_path=source_path,
        stem=stem,
        stem_dir=stem_dir,
        total_pages=total_pages,
        expected_pages=expected_pages,
        pending_pages=pending,
    )


# ---------- progress tracking --------------------------------------------


class Progress:
    """Thread-safe counters with one-line stdout updates per page completion.

    Output format: ``[stem]  N/T  p0042  q=14``. Cheap to parse, friendly
    to redirected logs (no carriage returns / ANSI).
    """

    def __init__(self, log_every: int = 1):
        self._lock = threading.Lock()
        self._totals: dict[str, int] = {}
        self._completed: dict[str, int] = {}
        self._log_every = max(1, log_every)

    def register_book(self, stem: str, total: int, already_done: int) -> None:
        with self._lock:
            self._totals[stem] = total
            self._completed[stem] = already_done

    def page_done(
        self, stem: str, page_num: int, queue_size: int = -1, error: bool = False
    ) -> None:
        with self._lock:
            self._completed[stem] = self._completed.get(stem, 0) + 1
            done = self._completed[stem]
            total = self._totals.get(stem, 0)
        if done % self._log_every != 0 and done != total:
            return
        flag = " ERR" if error else ""
        suffix = f"  q={queue_size}" if queue_size >= 0 else ""
        # Truncate long stems; keep alignment readable.
        stem_short = stem if len(stem) <= 40 else stem[:37] + "..."
        print(
            f"  [{stem_short:<40}]  {done}/{total}  p{page_num:04d}{flag}{suffix}",
            flush=True,
        )

    def book_assembled(self, stem: str, num_pages: int, num_chunks: int) -> None:
        stem_short = stem if len(stem) <= 40 else stem[:37] + "..."
        print(
            f"  ✓ assembled [{stem_short}]  {num_pages} pages, {num_chunks} chunks",
            flush=True,
        )


# ---------- worker -------------------------------------------------------


def _process_page(
    book: BookSpec,
    page_num: int,
    image: Image.Image,
    model: InferenceManager,
    generate_kwargs: dict,
) -> bool:
    """OCR one page and atomically write its per-page artifact.

    Returns ``True`` when the artifact records a clean OCR, ``False`` if the
    artifact records an error. Either way, the artifact IS written so the page
    counts as complete and the book can finish assembling.
    """
    try:
        item = BatchInputItem(
            image=image,
            prompt_type=generate_kwargs.get("prompt_type", "ocr_layout"),
            file_stem=book.stem,
            page_num=page_num,
        )
        # Strip kwargs that aren't accepted by InferenceManager.generate.
        gk = {k: v for k, v in generate_kwargs.items() if k != "prompt_type"}
        results = model.generate([item], **gk)
        result = results[0]

        # Save image crops directly to canonical location. Hash-based names
        # are deterministic so repeat runs land on the same files.
        image_files: list[str] = []
        for img_name, pil_img in (result.images or {}).items():
            try:
                pil_img.save(book.stem_dir / img_name)
                image_files.append(img_name)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "save image %s for %s p%04d", img_name, book.stem, page_num
                )

        artifact = {
            "page_num": page_num,
            "raw_html": result.raw,
            "markdown": result.markdown,
            "html_filtered": result.html,
            "page_box": result.page_box,
            "token_count": result.token_count,
            "error": bool(result.error),
            "error_message": None,
            "chunks": list(result.chunks or []),
            "image_files": image_files,
        }
        ok = not result.error
    except Exception as exc:  # noqa: BLE001 - workers MUST NOT crash the pool
        logger.exception("worker failed on %s p%04d", book.stem, page_num)
        artifact = {
            "page_num": page_num,
            "raw_html": "",
            "markdown": "",
            "html_filtered": "",
            "page_box": [0, 0, 0, 0],
            "token_count": 0,
            "error": True,
            "error_message": f"{type(exc).__name__}: {exc}",
            "chunks": [],
            "image_files": [],
        }
        ok = False

    manifest.atomic_write_json(
        manifest.page_artifact_path(book.stem_dir, page_num), artifact
    )
    return ok


# ---------- producer + pipeline runner -----------------------------------


_SENTINEL = object()


def _producer(
    books: Iterable[BookSpec],
    page_queue: "queue.Queue",
    n_workers: int,
    stop_event: threading.Event,
) -> None:
    """Render pending pages from each book into the queue, then signal exit.

    Backpressure: ``page_queue`` is bounded, so this thread blocks when
    workers fall behind. That's fine — it keeps memory bounded by queue size
    instead of by total page count.
    """
    try:
        for book in books:
            if stop_event.is_set():
                break
            if not book.pending_pages:
                continue
            # Build a 1-indexed page_range covering only pending pages so
            # iter_file_pages renders no wasted pages on resume.
            page_range = ",".join(str(p + 1) for p in book.pending_pages)
            config = {"page_range": page_range}
            try:
                pages_iter = iter_file_pages(str(book.source_path), config)
                for page_num, image in zip(book.pending_pages, pages_iter):
                    if stop_event.is_set():
                        break
                    page_queue.put((book, page_num, image))
            except Exception:  # noqa: BLE001
                logger.exception("producer failed on %s", book.source_path)
    finally:
        # One sentinel per worker so each thread exits cleanly.
        for _ in range(n_workers):
            try:
                page_queue.put(_SENTINEL, timeout=10)
            except queue.Full:
                logger.warning("producer couldn't enqueue all sentinels")
                break


def _worker(
    page_queue: "queue.Queue",
    model: InferenceManager,
    generate_kwargs: dict,
    progress: Progress,
) -> None:
    while True:
        item = page_queue.get()
        if item is _SENTINEL:
            return
        book, page_num, image = item
        ok = _process_page(book, page_num, image, model, generate_kwargs)
        progress.page_done(
            book.stem,
            page_num,
            queue_size=page_queue.qsize(),
            error=not ok,
        )


def _assembler(
    books: list[BookSpec],
    progress: Progress,
    paginate_output: bool,
    save_html: bool,
    stop_event: threading.Event,
    poll_seconds: float = 1.0,
) -> None:
    """Watch each book's ``.partial/`` directory; assemble when complete."""
    pending = {b.stem: b for b in books}
    while pending and not stop_event.is_set():
        for stem, book in list(pending.items()):
            done = manifest.read_partial_state(book.stem_dir)
            if done >= set(book.expected_pages):
                try:
                    stats = manifest.assemble_book(
                        book.stem_dir,
                        book.source_path.name,
                        paginate_output=paginate_output,
                        save_html=save_html,
                    )
                    progress.book_assembled(
                        stem, stats["num_pages"], stats["total_chunks"]
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("assembly failed for %s", stem)
                del pending[stem]
        if pending:
            time.sleep(poll_seconds)


def run_pipeline(
    books: list[BookSpec],
    model: InferenceManager,
    n_workers: int = 8,
    paginate_output: bool = False,
    save_html: bool = True,
    generate_kwargs: Optional[dict] = None,
    queue_size: Optional[int] = None,
    log_every: int = 1,
) -> dict:
    """Run the full pipeline. Blocks until every book is assembled (or until
    interrupted, in which case partial state remains on disk for next run).
    Returns a stats dict.
    """
    if not books:
        return {"books": 0, "pages_processed": 0, "pages_pending": 0}

    pending_pages_total = sum(len(b.pending_pages) for b in books)
    if pending_pages_total == 0:
        # Everything is already on disk; just need to assemble.
        return _assemble_only(books, paginate_output, save_html)

    if queue_size is None:
        queue_size = max(n_workers * 2, 16)
    page_queue: queue.Queue = queue.Queue(maxsize=queue_size)

    progress = Progress(log_every=log_every)
    for book in books:
        progress.register_book(book.stem, len(book.expected_pages), book.already_done)

    stop_event = threading.Event()
    gk = dict(generate_kwargs or {})

    assembler_thread = threading.Thread(
        target=_assembler,
        args=(books, progress, paginate_output, save_html, stop_event),
        name="chandra-assembler",
        daemon=False,
    )
    assembler_thread.start()

    try:
        with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="chandra-worker"
        ) as pool:
            worker_futures = [
                pool.submit(_worker, page_queue, model, gk, progress)
                for _ in range(n_workers)
            ]
            try:
                _producer(books, page_queue, n_workers, stop_event)
                # Wait for all workers to finish (they exit on sentinel).
                for fut in worker_futures:
                    fut.result()
            except KeyboardInterrupt:
                logger.warning("interrupted; signaling workers to exit")
                stop_event.set()
                # Drain queue and inject sentinels to unblock workers.
                while True:
                    try:
                        page_queue.get_nowait()
                    except queue.Empty:
                        break
                for _ in range(n_workers):
                    try:
                        page_queue.put_nowait(_SENTINEL)
                    except queue.Full:
                        break
                raise
    finally:
        # Allow the assembler to finish books that completed during this run.
        assembler_thread.join(timeout=60.0)
        if assembler_thread.is_alive():
            logger.warning("assembler still running after 60s; abandoning")

    # Pages still pending = pages that didn't make it onto disk for books
    # that haven't been assembled yet. Once a book assembles, .partial/ is
    # gone and the canonical .md is the truth — those pages count as done.
    pages_pending = 0
    for b in books:
        if (b.stem_dir / f"{b.stem}.md").exists():
            continue
        pages_pending += len(
            set(b.expected_pages) - manifest.read_partial_state(b.stem_dir)
        )

    return {
        "books": len(books),
        "pages_processed": pending_pages_total,
        "pages_pending": pages_pending,
    }


def _assemble_only(
    books: list[BookSpec], paginate_output: bool, save_html: bool
) -> dict:
    """Fast-path when every book's pages are already on disk: just assemble."""
    progress = Progress()
    for book in books:
        progress.register_book(book.stem, len(book.expected_pages), book.already_done)
        try:
            stats = manifest.assemble_book(
                book.stem_dir,
                book.source_path.name,
                paginate_output=paginate_output,
                save_html=save_html,
            )
            progress.book_assembled(
                book.stem, stats["num_pages"], stats["total_chunks"]
            )
        except Exception:  # noqa: BLE001
            logger.exception("assembly failed for %s", book.stem)
    return {"books": len(books), "pages_processed": 0, "pages_pending": 0}
