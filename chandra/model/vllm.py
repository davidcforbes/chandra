from __future__ import annotations

import base64
import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import List, Optional

from PIL import Image
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from chandra.model.schema import BatchInputItem, GenerationResult
from chandra.model.util import detect_repeat_token, scale_to_fit
from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings

logger = logging.getLogger(__name__)

# Errors that look transient and are worth retrying with the same temperature.
_TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


def image_to_base64(
    image: Image.Image,
    format: str = "JPEG",
    quality: int = 92,
) -> str:
    """Encode a PIL image as base64. Default JPEG (~10x faster than PNG)."""
    buffered = io.BytesIO()
    save_kwargs: dict = {"format": format}
    if format.upper() in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = quality
        if format.upper() == "JPEG":
            # JPEG can't carry alpha; ensure RGB.
            if image.mode != "RGB":
                image = image.convert("RGB")
    image.save(buffered, **save_kwargs)
    return base64.b64encode(buffered.getvalue()).decode()


_CLIENT_CACHE: dict[tuple, OpenAI] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


def get_openai_client(
    api_base: str,
    api_key: str,
    custom_headers: Optional[dict] = None,
) -> OpenAI:
    """Return a memoized OpenAI client keyed by (base, key, headers).

    Constructing a fresh ``OpenAI`` per request creates a new HTTP connection
    pool each time. This cache lets the executor reuse keep-alive connections
    across batches without thread-safety hazards (the OpenAI SDK is
    thread-safe).
    """
    headers_key = tuple(sorted(custom_headers.items())) if custom_headers else None
    cache_key = (api_base, api_key, headers_key)
    with _CLIENT_CACHE_LOCK:
        client = _CLIENT_CACHE.get(cache_key)
        if client is None:
            client = OpenAI(
                api_key=api_key, base_url=api_base, default_headers=custom_headers
            )
            _CLIENT_CACHE[cache_key] = client
        return client


def _classify_error(exc: Exception) -> tuple[str, bool]:
    """Categorize an exception. Returns (category, is_retryable)."""
    if isinstance(exc, AuthenticationError):
        return ("auth", False)
    if isinstance(exc, RateLimitError):
        return ("rate_limit", True)
    if isinstance(exc, APITimeoutError):
        return ("timeout", True)
    if isinstance(exc, APIConnectionError):
        return ("connection", True)
    if isinstance(exc, APIStatusError):
        # 5xx → retry, 4xx → no retry.
        retryable = 500 <= getattr(exc, "status_code", 0) < 600
        return (f"http_{getattr(exc, 'status_code', 'unknown')}", retryable)
    return ("unknown", True)


def generate_vllm(
    batch: List[BatchInputItem],
    max_output_tokens: int = None,
    max_retries: int = None,
    max_workers: int | None = None,
    custom_headers: dict | None = None,
    max_failure_retries: int | None = None,
    bbox_scale: int = settings.BBOX_SCALE,
    vllm_api_base: str = settings.VLLM_API_BASE,
    temperature: float = 0.0,
    top_p: float = 0.1,
    image_format: Optional[str] = None,
    image_quality: Optional[int] = None,
) -> List[GenerationResult]:
    client = get_openai_client(
        api_base=vllm_api_base,
        api_key=settings.VLLM_API_KEY,
        custom_headers=custom_headers,
    )
    model_name = settings.VLLM_MODEL_NAME

    if max_retries is None:
        max_retries = settings.MAX_VLLM_RETRIES
    if max_failure_retries is None:
        # By default, the same budget covers the soft-error path.
        max_failure_retries = max_retries

    if max_workers is None:
        max_workers = min(64, len(batch))

    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    fmt = (image_format or settings.VLLM_IMAGE_FORMAT).upper()
    quality = (
        image_quality if image_quality is not None else settings.VLLM_IMAGE_QUALITY
    )
    mime = "image/jpeg" if fmt == "JPEG" else f"image/{fmt.lower()}"

    def _generate(item: BatchInputItem, temperature, top_p) -> GenerationResult:
        prompt = item.prompt
        if not prompt:
            prompt = PROMPT_MAPPING[item.prompt_type]

        image = scale_to_fit(item.image)
        image_b64 = image_to_base64(image, format=fmt, quality=quality)

        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{image_b64}"},
            },
            {"type": "text", "text": prompt},
        ]

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            raw = completion.choices[0].message.content
            return GenerationResult(
                raw=raw,
                token_count=completion.usage.completion_tokens,
                error=False,
            )
        except Exception as exc:  # noqa: BLE001 — typed inside _classify_error
            category, _retryable = _classify_error(exc)
            logger.warning(
                "vLLM generation failed (%s): %s", category, exc, exc_info=True
            )
            return GenerationResult(raw="", token_count=0, error=True)

    def _should_retry(result: GenerationResult, retries: int) -> bool:
        # 1) Repeat-token loop is the most common bad-output failure mode.
        # detect_repeat_token now performs both the literal-tail and
        # trimmed-tail scans in one pass.
        has_repeat = detect_repeat_token(result.raw)
        if has_repeat and retries < max_retries:
            logger.info(
                "Detected repeat-token loop, retrying (attempt %d/%d)",
                retries + 1,
                max_retries,
            )
            return True

        # 2) Soft errors (network, 5xx, etc.) — retry up to max_failure_retries.
        if result.error and retries < max_failure_retries:
            logger.info(
                "vLLM error result, retrying (attempt %d/%d) after backoff",
                retries + 1,
                max_failure_retries,
            )
            time.sleep(2 * (retries + 1))
            return True

        return False

    def process_item(item, _max_retries, _max_failure_retries=None):
        result = _generate(item, temperature=temperature, top_p=top_p)
        retries = 0
        while _should_retry(result, retries):
            retry_temperature = min(temperature + 0.2 * (retries + 1), 0.8)
            result = _generate(item, temperature=retry_temperature, top_p=0.95)
            retries += 1
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                process_item, batch, repeat(max_retries), repeat(max_failure_retries)
            )
        )

    return results
