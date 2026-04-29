"""Unit tests for chandra.model.vllm — image encoding, client cache, retries."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from chandra.model import util as model_util
from chandra.model import vllm as vllm_mod
from chandra.model.schema import BatchInputItem


# ---------- image_to_base64 (chandra-cn0) ----------


class TestImageToBase64:
    def _decode(self, b64: str) -> bytes:
        return base64.b64decode(b64)

    def test_jpeg_smaller_than_png_on_photographic_content(self):
        # Random-noise image — the worst case for PNG (no run-length wins) and
        # the best case for lossy JPEG, mirroring real photographic OCR input.
        import random

        rng = random.Random(0)
        img = Image.new("RGB", (256, 256))
        pixels = [
            (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for _ in range(256 * 256)
        ]
        img.putdata(pixels)

        png_b64 = vllm_mod.image_to_base64(img, format="PNG")
        jpeg_b64 = vllm_mod.image_to_base64(img, format="JPEG", quality=80)
        assert len(self._decode(jpeg_b64)) < len(self._decode(png_b64))

    def test_jpeg_converts_rgba_to_rgb(self):
        # JPEG can't carry alpha; encoding must not raise.
        img = Image.new("RGBA", (50, 50))
        b64 = vllm_mod.image_to_base64(img, format="JPEG")
        decoded = self._decode(b64)
        # Decoded bytes start with JPEG SOI marker.
        assert decoded[:2] == b"\xff\xd8"

    def test_webp_supported(self):
        img = Image.new("RGB", (50, 50), "red")
        b64 = vllm_mod.image_to_base64(img, format="WEBP")
        assert len(b64) > 0

    def test_quality_parameter_affects_size(self):
        img = Image.new("RGB", (256, 256), "white")
        for x in range(0, 256, 2):
            img.putpixel((x, x), (255, 0, 0))
        small = vllm_mod.image_to_base64(img, format="JPEG", quality=20)
        large = vllm_mod.image_to_base64(img, format="JPEG", quality=95)
        assert len(self._decode(small)) < len(self._decode(large))


# ---------- get_openai_client (chandra-dk8) ----------


class TestClientCache:
    def setup_method(self):
        # Reset the cache so test isolation holds.
        vllm_mod._CLIENT_CACHE.clear()

    def test_returns_same_client_for_same_args(self):
        with patch.object(vllm_mod, "OpenAI") as fake_cls:
            fake_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw}")
            c1 = vllm_mod.get_openai_client("http://a", "k1")
            c2 = vllm_mod.get_openai_client("http://a", "k1")
            assert c1 is c2
            assert fake_cls.call_count == 1

    def test_different_endpoints_get_different_clients(self):
        with patch.object(vllm_mod, "OpenAI") as fake_cls:
            fake_cls.side_effect = lambda **kw: MagicMock()
            c1 = vllm_mod.get_openai_client("http://a", "k1")
            c2 = vllm_mod.get_openai_client("http://b", "k1")
            assert c1 is not c2

    def test_different_headers_get_different_clients(self):
        with patch.object(vllm_mod, "OpenAI") as fake_cls:
            fake_cls.side_effect = lambda **kw: MagicMock()
            c1 = vllm_mod.get_openai_client("http://a", "k", custom_headers={"x": "1"})
            c2 = vllm_mod.get_openai_client("http://a", "k", custom_headers={"x": "2"})
            assert c1 is not c2


# ---------- _classify_error (chandra-6y1) ----------


class TestClassifyError:
    def test_unknown_error_is_retryable(self):
        cat, retry = vllm_mod._classify_error(RuntimeError("?"))
        assert cat == "unknown"
        assert retry is True

    def test_auth_error_is_not_retryable(self):
        from openai import AuthenticationError

        exc = AuthenticationError.__new__(AuthenticationError)
        cat, retry = vllm_mod._classify_error(exc)
        assert cat == "auth"
        assert retry is False

    def test_rate_limit_is_retryable(self):
        from openai import RateLimitError

        exc = RateLimitError.__new__(RateLimitError)
        cat, retry = vllm_mod._classify_error(exc)
        assert cat == "rate_limit"
        assert retry is True

    def test_5xx_is_retryable_4xx_is_not(self):
        from openai import APIStatusError

        exc500 = APIStatusError.__new__(APIStatusError)
        exc500.status_code = 500
        cat, retry = vllm_mod._classify_error(exc500)
        assert "http_500" in cat
        assert retry is True

        exc400 = APIStatusError.__new__(APIStatusError)
        exc400.status_code = 400
        cat, retry = vllm_mod._classify_error(exc400)
        assert retry is False

    def test_timeout_is_retryable(self):
        from openai import APITimeoutError

        exc = APITimeoutError.__new__(APITimeoutError)
        cat, retry = vllm_mod._classify_error(exc)
        assert cat == "timeout"
        assert retry is True

    def test_connection_error_is_retryable(self):
        from openai import APIConnectionError

        exc = APIConnectionError.__new__(APIConnectionError)
        cat, retry = vllm_mod._classify_error(exc)
        assert cat == "connection"
        assert retry is True


# ---------- detect_repeat_token (chandra-kqn) ----------


class TestDetectRepeat:
    def test_detects_repeating_tail(self):
        # Long "abc" repeat at the end.
        text = "some normal text " + "abc" * 50
        assert model_util.detect_repeat_token(text) is True

    def test_clean_text_returns_false(self):
        assert model_util.detect_repeat_token("the quick brown fox") is False

    def test_detects_repeat_in_trimmed_tail(self):
        # The "raw end" has 50 chars of trailing junk that is not itself
        # repetitive, but the segment behind it is. Trim-aware pass should
        # catch it.
        clean_text = "some normal text " + "abc" * 50
        text = clean_text + "Q" * 20  # Trailing 20 chars of non-repeating noise.
        assert model_util.detect_repeat_token(text, also_check_trim=50) is True

    def test_does_not_double_call(self):
        """A single call now exercises both literal and trimmed paths."""
        with patch.object(
            model_util, "_detect_repeat_at", wraps=model_util._detect_repeat_at
        ) as spy:
            text = "x" * 20  # Short input, no trim needed.
            model_util.detect_repeat_token(text, also_check_trim=50)
            # Short text triggers only the literal scan.
            assert spy.call_count == 1

    def test_legacy_cut_from_end_path(self):
        # Backwards-compatible: when caller explicitly passes cut_from_end,
        # take the single trimmed scan and skip the also_check_trim logic.
        text = "abc" * 50 + "Q" * 30
        assert (
            model_util.detect_repeat_token(text, cut_from_end=30, also_check_trim=0)
            is True
        )


# ---------- generate_vllm with mocked client ----------


def _make_fake_completion(text: str, tokens: int = 5):
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = text
    completion.usage = MagicMock()
    completion.usage.completion_tokens = tokens
    return completion


def _patch_client(returner):
    """Return a context-managed patch on get_openai_client."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = returner
    return patch.object(
        vllm_mod, "get_openai_client", return_value=fake_client
    ), fake_client


def _make_batch(n: int = 1) -> list[BatchInputItem]:
    return [
        BatchInputItem(image=Image.new("RGB", (50, 50)), prompt_type="ocr_layout")
        for _ in range(n)
    ]


class TestGenerateVllm:
    def test_happy_path_one_call_per_item(self):
        ctx, fake_client = _patch_client(
            lambda **_: _make_fake_completion("<div>ok</div>", tokens=3)
        )
        with ctx:
            results = vllm_mod.generate_vllm(
                _make_batch(2), max_workers=1, max_retries=2
            )
        assert len(results) == 2
        assert all(not r.error for r in results)
        assert fake_client.chat.completions.create.call_count == 2

    def test_retries_on_repeat_token(self):
        # First response is a degenerate repeat. Second is fine.
        outputs = iter(
            [
                _make_fake_completion("abc" * 100, tokens=300),  # bad
                _make_fake_completion("<div>good</div>", tokens=4),  # good
            ]
        )
        ctx, fake_client = _patch_client(lambda **_: next(outputs))
        with ctx:
            results = vllm_mod.generate_vllm(
                _make_batch(1), max_workers=1, max_retries=3
            )
        assert len(results) == 1
        assert results[0].raw == "<div>good</div>"
        assert fake_client.chat.completions.create.call_count == 2

    def test_returns_error_result_on_exception(self):
        ctx, fake_client = _patch_client(
            lambda **_: (_ for _ in ()).throw(RuntimeError("network gone"))
        )
        with ctx, patch.object(vllm_mod, "time"):  # collapse retry sleep
            results = vllm_mod.generate_vllm(
                _make_batch(1),
                max_workers=1,
                max_retries=1,
                max_failure_retries=1,
            )
        assert len(results) == 1
        assert results[0].error is True
        # Initial call + 1 retry.
        assert fake_client.chat.completions.create.call_count == 2

    def test_default_max_workers_and_failure_retries(self):
        """Implicit defaults branch: pass nothing and let generate_vllm fill."""
        ctx, fake_client = _patch_client(
            lambda **_: _make_fake_completion("<div>ok</div>", tokens=2)
        )
        with ctx:
            results = vllm_mod.generate_vllm(_make_batch(1))
        assert len(results) == 1
        assert fake_client.chat.completions.create.call_count == 1

    def test_image_format_is_used(self):
        captured: list[dict] = []

        def fake_create(**kwargs):
            captured.append(kwargs)
            return _make_fake_completion("<div>ok</div>", tokens=2)

        ctx, _ = _patch_client(fake_create)
        with ctx:
            vllm_mod.generate_vllm(
                _make_batch(1),
                max_workers=1,
                max_retries=0,
                image_format="JPEG",
            )
        assert len(captured) == 1
        # The first content item is the image_url; the URL data prefix tells us
        # the encoding format.
        url = captured[0]["messages"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
