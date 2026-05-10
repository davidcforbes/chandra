from dotenv import find_dotenv
from pydantic_settings import BaseSettings
import os


def _resolve_env_file() -> str | None:
    """Find the project's env file.

    Prefers the conventional ``.env`` (python-dotenv default + ecosystem norm)
    and falls back to the legacy ``local.env`` for backwards compatibility.
    """
    for candidate in (".env", "local.env"):
        path = find_dotenv(candidate)
        if path:
            return path
    return None


class Settings(BaseSettings):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_DPI: int = 192
    MIN_PDF_IMAGE_DIM: int = 1024
    MIN_IMAGE_DIM: int = 1536
    MODEL_CHECKPOINT: str = "datalab-to/chandra-ocr-2"
    TORCH_DEVICE: str | None = None
    MAX_OUTPUT_TOKENS: int = 12384
    TORCH_ATTN: str | None = None
    BBOX_SCALE: int = 1000

    # Image encoding for the vLLM/openai-compatible request payloads. PNG is
    # lossless but slow to encode on multi-megapixel images; JPEG/WebP encode
    # 5-10x faster with negligible OCR-quality impact.
    VLLM_IMAGE_FORMAT: str = "JPEG"
    VLLM_IMAGE_QUALITY: int = 92

    # vLLM server settings
    VLLM_API_KEY: str = "EMPTY"
    VLLM_API_BASE: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "chandra"
    VLLM_GPUS: str = "0"
    MAX_VLLM_RETRIES: int = 6
    # Per-request timeout in seconds. The OpenAI SDK default of 600s is too
    # tight when running vLLM on a Windows desktop GPU: the compositor
    # periodically preempts the GPU, throughput drops to 0 tokens/s for
    # 10+ seconds at a time, and a long-generation page (close to
    # MAX_OUTPUT_TOKENS) accumulates enough stall time to trip the
    # client timeout. 1200s gives plenty of headroom while still
    # surfacing genuinely hung requests.
    VLLM_CLIENT_TIMEOUT: float = 1200.0

    class Config:
        env_file = _resolve_env_file()
        extra = "ignore"


settings = Settings()
