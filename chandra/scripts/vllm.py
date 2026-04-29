from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys

from chandra.settings import settings

logger = logging.getLogger(__name__)

# H100 80GB is the baseline for scaling.
BASELINE_VRAM_GB = 80
BASELINE_MAX_BATCHED_TOKENS = 8192
BASELINE_MAX_NUM_SEQS = 64

GPU_VRAM_GB = {
    "h100": 80,
    "a100-80": 80,
    "a100": 40,
    "a100-40": 40,
    "l40s": 48,
    "a10": 24,
    "l4": 24,
    "4090": 24,
    "3090": 24,
    "t4": 16,
}


def get_gpu_settings(gpu: str) -> tuple[int, int]:
    """Return (max_batched_tokens, max_num_seqs) for the given GPU type."""
    vram = GPU_VRAM_GB.get(gpu)
    if vram is None:
        available = ", ".join(sorted(GPU_VRAM_GB.keys()))
        raise SystemExit(f"Unknown GPU '{gpu}'. Available: {available}")

    ratio = vram / BASELINE_VRAM_GB
    raw_tokens = BASELINE_MAX_BATCHED_TOKENS * ratio
    max_batched_tokens = max(1024, 2 ** math.floor(math.log2(raw_tokens)))
    max_num_seqs = max(8, (int(BASELINE_MAX_NUM_SEQS * ratio) // 8) * 8)
    return max_batched_tokens, max_num_seqs


def docker_invocation(docker_bin: str = "docker") -> list[str]:
    """Return the command prefix for running docker.

    Tries ``<docker_bin> info`` without sudo first; if that fails AND ``sudo``
    is available, prepends sudo. This avoids the previous hardcoded ``sudo``
    that broke on Docker Desktop, rootless docker, and any setup with the
    user in the ``docker`` group.
    """
    if shutil.which(docker_bin) is None:
        raise SystemExit(
            f"docker binary {docker_bin!r} not found on PATH; "
            "install Docker / Docker Desktop or pass --docker-bin"
        )

    # If `docker info` works without sudo, we don't need it.
    try:
        result = subprocess.run(
            [docker_bin, "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        result = None

    if result is not None and result.returncode == 0:
        return [docker_bin]

    # Fall back to sudo if available.
    if shutil.which("sudo") is not None:
        logger.info("docker requires elevated privileges; using sudo")
        return ["sudo", docker_bin]

    # No sudo — return docker_bin and let the actual run fail with a
    # platform-appropriate error (Docker Desktop typically does not need sudo).
    return [docker_bin]


def build_command(
    args: argparse.Namespace,
    max_batched_tokens: int,
    max_num_seqs: int,
) -> list[str]:
    cmd = docker_invocation(args.docker_bin)
    cmd += [
        "run",
        "--rm",
    ]
    if args.gpu_runtime:
        cmd += ["--runtime", args.gpu_runtime]
    cmd += [
        "--gpus",
        f"device={settings.VLLM_GPUS}",
        "-v",
        f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
        "-p",
        f"{args.port}:8000",
        "--ipc=host",
        args.image,
        "--model",
        settings.MODEL_CHECKPOINT,
        "--no-enforce-eager",
        "--max-num-seqs",
        str(max_num_seqs),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "18000",
        "--max_num_batched_tokens",
        str(max_batched_tokens),
        "--gpu-memory-utilization",
        ".85",
        "--enable-prefix-caching",
        "--mm-processor-kwargs",
        json.dumps({"min_pixels": 3136, "max_pixels": 6291456}),
        "--served-model-name",
        settings.VLLM_MODEL_NAME,
    ]
    if args.mtp:
        spec_config = json.dumps({"method": "mtp", "num_speculative_tokens": 1})
        cmd.extend(["--speculative-config", spec_config])
    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch vLLM server for Chandra")
    parser.add_argument(
        "--gpu",
        default="h100",
        choices=sorted(GPU_VRAM_GB.keys()),
        help="GPU type for optimal settings (default: h100)",
    )
    parser.add_argument(
        "--mtp",
        action="store_true",
        default=False,
        help="Enable MTP speculative decoding (disabled by default, unstable with vLLM)",
    )
    parser.add_argument(
        "--docker-bin",
        default="docker",
        help="Docker binary name or path (default: docker)",
    )
    parser.add_argument(
        "--gpu-runtime",
        default="nvidia",
        help="Container GPU runtime (default: nvidia; pass empty to omit --runtime)",
    )
    parser.add_argument(
        "--image",
        default="vllm/vllm-openai:v0.17.0",
        help="Docker image to run (default: vllm/vllm-openai:v0.17.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Host port to publish (default: 8000)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the docker command and exit (no docker invocation)",
    )
    return parser.parse_args(argv)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    max_batched_tokens, max_num_seqs = get_gpu_settings(args.gpu)
    cmd = build_command(args, max_batched_tokens, max_num_seqs)

    vram = GPU_VRAM_GB[args.gpu]
    logger.info("GPU: %s (%dGB VRAM)", args.gpu, vram)
    logger.info(
        "max-num-batched-tokens: %d, max-num-seqs: %d",
        max_batched_tokens,
        max_num_seqs,
    )
    logger.info("MTP: %s", "enabled" if args.mtp else "disabled")
    logger.info("Command: %s", " ".join(cmd))

    if args.print_only:
        return

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Shutting down vLLM server...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        logger.error("vLLM server exited with error code %d", e.returncode)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
