#!/usr/bin/env python3
"""
Launch a vLLM OpenAI-compatible server for the Rabit0 fine-tuned model.

This script wraps `vllm.entrypoints.openai.api_server` and wires up the
base model with the Rabit0 LoRA adapters so you can query the model via
the OpenAI REST API (or the official OpenAI SDKs).

For vLLM-compatible models (trained with single_stage_finetune.py):
- Use merged models: --base-model ./rabit0-v1.0-qwen3-7b-vllm-merged --no-lora
- Or use LoRA adapters: --base-model Qwen/Qwen2.5-Coder-7B-Instruct --adapter-path ./rabit0-v1.0-qwen3-7b-vllm

Examples
--------
Start a local server on port 8000 with LoRA adapters:
    python src/main/serve/serve_rabit0_vllm.py

Serve a merged model (vLLM-compatible, no LoRA):
    python src/main/serve/serve_rabit0_vllm.py \
        --base-model ./rabit0-v1.0-qwen3-7b-vllm-merged \
        --no-lora \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192

Enable tool calling (for Qwen models):
    python src/main/serve/serve_rabit0_vllm.py \
        --base-model ./rabit0-v1.0-qwen3-7b-vllm-merged-fixed \
        --no-lora \
        --enable-tool-calling \
        --tool-call-parser hermes

Expose on a different port with tensor parallelism:
    python src/main/serve/serve_rabit0_vllm.py --port 9000 --tensor-parallel-size 2

Once running, you can query the model with:
    curl http://127.0.0.1:8000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{"model":"rabit0", "messages":[{"role":"user","content":"Ping!"}]}'
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LORA_PATH = REPO_ROOT / "rabit0-v1.0-qwen3-30b"

# During fine-tuning we used the 4-bit Unsloth build, but for vLLM we should
# target the original full-precision weights. vLLM does not yet support the
# Unsloth 4-bit checkpoints directly.
# 
# For 20GB GPU, consider using an AWQ-quantized base model:
# DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"  # If available
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose the Rabit0 LoRA model via the vLLM OpenAI server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base Hugging Face model to load with vLLM.",
    )
    parser.add_argument(
        "--adapter-path",
        default=str(DEFAULT_LORA_PATH),
        help="Absolute path to the Rabit0 LoRA adapter directory.",
    )
    parser.add_argument(
        "--served-model-name",
        default="rabit0",
        help="Model identifier exposed via the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP address for the vLLM server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port for the vLLM server.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallel inference.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Upper bound on GPU memory utilization (0-1).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=12288,
        help="Maximum sequence length for vLLM (tokens).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Computation dtype for vLLM (auto, float16, bfloat16, float32).",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trusting remote code when loading the base model.",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Skip loading the Rabit0 LoRA adapter (serve base model only).",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Quantization method (awq, gptq, fp8, etc.).",
    )
    parser.add_argument(
        "--enable-tool-calling",
        action="store_true",
        help="Enable tool/function calling support (requires --tool-call-parser).",
    )
    parser.add_argument(
        "--tool-call-parser",
        default=None,
        help="Tool call parser (hermes for Qwen models, minimax, deepseek_v3, etc.).",
    )
    parser.add_argument(
        "--tool-parser-plugin",
        default=None,
        help="Path to custom tool parser plugin file.",
    )
    parser.add_argument(
        "--extra-flag",
        action="append",
        default=[],
        help="Optional passthrough flags for the vLLM server (repeatable).",
    )
    return parser.parse_args()


def check_gpu_availability() -> bool:
    """Check if GPU/CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        return False


def ensure_vllm_available() -> None:
    if shutil.which("python") is None:
        raise RuntimeError("Python executable not found in PATH.")

    try:
        import vllm  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - import side effect
        raise RuntimeError(
            "vLLM is not installed. Install with `pip install vllm` "
            "or `pip install vllm==0.5.4` (or newer)."
        ) from exc
    
    # Check GPU availability
    if not check_gpu_availability():
        raise RuntimeError(
            "GPU/CUDA is not available. vLLM requires a GPU to run.\n"
            "Please ensure:\n"
            "  1. NVIDIA GPU is installed\n"
            "  2. NVIDIA drivers are installed\n"
            "  3. CUDA libraries are available\n"
            "  4. If using Docker, NVIDIA Container Toolkit is installed\n"
            "     and the container has GPU access (--gpus all or deploy.resources)"
        )


def validate_adapter_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"LoRA adapter path not found: {path}\n"
            "Make sure you have fine-tuned Rabit0 and the directory exists."
        )


def build_vllm_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.base_model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--dtype",
        args.dtype,
        "--served-model-name",
        args.served_model_name,
    ]

    if not args.no_trust_remote_code:
        cmd.append("--trust-remote-code")

    if args.quantization:
        cmd.extend(["--quantization", args.quantization])

    if args.enable_tool_calling:
        cmd.append("--enable-auto-tool-choice")
        if args.tool_parser_plugin:
            # Use custom parser plugin when provided
            plugin_path = Path(args.tool_parser_plugin).expanduser().resolve()
            if not plugin_path.exists():
                raise FileNotFoundError(f"Tool parser plugin not found: {plugin_path}")
            cmd.extend(["--tool-parser-plugin", str(plugin_path)])
        if args.tool_call_parser:
            cmd.extend(["--tool-call-parser", args.tool_call_parser])
        elif "qwen" in args.base_model.lower():
            # Default to hermes parser for Qwen models when no custom parser is set
            cmd.extend(["--tool-call-parser", "hermes"])

    if not args.no_lora:
        adapter_path = Path(args.adapter_path).expanduser().resolve()
        validate_adapter_path(adapter_path)
        cmd.extend(
            [
                "--enable-lora",
                "--lora-modules",
                f"{args.served_model_name}={adapter_path}",
            ]
        )

    if args.extra_flag:
        cmd.extend(args.extra_flag)

    return cmd


def main() -> int:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        ensure_vllm_available()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    command = build_vllm_command(args)

    logger.info("=" * 80)
    logger.info("Starting vLLM OpenAI server for Rabit0")
    logger.info("=" * 80)
    logger.info("Command: %s", " ".join(command))
    logger.info("=" * 80)

    # Forward stdout/stderr so the user can monitor server logs.
    process = subprocess.Popen(command)
    try:
        process.wait()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down vLLM server...")
        process.terminate()
        process.wait()
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())

