"""
Interactive prompt CLI for local/chat HuggingFace models.

Loads a model once, then runs a REPL loop:
- user enters prompts
- model returns answers
- typing quit/exit/q closes the script
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval.checkpoint_runner import (
    load_adapter_for_checkpoint,
    load_base_model,
    make_generate_fn,
    read_base_model_name,
)
from src.utilities.training_logger import Logger

_EXIT_COMMANDS = {"quit", "exit", "q"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive prompt loop for a local HuggingFace chat model."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--model",
        metavar="MODEL_ID_OR_PATH",
        help="Base HuggingFace model id or local model path.",
    )
    target.add_argument(
        "--checkpoint",
        metavar="CHECKPOINT_DIR",
        help=(
            "LoRA checkpoint directory (for example, Unsloth checkpoint-* folder). "
            "Base model is auto-read from adapter_config.json."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model precision (default: bfloat16).",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=False,
        help="Load base model in 4-bit quantization (BitsAndBytes).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Device selector: cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        metavar="N",
        help="Maximum new tokens for each answer.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=False,
        help="Enable sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (used when --do-sample is set).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling p value (used when --do-sample is set).",
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default=None,
        help="Optional system message inserted before each user prompt.",
    )
    return parser


def _is_exit_command(text: str) -> bool:
    return text.strip().lower() in _EXIT_COMMANDS


def _load_prompt_model(args) -> tuple[object, object]:
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        base_model_name = read_base_model_name(checkpoint_path)
        Logger.print_info("base model", base_model_name)
        Logger.print_info("checkpoint", str(checkpoint_path))
        model, tokenizer = load_base_model(
            base_model_name=base_model_name,
            dtype=args.dtype,
            use_4bit=args.use_4bit,
            gpu=args.gpu,
        )
        model = load_adapter_for_checkpoint(model, checkpoint_path)
        return model, tokenizer

    model, tokenizer = load_base_model(
        base_model_name=args.model,
        dtype=args.dtype,
        use_4bit=args.use_4bit,
        gpu=args.gpu,
    )
    return model, tokenizer


def main() -> None:
    args = _build_parser().parse_args()
    Logger.print_header("Interactive Prompt")
    if args.model:
        Logger.print_info("model", args.model)
    if args.checkpoint:
        Logger.print_info("model", "loaded from checkpoint adapter")
    Logger.print_info("dtype", args.dtype)
    Logger.print_info("4bit", args.use_4bit)
    Logger.print_info("gpu", args.gpu if args.gpu else "auto")

    model, tokenizer = _load_prompt_model(args)
    generate_fn = make_generate_fn(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        system_message=args.system_message,
    )

    Logger.print_success("Model loaded. Type your prompt.")
    Logger.print_info("exit commands", "quit | exit | q")

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            Logger.print_warning("Interrupted. Closing prompt session.")
            break

        if _is_exit_command(prompt):
            Logger.print_success("Prompt session closed.")
            break

        if not prompt:
            continue

        answer = generate_fn(prompt)
        Logger.print_section("Assistant")
        Logger.print_info("answer", answer)


if __name__ == "__main__":
    main()
