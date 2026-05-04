#!/usr/bin/env python3
"""
Query the Rabit0 fine-tuned Qwen model with a set of questions.

This script loads the latest Rabit0 LoRA adapters from
`rabit0-v1.0-qwen3-30b` and runs them against the recommended
base model (`unsloth/qwen2.5-coder-14b-instruct-bnb-4bit` by default).

Usage:
    python src/review/query_rabit0_model.py \
        --questions "Give me a red-team engagement outline" \
        --questions "How do you detect SSRF attacks?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List

logger = logging.getLogger(__name__)

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER_PATH = REPO_ROOT / "rabit0-v1.0-qwen3-30b"
DEFAULT_BASE_MODEL = "unsloth/qwen2.5-coder-14b-instruct-bnb-4bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions to the Rabit0 fine-tuned Qwen model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adapter-path",
        default=str(DEFAULT_ADAPTER_PATH),
        help="Absolute path to the Rabit0 LoRA adapter directory.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model identifier used when Rabit0 was fine-tuned.",
    )
    parser.add_argument(
        "--questions",
        nargs="*",
        default=[
            "What are the highest-impact red teaming tasks you can help automate?",
            "Outline a plan for detecting data exfiltration across a hybrid cloud environment.",
            "Summarize the latest SSRF mitigation techniques for Kubernetes workloads.",
        ],
        help="Questions to ask the model (provide multiple entries for multiple questions).",
    )
    parser.add_argument(
        "--questions-file",
        help="Optional path to a JSON or newline-delimited text file containing questions.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are Rabit0, an elite red-team security assistant. "
        "Provide concise, actionable answers grounded in real-world practices.",
        help="System message to steer the assistant's behavior.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate for each answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (set to 0 for greedy decoding).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling cumulative probability.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Penalty factor to discourage repeated phrases.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length for prompts (used during tokenization).",
    )
    parser.add_argument(
        "--merge-and-unload",
        action="store_true",
        help="Merge LoRA weights into the base model to speed up inference "
        "(requires additional memory).",
    )
    return parser.parse_args()


def load_questions(args: argparse.Namespace) -> List[str]:
    """Combine inline questions and optional file-based questions."""
    questions: List[str] = list(args.questions or [])

    if args.questions_file:
        file_path = Path(args.questions_file).expanduser().resolve()
        if not file_path.exists():
            msg = f"Questions file not found: {file_path}"
            raise FileNotFoundError(msg)

        if file_path.suffix.lower() in {".json", ".jsonl"}:
            with file_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                file_questions = list(data.values())
            elif isinstance(data, list):
                file_questions = data
            else:
                msg = "JSON questions file must contain a list or dict of strings."
                raise ValueError(msg)
        else:
            with file_path.open("r", encoding="utf-8") as fh:
                file_questions = [line.strip() for line in fh if line.strip()]

        for q in file_questions:
            if not isinstance(q, str):
                raise ValueError("All questions must be strings.")
        questions.extend(file_questions)

    if not questions:
        raise ValueError("No questions provided. Use --questions or --questions-file.")

    # Deduplicate while preserving order
    seen = set()
    unique_questions = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    return unique_questions


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: Path,
    max_seq_length: int,
    merge_and_unload: bool,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load the 4-bit base model and apply the Rabit0 LoRA adapters."""
    adapter_path = adapter_path.expanduser().resolve()
    if not adapter_path.exists():
        msg = f"Adapter path does not exist: {adapter_path}"
        raise FileNotFoundError(msg)

    logger.info("Loading tokenizer from %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model %s (4-bit quantized)", base_model)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = True
    model.config.max_position_embeddings = max_seq_length

    logger.info("Loading Rabit0 LoRA adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()

    if merge_and_unload:
        logger.info("Merging LoRA weights into the base model (this may take a moment)...")
        model = model.merge_and_unload()
        model.eval()

    return model, tokenizer


def build_generation_config(args: argparse.Namespace, tokenizer) -> GenerationConfig:
    do_sample = args.temperature > 0
    config_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        config_kwargs.update(
            {
                "temperature": max(args.temperature, 1e-5),
                "top_p": args.top_p,
            }
        )
    return GenerationConfig(**config_kwargs)


def prepare_prompt(tokenizer, system_prompt: str, question: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except AttributeError:
        # Fallback: construct a simple prompt if the tokenizer lacks a chat template
        prompt = ""
        if system_prompt:
            prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def move_inputs_to_device(inputs, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def generate_answer(
    model: torch.nn.Module,
    tokenizer,
    generation_config: GenerationConfig,
    prompt: str,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")

    device = get_model_device(model)
    inputs = move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response


def ask_questions(
    model: torch.nn.Module,
    tokenizer,
    generation_config: GenerationConfig,
    questions: Iterable[str],
    system_prompt: str,
) -> None:
    for idx, question in enumerate(questions, start=1):
        logger.info("=" * 80)
        logger.info("Question %s: %s", idx, question)
        prompt = prepare_prompt(tokenizer, system_prompt, question)
        answer = generate_answer(model, tokenizer, generation_config, prompt)
        logger.info("Answer: %s", answer)
        logger.info("=" * 80)


def get_model_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    if hasattr(model, "hf_device_map"):
        # Pick the first device from the device map
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, str):
            return torch.device(first_device)
    if hasattr(model, "base_model"):
        try:
            return get_model_device(model.base_model)  # type: ignore[return-value]
        except RecursionError:
            pass
    return next(model.parameters()).device


def main() -> int:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        questions = load_questions(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    try:
        model, tokenizer = load_model_and_tokenizer(
            base_model=args.base_model,
            adapter_path=Path(args.adapter_path),
            max_seq_length=args.max_seq_length,
            merge_and_unload=args.merge_and_unload,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.error("Failed to load model: %s", exc)
        return 1

    generation_config = build_generation_config(args, tokenizer)
    ask_questions(model, tokenizer, generation_config, questions, args.system_prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())

