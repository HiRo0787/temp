#!/usr/bin/env python3
"""
Merge LoRA adapter with base model and save in a format compatible with vLLM.

This script:
1. Loads the base model (full precision or quantized)
2. Loads and merges the LoRA adapter
3. Saves the merged model that can be used with vLLM

For 20GB GPU, you'll need to use quantization (AWQ/GPTQ) after merging.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Move Unsloth import to top to ensure optimizations are applied correctly
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

logger = logging.getLogger(__name__)

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


def merge_lora_adapter(
    base_model_name: str,
    adapter_path: Path,
    output_path: Path,
    use_4bit: bool = False,
    dtype: str = "bfloat16",
    use_unsloth: bool = False,
    device_map: str = "auto",
) -> None:
    """
    Merge LoRA adapter with base model and save.

    Args:
        base_model_name: HuggingFace model name or path
        adapter_path: Path to LoRA adapter directory
        output_path: Where to save the merged model
        use_4bit: Whether to load base model in 4-bit (for memory efficiency)
        dtype: Data type for model weights (bfloat16, float16, float32)
        use_unsloth: Whether to route the merge through Unsloth's fast engine
        device_map: Device map for loading the base model. Use "cpu" to keep the
            merge entirely in system RAM so GPU VRAM is free for subsequent inference.
            Use "auto" (default) to let transformers place layers on available devices.
    """
    adapter_path = adapter_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Merging LoRA Adapter for vLLM")
    logger.info("=" * 80)
    logger.info("Base Model: %s", base_model_name)
    logger.info("Adapter Path: %s", adapter_path)
    logger.info("Output Path: %s", output_path)
    logger.info("Quantization: %s", "4-bit" if use_4bit else "None")
    logger.info("Data Type: %s", dtype)
    logger.info("Using Unsloth Engine: %s", use_unsloth)
    logger.info("=" * 80)

    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = torch_dtype_map.get(dtype, torch.bfloat16)

    # --- NEW UNSLOTH MERGING PATH ---
    if use_unsloth:
        if not HAS_UNSLOTH:
            raise ImportError("Unsloth is not installed but --use-unsloth was used.")
            
        logger.info("Routing merge through Unsloth FastLanguageModel...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=8192,
            dtype=torch_dtype,
            load_in_4bit=use_4bit,
        )

        logger.info("Merging and saving model...")
        save_method = "merged_4bit" if use_4bit else "merged_16bit"
        
        # Advisory log added per user request regarding Input/Output timeout errors
        logger.info("NOTE: If you encounter an 'OSError: [Errno 5] Input/output error' during saving,")
        logger.info("this is due to network drive timeouts. To fix it, change max_shard_size to '3GB' in the script.")
        
        model.save_pretrained_merged(
            str(output_path), 
            tokenizer, 
            save_method=save_method, 
            max_shard_size="5GB",  # Reverted back to 5GB for speed as RunPod fixed the network bug
        )
        
        logger.info("Merge complete via Unsloth!")
        logger.info("Merged model saved to: %s", output_path)
        return

    # --- EXISTING STANDARD HUGGINGFACE/PEFT MERGING PATH (Fallback) ---
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    logger.info("Loading base model (device_map=%s)...", device_map)
    if use_4bit:
        logger.info("Using 4-bit quantization (for memory efficiency)")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quant_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        logger.info("Using %s precision", dtype)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    
    model.config.use_cache = True
    
    # Load LoRA adapter
    logger.info("Loading LoRA adapter from %s...", adapter_path)
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)

    # Merge LoRA weights
    logger.info("Merging LoRA weights into base model...")
    logger.info("This may take several minutes...")
    model = model.merge_and_unload()
    model.eval()
    
    # Save merged model
    logger.info("Saving merged model to %s...", output_path)
    
    # Advisory log added here as well for standard saving
    logger.info("NOTE: If you encounter an 'OSError: [Errno 5] Input/output error' during saving,")
    logger.info("change max_shard_size to '3GB' in the script.")
    
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB",  # Split into shards for large models
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    logger.info("Merge complete!")
    logger.info("Merged model saved to: %s", output_path)
    logger.info("Next steps:")
    logger.info("  1. For vLLM with 20GB GPU, you'll need to quantize this model")
    logger.info("  2. Use AWQ or GPTQ quantization tools")
    logger.info("  3. Or use vLLM with --quantization awq/gptq if available")
    logger.info("  4. Or use FP8 quantization: --quantization fp8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model for vLLM compatibility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-Coder-14B-Instruct",
        help="Base HuggingFace model name or path.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Path to LoRA adapter directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path where merged model will be saved.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Load base model in 4-bit (for memory efficiency during merge).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for model weights (if not using 4-bit).",
    )
    parser.add_argument(
        "--use-unsloth",
        action="store_true",
        help="Use Unsloth Engine for extremely fast merging and saving.",
    )
    
    args = parser.parse_args()
    
    try:
        merge_lora_adapter(
            base_model_name=args.base_model,
            adapter_path=args.adapter_path,
            output_path=args.output_path,
            use_4bit=args.use_4bit,
            dtype=args.dtype,
            use_unsloth=args.use_unsloth,
        )
        return 0
    except Exception as exc:
        logger.exception("Error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

