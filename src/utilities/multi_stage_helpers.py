"""
Multi-Stage Training Helpers (multi-stage logic only).

Used only by multi_stage_finetune.py. Single-stage training
(single_stage_finetune.py) does not use this module.

Helpers: stage learning rate, mixed dataset (replay), load existing LoRA.
"""

from typing import Any, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import DatasetDict, concatenate_datasets
from peft import PeftModel

from src.utilities.training_logger import Logger


def get_stage_learning_rate(stage: int) -> float:
    """Learning rate per stage (higher for stage 1, lower for refinement)."""
    if stage == 1:
        return 2e-4
    if stage == 2:
        return 5e-5
    return 1e-5


def create_mixed_dataset(
    previous_train_datasets: List[Any],
    current_dataset: DatasetDict,
    replay_ratio: float,
    seed: int = 42,
) -> DatasetDict:
    """
    Mix previous stage train data with current stage to reduce forgetting.

    Args:
        previous_train_datasets: List of train Dataset from previous stages.
        current_dataset: DatasetDict with "train" and "test".
        replay_ratio: Fraction of current batch that is replay (0.0-1.0).
        seed: Shuffle seed.

    Returns:
        DatasetDict with mixed "train" and same "test".
    """
    if replay_ratio <= 0 or not previous_train_datasets:
        return current_dataset

    all_previous = concatenate_datasets(previous_train_datasets)
    current_train = current_dataset["train"]
    num_replay = int(len(current_train) * replay_ratio / (1.0 - replay_ratio))
    num_replay = min(num_replay, len(all_previous))
    if num_replay <= 0:
        return current_dataset

    replay = all_previous.shuffle(seed=seed).select(range(num_replay))
    mixed_train = concatenate_datasets([replay, current_train]).shuffle(seed=seed)
    return DatasetDict({"train": mixed_train, "test": current_dataset["test"]})


def load_existing_lora(
    continue_from_lora: str,
    model_name: str,
    torch_dtype: Any,
) -> Tuple[Any, Any]:
    """
    Load base model and existing LoRA for continuation.
    Returns (model, tokenizer). Caller should set tuner.model and tuner.tokenizer.

    Args:
        continue_from_lora: Path to saved LoRA adapter directory.
        model_name: Base model name (e.g. from registry).
        torch_dtype: Torch dtype for base model.

    Returns:
        Tuple of (model, tokenizer).
    """
    Logger.print_section(f"Loading existing LoRA from {continue_from_lora} for continuation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, continue_from_lora)
    tokenizer = AutoTokenizer.from_pretrained(
        continue_from_lora,
        trust_remote_code=True,
    )
    model.train()
    Logger.print_success("LoRA loaded; ready to continue training.")
    return (model, tokenizer)
