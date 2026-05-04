"""
Auto-generate LoRA README.md with training metadata.

Called by single_stage_finetune and multi_stage_finetune after saving LoRA
adapters, so the README is populated automatically without manual edits.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from src.infra.model_registry import ModelConfig


def _read_adapter_config(lora_dir: Path) -> Dict[str, Any]:
    """Read adapter_config.json from LoRA directory."""
    cfg_path = lora_dir / "adapter_config.json"
    if not cfg_path.exists():
        return {}
    try:
        import json
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_trainer_state(trainer) -> Dict[str, Any]:
    """Extract training state from trainer."""
    out = {}
    if not hasattr(trainer, "state") or trainer.state is None:
        return out
    s = trainer.state
    out["global_step"] = getattr(s, "global_step", None)
    out["num_train_epochs"] = getattr(s, "num_train_epochs", None)
    out["train_batch_size"] = getattr(s, "train_batch_size", None)
    out["total_flos"] = getattr(s, "total_flos", None)
    out["log_history"] = getattr(s, "log_history", []) or []
    return out


def write_lora_readme(
    lora_dir: str | Path,
    trainer,
    model_config: ModelConfig,
    training_args,
    run_id: str = "",
    stage_label: Optional[str] = None,
) -> None:
    """
    Write README.md to the LoRA directory with training metadata.

    Overwrites the default PEFT template with run-specific data.

    Args:
        lora_dir: Path to LoRA output directory.
        trainer: HuggingFace Trainer (or SFTTrainer) after training.
        model_config: ModelConfig from registry.
        training_args: TrainingArguments used for training.
        run_id: Run identifier (e.g. rabit0-v1-run4).
        stage_label: Optional stage label for multi-stage (e.g. stage_1).
    """
    lora_path = Path(lora_dir).resolve()
    if not lora_path.is_dir():
        return
    lora_path_str = str(lora_path)

    adapter_cfg = _read_adapter_config(lora_path)
    state = _get_trainer_state(trainer)

    base_model = adapter_cfg.get("base_model_name_or_path", model_config.name)
    lora_r = adapter_cfg.get("r", "")
    lora_alpha = adapter_cfg.get("lora_alpha", "")
    lora_dropout = adapter_cfg.get("lora_dropout", "")
    target_modules = adapter_cfg.get("target_modules", model_config.lora_target_modules or [])
    peft_version = adapter_cfg.get("peft_version", "")

    global_step = state.get("global_step")
    num_epochs = state.get("num_train_epochs")
    batch_size = state.get("train_batch_size")
    total_flos = state.get("total_flos")
    log_history = state.get("log_history", [])
    lr_final = None
    for entry in reversed(log_history):
        if "learning_rate" in entry:
            lr_final = entry.get("learning_rate")
            break

    lr = getattr(training_args, "learning_rate", None) if training_args else None
    weight_decay = getattr(training_args, "weight_decay", None) if training_args else None
    grad_accum = getattr(training_args, "gradient_accumulation_steps", None) if training_args else None
    eff_batch = (batch_size * grad_accum) if (batch_size and grad_accum) else None
    bf16 = getattr(training_args, "bf16", False) if training_args else False
    fp16 = getattr(training_args, "fp16", False) if training_args else False
    regime = "bf16 mixed precision" if bf16 else ("fp16 mixed precision" if fp16 else "fp32")

    title = f"Rabit0 LoRA Adapters"
    if run_id:
        title = f"{title} ({run_id})"
    if stage_label:
        title = f"{title} - {stage_label}"

    target_modules_str = ", ".join(target_modules) if target_modules else ""

    content = f"""---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:{base_model}
- lora
- sft
- transformers
- trl
- unsloth
---

# {title}

LoRA (Low-Rank Adaptation) adapters for Rabit0, a red team security AI model fine-tuned from Qwen2.5-Coder-7B-Instruct. Rabit0 = Red team + Rabbit (fast, agile) + 0 (zero-day focus).

## Model Details

- **Developed by:** Rabit0 project
- **Model type:** LoRA adapter for causal language model
- **Base model:** {base_model} (Qwen2ForCausalLM)
- **Language(s):** English (security/red team domain)
- **License:** Apache 2.0
- **Finetuned from:** {base_model}

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | {lora_r} |
| LoRA alpha | {lora_alpha} |
| LoRA dropout | {lora_dropout} |
| Target modules | {target_modules_str} |
| Task type | CAUSAL_LM |
| PEFT version | {peft_version} |

## Uses

### Direct Use

Load these adapters on top of the base model for security-focused red team assistance, penetration testing guidance, and vulnerability exploitation education.

### Downstream Use

Can be merged into the base model for vLLM serving or further fine-tuning. See project `src/serve/serve_rabit0_vllm.py` for serving with vLLM.

### Out-of-Scope Use

Not intended for malicious use. For authorized security testing and education only.

## How to Get Started with the Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "{lora_path_str}")
tokenizer = AutoTokenizer.from_pretrained("{lora_path_str}")
```

Or serve with vLLM:

```bash
poetry run python src/serve/serve_rabit0_vllm.py \\
    --base-model {base_model} \\
    --adapter-path {lora_path_str}
```

## Training Details

### Training Data

JSONL format with `messages` arrays. Sources defined in `src/config/data_prep/datapoints.yaml`. Data split 90% train / 10% validation. Max sequence length: 2048.

### Training Procedure

#### Training Hyperparameters

- **Training regime:** {regime}
- **Epochs:** {num_epochs}
- **Global steps:** {global_step}
- **Train batch size:** {batch_size}
- **Learning rate:** {lr}
- **Learning rate (final):** {lr_final}
- **Weight decay:** {weight_decay}
- **Gradient accumulation steps:** {grad_accum}
- **Effective batch size:** {eff_batch}
- **Max sequence length:** 2048

#### Speeds, Sizes, Times

- **Total FLOPs:** {total_flos}
- **Framework:** Unsloth + TRL SFTTrainer + PEFT

## Technical Specifications

### Model Architecture

Base: Qwen2ForCausalLM. LoRA applied to: {target_modules_str or "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"}.

### Framework versions

- PEFT {peft_version}
"""

    readme_path = lora_path / "README.md"
    readme_path.write_text(content, encoding="utf-8")
