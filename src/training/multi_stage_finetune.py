"""
Multi-Stage Fine-Tuning for Rabit0 (vLLM-Compatible)

Progressive fine-tuning in sequential stages for better knowledge retention
and output quality. Each stage continues from the previous stage's LoRA.

Flow: Stage 1 (fresh LoRA) -> Stage 2 (continue from stage 1) -> ... -> optional merge

Standalone file: no imports from single_stage_finetune. Same tuner/training
logic inlined for independence.
"""

import sys
from pathlib import Path

try:
    import unsloth  # noqa: F401
except ImportError:
    pass

# Ensure project root is on sys.path when run as script (after unsloth may have patched sys.path)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import transformers
from typing import Optional, List, Any, Dict
from packaging import version

from transformers import (
    AutoModelForCausalLM,
    # AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import DatasetDict
from peft import PeftModel

from src.infra.model_registry import get_registry
from src.infra.project_paths import get_paths
from src.utilities.training_config import TrainingConfig
from src.utilities.training_logger import Logger
from src.utilities.model_loader import ModelLoader
from src.utilities.data_formatter import DataFormatter
from src.utilities.data_loader import DataLoader
from src.utilities.lora_preparer import LoRAPreparer
from src.utilities.multi_stage_helpers import (
    get_stage_learning_rate,
    create_mixed_dataset,
    load_existing_lora as load_existing_lora_from_path,
)
from src.utilities.artifact_paths import ensure_artifact_run_subdirs, get_artifact_run_dir
from src.utilities.lora_readme import write_lora_readme

class QwenVLLMFineTuner:
    """
    Fine-tuner for Qwen optimized for vLLM compatibility.
    Used by multi-stage pipeline: stage 1 applies LoRA; stage 2+ continue from saved LoRA.
    """

    def __init__(
        self,
        model_size: str = "14b",
        dtype: str = "bfloat16",
        use_unsloth: bool = True,
        gpu_memory: int = 20,
        model_version: str = "qwen3",
        model_key: str = None,
        use_4bit: bool = False,
        config_path: Path = None
    ):
        self.gpu_memory = gpu_memory
        self.dtype = dtype
        self.use_unsloth = use_unsloth
        self.use_4bit = use_4bit
        self.model_version = model_version.lower()
        self.torch_dtype = TrainingConfig.get_torch_dtype(dtype)

        registry = get_registry(config_path)
        defaults = registry.get_default_config()

        if dtype == "bfloat16" and "dtype" in defaults:
            dtype = defaults.get("dtype", dtype)
        if use_unsloth and "use_unsloth" in defaults:
            use_unsloth = defaults.get("use_unsloth", use_unsloth)

        if model_key:
            model_config = registry.get_model(model_key)
            if not model_config:
                raise ValueError(f"Model key not found: {model_key}. Available: {list(registry.list_models())}")
        else:
            model_config = registry.find_model(version=model_version.lower(), size=model_size)
            if not model_config:
                raise ValueError(
                    f"No model found for version={model_version}, size={model_size}. "
                    f"Available models: {[f'{m.version}-{m.size}' for m in registry.list_models()]}"
                )

        self.model_config = model_config
        self.model_name = model_config.name
        self.lora_target_modules = model_config.lora_target_modules

        fallback_chain = registry.get_fallback_chain(model_config.key)
        self.fallback_models = [m.name for m in fallback_chain[1:]]

        self._print_initialization_info()

        loader = ModelLoader(
            model_name=self.model_name,
            torch_dtype=self.torch_dtype,
            gpu_memory=gpu_memory,
            use_4bit=use_4bit,
            fallback_models=self.fallback_models
        )

        if use_unsloth:
            try:
                self.model, self.tokenizer = loader.load_with_unsloth()
            except (ImportError, RuntimeError):
                Logger.print_warning("Falling back to standard loading...")
                self.use_unsloth = False
                self.model, self.tokenizer = loader.load_standard()
        else:
            self.model, self.tokenizer = loader.load_standard()

    def _print_initialization_info(self):
        """Print initialization information"""
        Logger.print_header("Training Rabit0: Red Team Security AI (vLLM-Compatible)")
        Logger.print_info("Base Model", self.model_name)
        Logger.print_info("Model Key", self.model_config.key)
        Logger.print_info("Model Version", self.model_config.version.upper())
        Logger.print_info("Model Size", self.model_config.size)
        Logger.print_info("Output", f"Rabit0-v1.0-{self.model_config.version}-{self.model_config.size}-vllm")
        Logger.print_info("Architecture", self.model_config.architecture)
        Logger.print_info("Precision", f"{self.dtype} (vLLM-compatible)")
        Logger.print_info("Quantization", "None (full precision for vLLM)")
        Logger.print_info("Unsloth Optimization", self.use_unsloth)
        Logger.print_info("License", "Apache 2.0 (Fully unrestricted for SaaS)")
        Logger.print_info("Expected VRAM", self.model_config.expected_vram)
        if self.fallback_models:
            Logger.print_info("Fallback Models", ", ".join(self.fallback_models))
        Logger.print_section("=" * 80)

    def prepare_model_for_training(self):
        """Apply LoRA adapters for efficient fine-tuning"""
        preparer = LoRAPreparer(
            model=self.model,
            tokenizer=self.tokenizer,
            lora_target_modules=self.lora_target_modules,
            use_unsloth=self.use_unsloth,
            use_4bit=self.use_4bit
        )
        self.model = preparer.prepare()
        return self.model

    def load_training_data(self, data_file: str):
        """Load and prepare training dataset"""
        loader = DataLoader(
            tokenizer=self.tokenizer,
            use_4bit=self.use_4bit
        )
        return loader.load_and_tokenize(data_file)

    def train(
        self,
        train_dataset,
        output_dir: str = None,
        num_epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 8,
        merge_lora: bool = False
    ):
        if output_dir is None:
            paths = get_paths()
            model_name = paths.generate_model_name(
                version="v1.0",
                model_version=self.model_config.version,
                model_size=self.model_config.size,
                suffix="vllm"
            )
            output_dir = str(paths.get_model_path(model_name, model_type="lora"))

        dirs = ensure_artifact_run_subdirs(output_dir)
        Logger.print_section("Starting training...")
        Logger.print_info("Output directory", output_dir)
        Logger.print_info("Epochs", num_epochs)
        Logger.print_info("Batch size", batch_size)
        Logger.print_info("Gradient accumulation", gradient_accumulation_steps)
        Logger.print_info("Effective batch size", batch_size * gradient_accumulation_steps)
        Logger.print_info("Learning rate", learning_rate)
        Logger.print_info("Merge LoRA after training", merge_lora)

        if self.use_unsloth:
            Logger.print_info("Using Unsloth", "2x faster training!")

        TrainingConfig.setup_single_process_environment()
        TrainingConfig.create_accelerate_config(self.dtype)

        transformers_version = version.parse(transformers.__version__)
        eval_strategy_kwargs = (
            {"eval_strategy": "steps"}
            if transformers_version >= version.parse("4.46.0")
            else {"evaluation_strategy": "steps"}
        )

        optim_config = TrainingConfig.get_optimizer_config(self.dtype)

        training_args = TrainingArguments(
            output_dir=str(dirs["checkpoints"]),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            bf16=optim_config["bf16"],
            fp16=optim_config["fp16"],
            optim=optim_config["optim"],
            logging_steps=10,
            logging_dir=str(dirs["logs"]),
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            gradient_checkpointing=True,
            warmup_steps=100,
            report_to="tensorboard",
            ddp_find_unused_parameters=False,
            ddp_backend=None,
            local_rank=-1,
            no_cuda=False,
            dataloader_pin_memory=False,
            deepspeed=None,
            fsdp=None,
            fsdp_config=None,
            remove_unused_columns=False,
            **eval_strategy_kwargs,
        )

        if self.use_unsloth:
            trainer = self._create_unsloth_trainer(training_args, train_dataset)
        else:
            trainer = self._create_standard_trainer(training_args, train_dataset)

        Logger.print_section("Training in progress...")
        Logger.print_info("Monitor with", f"tensorboard --logdir {dirs['logs']}")

        trainer.train()

        lora_dir = str(dirs["lora"])
        Logger.print_section(f"Saving LoRA adapters to {lora_dir}")
        trainer.save_model(lora_dir)
        self.tokenizer.save_pretrained(lora_dir)

        run_id = Path(output_dir).name
        stage_label = run_id if run_id.startswith("stage_") else None
        write_lora_readme(
            lora_dir,
            trainer,
            self.model_config,
            training_args,
            run_id=run_id,
            stage_label=stage_label,
        )

        if merge_lora:
            self._merge_lora_adapters(lora_dir, merged_output_dir=str(dirs["merged"]))

        Logger.print_success("Training complete!")
        Logger.print_info("Model saved to", lora_dir)
        if merge_lora:
            Logger.print_info("Merged model (vLLM-ready)", str(dirs["merged"]))
        else:
            Logger.print_info("To merge LoRA for vLLM, run", "")
            Logger.print_section(f"   python src/main/serve/merge_lora_for_vllm.py --adapter-path {lora_dir} --output-path {dirs['merged']}")

        return trainer

    def _create_unsloth_trainer(self, training_args: TrainingArguments, train_dataset: DatasetDict):
        from trl import SFTTrainer
        return SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset["train"],
            eval_dataset=train_dataset["test"],
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

    def _create_standard_trainer(self, training_args: TrainingArguments, train_dataset: DatasetDict):
        class CustomDataCollator(DataCollatorForLanguageModeling):
            def __call__(self, features):
                batch = super().__call__(features)
                if "labels" in batch:
                    batch["labels"] = batch["labels"].clone().detach().requires_grad_(False)
                return batch

        data_collator = CustomDataCollator(
            tokenizer=self.tokenizer,
            mlm=False
        )

        self.model.train()
        self._verify_gradient_computation(train_dataset)

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                if hasattr(model, 'module'):
                    actual_model = model.module
                else:
                    actual_model = model

                actual_model.train()

                for param in actual_model.parameters():
                    if param.requires_grad:
                        param.requires_grad = True

                labels = inputs.get("labels")

                with torch.enable_grad():
                    outputs = actual_model(**inputs)

                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        loss = outputs.loss
                    else:
                        logits = outputs.get("logits")
                        if logits is None:
                            raise RuntimeError("Model outputs don't contain logits")

                        if not logits.requires_grad:
                            for name, param in actual_model.named_parameters():
                                if 'lora' in name.lower() and not param.requires_grad:
                                    param.requires_grad = True
                            outputs = actual_model(**inputs)
                            logits = outputs.get("logits")

                        loss_fct = torch.nn.CrossEntropyLoss()
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                if not loss.requires_grad:
                    trainable_count = sum(1 for p in actual_model.parameters() if p.requires_grad)
                    Logger.print_info("DEBUG: Trainable parameters", trainable_count)
                    Logger.print_info("DEBUG: Model training mode", actual_model.training)
                    Logger.print_info("DEBUG: Loss value", loss.item())
                    raise RuntimeError(
                        "Loss doesn't require gradients. Model forward pass is not building computation graph.\n"
                        "This usually happens when:\n"
                        "1. Model was loaded with device_map='auto' (use device_map=None for training)\n"
                        "2. Model parameters don't have requires_grad=True\n"
                        "3. Model is in eval() mode instead of train() mode"
                    )

                return (loss, outputs) if return_outputs else loss

        return CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset["train"],
            eval_dataset=train_dataset["test"],
            data_collator=data_collator,
        )

    def _verify_gradient_computation(self, train_dataset: DatasetDict):
        try:
            sample = train_dataset["train"][0]
            inputs = self.tokenizer(
                DataFormatter.format_training_data(sample["messages"]),
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding="max_length"
            ).to(next(self.model.parameters()).device)
            inputs["labels"] = inputs["input_ids"].clone()

            with torch.enable_grad():
                outputs = self.model(**inputs)
                loss = outputs.loss
                if loss.requires_grad:
                    Logger.print_info("Model forward pass computes gradients correctly", "")
                else:
                    Logger.print_warning("Model forward pass doesn't compute gradients - fixing...")
                    for param in self.model.parameters():
                        if param.requires_grad:
                            param.requires_grad = True
        except Exception as e:
            Logger.print_warning(f"Could not verify gradient computation: {e}")

    def _merge_lora_adapters(self, lora_dir: str, merged_output_dir: Optional[str] = None):
        Logger.print_section("Merging LoRA adapters into base model (for vLLM compatibility)...")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, lora_dir)
        merged_model = model.merge_and_unload()

        if merged_output_dir is None:
            paths = get_paths()
            model_name = Path(lora_dir).name.replace("-lora", "").replace("-vllm", "")
            merged_output_dir = str(paths.get_model_path(
                f"{model_name}-merged",
                model_type="merged"
            ))
        else:
            merged_output_dir = str(merged_output_dir)

        Logger.print_section(f"Saving merged model to {merged_output_dir}")
        merged_model.save_pretrained(
            merged_output_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        self.tokenizer.save_pretrained(merged_output_dir)
        Logger.print_success(f"Merged model saved! Use this with vLLM: {merged_output_dir}")

    def test_model(self, prompt: str, max_length: int = 512):
        """Test the fine-tuned model (base or LoRA-loaded)."""
        formatted_prompt = DataFormatter.format_prompt(prompt)
        device = getattr(self.model, "device", next(self.model.parameters()).device)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            response = response.split("<|im_end|>")[0].strip()

        return response


def _load_existing_lora_into_tuner(tuner: QwenVLLMFineTuner, continue_from_lora: str) -> None:
    """Load base model and existing LoRA; set tuner.model and tuner.tokenizer."""
    model, tokenizer = load_existing_lora_from_path(
        continue_from_lora,
        tuner.model_name,
        tuner.torch_dtype,
    )
    tuner.model = model
    tuner.tokenizer = tokenizer


def multi_stage_finetune(
    stage_data_paths: List[str],
    output_base_dir: str,
    *,
    replay_ratio: float = 0.0,
    merge_lora: bool = False,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    tuner_kwargs: Optional[Dict[str, Any]] = None,
) -> QwenVLLMFineTuner:
    """
    Run multi-stage fine-tuning: one stage per data file, continuing LoRA across stages.
    """
    tuner_kwargs = tuner_kwargs or {}
    tuner = QwenVLLMFineTuner(**tuner_kwargs)

    base_dir = Path(output_base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    previous_stage_dir: Optional[str] = None
    previous_train_datasets: List[Any] = []

    for stage_idx, data_path in enumerate(stage_data_paths, start=1):
        stage_dir = str(base_dir / f"stage_{stage_idx}")
        Logger.print_header(f"Multi-Stage Training: Stage {stage_idx} of {len(stage_data_paths)}")

        if stage_idx == 1:
            tuner.prepare_model_for_training()
            dataset = tuner.load_training_data(data_path)
        else:
            _load_existing_lora_into_tuner(tuner, previous_stage_dir)
            current = tuner.load_training_data(data_path)
            dataset = create_mixed_dataset(
                previous_train_datasets,
                current,
                replay_ratio=replay_ratio,
            )

        lr = get_stage_learning_rate(stage_idx)
        Logger.print_info("Stage learning rate", lr)
        if replay_ratio > 0 and stage_idx > 1:
            Logger.print_info("Replay ratio", replay_ratio)

        _, _ = tuner.train(
            train_dataset=dataset,
            output_dir=stage_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=lr,
            gradient_accumulation_steps=gradient_accumulation_steps,
            merge_lora=False,
        )

        previous_stage_dir = str(Path(stage_dir) / "lora")
        previous_train_datasets.append(dataset["train"])

    if merge_lora and previous_stage_dir:
        Logger.print_section("Merging LoRA after final stage...")
        merged_dir = str(base_dir / "merged")
        tuner._merge_lora_adapters(previous_stage_dir, merged_output_dir=merged_dir)

    Logger.print_success("Multi-stage fine-tuning complete.")
    return tuner

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-stage fine-tune Rabit0 (continue LoRA across stages)"
    )
    parser.add_argument(
        "--stage-data",
        nargs="+",
        required=False,
        help="Data files per stage (e.g. stage1.jsonl stage2.jsonl). All-in-one mode.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Single data file for current stage (used with --stage and --continue-from-lora).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Current stage number (1, 2, ...). Use with --continue-from-lora for stage 2+.",
    )
    parser.add_argument(
        "--continue-from-lora",
        type=str,
        default=None,
        help="Path to previous stage LoRA directory (for stage 2+).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output base directory (all-in-one) or stage output dir (resumable).",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.0,
        help="Fraction of previous stage data to mix into next (0.0-1.0, default 0).",
    )
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge LoRA into base model after the last stage.",
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Epochs per stage."
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=2, 
        help="Batch size."
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--model-size",
        choices=["7b", "8b", "14b", "30b"],
        default="8b",
        help="Model size.",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="Direct model key from registry.",
    )
    parser.add_argument(
        "--model-version",
        choices=["qwen3", "qwen2.5"],
        default="qwen3",
        help="Model version.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Data type.",
    )
    parser.add_argument(
        "--no-unsloth",
        action="store_true",
        help="Disable Unsloth.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA).",
    )
    parser.add_argument(
        "--gpu-memory",
        type=int,
        default=40,
        help="Available GPU memory in GiB.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List registry models and exit.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Load a LoRA (or base model) and run test prompts; use --continue-from-lora to test a trained LoRA.",
    )

    args = parser.parse_args()

    if args.list_models:
        registry = get_registry()
        models = registry.list_models(supported_only=True)
        Logger.print_header("Available Models in Registry")
        for model in models:
            Logger.print_section(f"Key: {model.key}")
            Logger.print_info("  Name", model.name)
            Logger.print_info("  Version", f"{model.version} | Size: {model.size}")
            Logger.print_info("  Architecture", model.architecture)
            Logger.print_info("  Expected VRAM", model.expected_vram)
            if model.fallbacks:
                Logger.print_info("  Fallbacks", ", ".join(model.fallbacks))
        Logger.print_section("=" * 80)
        return

    if args.test_only:
        Logger.print_section("Testing model...")
        tuner_kwargs = {
            "model_size": args.model_size,
            "dtype": args.dtype,
            "use_unsloth": not args.no_unsloth,
            "gpu_memory": args.gpu_memory,
            "model_version": args.model_version,
            "model_key": args.model_key,
            "use_4bit": args.use_4bit,
        }
        tuner = QwenVLLMFineTuner(**tuner_kwargs)
        if args.continue_from_lora:
            _load_existing_lora_into_tuner(tuner, args.continue_from_lora)
        tuner.model.eval()
        test_prompts = [
            "How do I perform privilege escalation on a Linux system?",
            "Explain SQL injection exploitation with examples",
            "What are the steps for Kerberoasting in Active Directory?",
            "How can I perform lateral movement in a Windows domain?",
        ]
        for prompt in test_prompts:
            Logger.print_section("=" * 80)
            Logger.print_info("Prompt", prompt)
            Logger.print_section("=" * 80)
            response = tuner.test_model(prompt)
            Logger.print_info("Response", f"\n{response}\n")
        return

    if args.stage is not None and args.continue_from_lora and args.data:
        if not args.output:
            parser.error("--output is required when using --stage and --continue-from-lora")
        tuner_kwargs = {
            "model_size": args.model_size,
            "dtype": args.dtype,
            "use_unsloth": not args.no_unsloth,
            "gpu_memory": args.gpu_memory,
            "model_version": args.model_version,
            "model_key": args.model_key,
            "use_4bit": args.use_4bit,
        }
        tuner = QwenVLLMFineTuner(**tuner_kwargs)
        _load_existing_lora_into_tuner(tuner, args.continue_from_lora)
        dataset = tuner.load_training_data(args.data)
        lr = get_stage_learning_rate(args.stage)
        _, _ = tuner.train(
            train_dataset=dataset,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=lr,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            merge_lora=args.merge_lora,
        )
        Logger.print_success("Stage %d complete." % args.stage)
        return

    if args.stage_data:
        output_base = args.output
        if not output_base:
            output_base = get_artifact_run_dir(
                "rabit0-multistage-vllm",
                run_prefix="rabit0-v1",
                update_latest=True,
            )
        multi_stage_finetune(
            stage_data_paths=args.stage_data,
            output_base_dir=output_base,
            replay_ratio=args.replay_ratio,
            merge_lora=args.merge_lora,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            tuner_kwargs={
                "model_size": args.model_size,
                "dtype": args.dtype,
                "use_unsloth": not args.no_unsloth,
                "gpu_memory": args.gpu_memory,
                "model_version": args.model_version,
                "model_key": args.model_key,
                "use_4bit": args.use_4bit,
            },
        )
        return

    parser.error(
        "Use either:\n"
        "  --stage-data file1.jsonl file2.jsonl [--output DIR] [--merge-lora]\n"
        "  or  --stage N --continue-from-lora DIR --data file.jsonl --output DIR\n"
        "  or  --test-only [--continue-from-lora DIR] to run test prompts"
    )


if __name__ == "__main__":
    import sys
    if "accelerate" not in " ".join(sys.argv):
        TrainingConfig.setup_single_process_environment()
    main()

"""
Example Usage:

# All-in-one: run stages 1 and 2 in one command (stage 2 continues from stage 1 LoRA)
poetry run python src/main/training/multi_stage_finetune.py \
    --stage-data stage1_data.jsonl stage2_data.jsonl \
    --output ./rabit0-multistage-vllm \
    --model-size 8b \
    --model-version qwen3 \
    --epochs 3 \
    --batch-size 2 \
    --gpu-memory 40

# Same with replay: mix 20% of previous stage data into next stage to reduce forgetting
poetry run python src/main/training/multi_stage_finetune.py \
    --stage-data stage1_data.jsonl stage2_data.jsonl \
    --output ./rabit0-multistage-vllm \
    --replay-ratio 0.2 \
    --epochs 3 \
    --batch-size 2

# All-in-one and merge LoRA after the last stage (vLLM-ready)
poetry run python src/main/training/multi_stage_finetune.py \
    --stage-data stage1_data.jsonl stage2_data.jsonl \
    --output ./rabit0-multistage-vllm \
    --merge-lora \
    --model-size 7b \
    --gpu-memory 20

# Resumable: run stage 1, then run stage 2 later with --continue-from-lora
# Stage 1:
poetry run python src/main/training/multi_stage_finetune.py \
    --stage 1 \
    --data stage1_data.jsonl \
    --output ./rabit0-multistage-vllm/stage_1

# Stage 2 (continues from stage 1 LoRA):
poetry run python src/main/training/multi_stage_finetune.py \
    --stage 2 \
    --continue-from-lora ./rabit0-multistage-vllm/stage_1 \
    --data stage2_data.jsonl \
    --output ./rabit0-multistage-vllm/stage_2 \
    --merge-lora

# Test prompts (base model only)
poetry run python src/main/training/multi_stage_finetune.py --test-only --model-size 8b

# Test prompts with a trained LoRA (e.g. after stage 1 or 2)
poetry run python src/main/training/multi_stage_finetune.py --test-only \
  --continue-from-lora ./rabit0-multistage-vllm/stage_1 --model-size 8b

# List available models and exit
poetry run python src/main/training/multi_stage_finetune.py --list-models
"""
