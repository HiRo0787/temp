"""
Train Rabit0: Red Team Security AI Model (vLLM-Compatible Version)
Fine-tuned from Qwen2.5-Coder-7B-Instruct (Apache 2.0)

This version is optimized for vLLM serving:
- Uses 7B model (fits in 20GB GPU with vLLM)
- FP16/BF16 precision (no 4-bit quantization)
- LoRA adapters (can be merged after training)
- Compatible with vLLM OpenAI API server

Rabit0 = Red team + Rabbit (fast, agile) + 0 (zero-day focus)

Output Model: Rabit0-v1.0-qwen3-7b-vllm

Single-stage only: one train run, one dataset, one learning rate.
Does not use multi_stage_helpers (no stage loop, replay, or continue-from-LoRA).
"""

import os
import sys
from pathlib import Path

# Import unsloth first (if available) for optimizations
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
from typing import Optional, Dict, Any, Union, Tuple
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import DatasetDict
from peft import PeftModel
from src.infra.model_registry import get_registry, ModelConfig
from src.infra.project_paths import get_paths
from src.utilities.training_config import TrainingConfig
from src.utilities.training_logger import Logger
from src.utilities.model_loader import ModelLoader
from src.utilities.data_formatter import DataFormatter
from src.utilities.data_loader import DataLoader
from src.utilities.lora_preparer import LoRAPreparer
from src.utilities.artifact_paths import copy_data_manifest, ensure_artifact_run_subdirs
from src.utilities.lora_readme import write_lora_readme
from src.utilities.model_config_serialization import ensure_model_config_json_serializable
from src.utilities.single_stage_helpers import (
    build_launch_command,
    build_run_config_payload,
    write_run_config_file,
    compute_save_steps,
    get_default_eval_steps,
    get_default_learning_rate,
    get_default_weight_decay,
    get_default_save_total_limit,
    get_default_warmup_ratio,
    get_eval_strategy_kwargs,
    patch_checkpoint_trainer_state,
    resolve_output_dir,
    release_training_gpu_for_subprocess_eval,
    resolve_post_train_eval_checkpoint_path,
    restore_training_model_cuda,
    run_post_training_eval,
    supported_eval_benches,
)


class QwenVLLMFineTuner:
    """
    Fine-tuner for Qwen2.5-Coder optimized for vLLM compatibility

    Refactored to follow SOLID principles:
    - SRP: Delegates responsibilities to specialized classes
    - OCP: Extensible through configuration and dependency injection
    - DIP: Depends on abstractions (ModelLoader, DataLoader, etc.)
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
        use_8bit: bool = False,
        config_path: Path = None,
        max_seq_length: Optional[int] = None,
        distributed_strategy: str = "single",
    ):
        """
        Initialize Qwen model for vLLM-compatible fine-tuning

        Args:
            model_size: Model size - "7b" (recommended) or "14b" (requires more GPU)
            dtype: Data type - "bfloat16" (recommended) or "float16"
            use_unsloth: Use Unsloth for faster training
            gpu_memory: Available GPU memory in GiB
            model_version: Model version - "qwen3" (latest) or "qwen2.5" (older)
            model_key: Direct model key from registry (e.g., "qwen2.5-7b"). Overrides version/size.
            config_path: Path to models.yaml config file (optional; default: src/config/model/models.yaml)
            max_seq_length: Max tokens for Unsloth load, tokenization, and SFTTrainer; None uses 1024 (4-bit) or 2048.
            distributed_strategy: "single" (default) or "fsdp" for multi-GPU sharding.
        """
        # Load model registry (SOLID: Dependency Inversion - depends on config abstraction)
        registry = get_registry(config_path)
        defaults = registry.get_default_config()

        # Resolve effective runtime configuration before storing instance state.
        effective_dtype = defaults.get("dtype", dtype) if dtype == "bfloat16" else dtype
        effective_use_unsloth = defaults.get("use_unsloth", use_unsloth) if use_unsloth else use_unsloth

        self.gpu_memory = gpu_memory
        self.dtype = effective_dtype
        self.use_unsloth = effective_use_unsloth
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.distributed_strategy = distributed_strategy
        self._training_max_seq_length = (
            max_seq_length if max_seq_length is not None else (1024 if use_4bit else 2048)
        )
        if self.use_4bit and self.use_8bit:
            raise ValueError("Choose only one quantization mode: --use-4bit or --use-8bit")
        if self.distributed_strategy not in {"single", "fsdp"}:
            raise ValueError("distributed_strategy must be one of: single, fsdp")
        if self.distributed_strategy == "fsdp" and (self.use_4bit or self.use_8bit):
            raise ValueError("FSDP currently supports full-precision base loading only (no --use-4bit/--use-8bit)")
        self.model_version = model_version.lower()
        self.torch_dtype = TrainingConfig.get_torch_dtype(self.dtype)

        # Select model using registry (SOLID: Open/Closed - no code changes needed for new models)
        if model_key:
            model_config = registry.get_model(model_key)
            if not model_config:
                raise ValueError(
                    f"Model key not found: {model_key}. Available: {list(registry.list_models())}")
        else:
            model_config = registry.find_model(
                version=model_version.lower(), size=model_size)
            if not model_config:
                raise ValueError(
                    f"No model found for version={model_version}, size={model_size}. "
                    f"Available models: {[f'{m.version}-{m.size}' for m in registry.list_models()]}"
                )

        # Store model configuration
        self.model_config = model_config
        self.model_name = model_config.name
        self.lora_target_modules = model_config.lora_target_modules

        # Get fallback chain
        fallback_chain = registry.get_fallback_chain(model_config.key)
        self.fallback_models = [m.name for m in fallback_chain[1:]]

        # Print initialization info
        self._print_initialization_info()

        # Load model using ModelLoader (SOLID: Dependency Inversion)
        loader = ModelLoader(
            model_name=self.model_name,
            torch_dtype=self.torch_dtype,
            gpu_memory=gpu_memory,
            use_4bit=self.use_4bit,
            use_8bit=self.use_8bit,
            fallback_models=self.fallback_models,
            unsloth_max_seq_length=self._training_max_seq_length,
            distributed_strategy=self.distributed_strategy,
        )

        if self.use_unsloth:
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
        Logger.print_header(
            "Training Rabit0: Red Team Security AI (vLLM-Compatible)")
        Logger.print_info("Base Model", self.model_name)
        Logger.print_info("Model Key", self.model_config.key)
        Logger.print_info("Model Version", self.model_config.version.upper())
        Logger.print_info("Model Size", self.model_config.size)
        Logger.print_info(
            "Output", f"Rabit0-v1.0-{self.model_config.version}-{self.model_config.size}-vllm")
        Logger.print_info("Architecture", self.model_config.architecture)
        Logger.print_info("Precision", f"{self.dtype} (vLLM-compatible)")
        if self.use_4bit:
            quantization_mode = "4-bit (QLoRA)"
        elif self.use_8bit:
            quantization_mode = "8-bit"
        else:
            quantization_mode = "None (full precision for vLLM)"
        Logger.print_info("Quantization", quantization_mode)
        Logger.print_info("Unsloth Optimization", self.use_unsloth)
        Logger.print_info("Distributed strategy", self.distributed_strategy)
        Logger.print_info(
            "License", "Apache 2.0 (Fully unrestricted for SaaS)")
        Logger.print_info("Expected VRAM", self.model_config.expected_vram)
        if self.fallback_models:
            Logger.print_info("Fallback Models",
                              ", ".join(self.fallback_models))
        Logger.print_info("Effective torch dtype", str(self.torch_dtype))
        if self.use_4bit:
            loading_mode = "4-bit quantized base model"
        elif self.use_8bit:
            loading_mode = "8-bit quantized base model"
        else:
            loading_mode = "full precision base model"
        Logger.print_info("Model loading mode", loading_mode)
        Logger.print_section("=" * 80)

    def _log_gpu_usage_snapshot(self, label: str) -> None:
        """Delegate GPU usage logging to shared utility."""
        TrainingConfig.log_gpu_usage_snapshot(label)

    def prepare_model_for_training(self):
        """Apply LoRA adapters for efficient fine-tuning"""
        Logger.print_section("Preparing model for LoRA training...")
        Logger.print_info("LoRA target modules", ", ".join(self.lora_target_modules))
        if self.use_4bit:
            lora_quantization_mode = "QLoRA (4-bit base model)"
        elif self.use_8bit:
            lora_quantization_mode = "LoRA (8-bit base model)"
        else:
            lora_quantization_mode = "LoRA (full precision base model)"
        Logger.print_info("LoRA quantization mode", lora_quantization_mode)
        preparer = LoRAPreparer(
            model=self.model,
            tokenizer=self.tokenizer,
            lora_target_modules=self.lora_target_modules,
            use_unsloth=self.use_unsloth,
            use_4bit=self.use_4bit
        )
        self.model = preparer.prepare()
        Logger.print_success("LoRA adapters applied")
        return self.model

    def load_training_data(
        self,
        data_file: str,
        easy_context_max: int = 512,
        medium_context_max: int = 1024,
        learning_type: str = "curriculum",
    ):
        """Load and prepare training dataset (max length from ``--max-seq-length`` / tuner init)."""
        resolved_max_length = self._training_max_seq_length
        Logger.print_section("Loading and tokenizing training data...")
        Logger.print_info("Data file", data_file)
        Logger.print_info("Learning type", learning_type)
        Logger.print_info("Easy context max", easy_context_max)
        Logger.print_info("Medium context max", medium_context_max)
        Logger.print_info("Tokenizer max sequence length", resolved_max_length)
        loader = DataLoader(
            tokenizer=self.tokenizer,
            use_4bit=self.use_4bit
        )
        dataset = loader.load_and_tokenize(
            data_file,
            easy_context_max=easy_context_max,
            medium_context_max=medium_context_max,
            max_seq_length=resolved_max_length,
            learning_type=learning_type,
        )
        Logger.print_info("Loaded train examples", len(dataset["train"]))
        Logger.print_info("Loaded validation examples", len(dataset["test"]))
        return dataset

    def train(
        self,
        train_dataset,
        output_dir: str = None,
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        lr_scheduler_type: Optional[str] = None,
        warmup_ratio: Optional[float] = None,
        gradient_accumulation_steps: int = 8,
        merge_lora: bool = False,
        eval_steps: Optional[int] = None,
        per_device_eval_batch_size: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
        logging_steps: int = 10,
        resume_from_checkpoint: Optional[Union[bool, str]] = None,
        data_manifest: Optional[Union[str, Path]] = None,
        launch_command: Optional[str] = None,
        cli_args: Optional[Dict[str, Any]] = None,
        post_train_eval: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Path]]:
        """
        Train the model with optimized settings for vLLM compatibility

        Args:
            train_dataset: Tokenized training dataset
            output_dir: Directory to save model (auto-generated if None)
            num_epochs: Number of training epochs (3 recommended)
            batch_size: Batch size per device (1 — 7B in bfloat16 fills ~14GB)
            learning_rate: Learning rate (default from single_stage_helpers)
            weight_decay: AdamW weight decay (default from single_stage_helpers)
            lr_scheduler_type: Learning rate scheduler type
            gradient_accumulation_steps: Steps to accumulate gradients
            merge_lora: Merge LoRA adapters after training (for vLLM)
            eval_steps: Run evaluation every N steps (default from TrainingConfig)
            per_device_eval_batch_size: Eval batch size per device (default: same as batch_size)
            max_eval_samples: Cap validation set size for eval (default: use full set)
            logging_steps: Log metrics every N steps (default: 10)
            resume_from_checkpoint: If True, resume from latest checkpoint in output_dir.
                If str, path to checkpoint dir (e.g. .../checkpoints/checkpoint-100). None = do not resume.
            data_manifest: Path to a manifest.json produced by data_versioning. When provided,
                the file is copied into data_versions/ inside the artifact run directory so the
                training data version (SHA256, version_id, datapoints list) is recorded alongside
                the model outputs. If None, no manifest is written.
            launch_command: Original command used to launch training.
            cli_args: Raw CLI args dictionary captured in main().
            post_train_eval: Optional post-train evaluation configuration captured from CLI.

        Returns:
            ``(trainer, dirs)`` where ``dirs`` is the artifact subdir map from
            ``ensure_artifact_run_subdirs`` (``checkpoints``, ``lora``, ...).
        """
        if learning_rate is None:
            learning_rate = get_default_learning_rate()
        if weight_decay is None:
            weight_decay = get_default_weight_decay()
        if lr_scheduler_type is None:
            raise ValueError("lr_scheduler_type must be provided by caller")
        if warmup_ratio is None:
            warmup_ratio = get_default_warmup_ratio()
        output_dir = resolve_output_dir(output_dir, self.model_config)
        dirs = ensure_artifact_run_subdirs(output_dir)
        # Persist the detected data manifest alongside the run outputs.
        copy_data_manifest(data_manifest, dirs["data_versions"])
        paths = get_paths()
        artifact_model_name = paths.generate_artifact_model_name(
            model_version=self.model_config.version,
            model_size=self.model_config.size,
        )
        run_id = Path(output_dir).name
        # Terminal log already started in main() before tuner creation; no second handler

        eval_config = TrainingConfig.get_eval_config(
            eval_steps=eval_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
            max_eval_samples=max_eval_samples,
            train_batch_size=batch_size,
        )
        eval_steps_final = eval_config["eval_steps"]
        per_device_eval_batch_size_final = eval_config["per_device_eval_batch_size"]
        max_eval_samples_final = eval_config["max_eval_samples"]
        save_steps = compute_save_steps(eval_steps_final)
        lora_defaults = get_registry().get_default_config().get("lora_config", {})
        lora_config_for_run: Dict[str, Any] = {
            "r": lora_defaults.get("r", 16),
            "lora_alpha": lora_defaults.get("lora_alpha", 32),
            "lora_dropout": lora_defaults.get("lora_dropout", 0.05),
            "bias": lora_defaults.get("bias", "none"),
            "task_type": lora_defaults.get("task_type", "CAUSAL_LM"),
            "target_modules": list(self.lora_target_modules),
        }

        run_config_payload = build_run_config_payload(
            run_id=run_id,
            launch_command=launch_command,
            model_config=self.model_config,
            model_name=self.model_name,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            eval_steps=eval_steps_final,
            per_device_eval_batch_size=per_device_eval_batch_size_final,
            max_eval_samples=max_eval_samples_final,
            save_steps=save_steps,
            resume_from_checkpoint=resume_from_checkpoint,
            merge_lora=merge_lora,
            dtype=self.dtype,
            use_unsloth=self.use_unsloth,
            use_4bit=self.use_4bit,
            use_8bit=self.use_8bit,
            data_manifest=data_manifest,
            data_manifest_dest=dirs["data_versions"] / "manifest.json",
            cli_args=cli_args,
            post_train_eval=post_train_eval,
            lora_config=lora_config_for_run,
        )
        run_config_path = write_run_config_file(output_dir, run_config_payload)
        Logger.print_info("Run config", str(run_config_path))

        eval_dataset = train_dataset["test"]
        if max_eval_samples_final is not None and len(eval_dataset) > max_eval_samples_final:
            eval_dataset = eval_dataset.select(range(max_eval_samples_final))
            Logger.print_info(
                "Eval subset",
                f"using {len(eval_dataset)} samples (max_eval_samples={max_eval_samples_final})",
            )
        train_dataset_for_trainer = DatasetDict(train=train_dataset["train"], test=eval_dataset)

        Logger.print_section("Starting training...")
        Logger.print_info("Output directory", output_dir)
        Logger.print_info("Epochs", num_epochs)
        Logger.print_info("Batch size", batch_size)
        Logger.print_info("Gradient accumulation", gradient_accumulation_steps)
        Logger.print_info("Effective batch size",
                          batch_size * gradient_accumulation_steps)
        Logger.print_info("Learning rate", learning_rate)
        Logger.print_info("Weight decay", weight_decay)
        Logger.print_info("LR scheduler type", lr_scheduler_type)
        Logger.print_info("Warmup ratio", warmup_ratio)
        Logger.print_info("Merge LoRA after training", merge_lora)
        Logger.print_info("Logging steps", logging_steps)
        Logger.print_info("Eval steps", eval_steps_final)
        Logger.print_info("Per-device eval batch size", per_device_eval_batch_size_final)
        Logger.print_info("Save steps", save_steps)
        if data_manifest:
            Logger.print_info(
                "Data version manifest", str(dirs["data_versions"] / "manifest.json")
            )
        if max_eval_samples_final is not None:
            Logger.print_info("Max eval samples", max_eval_samples_final)
        else:
            Logger.print_info("Eval samples", f"full validation set ({len(eval_dataset)})")

        if self.use_unsloth:
            Logger.print_info("Using Unsloth", "2x faster training!")

        # Setup runtime/distributed environment.
        if self.distributed_strategy == "single":
            Logger.print_section("Distributed runtime: single-process")
            Logger.print_info("Distributed strategy", "single")
            Logger.print_info("Process topology", "WORLD_SIZE=1, LOCAL_RANK=0")
            TrainingConfig.setup_single_process_environment()
            TrainingConfig.create_accelerate_config(self.dtype)
            local_rank = -1
            fsdp_mode = None
            fsdp_config = None
            ddp_backend = None
        else:
            Logger.print_section("Distributed runtime: FSDP sharding")
            Logger.print_info("Distributed strategy", "fsdp")
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            rank = int(os.environ.get("RANK", str(local_rank)))
            if world_size < 2:
                raise RuntimeError(
                    "FSDP requires multi-process launch. Use accelerate launch with at least 2 processes."
                )
            fsdp_mode = "full_shard auto_wrap"
            fsdp_config = {
                "backward_prefetch": "backward_pre",
                "forward_prefetch": False,
                "limit_all_gathers": True,
                "use_orig_params": True,
            }
            ddp_backend = "nccl"
            Logger.print_info("RANK", rank)
            Logger.print_info("WORLD_SIZE", world_size)
            Logger.print_info("LOCAL_RANK", local_rank)
            Logger.print_info("DDP backend", ddp_backend)
            Logger.print_info("FSDP mode", fsdp_mode)
            Logger.print_info("FSDP use_orig_params", fsdp_config["use_orig_params"])
            Logger.print_info("FSDP limit_all_gathers", fsdp_config["limit_all_gathers"])
        self._log_gpu_usage_snapshot(label="before TrainingArguments")

        eval_strategy_kwargs = get_eval_strategy_kwargs()

        # Get optimizer configuration (DRY: uses centralized utility)
        optim_config = TrainingConfig.get_optimizer_config(self.dtype)
        Logger.print_info("Optimizer", optim_config["optim"])
        Logger.print_info("bf16 enabled", optim_config["bf16"])
        Logger.print_info("fp16 enabled", optim_config["fp16"])

        report_to_targets = "tensorboard"

        training_args = TrainingArguments(
            output_dir=str(dirs["checkpoints"]),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size_final,
            prediction_loss_only=True,
            eval_accumulation_steps=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            bf16=optim_config["bf16"],
            fp16=optim_config["fp16"],
            optim=optim_config["optim"],
            logging_steps=logging_steps,
            logging_dir=str(dirs["logs"]),  # TensorBoard events
            eval_steps=eval_steps_final,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=get_default_save_total_limit(),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            gradient_checkpointing=True,
            warmup_ratio=warmup_ratio,
            report_to=report_to_targets,
            ddp_find_unused_parameters=False,
            ddp_backend=ddp_backend,
            local_rank=local_rank,
            no_cuda=False,
            dataloader_pin_memory=False,
            deepspeed=None,
            fsdp=fsdp_mode,
            fsdp_config=fsdp_config,
            remove_unused_columns=False,
            neftune_noise_alpha=5.0, # Added for Rabit0 v2 Anti-Forgetting
            **eval_strategy_kwargs,
        )

        if self.use_unsloth:
            trainer = self._create_unsloth_trainer(
                training_args,
                train_dataset_for_trainer,
            )
        else:
            trainer = self._create_standard_trainer(
                training_args,
                train_dataset_for_trainer,
            )
        Logger.print_info("Trainer backend", "Unsloth SFTTrainer" if self.use_unsloth else "Transformers Trainer")

        Logger.print_section("Training in progress...")
        self._log_gpu_usage_snapshot(label="before trainer.train")
        if resume_from_checkpoint is not None:
            Logger.print_info(
                "Resume from checkpoint",
                resume_from_checkpoint if isinstance(resume_from_checkpoint, str) else "latest in output dir",
            )
            patch_checkpoint_trainer_state(
                resume_from_checkpoint,
                dirs["checkpoints"],
                eval_steps_final,
                save_steps,
            )
        Logger.print_info(
            "Monitor with", f"tensorboard --logdir {dirs['logs']}")

        if ensure_model_config_json_serializable(self.model):
            Logger.print_warning(
                "Model config contained non-serializable values; applied safe JSON serialization patch."
            )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save LoRA adapters
        lora_dir = str(dirs["lora"])
        Logger.print_section(f"Saving LoRA adapters to {lora_dir}")
        trainer.save_model(lora_dir)
        self.tokenizer.save_pretrained(lora_dir)

        # Auto-generate README with training metadata (overwrites PEFT template)
        write_lora_readme(
            lora_dir,
            trainer,
            self.model_config,
            training_args,
            run_id=run_id,
        )

        # Optionally merge LoRA adapters for vLLM
        if merge_lora:
            self._merge_lora_adapters(lora_dir, merged_output_dir=str(dirs["merged"]))

        Logger.print_success("Training complete!")
        Logger.print_info("Model saved to", lora_dir)
        if merge_lora:
            Logger.print_info("Merged model (vLLM-ready)", str(dirs["merged"]))
        else:
            Logger.print_info("To merge LoRA for vLLM, run", "")
            Logger.print_section(
                f"   python src/main/serve/merge_lora_for_vllm.py --adapter-path {lora_dir} --output-path {dirs['merged']}")

        return trainer, dirs

    def _create_unsloth_trainer(
        self,
        training_args: TrainingArguments,
        train_dataset: DatasetDict,
        ):
        """Create Unsloth-optimized trainer"""
        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset["train"],
            eval_dataset=train_dataset["test"],
            dataset_text_field="text",
            max_seq_length=self._training_max_seq_length,
            dataset_num_proc=4,
            packing=False, # JSON trajectory boundary preservation
            args=training_args,
        )
        _orig_training_step = trainer.training_step

        def _training_step(model, inputs, *args, **kwargs):
            inputs.pop("datapoint_id", None)
            return _orig_training_step(model, inputs, *args, **kwargs)

        trainer.training_step = _training_step
        return trainer

    def _create_standard_trainer(
        self,
        training_args: TrainingArguments,
        train_dataset: DatasetDict,
        ):
        """Create standard HuggingFace trainer"""
        class CustomDataCollator(DataCollatorForLanguageModeling):
            def __call__(self, features):
                batch = super().__call__(features)
                if "labels" in batch:
                    batch["labels"] = batch["labels"].clone(
                    ).detach().requires_grad_(False)
                return batch

        data_collator = CustomDataCollator(
            tokenizer=self.tokenizer,
            mlm=False
        )

        self.model.train()
        self._verify_gradient_computation(train_dataset)

        class CustomTrainer(Trainer):
            def compute_loss(self,
                             model,
                             inputs,
                             return_outputs=False,
                             num_items_in_batch=None):
                """Custom loss computation to ensure gradients work"""
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
                            raise RuntimeError(
                                "Model outputs don't contain logits")

                        if not logits.requires_grad:
                            for name, param in actual_model.named_parameters():
                                if 'lora' in name.lower() and not param.requires_grad:
                                    param.requires_grad = True
                            outputs = actual_model(**inputs)
                            logits = outputs.get("logits")

                        loss_fct = torch.nn.CrossEntropyLoss()
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                if not loss.requires_grad:
                    trainable_count = sum(
                        1 for p in actual_model.parameters() if p.requires_grad)
                    Logger.print_info(
                        "DEBUG: Trainable parameters", trainable_count)
                    Logger.print_info(
                        "DEBUG: Model training mode", actual_model.training)
                    Logger.print_info("DEBUG: Loss value", loss.item())
                    raise RuntimeError(
                        "Loss doesn't require gradients. Model forward pass is not building computation graph.\n"
                        "This usually happens when:\n"
                        "1. Model was loaded with device_map='auto' (use device_map=None for training)\n"
                        "2. Model parameters don't have requires_grad=True\n"
                        "3. Model is in eval() mode instead of train() mode"
                    )

                return (loss, outputs) if return_outputs else loss

            def training_step(self, model, inputs, *args, **kwargs):
                inputs.pop("datapoint_id", None)
                return super().training_step(model, inputs, *args, **kwargs)

        return CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset["train"],
            eval_dataset=train_dataset["test"],
            data_collator=data_collator,
        )

    def _verify_gradient_computation(self, train_dataset: DatasetDict):
        """Verify model can compute gradients"""
        try:
            sample = train_dataset["train"][0]
            inputs = self.tokenizer(
                DataFormatter.format_training_data(sample["messages"], tokenizer=self.tokenizer),                
                return_tensors="pt",
                truncation=True,
                max_length=self._training_max_seq_length,
                padding="max_length"
            ).to(next(self.model.parameters()).device)
            inputs["labels"] = inputs["input_ids"].clone()

            with torch.enable_grad():
                outputs = self.model(**inputs)
                loss = outputs.loss
                if loss.requires_grad:
                    Logger.print_info(
                        "Model forward pass computes gradients correctly", "")
                else:
                    Logger.print_warning(
                        "Model forward pass doesn't compute gradients - fixing...")
                    for param in self.model.parameters():
                        if param.requires_grad:
                            param.requires_grad = True
        except Exception as e:
            Logger.print_warning(f"Could not verify gradient computation: {e}")

    def _merge_lora_adapters(
        self, 
        lora_dir: str, 
        merged_output_dir: Optional[str] = None
        ):
        """Merge LoRA adapters into base model. If merged_output_dir is set, save there; else use project models/merged."""
        Logger.print_section(
            "Merging LoRA adapters into base model (for vLLM compatibility)...")

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
            model_name = Path(lora_dir).name.replace(
                "-lora", "").replace("-vllm", "")
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
        Logger.print_success(
            f"Merged model saved! Use this with vLLM: {merged_output_dir}")

    def test_model(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 1.0,
    ):
        """Test the fine-tuned model"""
        # Pass the tokenizer to get the correct generation prompt
        formatted_prompt = DataFormatter.format_prompt(prompt, tokenizer=self.tokenizer)
        inputs = self.tokenizer(
            formatted_prompt, return_tensors="pt").to(self.model.device)

        # temperature=0 means greedy decoding; do_sample must be False in that case.
        use_sampling = temperature > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_length,
            "num_return_sequences": 1,
            "do_sample": use_sampling,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if use_sampling:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Gemma 4 uses "model", Qwen uses "assistant"
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1].strip()
        elif "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()

        return response

    def load_checkpoint_for_inference(
        self,
        checkpoint_path: Union[str, Path],
    ) -> None:
        """
        Load LoRA adapter from a checkpoint directory for inference (no training).

        On the first call, wraps the base model with PeftModel. On subsequent calls,
        replaces the adapter weights in-place using load_adapter() + set_adapter()
        rather than unload() + from_pretrained(). The in-place swap is more reliable
        because unload() modifies the model's module structure, making re-wrapping
        fragile across PEFT versions and Unsloth-patched models.
        """
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"Checkpoint path is not a directory: {checkpoint_path}"
            )
        adapter_config = checkpoint_path / "adapter_config.json"
        if not adapter_config.exists():
            raise FileNotFoundError(
                f"Not a LoRA checkpoint (no adapter_config.json): {checkpoint_path}"
            )
        Logger.print_section(f"Loading LoRA from checkpoint: {checkpoint_path}")

        if isinstance(self.model, PeftModel):
            # Adapter already present: swap weights in-place without touching module structure.
            # This is the correct PEFT way to switch checkpoints and avoids the
            # unload() -> from_pretrained() round-trip that causes all checkpoints to
            # produce identical outputs.
            self.model.load_adapter(
                str(checkpoint_path),
                adapter_name="default",
                is_trainable=False,
            )
            self.model.set_adapter("default")
        else:
            # First call: wrap the base model.
            self.model = PeftModel.from_pretrained(
                self.model,
                str(checkpoint_path),
                adapter_name="default",
                is_trainable=False,
            )

        self.model.eval()
        tokenizer_path = checkpoint_path / "tokenizer_config.json"
        if tokenizer_path.exists():
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(checkpoint_path), trust_remote_code=True
            )
            Logger.print_info("Tokenizer", f"loaded from {checkpoint_path}")
        Logger.print_success("Checkpoint loaded; ready for inference.")


def main():
    """Main training pipeline"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Rabit0: Red Team Security AI (vLLM-Compatible)"
    )
    parser.add_argument(
        "--data",         
        required=False,
        default="output/all_training_data.jsonl",
        help="Training data JSONL file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for Rabit0 model (auto-generated if not specified)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1, 
        help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps", 
        type=int, 
        default=8, 
        help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=get_default_learning_rate(),
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=get_default_weight_decay(),
        help="AdamW weight decay (default: 0.0)",
    )
    scheduler_choices = ["constant", "linear", "cosine", "step"]
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        choices=scheduler_choices,
        default="cosine",
        help=f"Learning rate scheduler type (default: cosine). Choices: {', '.join(scheduler_choices)}",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log training metrics every N steps (default: 10)"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=get_default_warmup_ratio(),
        help="Warmup ratio between 0.0 and 1.0 (default: 0.05)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override tokenizer max sequence length (default: auto: 1024 with --use-4bit, else 2048)"
    )
    parser.add_argument(
        "--model-size",
        choices=["7b", "8b", "14b", "30b", "31b", "32b"],  
        default="8b",
        help="Model size: 7b/8b, 14b, 30b, 31b, or 32b"
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="Direct model key from registry (e.g., 'qwen2.5-7b'). Overrides --model-version and --model-size."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models from registry and exit"
    )
    parser.add_argument(
        "--model-version",
        choices=["qwen3", "qwen2.5", "gemma4"],
        default="qwen3",
        help="Model version: qwen3, qwen2.5, or gemma4"
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Data type for training and post-train eval base load: bfloat16 (recommended), float16, or float32"
    )
    parser.add_argument(
        "--no-unsloth",
        action="store_true",
        help="Disable Unsloth optimization"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA) for large models to fit in limited VRAM"
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization for reduced VRAM usage"
    )
    parser.add_argument(
        "--gpu-memory",
        type=int,
        default=40,
        help="Available GPU memory in GiB (default: 40 for 14B model)"
    )
    parser.add_argument(
        "--distributed-strategy",
        choices=["single", "fsdp"],
        default="single",
        help=(
            "Training distribution mode: single (default) or fsdp (multi-GPU full-shard). "
            "Use accelerate launch for fsdp."
        ),
    )
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge LoRA adapters after training (for vLLM)"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help=f"Run evaluation every N steps (default: {get_default_eval_steps()})"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Per-device eval batch size (default: same as --batch-size)"
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Cap validation set size for each eval (default: use full set)"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help="Resume training: no value = latest checkpoint in output dir, or path to checkpoint dir (e.g. .../checkpoints/checkpoint-100)"
    )
    parser.add_argument(
        "--test-only",
        nargs="?",
        default=None,
        const="",
        metavar="CHECKPOINT_PATH",
        help="Test model without training. If CHECKPOINT_PATH is given (e.g. .../checkpoints/checkpoint-1500), load that LoRA checkpoint and run test prompts.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for --test-only inference (default: 0.7)",
    )
    parser.add_argument(
        "--easy-context-max",
        type=int,
        default=512,
        help="Context-length threshold for easy datapoints (default: 512)",
    )
    parser.add_argument(
        "--medium-context-max",
        type=int,
        default=1024,
        help="Context-length threshold for medium datapoints (default: 1024)",
    )
    parser.add_argument(
        "--learning-type",
        choices=[
            "random-shuffled",
            "blocked-learning",
            "interleaved-learning",
            "curriculum",
            "blocked-curriculum",
            "interleaved-curriculum",
        ],
        default="curriculum",
        help="Data ordering mode: random-shuffled, blocked-learning, interleaved-learning, curriculum, blocked-curriculum, interleaved-curriculum",
    )
    parser.add_argument(
        "--eval-after-train",
        action="store_true",
        help=(
            "After successful training, run benchmark eval via ``python -m src.eval.run_eval`` "
            "for each selected bench (separate subprocess per bench)."
        ),
    )
    parser.add_argument(
        "--eval-checkpoint-source",
        choices=["best", "final"],
        default="best",
        help=(
            "Checkpoint for post-train eval: best (lowest eval_loss on disk, else lora/) vs "
            "final (latest checkpoints/checkpoint-*)."
        ),
    )
    parser.add_argument(
        "--eval-bench",
        nargs="*",
        metavar="BENCH",
        default=None,
        help=(
            "Benchmarks for post-train eval (default: all supported). "
            "Example: --eval-bench secqa cybermetric"
        ),
    )
    parser.add_argument(
        "--eval-gpu",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Device for post-train eval (cpu, cuda, cuda:N). Default: run_eval auto device map.",
    )
    parser.add_argument(
        "--eval-use-4bit",
        action="store_true",
        help="Load base model in 4-bit for post-train eval (BitsAndBytes).",
    )
    parser.add_argument(
        "--eval-task-count",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Limit post-train eval to first N tasks per bench using benchmarks_catalog.yaml "
            "(default: 3; e.g. N=2 -> first two tasks; benches with fewer tasks run all available tasks)."
        ),
    )
    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        registry = get_registry()
        models = registry.list_models(supported_only=True)
        Logger.print_header("Available Models in Registry")
        for model in models:
            Logger.print_section(f"Key: {model.key}")
            Logger.print_info("  Name", model.name)
            Logger.print_info(
                "  Version", f"{model.version} | Size: {model.size}")
            Logger.print_info("  Architecture", model.architecture)
            Logger.print_info("  Expected VRAM", model.expected_vram)
            if model.fallbacks:
                Logger.print_info("  Fallbacks", ", ".join(model.fallbacks))
        Logger.print_section("=" * 80)
        Logger.print_section("Usage examples:")
        Logger.print_section("  --model-key qwen2.5-7b")
        Logger.print_section("  --model-version qwen2.5 --model-size 7b")
        Logger.print_section("  --model-version qwen3 --model-size 8b")
        Logger.print_section("=" * 80)
        return

    if not args.data and not args.test_only:
        parser.error(
            "--data is required for training (or use --list-models to see available models)")
    if args.easy_context_max < 1:
        parser.error("--easy-context-max must be >= 1")
    if args.medium_context_max < args.easy_context_max:
        parser.error("--medium-context-max must be >= --easy-context-max")
    if args.warmup_ratio < 0.0 or args.warmup_ratio > 1.0:
        parser.error("--warmup-ratio must be between 0.0 and 1.0")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay must be >= 0")
    if args.logging_steps < 1:
        parser.error("--logging-steps must be >= 1")
    if args.max_seq_length is not None and args.max_seq_length < 1:
        parser.error("--max-seq-length must be >= 1")
    if args.use_4bit and args.use_8bit:
        parser.error("Choose only one quantization mode: --use-4bit or --use-8bit")
    if args.eval_after_train and args.test_only is not None:
        parser.error("--eval-after-train cannot be used with --test-only")
    if args.eval_after_train and args.eval_bench:
        supported = set(supported_eval_benches())
        unknown = [b for b in args.eval_bench if b not in supported]
        if unknown:
            parser.error(
                f"Unknown --eval-bench name(s): {unknown}. "
                f"Supported: {sorted(supported)}"
            )
    if args.eval_task_count is not None and args.eval_task_count < 1:
        parser.error("--eval-task-count must be >= 1")
    if args.distributed_strategy == "fsdp" and (args.use_4bit or args.use_8bit):
        parser.error("--distributed-strategy fsdp does not support --use-4bit/--use-8bit")

    if args.test_only is not None:
        Logger.print_section("Testing model...")
        tuner = QwenVLLMFineTuner(
            model_size=args.model_size,
            dtype=args.dtype,
            use_unsloth=not args.no_unsloth,
            gpu_memory=args.gpu_memory,
            model_version=args.model_version,
            model_key=args.model_key,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit,
            max_seq_length=args.max_seq_length,
            distributed_strategy=args.distributed_strategy,
        )
        checkpoint_path = (args.test_only or "").strip()
        if checkpoint_path:
            tuner.load_checkpoint_for_inference(checkpoint_path)
        test_prompts = [
            # "How do I perform privilege escalation on a Linux system?",
            # "Explain SQL injection exploitation with examples",
            # "What are the steps for Kerberoasting in Active Directory?",
            # "How can I perform lateral movement in a Windows domain?"
            # "How does SQL injection allow attackers to enumerate database tables and extract sensitive data from a vulnerable web application?",
            # "How does a reverse shell work and why do attackers prefer it after exploiting a vulnerability",
            # "What is the purpose of a Web Application Firewall (WAF)",
            # "What is the difference between a SQL injection and a cross-site scripting (XSS) attack?",
            # "How we hack a web application using SQL injection and find the vulnerable code?",
            #"how to bypass waf using sql injection?",
           # "how to bypass waf using xss?",
            #"how to bypass waf using command injection?",
            #"how to bypass waf using file inclusion?",
            #"how to bypass waf using directory traversal?",
            #"how to bypass waf using path traversal?",
           # "how to bypass waf using code injection?",
           # "how to bypass waf using command injection?",
            #"how to bypass waf using file inclusion?",
            #"how to bypass waf using directory traversal?",
            "How to perform privilege escalation on a Linux system?",
            "How to do phishing attack using social engineering?",
            "Create payload to perform sql injection.",


        ]

        for prompt in test_prompts:
            Logger.print_section("=" * 80)
            Logger.print_info("Prompt", prompt)
            Logger.print_section("=" * 80)
            response = tuner.test_model(prompt, temperature=args.temperature)
            Logger.print_info("Response", f"\n{response}")
    else:
        # Full training pipeline: start log file before creating tuner so full terminal output is captured
        registry = get_registry()
        if args.model_key:
            model_config = registry.get_model(args.model_key)
            if not model_config:
                raise ValueError(
                    f"Model key not found: {args.model_key}. Available: {[m.key for m in registry.list_models()]}"
                )
        else:
            model_config = registry.find_model(
                version=args.model_version.lower(), size=args.model_size
            )
            if not model_config:
                raise ValueError(
                    f"No model found for version={args.model_version}, size={args.model_size}. "
                    f"Available: {[f'{m.version}-{m.size}' for m in registry.list_models()]}"
                )
        output_dir = resolve_output_dir(
            args.output if args.output else None,
            model_config,
        )
        paths = get_paths()
        artifact_model_name = paths.generate_artifact_model_name(
            model_version=model_config.version,
            model_size=model_config.size,
        )
        run_id = Path(output_dir).name
        Logger.start_training_log(model_name=artifact_model_name, run_id=run_id)
        tuner = QwenVLLMFineTuner(
            model_size=args.model_size,
            dtype=args.dtype,
            use_unsloth=not args.no_unsloth,
            gpu_memory=args.gpu_memory,
            model_version=args.model_version,
            model_key=args.model_key,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit,
            max_seq_length=args.max_seq_length,
            distributed_strategy=args.distributed_strategy,
        )
        # Auto-detect data manifest from `--data`:
        # - if `--data` is a file: look for `manifest.json` next to it
        # - if `--data` is a directory: look for `manifest.json` inside that directory
        #   (matches versioned combine output folders: combine_output/<version_id>/manifest.json)
        data_manifest: Optional[str] = None
        if args.data:
            data_path = Path(args.data).resolve()
            candidate = (
                data_path / "manifest.json"
                if data_path.is_dir()
                else data_path.parent / "manifest.json"
            )
            if candidate.is_file():
                data_manifest = str(candidate)
                Logger.print_info("Data manifest found", data_manifest)

        try:
            tuner.prepare_model_for_training()
            dataset = tuner.load_training_data(
                args.data,
                easy_context_max=args.easy_context_max,
                medium_context_max=args.medium_context_max,
                learning_type=args.learning_type,
            )

            resume_from_checkpoint = args.resume_from_checkpoint
            if isinstance(resume_from_checkpoint, str) and not resume_from_checkpoint.strip():
                resume_from_checkpoint = None
            elif isinstance(resume_from_checkpoint, str):
                resume_from_checkpoint = resume_from_checkpoint.strip()
            scheduler_aliases = {
                "constant": "constant",
                "linear": "linear",
                "cosine": "cosine",
                # Maps user-facing "step" to a supported HF scheduler.
                "step": "constant_with_warmup",
            }
            Logger.print_info("Requested LR scheduler", args.lr_scheduler_type)
            Logger.print_info(
                "Resolved LR scheduler", scheduler_aliases[args.lr_scheduler_type]
            )
            post_train_eval_config: Dict[str, Any] = {
                "enabled": True,
                "checkpoint_source": args.eval_checkpoint_source,
                "benches": list(args.eval_bench) if args.eval_bench else supported_eval_benches(),
                "gpu": args.eval_gpu,
                "dtype": args.dtype,
                "use_4bit": args.eval_use_4bit,
                "task_count": args.eval_task_count,
            }

            trainer, train_dirs = tuner.train(
                train_dataset=dataset,
                output_dir=output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                lr_scheduler_type=scheduler_aliases[args.lr_scheduler_type],
                warmup_ratio=args.warmup_ratio,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                merge_lora=args.merge_lora,
                eval_steps=args.eval_steps,
                per_device_eval_batch_size=args.eval_batch_size,
                max_eval_samples=args.max_eval_samples,
                logging_steps=args.logging_steps,
                resume_from_checkpoint=resume_from_checkpoint,
                data_manifest=data_manifest,
                launch_command=build_launch_command(sys.argv),
                cli_args=vars(args),
                post_train_eval=post_train_eval_config,
            )
            if args.eval_after_train:
                best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
                eval_ckpt_path = resolve_post_train_eval_checkpoint_path(
                    train_dirs,
                    args.eval_checkpoint_source,
                    best_model_checkpoint=best_ckpt,
                )
                if args.eval_bench:
                    benches_to_run = list(args.eval_bench)
                else:
                    benches_to_run = supported_eval_benches()
                Logger.print_header("Post-training benchmark evaluation")
                Logger.print_info("Eval checkpoint source", args.eval_checkpoint_source)
                Logger.print_info("Resolved checkpoint", str(eval_ckpt_path))
                Logger.print_info("Benches", " ".join(benches_to_run))
                release_training_gpu_for_subprocess_eval(trainer, tuner)
                del trainer
                try:
                    run_post_training_eval(
                        eval_ckpt_path,
                        benches_to_run,
                        gpu=args.eval_gpu,
                        dtype=args.dtype,
                        use_4bit=args.eval_use_4bit,
                        task_count=args.eval_task_count,
                    )
                finally:
                    restore_training_model_cuda(tuner)
            # Test the model
            Logger.print_section("Testing fine-tuned model...")
            test_prompts = [
                "How to perform privilege escalation on a Linux system?",
                "How to do phishing attack using social engineering?",
                "Create payload to perform sql injection.",
                "How to create bomb payload to exploit vulnerable software?",
            ]
            for test_prompt in test_prompts:
                response = tuner.test_model(test_prompt)
                Logger.print_info("Test Prompt", test_prompt)
                Logger.print_info("Response", f"\n{response}")
        finally:
            Logger.remove_file_handler()

if __name__ == "__main__":
    # Check if we're being run with accelerate launch
    import sys

    # If not using accelerate launch, set environment to force single process
    requested_fsdp = "--distributed-strategy" in sys.argv and "fsdp" in sys.argv
    if "accelerate" not in " ".join(sys.argv) and not requested_fsdp:
        # Force single process mode for device_map='auto' compatibility (DRY: reused utility)
        TrainingConfig.setup_single_process_environment()

    main()