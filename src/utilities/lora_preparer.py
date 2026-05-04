"""
LoRA Preparer Utility

Handles preparation and configuration of LoRA adapters for model fine-tuning.
"""

from typing import Any, Dict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.infra.model_registry import get_registry
from src.utilities.training_logger import Logger


class LoRAPreparer:
    """LoRA preparation utility (SRP: single responsibility for LoRA setup)"""
    
    def __init__(self, model: Any, tokenizer: Any, lora_target_modules: list[str], use_unsloth: bool, use_4bit: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.lora_target_modules = lora_target_modules
        self.use_unsloth = use_unsloth
        self.use_4bit = use_4bit
    
    def prepare(self) -> Any:
        """Apply LoRA adapters for efficient fine-tuning"""
        Logger.print_section("Preparing model with LoRA adapters...")
        
        registry = get_registry()
        defaults = registry.get_default_config()
        lora_defaults = defaults.get("lora_config", {})
        
        if self.use_unsloth:
            self._prepare_unsloth_lora(lora_defaults)
        else:
            self._prepare_standard_lora(lora_defaults)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        trainable_percent = 100 * trainable_params / all_params
        
        Logger.print_success("LoRA adapters applied")
        Logger.print_info("Trainable params", f"{trainable_params:,} ({trainable_percent:.2f}%)")
        Logger.print_info("Total params", f"{all_params:,}")
        
        if not self.model.training:
            self.model.train()
            Logger.print_info("Model set to training mode", "")
        
        trainable_count = sum(1 for p in self.model.parameters() if p.requires_grad)
        Logger.print_info("Parameters requiring grad", trainable_count)
        
        return self.model
    
    def _prepare_unsloth_lora(self, lora_defaults: Dict[str, Any]):
        """Prepare LoRA with Unsloth"""
        from unsloth import FastLanguageModel
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_defaults.get("r", 16),
            target_modules=self.lora_target_modules,
            lora_alpha=lora_defaults.get("lora_alpha", 32),
            lora_dropout=lora_defaults.get("lora_dropout", 0.05),
            bias=lora_defaults.get("bias", "none"),
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )
    
    def _prepare_standard_lora(self, lora_defaults: Dict[str, Any]):
        """Prepare LoRA with standard PEFT"""
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=lora_defaults.get("r", 16),
            lora_alpha=lora_defaults.get("lora_alpha", 32),
            target_modules=self.lora_target_modules,
            lora_dropout=lora_defaults.get("lora_dropout", 0.05),
            bias=lora_defaults.get("bias", "none"),
            task_type=lora_defaults.get("task_type", "CAUSAL_LM")
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
        
        self.model.train()
        
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
