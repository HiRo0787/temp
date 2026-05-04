"""
Model Loader Utility

Handles loading of models with Unsloth or standard HuggingFace transformers,
including fallback model support.
"""

import torch
from typing import Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utilities.training_logger import Logger


class ModelLoader:
    """Model loading utility (SRP: single responsibility for model loading)"""
    
    def __init__(
        self,
        model_name: str,
        torch_dtype: torch.dtype,
        gpu_memory: int,
        use_4bit: bool,
        fallback_models: list[str],
        use_8bit: bool = False,
        unsloth_max_seq_length: int = 2048,
        distributed_strategy: str = "single",
    ):
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.gpu_memory = gpu_memory
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.fallback_models = fallback_models
        self.unsloth_max_seq_length = unsloth_max_seq_length
        self.distributed_strategy = distributed_strategy
        self.model = None
        self.tokenizer = None
    
    def load_with_unsloth(self) -> Tuple[Any, Any]:
        """Load model with Unsloth optimization"""
        try:
            from unsloth import FastLanguageModel
            
            if self.use_4bit:
                mode_label = "4-bit quantized (QLoRA)"
            elif self.use_8bit:
                mode_label = "8-bit quantized"
            else:
                mode_label = "full precision"
            Logger.print_section(f"Loading with Unsloth optimization ({mode_label})...")
            Logger.print_info("Attempting to load", self.model_name)
            
            load_in_4bit = self.use_4bit
            load_in_8bit = self.use_8bit
            use_fsdp_sharding = self.distributed_strategy == "fsdp"
            device_map = None if use_fsdp_sharding else "auto"
            max_memory = (
                None
                if use_fsdp_sharding or load_in_4bit or load_in_8bit
                else {0: f"{self.gpu_memory}GiB"}
            )
            Logger.print_info(
                "LOAD_REQUEST",
                f"torch_dtype={self.torch_dtype} | load_in_4bit={load_in_4bit} | load_in_8bit={load_in_8bit} | device_map={device_map} | backend=Unsloth",
            )
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.unsloth_max_seq_length,
                dtype=self.torch_dtype,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
                device_map=device_map,
                max_memory=max_memory,
            )
            
            Logger.print_success("Model loaded with Unsloth! (2x faster training, vLLM-compatible)")
            self._log_runtime_precision_state(loader_name="Unsloth")
            return self.model, self.tokenizer
            
        except ImportError:
            Logger.print_warning("Unsloth not installed. Install with:")
            Logger.print_info("Install", 'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
            Logger.print_section("Falling back to standard loading...")
            raise
        except Exception as e:
            Logger.print_error(f"Error loading {self.model_name} with Unsloth: {e}")
            return self._try_fallback_models_unsloth()
    
    def _try_fallback_models_unsloth(self) -> Tuple[Any, Any]:
        """Try loading fallback models with Unsloth"""
        from unsloth import FastLanguageModel
        
        load_in_4bit = self.use_4bit
        load_in_8bit = self.use_8bit
        use_fsdp_sharding = self.distributed_strategy == "fsdp"
        device_map = None if use_fsdp_sharding else "auto"
        max_memory = (
            None
            if use_fsdp_sharding or load_in_4bit or load_in_8bit
            else {0: f"{self.gpu_memory}GiB"}
        )
        for fallback in self.fallback_models:
            try:
                Logger.print_info("Trying fallback model", fallback)
                self.model_name = fallback
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_name,
                    max_seq_length=self.unsloth_max_seq_length,
                    dtype=self.torch_dtype,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    trust_remote_code=True,
                    device_map=device_map,
                    max_memory=max_memory,
                )
                Logger.print_success(f"Successfully loaded fallback model: {fallback}")
                self._log_runtime_precision_state(loader_name="Unsloth fallback")
                return self.model, self.tokenizer
            except Exception as fallback_error:
                Logger.print_warning(f"Fallback {fallback} failed: {fallback_error}")
                continue
        
        raise RuntimeError("All fallback models failed to load with Unsloth")
    
    def load_standard(self) -> Tuple[Any, Any]:
        """Load model with standard HuggingFace transformers"""
        Logger.print_section("Loading tokenizer...")
        Logger.print_info("Model", self.model_name)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            Logger.print_section(f"Loading model in {self.torch_dtype} precision (this may take a few minutes)...")
            
            if self.use_4bit:
                self._load_with_4bit()
            elif self.use_8bit:
                self._load_with_8bit()
            else:
                self._load_full_precision()
            
            self.model.train()
            Logger.print_success("Model loaded successfully!")
            self._log_runtime_precision_state(loader_name="Transformers")
            return self.model, self.tokenizer
            
        except Exception as e:
            Logger.print_error(f"Error loading {self.model_name}: {e}")
            return self._try_fallback_models_standard()
    
    def _load_with_4bit(self):
        """Load model with 4-bit quantization"""
        from transformers import BitsAndBytesConfig
        
        Logger.print_info("Using 4-bit quantization (QLoRA) for memory efficiency", "")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        Logger.print_info("Model loaded with 4-bit quantization", "")

    def _load_with_8bit(self):
        """Load model with 8-bit quantization"""
        from transformers import BitsAndBytesConfig

        Logger.print_info("Using 8-bit quantization for memory efficiency", "")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        Logger.print_info("Model loaded with 8-bit quantization", "")
    
    def _load_full_precision(self):
        """Load model in full precision"""
        if self.distributed_strategy == "fsdp":
            Logger.print_info(
                "Loading for FSDP",
                "keeping model unplaced so Trainer can shard across multiple GPUs",
            )
        else:
            Logger.print_info("Loading on single GPU (required for training gradients)", "")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=None,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available() and self.distributed_strategy != "fsdp":
            self.model = self.model.to("cuda:0")
        if self.distributed_strategy == "fsdp":
            Logger.print_info("Model loaded for distributed sharding", "")
        else:
            Logger.print_info("Model loaded on single GPU and set to training mode", "")
    
    def _try_fallback_models_standard(self) -> Tuple[Any, Any]:
        """Try loading fallback models with standard loader"""
        for fallback in self.fallback_models:
            try:
                Logger.print_info("Trying fallback model", fallback)
                self.model_name = fallback
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: f"{self.gpu_memory}GiB"} if self.gpu_memory else None
                )
                Logger.print_success(f"Successfully loaded fallback model: {fallback}")
                self._log_runtime_precision_state(loader_name="Transformers fallback")
                return self.model, self.tokenizer
            except Exception as fallback_error:
                Logger.print_warning(f"Fallback {fallback} failed: {fallback_error}")
                continue
        
        raise RuntimeError("All fallback models failed to load")

    def _log_runtime_precision_state(self, loader_name: str):
        """Log actual loaded precision/quantization state from runtime objects."""
        quantized_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        quantized_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        model_dtype = getattr(self.model, "dtype", None)
        model_dtype_str = str(model_dtype) if model_dtype is not None else "unknown"
        Logger.print_info(
            "RUNTIME_SUMMARY",
            f"use_4bit={self.use_4bit} | use_8bit={self.use_8bit} | is_loaded_in_4bit={quantized_4bit} | is_loaded_in_8bit={quantized_8bit} | model_dtype={model_dtype_str} | backend={loader_name}",
        )
