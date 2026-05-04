"""
Training Configuration Utility

Centralizes training configuration settings and utilities.
"""

import os
import json
import torch
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingConfig:
    """Configuration utility for training settings (DRY: centralizes config)"""
    
    DTYPE_MAP = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    
    @staticmethod
    def get_torch_dtype(dtype: str) -> torch.dtype:
        """Convert dtype string to torch dtype"""
        return TrainingConfig.DTYPE_MAP.get(dtype, torch.bfloat16)
    
    @staticmethod
    def setup_single_process_environment():
        """Setup environment variables for single-process training (DRY: reused)"""
        os.environ["ACCELERATE_USE_CPU"] = "false"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    @staticmethod
    def create_accelerate_config(dtype: str) -> Path:
        """Create accelerate config file for single-process training"""
        accelerate_config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "NO",
            "downcast_bf16": "no",
            "gpu_ids": "0",
            "machine_rank": 0,
            "main_training_function": "main",
            "mixed_precision": "bf16" if dtype == "bfloat16" else "fp16" if dtype == "float16" else "no",
            "num_machines": 1,
            "num_processes": 1,
            "rdzv_backend": "static",
            "same_network": True,
            "tpu_env": [],
            "tpu_use_cluster": False,
            "tpu_use_sudo": False,
            "use_cpu": False
        }
        
        config_path = Path("/tmp/accelerate_config.json")
        with open(config_path, "w") as f:
            json.dump(accelerate_config, f)
        os.environ["ACCELERATE_CONFIG_FILE"] = str(config_path)
        return config_path
    
    @staticmethod
    def get_optimizer_config(dtype: str) -> Dict[str, Any]:
        """Get optimizer and precision configuration based on dtype (DRY: centralizes logic)"""
        if dtype == "bfloat16":
            return {
                "optim": "adamw_torch",
                "bf16": True,
                "fp16": False
            }
        elif dtype == "float16":
            return {
                "optim": "adamw_torch",
                "bf16": False,
                "fp16": True
            }
        else:
            return {
                "optim": "adamw_torch",
                "bf16": False,
                "fp16": False
            }

    @staticmethod
    def log_gpu_usage_snapshot(label: str) -> None:
        """Log per-GPU memory usage snapshot for terminal diagnostics."""
        from src.utilities.training_logger import Logger

        if not torch.cuda.is_available():
            Logger.print_info(f"GPU usage ({label})", "CUDA not available")
            return

        gpu_count = torch.cuda.device_count()
        Logger.print_section(f"GPU usage snapshot ({label})")
        for device_idx in range(gpu_count):
            props = torch.cuda.get_device_properties(device_idx)
            allocated_gb = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
            total_gb = props.total_memory / (1024 ** 3)
            Logger.print_info(
                f"GPU {device_idx}",
                f"allocated={allocated_gb:.2f} GiB | reserved={reserved_gb:.2f} GiB | total={total_gb:.2f} GiB",
            )

    # Evaluation defaults: run eval less often and optionally cap samples to avoid long runs / OOM
    EVAL_STEPS_DEFAULT = 500
    MAX_EVAL_SAMPLES_DEFAULT = None  # None = use full validation set

    @classmethod
    def get_eval_config(
        cls,
        eval_steps: Optional[int] = None,
        per_device_eval_batch_size: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
        train_batch_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Get evaluation configuration. Callers can override any value; others fall back to defaults.
        per_device_eval_batch_size defaults to train_batch_size when not set (same as before).
        """
        return {
            "eval_steps": eval_steps if eval_steps is not None else cls.EVAL_STEPS_DEFAULT,
            "per_device_eval_batch_size": (
                per_device_eval_batch_size
                if per_device_eval_batch_size is not None
                else train_batch_size
            ),
            "max_eval_samples": max_eval_samples if max_eval_samples is not None else cls.MAX_EVAL_SAMPLES_DEFAULT,
        }
