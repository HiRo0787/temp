#!/usr/bin/env python3
"""
Project Paths - Standardized directory structure management

This module provides a centralized way to manage project paths,
ensuring all scripts use the same directory structure.

Standard Structure:
    <project_root>/
    ├── artifacts/         # Finetune outputs: <model_name>/rabit0-v1-run1, run2, ...; logs in separate folder
    ├── combine_output/   # Combined training data (e.g. all_training_data.jsonl)
    ├── data/              # Raw training data
    ├── data_points/       # Data point categories (exploit, payload, QA, tool, unrestriction)
    ├── docker/            # Docker config and scripts (root)
    ├── docs/              # Documentation
    ├── logs/              # Training and run logs (root)
    ├── mlruns/            # MLflow runs
    ├── output/            # Generated outputs (JSONL, reports)
    ├── sources/           # Training data sources
    ├── src/               # Source code
    │   ├── config/        # Config files
    │   │   ├── data_prep/  # Data prep config (system_prompts, quality_guidelines, etc.)
    │   │   └── model/      # Model config (models.yaml, rabit0_config.yaml)
    │   ├── data_prep/     # Data preparation scripts
    │   ├── docker/        # Docker scripts (src)
    │   ├── eval/          # Evaluation scripts
    │   ├── infra/         # Infrastructure (project_paths, model_registry, etc.)
    │   ├── review/        # Review and MLflow integration
    │   ├── serve/         # Serving (vLLM, merge_lora)
    │   ├── training/      # Single- and multi-stage fine-tuning
    │   └── utilities/     # Shared utilities
    └── tests/             # Tests
"""

from pathlib import Path
from typing import Optional
from datetime import datetime


class ProjectPaths:
    """
    Centralized project path management
    
    Provides standardized paths for all project directories.
    All paths are relative to the project root.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize project paths
        
        Args:
            project_root: Root directory of the project. If None, auto-detects.
        """
        if project_root is None:
            # Auto-detect project root (directory containing src/, e.g. rabit0_rsr)
            # __file__ is src/infra/project_paths.py -> parent.parent.parent = project root
            self.root = Path(__file__).parent.parent.parent.resolve()
        else:
            self.root = Path(project_root).resolve()
        
        # Ensure root exists
        self.root.mkdir(parents=True, exist_ok=True)
    
    # Core directories
    @property
    def artifacts(self) -> Path:
        """Artifacts root (finetune outputs: artifacts/<model_name>/rabit0-v1-run1, run2, ...)"""
        return self._ensure_dir(self.root / "artifacts")

    @property
    def combine_output(self) -> Path:
        """Combined training data output (e.g. all_training_data.jsonl)"""
        return self._ensure_dir(self.root / "combine_output")

    @property
    def config(self) -> Path:
        """Model configuration directory (src/config/model: models.yaml, rabit0_config.yaml)"""
        return self._ensure_dir(self.root / "src" / "config" / "model")

    @property
    def config_data_prep(self) -> Path:
        """Data prep config (src/config/data_prep: system_prompts, quality_guidelines, etc.)"""
        return self._ensure_dir(self.root / "src" / "config" / "data_prep")

    @property
    def data(self) -> Path:
        """Training data directory (raw data)"""
        return self._ensure_dir(self.root / "data")

    @property
    def data_points(self) -> Path:
        """Data point categories (exploit, payload, QA, tool, unrestriction)"""
        return self._ensure_dir(self.root / "data_points")

    @property
    def docker(self) -> Path:
        """Docker directory at project root"""
        return self._ensure_dir(self.root / "docker")

    @property
    def docs(self) -> Path:
        """Documentation directory"""
        return self._ensure_dir(self.root / "docs")

    @property
    def logs(self) -> Path:
        """Logs directory at project root (training and run logs)"""
        return self._ensure_dir(self.root / "logs")

    @property
    def output(self) -> Path:
        """Generated outputs directory (JSONL, reports). Not auto-created."""
        return self.root / "output"

    @property
    def sources(self) -> Path:
        """Training data sources directory. Not auto-created."""
        return self.root / "sources"

    @property
    def src(self) -> Path:
        """Source code directory"""
        return self._ensure_dir(self.root / "src")

    @property
    def utilities(self) -> Path:
        """Utilities directory (src/utilities)"""
        return self._ensure_dir(self.root / "src" / "utilities")

    @property
    def tests(self) -> Path:
        """Tests directory"""
        return self._ensure_dir(self.root / "tests")
    
    # Models directory structure (paths only; not created - use artifacts for outputs)
    @property
    def models(self) -> Path:
        """Main models directory (not auto-created; artifacts folder is used instead)"""
        return self.root / "models"

    @property
    def models_checkpoints(self) -> Path:
        """Training checkpoints directory (not auto-created)"""
        return self.models / "checkpoints"

    @property
    def models_merged(self) -> Path:
        """Merged models directory for vLLM (not auto-created)"""
        return self.models / "merged"

    @property
    def models_lora(self) -> Path:
        """LoRA adapters directory (not auto-created)"""
        return self.models / "lora"

    # MLflow directory
    @property
    def mlruns(self) -> Path:
        """MLflow runs directory"""
        return self._ensure_dir(self.root / "mlruns")
    
    # Helper methods
    def _ensure_dir(self, path: Path) -> Path:
        """Ensure directory exists and return Path"""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_model_path(
        self,
        model_name: str,
        model_type: str = "checkpoints",
        create: bool = True
    ) -> Path:
        """
        Get path for a specific model
        
        Args:
            model_name: Name of the model (e.g., "rabit0-v1.0-qwen2.5-7b")
            model_type: Type of model - "checkpoints", "merged", or "lora"
            create: Whether to create the directory if it doesn't exist
        
        Returns:
            Path to the model directory
        """
        if model_type == "checkpoints":
            base = self.models_checkpoints
        elif model_type == "merged":
            base = self.models_merged
        elif model_type == "lora":
            base = self.models_lora
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Use 'checkpoints', 'merged', or 'lora'")
        
        path = base / model_name
        # Do not create models/ subdirs; artifacts folder is used for outputs
        if create and model_type not in ("checkpoints", "merged", "lora"):
            self._ensure_dir(path)

        return path
    
    def generate_model_name(
        self,
        version: str = "v1.0",
        model_version: str = "qwen2.5",
        model_size: str = "7b",
        suffix: Optional[str] = None
    ) -> str:
        """
        Generate standardized model name
        
        Args:
            version: Model version (e.g., "v1.0")
            model_version: Base model version (e.g., "qwen2.5")
            model_size: Model size (e.g., "7b")
            suffix: Optional suffix (e.g., "optimized", "merged")
        
        Returns:
            Standardized model name (e.g., "rabit0-v1.0-qwen2.5-7b-optimized")
        """
        name = f"rabit0-{version}-{model_version}-{model_size}"
        if suffix:
            name = f"{name}-{suffix}"
        return name

    def generate_artifact_prefix(self, version: str = "v1.0") -> str:
        """
        Generate short prefix for run dir names (e.g. rabit0-v1-run1).

        Args:
            version: Model version (e.g. "v1.0" -> prefix "rabit0-v1").

        Returns:
            Prefix string (e.g. "rabit0-v1").
        """
        v = version.split(".")[0] if version else "v1"
        return f"rabit0-{v}"

    def generate_artifact_model_name(self, model_version: str = "qwen2.5", model_size: str = "7b") -> str:
        """
        Generate model subfolder name for artifacts (e.g. artifacts/qwen2.5-7b/...).

        Args:
            model_version: Base model version (e.g. "qwen2.5").
            model_size: Model size (e.g. "7b").

        Returns:
            Subfolder name (e.g. "qwen2.5-7b").
        """
        return f"{model_version}-{model_size}"

    def list_models(self, model_type: str = "all") -> list[Path]:
        """
        List all models in the models directory
        
        Args:
            model_type: Type to list - "checkpoints", "merged", "lora", or "all"
        
        Returns:
            List of model paths
        """
        models = []
        for kind, base in [("checkpoints", self.models_checkpoints), ("merged", self.models_merged), ("lora", self.models_lora)]:
            if model_type not in ("all", kind):
                continue
            if base.exists() and base.is_dir():
                models.extend(base.iterdir())
        return [p for p in models if p.is_dir()]
    
    def get_timestamped_path(self, base_name: str, directory: Path) -> Path:
        """
        Get a timestamped path for temporary or versioned files
        
        Args:
            base_name: Base name for the file/directory
            directory: Directory to create the path in
        
        Returns:
            Path with timestamp appended
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{base_name}_{timestamp}"
        return directory / name


# Global instance
_paths: Optional[ProjectPaths] = None


def get_paths(project_root: Optional[Path] = None) -> ProjectPaths:
    """
    Get or create global ProjectPaths instance
    
    Args:
        project_root: Optional project root (only used on first call)
    
    Returns:
        ProjectPaths instance
    """
    global _paths
    if _paths is None:
        _paths = ProjectPaths(project_root)
    return _paths


def reset_paths() -> None:
    """Reset global paths instance (useful for testing)"""
    global _paths
    _paths = None

