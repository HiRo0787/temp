#!/usr/bin/env python3
"""
Model Registry - SOLID-compliant model configuration management

This module implements the Open/Closed Principle by allowing new models
to be added via configuration files without modifying code.

Principles:
- Single Responsibility: Handles only model configuration and selection
- Open/Closed: Open for extension (via config), closed for modification
- Dependency Inversion: Code depends on abstractions (config), not concrete models
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration data class"""
    key: str
    name: str
    version: str
    size: str
    architecture: str
    expected_vram: str
    fallbacks: List[str]
    lora_target_modules: List[str]
    supported: bool
    trust_remote_code: bool

    @classmethod
    def from_dict(cls, key: str, data: Dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from dictionary"""
        return cls(
            key=key,
            name=data["name"],
            version=data["version"],
            size=data["size"],
            architecture=data["architecture"],
            expected_vram=data["expected_vram"],
            fallbacks=data.get("fallbacks", []),
            lora_target_modules=data.get("lora_target_modules", []),
            supported=data.get("supported", True),
            trust_remote_code=data.get("trust_remote_code", True),
        )


class ModelRegistry:
    """
    Model Registry - Factory pattern for model selection
    
    This class implements:
    - Open/Closed Principle: New models added via config, not code
    - Single Responsibility: Only handles model configuration
    - Dependency Inversion: Depends on config abstraction
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize model registry from configuration file
        
        Args:
            config_path: Path to models.yaml config file. If None, uses default location.
        """
        if config_path is None:
            # Default to src/config/model/models.yaml via ProjectPaths
            from .project_paths import ProjectPaths
            config_path = ProjectPaths().config / "models.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._models: Dict[str, ModelConfig] = {}
        self._aliases: Dict[str, str] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Model configuration file not found: {self.config_path}\n"
                f"Please create src/config/model/models.yaml with model definitions."
            )
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Load models
        models_dict = self._config.get("models", {})
        for key, model_data in models_dict.items():
            self._models[key] = ModelConfig.from_dict(key, model_data)
        
        # Load aliases
        self._aliases = self._config.get("aliases", {})
    
    def get_model(self, identifier: str) -> Optional[ModelConfig]:
        """
        Get model configuration by identifier or alias
        
        Args:
            identifier: Model key (e.g., "qwen2.5-7b") or alias (e.g., "7b", "qwen2.5")
        
        Returns:
            ModelConfig if found, None otherwise
        """
        # Check direct key first
        if identifier in self._models:
            return self._models[identifier]
        
        # Check aliases
        if identifier in self._aliases:
            alias_target = self._aliases[identifier]
            return self._models.get(alias_target)
        
        return None
    
    def find_model(self, version: Optional[str] = None, size: Optional[str] = None) -> Optional[ModelConfig]:
        """
        Find model by version and/or size
        
        Args:
            version: Model version (e.g., "qwen2.5", "qwen3")
            size: Model size (e.g., "7b", "14b", "30b")
        
        Returns:
            First matching ModelConfig, or None if not found
        """
        for model in self._models.values():
            if not model.supported:
                continue
            
            version_match = version is None or model.version == version
            size_match = size is None or model.size == size
            
            if version_match and size_match:
                return model
        
        return None
    
    def get_fallback_chain(self, model_key: str) -> List[ModelConfig]:
        """
        Get fallback chain for a model
        
        Args:
            model_key: Primary model key
        
        Returns:
            List of ModelConfig objects in fallback order (primary first)
        """
        chain = []
        current_key = model_key
        
        # Add primary model
        primary = self.get_model(current_key)
        if primary:
            chain.append(primary)
        
        # Follow fallback chain
        visited = {current_key}
        while current_key in self._models:
            model = self._models[current_key]
            for fallback_key in model.fallbacks:
                if fallback_key not in visited:
                    visited.add(fallback_key)
                    fallback = self.get_model(fallback_key)
                    if fallback:
                        chain.append(fallback)
                        current_key = fallback_key
                        break
            else:
                break
        
        return chain
    
    def list_models(self, version: Optional[str] = None, supported_only: bool = True) -> List[ModelConfig]:
        """
        List all available models
        
        Args:
            version: Filter by version (optional)
            supported_only: Only return supported models
        
        Returns:
            List of ModelConfig objects
        """
        models = []
        for model in self._models.values():
            if supported_only and not model.supported:
                continue
            if version and model.version != version:
                continue
            models.append(model)
        
        return sorted(models, key=lambda m: (m.version, m.size))
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return self._config.get("defaults", {})
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self._load_config()


# Global registry instance (singleton pattern)
_registry: Optional[ModelRegistry] = None


def get_registry(config_path: Optional[Path] = None) -> ModelRegistry:
    """
    Get or create global model registry instance
    
    Args:
        config_path: Optional path to config file (only used on first call)
    
    Returns:
        ModelRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry(config_path)
    return _registry


def reset_registry() -> None:
    """Reset global registry (useful for testing)"""
    global _registry
    _registry = None

