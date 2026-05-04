"""
Utilities for making model configs JSON-serializable.
"""

import json
from typing import Any


def _to_json_safe(value: Any):
    """Recursively convert values to JSON-serializable forms."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return f"<non-serializable: {type(value).__name__}>"


def ensure_model_config_json_serializable(model: Any) -> bool:
    """
    Patch model.config.to_json_string when config has non-serializable values.

    Returns True when a patch is applied, False otherwise.
    """
    config = getattr(model, "config", None)
    if config is None:
        return False

    try:
        config.to_json_string()
        return False
    except TypeError:
        pass

    def _safe_to_json_string():
        if hasattr(config, "to_dict"):
            safe_dict = _to_json_safe(config.to_dict())
        else:
            safe_dict = _to_json_safe(getattr(config, "__dict__", {}))
        return json.dumps(safe_dict, indent=2, sort_keys=True) + "\n"

    config.to_json_string = _safe_to_json_string
    return True
