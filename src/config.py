import os
import yaml
from typing import Any, Dict, Optional


_CONFIG: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get(key: str, default: Any = None) -> Any:
    cfg = load_config()
    parts = key.split(".")
    val = cfg
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return default
    return val if val is not None else default


def reload_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    global _CONFIG
    _CONFIG = None
    return load_config(config_path)
