# models/__init__.py

import importlib
import pkgutil
from pathlib import Path

from .registry import MODEL_REGISTRY, register_model  # import the registry

# Automatically import all .py files in this folder (except __init__.py and registry.py)
_pkg_dir = Path(__file__).resolve().parent

for _, module_name, _ in pkgutil.iter_modules([str(_pkg_dir)]):
    if module_name not in {"__init__", "registry", "Jacques"}:
        importlib.import_module(f"{__name__}.{module_name}")
