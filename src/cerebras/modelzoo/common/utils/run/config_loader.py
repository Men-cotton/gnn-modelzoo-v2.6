# Men-cotton Original file.
# Have no relationships with Cerebras System.

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cerebras.modelzoo.common.utils.utils import UniqueKeyLoader


EXTENDS_KEY = "extends"


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged[key], value) if key in merged else value
        return merged

    if (
        isinstance(base, list)
        and isinstance(override, list)
        and len(base) == len(override)
        and all(
            isinstance(base_item, dict) and isinstance(override_item, dict)
            for base_item, override_item in zip(base, override)
        )
    ):
        return [
            _deep_merge(base_item, override_item)
            for base_item, override_item in zip(base, override)
        ]

    return override


def _resolve_parent(path: Path, parent: str) -> Path:
    parent_path = Path(parent)
    if parent_path.is_absolute():
        return parent_path
    return path.parent / parent_path


def _normalize_extends(value: Any, path: Path) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise TypeError(
        f"{path}: top-level `{EXTENDS_KEY}` must be a string or a list of strings"
    )


def load_params_file(params_file: str | Path) -> dict:
    """Load a params YAML file, resolving top-level extends entries."""
    return _load_params_file(Path(params_file).resolve(), stack=[])


def _load_params_file(path: Path, stack: list[Path]) -> dict:
    if path in stack:
        cycle = " -> ".join(str(item) for item in [*stack, path])
        raise ValueError(f"Found cyclic YAML extends chain: {cycle}")

    with path.open("r") as stream:
        params = yaml.load(stream, Loader=UniqueKeyLoader) or {}

    if not isinstance(params, dict):
        raise TypeError(f"{path}: params file must contain a YAML mapping")

    parent_refs = _normalize_extends(params.pop(EXTENDS_KEY, None), path)
    merged: dict[str, Any] = {}
    for parent_ref in parent_refs:
        parent_path = _resolve_parent(path, parent_ref).resolve()
        merged = _deep_merge(
            merged,
            _load_params_file(parent_path, stack=[*stack, path]),
        )

    return _deep_merge(merged, params)
