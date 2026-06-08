from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple, Type

import torch
import torch.nn as nn

from ..task.adapters import AdaptedBatch, GNNBatch


BuildArchitectureFn = Callable[[Any], nn.Module]
AdaptBatchFn = Callable[
    [GNNBatch, torch.device, torch.dtype, str],
    AdaptedBatch,
]
PostprocessLogitsFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ArchitectureSpec:
    """Single ownership point for architecture-specific task integration."""

    name: str
    aliases: Tuple[str, ...]
    config_types: Tuple[Type[Any], ...]
    model_cls: Type[nn.Module]
    build_model: BuildArchitectureFn
    adapt_batch: AdaptBatchFn
    postprocess_logits: PostprocessLogitsFn

    def matches_config(self, config: Any) -> bool:
        return isinstance(config, self.config_types)

    def matches_name(self, name: str) -> bool:
        normalized = name.lower()
        return normalized == self.name.lower() or normalized in self.aliases


def identity_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits


def float32_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dtype != torch.float32:
        return logits.to(torch.float32)
    return logits


__all__ = [
    "AdaptBatchFn",
    "ArchitectureSpec",
    "BuildArchitectureFn",
    "PostprocessLogitsFn",
    "float32_logits",
    "identity_logits",
]
