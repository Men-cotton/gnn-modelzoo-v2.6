from __future__ import annotations

import logging

from .data_processing.batches import FullGraphBatch, GraphSAGEBatch
from .task.config import GNNArchConfig, GNNModelConfig
from .task.wrapper import GNNTaskWrapper

logger = logging.getLogger(__name__)


class GNNModel(GNNTaskWrapper):
    """Generic trainer-facing GNN model driven by architecture config."""

    pass


__all__ = [
    "FullGraphBatch",
    "GNNArchConfig",
    "GNNModel",
    "GNNModelConfig",
    "GraphSAGEBatch",
]
