from __future__ import annotations

import logging

from .data_processing.batches import FullGraphBatch, GraphSAGEBatch
from .task.config import GNNArchConfig, GNNModelConfig
from .task.wrapper import GNNTaskWrapper

logger = logging.getLogger(__name__)


class GATv2Model(GNNTaskWrapper):
    """Alias model registered separately for GATv2 experiments."""

    pass


class GCNModel(GNNTaskWrapper):
    """Alias model registered separately for GCN experiments."""

    pass


class GraphSAGEModel(GNNTaskWrapper):
    """Alias model registered separately for GraphSAGE experiments."""

    pass


__all__ = [
    "FullGraphBatch",
    "GATv2Model",
    "GCNModel",
    "GNNArchConfig",
    "GNNModelConfig",
    "GraphSAGEBatch",
    "GraphSAGEModel",
]
