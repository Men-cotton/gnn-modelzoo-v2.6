from __future__ import annotations

from typing import Literal, Union

from annotated_types import Ge, Le
from cerebras.modelzoo.config import ModelConfig
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated


class _GNNNestedConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class GCNConfig(_GNNNestedConfig):
    """GCN architecture configuration."""

    type: Literal["gcn", "GCN"] = "gcn"
    n_feat: int
    n_class: int
    n_hid: int = 16
    dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.5
    activation_fn_hidden: Literal["relu", "none"] = "relu"
    activation_fn_output: Literal["relu", "none"] = "none"
    use_bias: bool = True


class GCNSparseMatMulConfig(_GNNNestedConfig):
    """GCN sparse_matmul architecture configuration."""

    type: Literal[
        "gcn_sparse_matmul",
        "gcnsparsematmul",
        "GCNSparseMatMul",
    ] = "gcn_sparse_matmul"
    n_feat: int
    n_class: int
    n_hid: int = 16
    dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.5
    activation_fn_hidden: Literal["relu", "none"] = "relu"
    activation_fn_output: Literal["relu", "none"] = "none"
    use_bias: bool = True


class GATv2Config(_GNNNestedConfig):
    """GATv2 architecture configuration."""

    type: Literal["gatv2", "GATv2"] = "gatv2"
    n_feat: int
    n_class: int
    n_hid: int = 16
    num_heads: int = 8
    dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.5
    activation_fn_hidden: Literal["relu", "none"] = "relu"
    activation_fn_output: Literal["relu", "none"] = "none"
    use_bias: bool = True


class GraphTransformerConfig(_GNNNestedConfig):
    """Graph Transformer architecture configuration."""

    type: Literal[
        "graph_transformer",
        "graphtransformer",
        "GraphTransformer",
    ] = "graph_transformer"
    n_feat: int
    n_class: int
    n_hid: int = 64
    num_heads: int = 8
    dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.5
    activation_fn_hidden: Literal["relu", "none"] = "relu"
    activation_fn_output: Literal["relu", "none"] = "none"
    use_bias: bool = True
    beta: bool = True
    root_weight: bool = True


class GraphSAGEConfig(_GNNNestedConfig):
    """GraphSAGE architecture configuration."""

    type: Literal["graphsage", "GraphSAGE"] = "graphsage"
    n_feat: int
    n_class: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: Annotated[float, Ge(0), Le(1)] = 0.5
    aggregator: Literal["mean", "sum", "max"] = "mean"


ArchitectureConfig = Union[
    GraphTransformerConfig,
    GATv2Config,
    GraphSAGEConfig,
    GCNSparseMatMulConfig,
    GCNConfig,
]


class GNNTaskConfig(_GNNNestedConfig):
    """Trainer-facing GNN task policy."""

    to_float16: bool = False
    disable_log_softmax: bool = False
    compute_eval_metrics: bool = True


class GNNArchConfig(ModelConfig):
    """Base trainer-facing GNN configuration."""

    name: Literal["gnn"] = "gnn"
    architecture: ArchitectureConfig
    task: GNNTaskConfig = Field(default_factory=GNNTaskConfig)


class GNNModelConfig(GNNArchConfig):
    """Trainer-facing GNN model config."""

    @property
    def architecture_config(
        self,
    ) -> Union[
        GraphTransformerConfig,
        GATv2Config,
        GCNConfig,
        GCNSparseMatMulConfig,
        GraphSAGEConfig,
    ]:
        return self.architecture

    @property
    def to_float16(self) -> bool:
        return self.task.to_float16

    @property
    def disable_log_softmax(self) -> bool:
        return self.task.disable_log_softmax

    @property
    def compute_eval_metrics(self) -> bool:
        return self.task.compute_eval_metrics

    @property
    def __model_cls__(self):
        from cerebras.modelzoo.models.gnn.model import GNNModel

        return GNNModel


__all__ = [
    "GATv2Config",
    "ArchitectureConfig",
    "GCNConfig",
    "GCNSparseMatMulConfig",
    "GNNArchConfig",
    "GNNModelConfig",
    "GNNTaskConfig",
    "GraphTransformerConfig",
    "GraphSAGEConfig",
]
