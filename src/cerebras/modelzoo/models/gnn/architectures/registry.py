from __future__ import annotations

from typing import Any, Dict, Literal, Type

import torch.nn as nn

from ..data_processing.batches import FullGraphBatch, GraphSAGEBatch
from ..task.adapters import adapt_full_graph_batch, adapt_graphsage_batch
from ..task.adapters import coerce_gnn_batch
from ..task.config import GATv2Config, GCNConfig, GCNSparseMatMulConfig, GraphSAGEConfig
from .gatv2 import GATv2
from .gcn import GCN
from .gcn_sparse_matmul import GCNSparseMatMul
from .graphsage import GraphSAGE
from .spec import ArchitectureSpec, float32_logits, identity_logits

ArchitectureName = Literal["GATv2", "GCN", "GCNSparseMatMul", "GraphSAGE"]

_ACTIVATION_FN_MAP: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "none": nn.Identity,
}


_ARCHITECTURE_REGISTRY: Dict[str, Type[nn.Module]] = {
    "gatv2": GATv2,
    "gcn": GCN,
    "gcnsparsematmul": GCNSparseMatMul,
    "gcn_sparse_matmul": GCNSparseMatMul,
    "graphsage": GraphSAGE,
}


def _build_gcn(config: GCNConfig) -> nn.Module:
    activation_hidden = _ACTIVATION_FN_MAP[config.activation_fn_hidden]()
    activation_output = _ACTIVATION_FN_MAP[config.activation_fn_output]()
    return GCN(
        in_dim=config.n_feat,
        hidden_dim=config.n_hid,
        num_classes=config.n_class,
        dropout_rate=config.dropout_rate,
        activation_hidden=activation_hidden,
        activation_output=activation_output,
        use_bias=config.use_bias,
    )


def _build_gatv2(config: GATv2Config) -> nn.Module:
    activation_hidden = _ACTIVATION_FN_MAP[config.activation_fn_hidden]()
    activation_output = _ACTIVATION_FN_MAP[config.activation_fn_output]()
    return GATv2(
        in_dim=config.n_feat,
        hidden_dim=config.n_hid,
        num_classes=config.n_class,
        num_heads=config.num_heads,
        dropout_rate=config.dropout_rate,
        activation_hidden=activation_hidden,
        activation_output=activation_output,
        use_bias=config.use_bias,
    )


def _build_graphsage(config: GraphSAGEConfig) -> nn.Module:
    return GraphSAGE(
        input_dim=config.n_feat,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        aggregator=config.aggregator,
        num_classes=config.n_class,
    )


def _build_gcn_sparse_matmul(config: GCNSparseMatMulConfig) -> nn.Module:
    raise NotImplementedError(
        "gcn_sparse_matmul is unsupported: the PubMed full-graph GCN "
        "mapping emits cirh.SparseMatMul with graph nodes as the sparse "
        "dimension, which does not lower to WAF on CSX and is "
        "impractical on CPU."
    )


def _adapt_full_graph(batch, device, model_dtype, architecture):
    batch = coerce_gnn_batch(batch)
    if not isinstance(batch, FullGraphBatch):
        raise TypeError(
            f"Architecture '{architecture}' expects a full-graph batch, "
            f"received '{type(batch).__name__}'."
        )
    return adapt_full_graph_batch(
        batch,
        architecture=architecture,
        device=device,
    )


def _adapt_graphsage(batch, device, model_dtype, architecture):
    batch = coerce_gnn_batch(batch)
    if not isinstance(batch, GraphSAGEBatch):
        raise TypeError(
            f"Architecture '{architecture}' expects a GraphSAGE neighbor batch, "
            f"received '{type(batch).__name__}'."
        )
    return adapt_graphsage_batch(
        batch,
        device=device,
        model_dtype=model_dtype,
    )


_ARCHITECTURE_SPECS: Dict[str, ArchitectureSpec] = {
    "gcn": ArchitectureSpec(
        name="GCN",
        aliases=("gcn",),
        config_types=(GCNConfig,),
        model_cls=GCN,
        build_model=_build_gcn,
        adapt_batch=_adapt_full_graph,
        postprocess_logits=float32_logits,
    ),
    "gatv2": ArchitectureSpec(
        name="GATv2",
        aliases=("gatv2",),
        config_types=(GATv2Config,),
        model_cls=GATv2,
        build_model=_build_gatv2,
        adapt_batch=_adapt_full_graph,
        postprocess_logits=identity_logits,
    ),
    "gcnsparsematmul": ArchitectureSpec(
        name="GCNSparseMatMul",
        aliases=("gcnsparsematmul", "gcn_sparse_matmul"),
        config_types=(GCNSparseMatMulConfig,),
        model_cls=GCNSparseMatMul,
        build_model=_build_gcn_sparse_matmul,
        adapt_batch=_adapt_full_graph,
        postprocess_logits=float32_logits,
    ),
    "graphsage": ArchitectureSpec(
        name="GraphSAGE",
        aliases=("graphsage",),
        config_types=(GraphSAGEConfig,),
        model_cls=GraphSAGE,
        build_model=_build_graphsage,
        adapt_batch=_adapt_graphsage,
        postprocess_logits=identity_logits,
    ),
}

_ARCHITECTURE_SPEC_ALIASES: Dict[str, ArchitectureSpec] = {
    alias: spec
    for spec in _ARCHITECTURE_SPECS.values()
    for alias in (spec.name.lower(), *spec.aliases)
}


def get_architecture_class(name: str) -> Type[nn.Module]:
    try:
        return _ARCHITECTURE_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported core architecture '{name}'.") from exc


def get_architecture_spec(name: str) -> ArchitectureSpec:
    try:
        return _ARCHITECTURE_SPEC_ALIASES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported core architecture '{name}'.") from exc


def get_architecture_spec_for_config(config: Any) -> ArchitectureSpec:
    for spec in _ARCHITECTURE_SPECS.values():
        if spec.matches_config(config):
            return spec
    raise ValueError(
        f"Unsupported architecture config type '{type(config).__name__}'."
    )


__all__ = [
    "ArchitectureName",
    "get_architecture_class",
    "get_architecture_spec",
    "get_architecture_spec_for_config",
]
