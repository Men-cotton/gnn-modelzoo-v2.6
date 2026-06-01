from __future__ import annotations

from contextlib import nullcontext
from typing import Dict

import cerebras.pytorch as cstorch
from cerebras.pytorch.nn.functional import sparse_matmul
import torch
import torch.nn as nn

try:
    from torch.cuda.amp import autocast as cuda_autocast
except ImportError:
    cuda_autocast = None


class GCNSparseMatMulLayer(nn.Module):
    """GCN layer that performs adjacency propagation with cstorch sparse_matmul."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        use_bias: bool,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor, adjacency: Dict[str, torch.Tensor]):
        output_dtype = features.dtype
        if features.dim() == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        if features.dim() != 2:
            raise ValueError(
                f"GCNSparseMatMulLayer expects features with shape [N, F], got {features.shape}."
            )

        support = torch.matmul(features, self.weight)
        indices, values = self._prepare_sparse_adjacency(
            adjacency,
            device=support.device,
            dtype=support.dtype,
            num_nodes=support.size(0),
        )

        ctx = (
            cuda_autocast(enabled=False)
            if cuda_autocast is not None and torch.cuda.is_available()
            else nullcontext()
        )
        with ctx:
            output = self._propagate_sparse_matmul(support, indices, values)

        if self.bias is not None:
            output = output + self.bias
        output = self.activation(output)
        if output.dtype != output_dtype:
            output = output.to(output_dtype)
        return output

    def _prepare_sparse_adjacency(
        self,
        adjacency: Dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(adjacency, dict):
            raise TypeError(
                "GCNSparseMatMul requires adjacency_format: sparse_matmul from the data processor."
            )
        try:
            indices = adjacency["indices"]
            values = adjacency["values"]
        except KeyError as exc:
            raise KeyError(
                "GCNSparseMatMul adjacency must contain 'indices' and 'values'."
            ) from exc

        if indices.dim() == 3 and indices.size(0) == 1:
            indices = indices.squeeze(0)
        if values.dim() == 3 and values.size(0) == 1:
            values = values.squeeze(0)
        if values.dim() != 2:
            raise ValueError(
                f"SparseMatMul adjacency values must have shape [N, fanout], got {values.shape}."
            )
        if indices.dim() != 2:
            raise ValueError(
                f"SparseMatMul adjacency indices must have shape [N, fanout], got {indices.shape}."
            )
        if indices.shape != values.shape:
            raise ValueError(
                "SparseMatMul adjacency indices and values shape mismatch: "
                f"{tuple(indices.shape)} vs {tuple(values.shape)}."
            )
        if indices.size(0) != num_nodes:
            raise ValueError(
                "SparseMatMul adjacency row count does not match feature tensor: "
                f"{indices.size(0)} vs {num_nodes}."
            )

        index_dtype = torch.int32 if cstorch.use_cs() else torch.long
        return (
            indices.to(device=device, dtype=index_dtype),
            values.to(device=device, dtype=dtype),
        )

    def _propagate_sparse_matmul(
        self,
        support: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        input_values = values.unsqueeze(-1)
        weight = support.t().unsqueeze(-1).contiguous()
        messages = sparse_matmul(input_values, indices, weight)
        return messages.sum(dim=-2)


class GCNSparseMatMul(nn.Module):
    """Two-layer GCN variant backed by cstorch sparse_matmul adjacency propagation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout_rate: float,
        activation_hidden: nn.Module,
        activation_output: nn.Module,
        use_bias: bool,
    ):
        super().__init__()
        self.gc1 = GCNSparseMatMulLayer(
            in_features=in_dim,
            out_features=hidden_dim,
            activation=activation_hidden,
            use_bias=use_bias,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.gc2 = GCNSparseMatMulLayer(
            in_features=hidden_dim,
            out_features=num_classes,
            activation=activation_output,
            use_bias=use_bias,
        )

    def forward(self, features: torch.Tensor, adjacency: Dict[str, torch.Tensor]):
        ctx = (
            cuda_autocast(enabled=False)
            if cuda_autocast is not None and torch.cuda.is_available()
            else nullcontext()
        )
        with ctx:
            hidden = self.gc1(features, adjacency)
            hidden = self.dropout(hidden)
            logits = self.gc2(hidden, adjacency)
        return logits


__all__ = ["GCNSparseMatMul", "GCNSparseMatMulLayer"]
