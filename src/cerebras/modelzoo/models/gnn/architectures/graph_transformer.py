from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple, Union

import cerebras.pytorch as cstorch
import cerebras.pytorch.nn.functional as csF
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data_processing.runtime.torch import EdgeIndexAdjacency
from .ops import segment_softmax

try:
    from torch.amp import autocast as torch_autocast
except ImportError:
    torch_autocast = None

try:
    from torch.cuda.amp import autocast as cuda_autocast
except ImportError:
    cuda_autocast = None

AdjacencyInput = Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Dict[str, torch.Tensor],
    EdgeIndexAdjacency,
]


def _disabled_cuda_autocast():
    if not torch.cuda.is_available():
        return nullcontext()
    if torch_autocast is not None:
        return torch_autocast("cuda", enabled=False)
    if cuda_autocast is not None:
        return cuda_autocast(enabled=False)
    return nullcontext()


class GraphTransformerLayer(nn.Module):
    """PyG TransformerConv semantics expanded into local tensor operations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        root_weight: bool = True,
    ):
        super().__init__()
        if heads <= 0:
            raise ValueError("GraphTransformerLayer requires heads > 0.")

        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.beta = beta and root_weight
        self.dropout_rate = dropout_rate
        self.root_weight = root_weight

        self.lin_key = nn.Linear(in_features, heads * out_features, bias=use_bias)
        self.lin_query = nn.Linear(in_features, heads * out_features, bias=use_bias)
        self.lin_value = nn.Linear(in_features, heads * out_features, bias=use_bias)

        total_out_channels = out_features * (heads if concat else 1)
        if root_weight:
            self.lin_skip = nn.Linear(in_features, total_out_channels, bias=use_bias)
        else:
            self.lin_skip = None
        if self.beta:
            self.lin_beta = nn.Linear(3 * total_out_channels, 1, bias=False)
        else:
            self.lin_beta = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.lin_key, self.lin_query, self.lin_value):
            nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
            if layer.bias is not None:
                fan_in = layer.weight.size(1)
                bound = fan_in**-0.5 if fan_in > 0 else 0
                nn.init.uniform_(layer.bias, -bound, bound)
        if self.lin_skip is not None:
            nn.init.kaiming_uniform_(self.lin_skip.weight, a=5**0.5)
            if self.lin_skip.bias is not None:
                fan_in = self.lin_skip.weight.size(1)
                bound = fan_in**-0.5 if fan_in > 0 else 0
                nn.init.uniform_(self.lin_skip.bias, -bound, bound)
        if self.lin_beta is not None:
            nn.init.kaiming_uniform_(self.lin_beta.weight, a=5**0.5)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: AdjacencyInput,
    ) -> torch.Tensor:
        output_dtype = features.dtype
        if features.dim() == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        if features.dim() != 2:
            raise ValueError(
                "GraphTransformerLayer expects node features with shape "
                f"[N, F], got {features.shape}."
            )

        if cstorch.use_cs():
            dense_adjacency = self._prepare_dense_adjacency(adjacency, features.device)
            if dense_adjacency is None:
                raise TypeError(
                    "GraphTransformer on CSX requires dense full-graph adjacency from the "
                    "ModelZoo data pipeline."
                )
            out = self._forward_dense(features, dense_adjacency)
        else:
            edge_index = self._prepare_edge_index(adjacency, features.device)
            out = self._forward_edge_index(features, edge_index)

        if out.dtype != output_dtype:
            out = out.to(output_dtype)
        return out

    def _linear(self, layer: nn.Linear, features: torch.Tensor) -> torch.Tensor:
        projected = csF.matmul(layer.weight, features.transpose(0, 1)).transpose(0, 1)
        if layer.bias is not None:
            projected = projected + layer.bias
        return projected

    def _project(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self._linear(self.lin_query, features).view(
            -1, self.heads, self.out_features
        )
        key = self._linear(self.lin_key, features).view(
            -1, self.heads, self.out_features
        )
        value = self._linear(self.lin_value, features).view(
            -1, self.heads, self.out_features
        )
        return query, key, value

    def _finalize(self, features: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if self.concat:
            out = out.reshape(-1, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)

        if self.lin_skip is not None:
            skip = self._linear(self.lin_skip, features)
            if self.lin_beta is not None:
                beta = self._linear(
                    self.lin_beta,
                    torch.cat((out, skip, out - skip), dim=-1),
                ).sigmoid()
                out = beta * skip + (1 - beta) * out
            else:
                out = out + skip
        return out

    def _forward_edge_index(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = features.size(0)
        source_selector = edge_index.new_tensor([0])
        target_selector = edge_index.new_tensor([1])
        source_nodes = torch.index_select(edge_index, 0, source_selector).squeeze(0)
        target_nodes = torch.index_select(edge_index, 0, target_selector).squeeze(0)

        query, key, value = self._project(features)
        query_i = query.index_select(0, target_nodes)
        key_j = key.index_select(0, source_nodes)
        value_j = value.index_select(0, source_nodes)

        scores = (query_i * key_j).sum(dim=-1) * (self.out_features**-0.5)
        alpha = segment_softmax(scores, target_nodes, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        messages = value_j * alpha.unsqueeze(-1)

        out = features.new_zeros((num_nodes, self.heads, self.out_features))
        out.index_add_(0, target_nodes, messages)
        return self._finalize(features, out)

    def _forward_dense(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        if adjacency.dim() == 3 and adjacency.size(0) == 1:
            adjacency = adjacency.squeeze(0)
        if adjacency.dim() != 2:
            raise ValueError(
                f"Dense adjacency must have shape [N, N], got {adjacency.shape}."
            )

        query, key, value = self._project(features)
        edge_mask = adjacency != 0
        scores = (query.unsqueeze(0) * key.unsqueeze(1)).sum(dim=-1)
        scores = scores * (self.out_features**-0.5)
        scores = scores.masked_fill(~edge_mask.unsqueeze(-1), float("-inf"))
        has_incoming = edge_mask.any(dim=0).view(1, -1, 1)
        scores = torch.where(has_incoming, scores, torch.zeros_like(scores))
        alpha = torch.softmax(scores, dim=0)
        alpha = alpha.masked_fill(~edge_mask.unsqueeze(-1), 0.0)
        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)

        out_by_head = []
        for head_idx in range(self.heads):
            out_by_head.append(
                csF.matmul(
                    alpha[:, :, head_idx].transpose(0, 1),
                    value[:, head_idx, :],
                )
            )
        out = torch.stack(out_by_head, dim=1)
        return self._finalize(features, out)

    def _prepare_dense_adjacency(
        self,
        adjacency: AdjacencyInput,
        device: torch.device,
    ) -> torch.Tensor | None:
        if torch.is_tensor(adjacency) and not adjacency.is_sparse:
            return adjacency.to(device=device)
        return None

    def _prepare_edge_index(
        self,
        adjacency: AdjacencyInput,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(adjacency, EdgeIndexAdjacency):
            edge_index = adjacency.edge_index
        elif isinstance(adjacency, dict):
            edge_index = adjacency.get("edge_index")
            if edge_index is None:
                raise KeyError("Expected 'edge_index' in adjacency dictionary.")
        elif isinstance(adjacency, (tuple, list)):
            if len(adjacency) != 2:
                raise ValueError(
                    "Adjacency tuple must contain (edge_index, edge_weight)."
                )
            edge_index = adjacency[0]
        elif torch.is_tensor(adjacency):
            if adjacency.dim() == 3 and adjacency.size(0) == 1:
                adjacency = adjacency.squeeze(0)
            if adjacency.is_sparse:
                edge_index = adjacency.coalesce().indices()
            else:
                edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
        else:
            raise TypeError(
                "Unsupported adjacency type. Expected tuple, dict, EdgeIndexAdjacency, "
                "or torch.Tensor."
            )

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E]; received {tuple(edge_index.shape)}"
            )
        return edge_index.to(device=device, dtype=torch.long)


class GraphTransformer(nn.Module):
    """Graph Transformer node classifier backed by local TransformerConv layers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int,
        dropout_rate: float,
        activation_hidden: nn.Module,
        activation_output: nn.Module,
        use_bias: bool,
        beta: bool,
        root_weight: bool,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("GraphTransformer requires num_heads > 0.")

        self.conv1 = GraphTransformerLayer(
            in_features=in_dim,
            out_features=hidden_dim,
            heads=num_heads,
            concat=True,
            beta=beta,
            dropout_rate=dropout_rate,
            root_weight=root_weight,
            use_bias=use_bias,
        )
        self.activation_hidden = activation_hidden
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = GraphTransformerLayer(
            in_features=hidden_dim * num_heads,
            out_features=num_classes,
            heads=1,
            concat=False,
            beta=beta,
            dropout_rate=dropout_rate,
            root_weight=root_weight,
            use_bias=use_bias,
        )
        self.activation_output = activation_output

    def forward(
        self,
        features: torch.Tensor,
        adjacency: AdjacencyInput,
    ) -> torch.Tensor:
        if features.dim() == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        if features.dim() != 2:
            raise ValueError(
                f"GraphTransformer expects node features with shape [N, F], got {features.shape}."
            )

        ctx = _disabled_cuda_autocast()
        with ctx:
            hidden = self.conv1(features, adjacency)
            hidden = self.activation_hidden(hidden)
            hidden = self.dropout(hidden)
            logits = self.conv2(hidden, adjacency)
        return self.activation_output(logits)


__all__ = ["GraphTransformer", "GraphTransformerLayer"]
