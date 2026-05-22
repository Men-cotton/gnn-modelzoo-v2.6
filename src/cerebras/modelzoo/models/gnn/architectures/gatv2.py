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


class GATv2Layer(nn.Module):
    """A single GATv2 layer matching PyG's basic tensor-edge semantics."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        activation: nn.Module = nn.Identity(),
        use_bias: bool = True,
        concat: bool = True,
        dropout_rate: float = 0.0,
        share_weights: bool = False,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        residual: bool = False,
    ):
        super().__init__()
        if heads <= 0:
            raise ValueError("GATv2Layer requires heads > 0.")
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.share_weights = share_weights
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.residual = residual

        self.lin_l = nn.Linear(in_features, heads * out_features, bias=use_bias)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = nn.Linear(in_features, heads * out_features, bias=use_bias)
        self.att = nn.Parameter(torch.empty(1, heads, out_features))

        total_out_channels = out_features * (heads if concat else 1)
        if residual:
            self.res = nn.Linear(in_features, total_out_channels, bias=False)
        else:
            self.res = None
        if use_bias:
            self.bias = nn.Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter("bias", None)

        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_l.weight)
        if self.lin_l.bias is not None:
            nn.init.zeros_(self.lin_l.bias)
        if not self.share_weights:
            nn.init.xavier_uniform_(self.lin_r.weight)
            if self.lin_r.bias is not None:
                nn.init.zeros_(self.lin_r.bias)
        if self.res is not None:
            nn.init.xavier_uniform_(self.res.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

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
                f"GATv2Layer expects node features with shape [N, F], got {features.shape}."
            )

        if cstorch.use_cs():
            dense_adjacency = self._prepare_dense_adjacency(adjacency, features.device)
            if dense_adjacency is None:
                raise TypeError(
                    "GATv2 on CSX requires dense full-graph adjacency from the "
                    "ModelZoo data pipeline."
                )
            out = self._forward_dense(features, dense_adjacency)
        else:
            edge_index = self._prepare_edge_index(adjacency, features.device)
            out = self._forward_edge_index(features, edge_index)

        out = self.activation(out)
        if out.dtype != output_dtype:
            out = out.to(output_dtype)
        return out

    def _linear(self, layer: nn.Linear, features: torch.Tensor) -> torch.Tensor:
        projected = csF.matmul(layer.weight, features.transpose(0, 1)).transpose(0, 1)
        if layer.bias is not None:
            projected = projected + layer.bias
        return projected

    def _project(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_l = self._linear(self.lin_l, features).view(-1, self.heads, self.out_features)
        h_r = self._linear(self.lin_r, features).view(-1, self.heads, self.out_features)
        return h_l, h_r

    def _finalize(self, features: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if self.concat:
            out = out.reshape(-1, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)
        if self.res is not None:
            out = out + self.res(features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _forward_edge_index(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = features.size(0)
        edge_index = self._maybe_add_self_loops(edge_index, num_nodes, features.device)

        # Avoid direct indexing into integer tensors on WSE-facing code paths.
        source_selector = edge_index.new_tensor([0])
        target_selector = edge_index.new_tensor([1])
        source_nodes = torch.index_select(edge_index, 0, source_selector).squeeze(0)
        target_nodes = torch.index_select(edge_index, 0, target_selector).squeeze(0)

        h_l, h_r = self._project(features)
        x_j = h_l.index_select(0, source_nodes)
        x_i = h_r.index_select(0, target_nodes)

        attention_input = F.leaky_relu(
            x_i + x_j,
            negative_slope=self.negative_slope,
        )
        scores = (attention_input * self.att).sum(dim=-1)

        alpha = segment_softmax(scores, target_nodes, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        messages = x_j * alpha.unsqueeze(-1)

        out = features.new_zeros((num_nodes, self.heads, self.out_features))
        out.index_add_(0, target_nodes, messages)
        return self._finalize(features, out)

    def _forward_dense(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = features.size(0)
        if adjacency.dim() == 3 and adjacency.size(0) == 1:
            adjacency = adjacency.squeeze(0)
        if adjacency.dim() != 2:
            raise ValueError(
                f"Dense adjacency must have shape [N, N], got {adjacency.shape}."
            )

        h_l, h_r = self._project(features)
        edge_mask = adjacency != 0
        if self.add_self_loops:
            eye = torch.eye(num_nodes, dtype=torch.bool, device=features.device)
            edge_mask = torch.logical_or(edge_mask, eye)

        attention_input = F.leaky_relu(
            h_r.unsqueeze(0) + h_l.unsqueeze(1),
            negative_slope=self.negative_slope,
        )
        scores = (attention_input * self.att).sum(dim=-1)
        scores = scores.masked_fill(~edge_mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(scores, dim=0)
        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        out_by_head = []
        for head_idx in range(self.heads):
            out_by_head.append(
                csF.matmul(
                    alpha[:, :, head_idx].transpose(0, 1),
                    h_l[:, head_idx, :],
                )
            )
        out = torch.stack(out_by_head, dim=1)
        return self._finalize(features, out)

    def _maybe_add_self_loops(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.add_self_loops:
            return edge_index
        source_nodes = torch.index_select(
            edge_index, 0, edge_index.new_tensor([0])
        ).squeeze(0)
        target_nodes = torch.index_select(
            edge_index, 0, edge_index.new_tensor([1])
        ).squeeze(0)
        non_loop_mask = source_nodes != target_nodes
        edge_index = edge_index[:, non_loop_mask]
        loops = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
        loops = loops.unsqueeze(0).expand(2, -1)
        return torch.cat((edge_index, loops), dim=1)

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


class GATv2(nn.Module):
    """Standard two-layer Graph Attention Network v2."""

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
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("GATv2 requires num_heads > 0.")
        self.gat1 = GATv2Layer(
            in_features=in_dim,
            out_features=hidden_dim,
            heads=num_heads,
            activation=activation_hidden,
            use_bias=use_bias,
            concat=True,
            dropout_rate=dropout_rate,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.gat2 = GATv2Layer(
            in_features=hidden_dim * num_heads,
            out_features=num_classes,
            heads=1,
            activation=activation_output,
            use_bias=use_bias,
            concat=False,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        features: torch.Tensor,
        adjacency: AdjacencyInput,
    ) -> torch.Tensor:
        ctx = _disabled_cuda_autocast()
        with ctx:
            hidden = self.gat1(features, adjacency)
            hidden = self.dropout(hidden)
            logits = self.gat2(hidden, adjacency)
        return logits


__all__ = ["GATv2", "GATv2Layer"]
