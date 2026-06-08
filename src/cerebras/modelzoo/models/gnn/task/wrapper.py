from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from cerebras.pytorch.metrics import AccuracyMetric

from ..architectures.registry import get_architecture_spec_for_config
from ..architectures.spec import ArchitectureSpec
from .adapters import GNNBatch
from .config import GNNModelConfig


class GNNTaskWrapper(nn.Module):
    """Trainer-facing GNN wrapper for architecture adapters, loss, and metrics."""

    def __init__(self, config: GNNModelConfig):
        super().__init__()
        if isinstance(config, dict):
            model_dict = config.get("model", config)
            if not isinstance(model_dict, dict):
                raise TypeError("Expected model configuration dictionary.")
            self.config = GNNModelConfig(**model_dict)
        else:
            self.config = config

        self.architecture_config = self.config.architecture_config
        self.architecture_spec = get_architecture_spec_for_config(
            self.architecture_config
        )
        self.model = self.build_model(self.architecture_spec, self.architecture_config)
        self.nll_loss_fn = nn.NLLLoss(ignore_index=-100)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.accuracy_metric = (
            AccuracyMetric(name="eval/masked_accuracy")
            if self.config.compute_eval_metrics
            else None
        )

    def build_model(
        self,
        architecture_spec: ArchitectureSpec,
        architecture_config,
    ) -> nn.Module:
        return architecture_spec.build_model(architecture_config)

    def forward(self, batch: GNNBatch) -> torch.Tensor:
        param = next(self.parameters())
        adapted = self.architecture_spec.adapt_batch(
            batch,
            param.device,
            param.dtype,
            self.architecture_spec.name,
        )
        logits = self.model(*adapted.model_args)
        logits = self.architecture_spec.postprocess_logits(logits)

        labels_long = adapted.labels.to(torch.long)
        mask = adapted.target_mask.to(torch.bool)
        ignore_filled = torch.full_like(labels_long, self.nll_loss_fn.ignore_index)
        labels_with_ignore = torch.where(mask, labels_long, ignore_filled)
        if not self.config.disable_log_softmax:
            log_probs = F.log_softmax(logits, dim=1)
            loss = self.nll_loss_fn(log_probs, labels_with_ignore)
        else:
            log_probs = logits
            loss = self.ce_loss_fn(logits, labels_with_ignore)

        if not self.training and self.accuracy_metric is not None:
            predictions = log_probs.argmax(dim=-1).to(labels_long.dtype).detach()
            weights = mask.to(log_probs.dtype)
            self.accuracy_metric(
                labels=labels_long.clone().detach(),
                predictions=predictions,
                weights=weights,
            )

        return loss


__all__ = ["GNNTaskWrapper"]
