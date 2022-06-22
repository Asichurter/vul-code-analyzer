from typing import Optional

import torch
from allennlp.training.metrics import Metric


@Metric.register('unified_mask_accuracy')
class UnifiedMaskAccuracy(Metric):
    def __init__(self):
        self.total = 0
        self.correct = 0

    def _check_shape(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.BoolTensor]):
        assert predictions.shape == gold_labels.shape == mask.shape

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.BoolTensor]):
        # First detach tensors to avoid gradient flow.
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)


        self._check_shape(predictions, gold_labels, mask)
        masked_pred = torch.masked_select(predictions, mask).long()
        masked_label = torch.masked_select(gold_labels, mask).long()
        total = mask.sum().item()
        correct = (masked_pred == masked_label).sum().item()

        self.correct += correct
        self.total += total

    def get_metric(self, reset: bool):
        if self.total > 0:
            ret = float(self.correct) / float(self.total)
        else:
            ret = 0.
        if reset:
            self.reset()
        return {'masked_accuracy': ret}

    def reset(self) -> None:
        self.total = self.correct = 0

