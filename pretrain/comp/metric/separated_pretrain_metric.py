from typing import Optional

import torch
from allennlp.training.metrics import Metric


@Metric.register('separated_mask_accuracy')
class SeparatedMaskAccuracy(Metric):
    def __init__(self):
        self.data_total = 0
        self.data_correct = 0
        self.ctrl_total = 0
        self.ctrl_correct = 0

    def _check_shape(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.BoolTensor]):
        assert predictions.shape == gold_labels.shape == mask.shape

    def _mask_count(self, pred, label, mask):
        masked_pred = torch.masked_select(pred, mask).long()
        masked_label = torch.masked_select(label, mask).long()
        total = mask.sum().item()
        correct = (masked_pred == masked_label).sum().item()
        return correct, total

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.BoolTensor]):
        # First detach tensors to avoid gradient flow.
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        self._check_shape(predictions, gold_labels, mask)

        masked_data_pred, masked_ctrl_pred = predictions[:, 0], predictions[:, 1]
        masked_data_label, masked_ctrl_label = gold_labels[:, 0], gold_labels[:, 1]
        masked_data_mask, masked_ctrl_mask = mask[:, 0], mask[:, 1]

        data_correct, data_total = self._mask_count(masked_data_pred, masked_data_label, masked_data_mask)
        ctrl_correct, ctrl_total = self._mask_count(masked_ctrl_pred, masked_ctrl_label, masked_ctrl_mask)

        self.data_total += data_total
        self.data_correct += data_correct
        self.ctrl_total += ctrl_total
        self.ctrl_correct += ctrl_correct
    
    def get_single_metric(self, total, correct):
        if total > 0:
            ret = float(correct) / float(total)
        else:
            ret = 0.
        return ret

    def get_metric(self, reset: bool):
        data_metric = self.get_single_metric(self.data_total, self.data_correct)
        ctrl_metric = self.get_single_metric(self.ctrl_total, self.ctrl_correct)
        if reset:
            self.reset()
        return {
            'masked_data_accuracy': data_metric,
            'masked_ctrl_accuracy': ctrl_metric,
        }

    def reset(self) -> None:
        self.data_total = self.data_correct = 0
        self.ctrl_total = self.ctrl_correct = 0

