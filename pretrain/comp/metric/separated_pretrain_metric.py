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

class F1Measure:
    def __init__(self, name):
        self.name = name
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update_metric(self,
                      preds: torch.Tensor,
                      labels: torch.Tensor):
        positive_idxes = (preds == 1).int().nonzero().squeeze(-1)
        negative_idxes = (preds == 0).int().nonzero().squeeze(-1)

        true_pos = labels[positive_idxes].sum().item()  # pred=1, label=1
        false_pos = len(positive_idxes) - true_pos      # pred=1, label!=1
        false_neg = labels[negative_idxes].sum().item() # pred=0, label=1
        true_neg = len(negative_idxes) - false_neg      # pred=0, label!=1

        self.tp += true_pos
        self.fp += false_pos
        self.tn += true_neg
        self.fn += false_neg

    def get_metric(self):
        try:
            precision = self.tp / (self.tp + self.fp)
            recall = self.tp / (self.tp + self.fn)
            f1 = 2*precision*recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0

        return {
            f'{self.name}_f1': f1
        }

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0

@Metric.register('separated_mask_f1')
class SeparatedMaskF1(Metric):
    def __init__(self):
        self.data_f1 = F1Measure('masked_data')
        self.ctrl_f1 = F1Measure('masked_ctrl')

    def _check_shape(self,
                     predictions: torch.Tensor,
                     gold_labels: torch.Tensor,
                     mask: Optional[torch.BoolTensor]):
        assert predictions.shape == gold_labels.shape == mask.shape

    def _mask_select(self, pred, label, mask):
        masked_pred = torch.masked_select(pred, mask).long()
        masked_label = torch.masked_select(label, mask).long()
        return masked_pred, masked_label

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

        data_pred, data_label = self._mask_select(masked_data_pred, masked_data_label, masked_data_mask)
        ctrl_pred, ctrl_label = self._mask_select(masked_ctrl_pred, masked_ctrl_label, masked_ctrl_mask)

        self.data_f1.update_metric(data_pred, data_label)
        self.ctrl_f1.update_metric(ctrl_pred, ctrl_label)

    def get_metric(self, reset: bool):
        data_metric = self.data_f1.get_metric()
        ctrl_metric = self.ctrl_f1.get_metric()
        if reset:
            self.reset()
        return dict(**data_metric, **ctrl_metric)

    def reset(self) -> None:
        self.data_f1.reset()
        self.ctrl_f1.reset()

@Metric.register('separated_single_mask_f1')
class SeparatedSingleMaskF1(Metric):
    def __init__(self):
        self.f1 = F1Measure('masked_data')

    def _check_shape(self,
                     predictions: torch.Tensor,
                     gold_labels: torch.Tensor,
                     mask: Optional[torch.BoolTensor]):
        assert predictions.shape == gold_labels.shape == mask.shape

    def _mask_select(self, pred, label, mask):
        masked_pred = torch.masked_select(pred, mask).long()
        masked_label = torch.masked_select(label, mask).long()
        return masked_pred, masked_label

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.BoolTensor]):
        # First detach tensors to avoid gradient flow.
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        self._check_shape(predictions, gold_labels, mask)

        masked_pred, masked_label = self._mask_select(predictions, gold_labels, mask)

        self.f1.update_metric(masked_pred, masked_label)

    def get_metric(self, reset: bool):
        metric = self.f1.get_metric()
        if reset:
            self.reset()
        return metric

    def reset(self) -> None:
        self.f1.reset()