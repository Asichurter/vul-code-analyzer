from typing import Union

import numpy
import torch
from allennlp.training.metrics import Metric

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@Metric.register('multiclass_classification')
class MulticlassClassificationMetric(Metric):
    def __init__(self,
                 average: str = 'macro',
                 zero_division: Union[str, int] = 0):
        self.average = average
        self.zero_division = zero_division

        self.predictions = []
        self.labels = []

    def __call__(self, predictions, gold_labels, mask):
        # First detach tensors to avoid gradient flow.
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.bool()

        masked_preds = torch.masked_select(predictions, mask).cpu().tolist()
        masked_labels = torch.masked_select(gold_labels, mask).cpu().tolist()
        self.predictions.extend(masked_preds)
        self.labels.extend(masked_labels)

    def reset(self) -> None:
        self.predictions.clear()
        self.labels.clear()

    def get_metric(self, reset: bool):
        accuracy = numpy.round(accuracy_score(self.labels, self.predictions), 4)
        precision = precision_score(self.labels, self.predictions, average=self.average, zero_division=self.zero_division).round(4)
        recall = recall_score(self.labels, self.predictions, average=self.average, zero_division=self.zero_division).round(4)
        f1 = f1_score(self.labels, self.predictions, average=self.average, zero_division=self.zero_division).round(4)

        if reset:
            self.reset()
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
