import torch
from allennlp.training.metrics import Metric

from sklearn.metrics import matthews_corrcoef

@Metric.register('mcc')
class MatthewsCorrcoefMetric(Metric):
    def __init__(self):
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
        mcc = matthews_corrcoef(self.labels, self.predictions).round(4)
        if reset:
            self.reset()

        return {
            'mcc': mcc,
        }