import torch
from torch.nn import functional
from functools import reduce

from allennlp.common import Registrable


# TODO: Modules and functions in this file should be validated...

class LossFunc(Registrable, torch.nn.Module):
    def __init__(self, ignore_index: int=-100):
        self.ignore_index = ignore_index
        super().__init__()

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def align_pred_label_batch_size(self, pred, label):
        '''
        Call this method to align the batch dimension of predictions and
        labels, dealing with unmatched batch size in tail batch caused by
        different padding implementations of different fields (such as
        'TensorField' type labels will not be padded).
        # [Note]: This method assumes predictions and labels are matched
        #         at corresponding dimension, which may not be true.
        '''
        pred_size, label_size = pred.size(0), label.size(0)
        if pred_size == label_size:
            return pred, label
        else:
            smaller_batch_size = min(pred_size, label_size)
            return pred[:smaller_batch_size], \
                   label[:smaller_batch_size]


@LossFunc.register('binary_cross_entropy')
class BinaryCrossEntropyLoss(LossFunc):
    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        return functional.binary_cross_entropy(pred, label, **kwargs)


@LossFunc.register('cross_entropy')
class CrossEntropyLoss(LossFunc):
    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        return functional.cross_entropy(pred, label, ignore_index=self.ignore_index, **kwargs)


@LossFunc.register('nll')
class NllLoss(LossFunc):
    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        pred_shape = pred.shape
        label_shape = label.shape

        # Adapt multi-dimensional pred/label tensor.
        if len(pred.shape) > 2:
            dummy_pred_batch_size = reduce(lambda x,y: x*y, pred_shape[:-1], 1)
            dummy_label_batch_size = reduce(lambda x,y: x*y, label_shape, 1)
            assert dummy_pred_batch_size==dummy_pred_batch_size, f'{dummy_pred_batch_size}!={dummy_pred_batch_size}'
            pred = pred.reshape((dummy_pred_batch_size, pred_shape[-1]))
            label = label.reshape(dummy_label_batch_size,)

        loss = functional.nll_loss(pred, label, ignore_index=self.ignore_index, **kwargs)

        # Adapt dimension keeping.
        if 'reduction' in kwargs and kwargs['reduction'] == 'none':
            return loss.reshape(label_shape)
        else:
            return loss



@LossFunc.register('mean_square')
class MeanSquareLoss(LossFunc):
    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        pred = pred.squeeze()
        assert pred.size() == label.size(), f'MSE assumes logit and label have the same size,' \
                                             f'but got {pred.size()} and {label.size()}'

        return (pred - label) ** 2


@LossFunc.register('bce_logits')
class BCEWithLogitsLoss(LossFunc):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        # align batch_size of predictions and labels in tail batch
        pred, label = self.align_pred_label_batch_size(pred, label)
        return self._loss(pred, label, **kwargs)


@LossFunc.register('bce')
class BCELoss(LossFunc):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
        # align batch_size of predictions and labels in tail batch
        # pred, label = self.align_pred_label_batch_size(pred, label)
        # pred = pred.view(pred.size(0),)
        # label = label.view(pred.size(0),)
        return torch.nn.functional.binary_cross_entropy(pred, label.float(), **kwargs)  # float type tensor is expected for 'label'