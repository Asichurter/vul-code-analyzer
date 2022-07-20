from typing import Callable, Dict, List

import torch

from allennlp.common.registrable import Registrable
from allennlp.data import TextFieldTensors, Vocabulary

from pretrain.comp.metric.objective_pretrain_metric import ObjectiveLoss


class CodeObjective(torch.nn.Module, Registrable):
    def __init__(self,
                 vocab: Vocabulary,
                 name: str,
                 as_code_embedder: bool = False,
                 forward_from_where: str = 'embedding',
                 loss_coeff: float = 1.,
                 loss_epoch_range: List[int] = [-1,-1],
                 **kwargs):
        self.name = name
        self.as_code_embedder = as_code_embedder
        self.loss_coeff = loss_coeff
        self.loss_metric = ObjectiveLoss(name)
        self.loss_epoch_range = loss_epoch_range

        assert forward_from_where in ['token', 'embedding']
        self.forward_from_where = forward_from_where
        super().__init__()

    def forward_from_token(self,
                           code: TextFieldTensors,
                           code_embed_func: Callable,
                           epoch: int,
                           **kwargs) -> Dict:
        raise NotImplementedError

    def forward_from_embedding(self,
                               token_embedding: torch.Tensor,
                               token_mask: torch.Tensor,
                               tensor_dict: Dict[str, torch.Tensor],
                               epoch: int,
                               **kwargs) -> Dict:
        raise NotImplementedError

    def forward(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def reset_metric(self):
        self.loss_metric.reset()

    def get_metric(self, reset: bool):
        return self.loss_metric.get_metric(reset)

    def update_metric(self, *args, **kwargs):
        self.loss_metric(*args, **kwargs)

    def rectify_loss_based_on_range(self, loss, epoch):
        # Default behavior: Always in range.
        if self.loss_epoch_range[0] == self.loss_epoch_range[1] == -1:
            return loss

        if self.loss_epoch_range[0] <= epoch <= self.loss_epoch_range[1]:
            return loss
        else:
            return 0    # todo: Maybe 'torch.zeros_as(loss)' ?
