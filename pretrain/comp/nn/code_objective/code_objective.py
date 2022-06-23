from typing import Callable, Dict

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
                 **kwargs):
        self.name = name
        self.as_code_embedder = as_code_embedder
        self.loss_coeff = loss_coeff
        self.loss_metric = ObjectiveLoss(name)

        assert forward_from_where in ['token', 'embedding']
        self.forward_from_where = forward_from_where
        super().__init__()

    def forward_from_token(self,
                           code: TextFieldTensors,
                           code_embed_func: Callable,
                           **kwargs) -> Dict:
        raise NotImplementedError

    def forward_from_embedding(self,
                               token_embedding: torch.Tensor,
                               token_mask: torch.Tensor,
                               tensor_dict: Dict[str, torch.Tensor],
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