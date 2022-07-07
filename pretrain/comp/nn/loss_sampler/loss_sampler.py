from typing import Tuple

import torch

from allennlp.common.registrable import Registrable

from common.nn.loss_func import LossFunc
from pretrain.comp.nn.utils import replace_int_value


class LossSampler(Registrable, torch.nn.Module):
    def __init__(self,
                 loss_func: LossFunc,
                 **kwargs):
        super().__init__()
        self.loss_func = loss_func

    def get_loss(self,
                 edge_matrix: torch.Tensor,
                 predicted_matrix: torch.Tensor,
                 vertice_num: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        raise NotImplementedError

    def cal_matrix_masked_loss_mean(self,
                                    predicted_matrix: torch.Tensor,
                                    edge_matrix: torch.Tensor,
                                    loss_mask: torch.Tensor) -> torch.Tensor:
        # Replace -1 with an arbitrary label to prevent error.
        edge_matrix = replace_int_value(edge_matrix, -1, 0)
        loss_matrix = self.loss_func(predicted_matrix, edge_matrix, reduction='none')
        loss_mask = loss_mask.int()
        return (loss_matrix * loss_mask).sum() / loss_mask.sum()
