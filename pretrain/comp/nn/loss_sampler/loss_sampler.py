import torch

from allennlp.common.registrable import Registrable

from common.nn.loss_func import LossFunc


class LossSampler(Registrable):
    def __init__(self,
                 loss_func: LossFunc,
                 **kwargs):
        self.loss_func = loss_func

    def get_loss(self,
                 edge_matrix: torch.Tensor,
                 predicted_matrix: torch.Tensor,
                 vertice_num: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
