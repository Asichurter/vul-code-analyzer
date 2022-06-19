from typing import Dict

import torch

from common.nn.loss_func import LossFunc
from pretrain.comp.nn.loss_sampler.loss_sampler import LossSampler


@LossSampler.register('full')
class FullLossSampler(LossSampler):
    def __init__(self, loss_func: LossFunc, **kwargs):
        # Set ignore_index to -1 to exclude padded vertices when calculating loss
        loss_func.ignore_index = -1
        super().__init__(loss_func, **kwargs)

    def get_loss(self,
                 edge_matrix: torch.Tensor,
                 predicted_matrix: torch.Tensor,
                 vertice_num: torch.Tensor) -> torch.Tensor:
        """
        :param edge_matrix: [batch, max_vertice, max_vertice]
        :param predicted_matrix: [batch, max_vertice, max_vertice, edge_type_num]
        :param vertice_num: Shape: [batch, 1]
        :return:
        """
        assert edge_matrix.shape == predicted_matrix.shape[:3], "Unmatched shape between label edges and predicted edges"
        assert predicted_matrix.size(-1) == 4, "Last dimension of predicted edge matrix must be 4"
        # Since edge matrix contains padded zeros, we minus 1 to make them to be -1,
        # and exclude them when calculating loss.
        edge_matrix -= 1

        return self.loss_func(predicted_matrix, edge_matrix)