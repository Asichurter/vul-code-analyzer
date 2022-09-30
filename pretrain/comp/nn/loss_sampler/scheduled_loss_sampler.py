from typing import Dict

import torch

from pretrain.comp.nn.loss_sampler.loss_sampler import LossSampler


@LossSampler.register('epoch_scheduled')
class EpochScheduledLossSampler(LossSampler):
    def __init__(self,
                 sampling_ratio_schedule: Dict[int,float],
                 **kwargs):
        super().__init__(**kwargs)
        self.sampling_ratio_schedule = sampling_ratio_schedule
        self.epoch_count = 0
        self.current_sampling_ratio = 0.

    def get_loss(self, edge_matrix: torch.Tensor, predicted_matrix: torch.Tensor) -> torch.Tensor:
        """
        :param edge_matrix: [batch, max_vertice, max_vertice]
        :param predicted_matrix: [batch, max_vertice, max_vertice, edge_type_num]
        :return:
        """
        pass
        # edge_matrix_flat = edge_matrix.flatten(1)
        # edge_matrix_flat