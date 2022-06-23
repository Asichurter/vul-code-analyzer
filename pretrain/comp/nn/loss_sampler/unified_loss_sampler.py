from typing import Dict, Tuple

import torch

from common.nn.loss_func import LossFunc
from pretrain.comp.nn.utils import stat_true_count_in_batch_dim, sample_2D_mask_by_count_in_batch_dim
from pretrain.comp.nn.loss_sampler.loss_sampler import LossSampler


@LossSampler.register('unified_full')
class UnifiedFullLossSampler(LossSampler):
    def __init__(self, loss_func: LossFunc, **kwargs):
        # Set ignore_index to -1 to exclude padded vertices when calculating loss
        # loss_func.ignore_index = -1
        super().__init__(loss_func, **kwargs)

    def get_loss(self,
                 edge_matrix: torch.Tensor,
                 predicted_matrix: torch.Tensor,
                 vertice_num: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        :param edge_matrix: [batch, max_vertice, max_vertice]
        :param predicted_matrix: [batch, max_vertice, max_vertice, edge_type_num]
        :param vertice_num: Shape: [batch, 1]
        :return:
        """
        # Drop 0-th row and column, since line index start from 1.
        # edge_matrix = edge_matrix[:, 1:, 1:]
        # Since edge matrix contains padded zeros, we minus 1 to make them to be -1,
        # and exclude them when calculating loss.
        edge_matrix -= 1
        assert edge_matrix.shape == predicted_matrix.shape[:3], "Unmatched shape between label edges and predicted edges"
        assert predicted_matrix.size(-1) == 4, "Last dimension of predicted edge matrix must be 4"

        # Manually operating "masked_mean"
        loss_mask = (edge_matrix != -1).int()
        return self.cal_matrix_masked_loss_mean(predicted_matrix, edge_matrix, loss_mask), \
               loss_mask


@LossSampler.register('unified_balanced')
class UnifiedBalancedLossSampler(LossSampler):
    """
    This sampler balances edged pairs and non-edged pairs by sampling partial non-edged pairs
    from all the empty positions to make them have the same size.
    """
    def __init__(self, loss_func: LossFunc, **kwargs):
        super().__init__(loss_func, **kwargs)

    def get_loss(self,
                 edge_matrix: torch.Tensor,
                 predicted_matrix: torch.Tensor,
                 vertice_num: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        # Drop 0-th row and column, since line index start from 1.
        # edge_matrix = edge_matrix[:, 1:, 1:]
        # Since edge matrix contains padded zeros, we minus 1 to make them to be -1,
        # and exclude them when calculating loss.
        edge_matrix -= 1
        assert edge_matrix.shape == predicted_matrix.shape[:3], "Unmatched shape between label edges and predicted edges"
        assert predicted_matrix.size(-1) == 4, "Last dimension of predicted edge matrix must be 4"

        bsz, v_num = edge_matrix.shape[:2]
        edge_matrix = edge_matrix.reshape(bsz, v_num*v_num)

        # Get indexes for edge/non-edge positions.
        # Note non-edged positions have index 3.
        non_edge_mask = (edge_matrix == 3)
        # no_edge_idx_list = non_edge_mask.nonzero()
        edge_mask = ((edge_matrix >= 0) & (edge_matrix < 3))

        # Count number of edges per instance in the batch.
        edge_count = stat_true_count_in_batch_dim(edge_mask)

        sampled_non_edge_mask = sample_2D_mask_by_count_in_batch_dim(non_edge_mask, edge_count)
        # We use all the edged pairs and sampled partial edged pairs to compute final loss.
        # W.r.t: len(edge) >= len(non_edge)
        sampled_mask = sampled_non_edge_mask.bool() | edge_mask
        return self.cal_matrix_masked_loss_mean(predicted_matrix, edge_matrix, sampled_mask), \
               sampled_mask.view(bsz, v_num, v_num)