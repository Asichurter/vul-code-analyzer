from typing import Dict

import torch

from common.nn.loss_func import LossFunc
from pretrain.comp.nn.loss_sampler.loss_sampler import LossSampler


@LossSampler.register('unified_full')
class UnifiedFullLossSampler(LossSampler):
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
        # Drop 0-th row and column, since line index start from 1.
        edge_matrix = edge_matrix[:, 1:, 1:]
        # Since edge matrix contains padded zeros, we minus 1 to make them to be -1,
        # and exclude them when calculating loss.
        edge_matrix -= 1

        return self.loss_func(predicted_matrix, edge_matrix)


@LossSampler.register('unified_balanced')
class UnifiedBalancedLossSampler(LossSampler):
    """
    This sampler balances edged pairs and non-edged pairs by sampling partial non-edged pairs
    from all the empty positions to make the have the same size.
    """
    def __init__(self, loss_func: LossFunc, **kwargs):
        super().__init__(loss_func, **kwargs)

    def get_loss(self,
                 edge_matrix: torch.Tensor,
                 predicted_matrix: torch.Tensor,
                 vertice_num: torch.Tensor) -> torch.Tensor:
        assert edge_matrix.shape == predicted_matrix.shape[:3], "Unmatched shape between label edges and predicted edges"
        assert predicted_matrix.size(-1) == 4, "Last dimension of predicted edge matrix must be 4"
        # Drop 0-th row and column, since line index start from 1.
        edge_matrix = edge_matrix[:, 1:, 1:]
        # Since edge matrix contains padded zeros, we minus 1 to make them to be -1,
        # and exclude them when calculating loss.
        edge_matrix -= 1

        bsz, v_num = edge_matrix.shape[:2]
        edge_matrix = edge_matrix.view(bsz, v_num*v_num)

        # Get indexes for edge/non-edge positions.
        # Note non-edged positions have index 3.
        no_edge_idx_list = (edge_matrix == 3).nonzero()
        edge_mask = ((edge_matrix >= 0) & (edge_matrix < 3))
        sampled_mask = edge_mask

        # Count number of edges per instance in the batch
        zeros_to_be_filled = torch.zeros((bsz), dtype=torch.int, device=edge_matrix.device)
        ones_to_fill = torch.ones_like(no_edge_idx_list[:,0], dtype=torch.int, device=edge_matrix.device)
        edge_count = torch.scatter_add(zeros_to_be_filled, 0, no_edge_idx_list[:,0], ones_to_fill)

        # Sample the same number of non-edged pair as the edged pair.
        # This operation can not be easily implemented in a batched manner, thus
        # sampling is done iteratively in the batch.
        sampled_non_edge_idxes = []
        for i in range(bsz):
            edge_count_i = int(edge_count[i].item())
            # Selected indexes are in 1D form, reshape to 2D.
            i_non_edge_idxes = torch.masked_select(no_edge_idx_list, no_edge_idx_list[:,0:1]==i).view(-1, 2)
            sampled_i_non_edge_idxes = i_non_edge_idxes[torch.randperm(i_non_edge_idxes.size(0))[:edge_count_i]]
            sampled_non_edge_idxes.append(sampled_i_non_edge_idxes)

        # Set sampled non-edged positions to be True,
        # and mask the labels of False positions to be -1.
        # [TODO]: use torch.stack?
        sampled_non_edge_idxes = torch.stack(sampled_non_edge_idxes, dim=0)
        sampled_mask[sampled_non_edge_idxes[:,0], sampled_non_edge_idxes[:,1]] = True
        edge_matrix.masked_fill_(~sampled_mask, -1)
        return self.loss_func(predicted_matrix, edge_matrix)