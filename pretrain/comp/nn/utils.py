from typing import Optional

import torch


def stat_true_count_in_batch_dim(mask: torch.Tensor):
    """
    Count the number of true elements among each row of the mask.
    Input can be arbitrary shape.
    It wil return a 1D int tensor to indicate the count of 'True' in the batch.
    """
    bsz = mask.size(0)
    idx_list = mask.nonzero()

    # Count number of edges per instance in the batch.
    zeros_to_be_filled = torch.zeros((bsz), dtype=torch.int, device=mask.device)
    ones_to_fill = torch.ones_like(idx_list[:, 0], dtype=torch.int, device=mask.device)
    true_count = torch.scatter_add(zeros_to_be_filled, 0, idx_list[:, 0], ones_to_fill)

    return true_count


def sample_2D_mask_by_count_along_batch_dim(
        source_mask: torch.Tensor,
        sample_num_list: torch.Tensor
) -> torch.Tensor:
    """
    Sample 'True' value from from the 2D input mask based on the given
    count of each row.
    Source mask must be in 2D shape and count tensor should be derived from
    torch.nonzero() method.
    Return a new mask that only sampled 'True' positions are of True value.
    """

    sampled_mask = torch.zeros_like(source_mask)
    bsz = source_mask.size(0)
    batch_candidate_idx_list = source_mask.nonzero()

    # Sample from each item in the batch by given count.
    # This operation can not be easily implemented in a batched manner, thus
    # sampling is done iteratively in the batch.
    sampled_idxes = []
    for i in range(bsz):
        sampled_num_i = int(sample_num_list[i].item())
        # Skip null sampling.
        if sampled_num_i == 0:
            continue
        # Selected indexes are in 1D form, reshape to 2D.
        i_idxes_to_be_sampled = torch.masked_select(batch_candidate_idx_list, batch_candidate_idx_list[:, 0:1] == i).view(-1, 2)
        i_sampled_idxes = i_idxes_to_be_sampled[torch.randperm(i_idxes_to_be_sampled.size(0))[:sampled_num_i]]
        sampled_idxes.append(i_sampled_idxes)

    # Set sampled non-edged positions to be True.
    # [Note]
    # Sampled items may be less than given count, since the number of total candidates implied
    # by the source mask may be less than required count.
    sampled_idxes = torch.cat(sampled_idxes, dim=0)
    # Set sampled positions to be 1.
    sampled_mask[sampled_idxes[:, 0], sampled_idxes[:, 1]] = 1
    return sampled_mask


def multinomial_sample_2D_mask_by_count_along_batch_dim(
        source_mask: torch.Tensor,
        sample_num_list: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample 'True' value from from the 2D input mask based on the given
    count of each row, supporting sampling weights of multinomial distribution.
    Source mask must be in 2D shape and count tensor should be derived from
    torch.nonzero() method.
    Return a new mask that only sampled 'True' positions are of True value.
    """
    if weight is None:
        weight = torch.ones_like(source_mask)

    sampled_mask = torch.zeros_like(source_mask)
    bsz = source_mask.size(0)
    batch_candidate_idx_list = source_mask.nonzero()
    batch_candidate_weight = weight[batch_candidate_idx_list[:,0], batch_candidate_idx_list[:,1]]

    # Sample from each item in the batch by given count.
    # This operation can not be easily implemented in a batched manner, thus
    # sampling is done iteratively in the batch.
    sampled_idxes = []
    for i in range(bsz):
        sampled_num_i = int(sample_num_list[i].item())
        # Skip null sampling.
        if sampled_num_i == 0:
            continue
        # Selected indexes are in 1D form, reshape to 2D.
        i_idxes_to_be_sampled = torch.masked_select(batch_candidate_idx_list, batch_candidate_idx_list[:, 0:1] == i).view(-1, 2)
        # Select weights
        i_idxes_sample_weight = torch.masked_select(batch_candidate_weight, batch_candidate_idx_list[:, 0] == i)
        # Sample from indexes according to weights
        i_sampled_inner_idxes = torch.multinomial(i_idxes_sample_weight, sampled_num_i)
        i_real_sampled_idxes = i_idxes_to_be_sampled[i_sampled_inner_idxes]
        # i_sampled_idxes = i_idxes_to_be_sampled[torch.randperm(i_idxes_to_be_sampled.size(0))[:sampled_num_i]]
        sampled_idxes.append(i_real_sampled_idxes)

    # Set sampled non-edged positions to be True.
    # [Note]
    # Sampled items may be less than given count, since the number of total candidates implied
    # by the source mask may be less than required count.
    sampled_idxes = torch.cat(sampled_idxes, dim=0)
    # Set sampled positions to be 1.
    sampled_mask[sampled_idxes[:, 0], sampled_idxes[:, 1]] = 1
    return sampled_mask


def replace_int_value(tensor: torch.Tensor,
                      replaced_value: int,
                      new_value: int) -> torch.Tensor:
    replace_mask = tensor == replaced_value
    return torch.masked_fill(tensor, replace_mask, new_value)


if __name__ == "__main__":
    source_mask = torch.Tensor([[1,1,1,1,1,0], [1,1,1,1,0,0]])
    sampled_num_list = torch.IntTensor([3,2])
    weight = torch.Tensor([[1,2,1,1,1,1], [2,1,1,1,1,1]])
    sampled_mask = multinomial_sample_2D_mask_by_count_along_batch_dim(source_mask, sampled_num_list, weight=weight)