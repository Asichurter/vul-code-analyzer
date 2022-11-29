from typing import Optional

import torch

from allennlp.common.registrable import Registrable

class Pooler(Registrable, torch.nn.Module):
    def __init__(self, dim: int):
        self.dim = dim
        super().__init__()

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


@Pooler.register('cls')
class ClsPooler(Pooler):
    """
    Only support head cls pooling.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.select(input, self.dim, 0)


@Pooler.register('mean')
class MeanPooler(Pooler):
    def __init__(self, dim: int = 1):
        super().__init__(dim)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param input: N dims: [n1, n2, ..., nk], assert last dim is feature dim
        :param mask:  N-1 dims: [n1, n2, ..., nk-1], last dim is omit
        """
        avg_count = mask.sum(self.dim)
        avg_count = torch.max(avg_count, avg_count.new_ones(1))  # Avoid zero-division
        avg_out = (input * mask.unsqueeze(-1)).sum(self.dim) / avg_count.unsqueeze(self.dim)
        return avg_out
