from typing import Dict, Tuple, Optional, List, Iterable, Callable

import torch
from allennlp.common import Registrable
from allennlp.data import Vocabulary


class CodeAnalysis(torch.nn.Module, Registrable):
    def __init__(self,
                 vocab: Vocabulary,
                 name: str,
                 **kwargs):
        self.name = name
        self.cur_epoch = 0
        super().__init__()

    def forward(self,
                code_features: Dict[str, torch.Tensor],
                code_labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def check_loss_in_range(self, range_to_check: List[int]):
        # Default behavior: Always in range.
        if range_to_check[0] == range_to_check[1] == -1:
            return True
        return range_to_check[0] <= self.cur_epoch <= range_to_check[1]

    def get_metrics(self, reset: bool = False) -> Dict:
        return {}