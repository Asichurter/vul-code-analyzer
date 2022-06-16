from typing import Dict

import torch

from allennlp.common.registrable import Registrable

class NodeEncoder(torch.nn.Module, Registrable):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self,
                node_features: torch.Tensor,
                node_extra_features: Dict[str, torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_output_dim(self):
        return self.out_dim


