from typing import Dict, Optional
import torch

from pretrain.comp.nn.node_encoder.node_encoder import NodeEncoder


@NodeEncoder.register('pass_through')
class PassThroughNodeEncoder(NodeEncoder):
    """
    Just pass through node features, without any operations.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: Optional[int] = None,
                 **kwargs):
        output_dim = output_dim or input_dim
        super().__init__(input_dim, output_dim, **kwargs)

    def forward(self,
                node_features: torch.Tensor,
                node_mask: Optional[torch.Tensor] = None,
                node_extra_features: Dict[str, torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        return node_features
