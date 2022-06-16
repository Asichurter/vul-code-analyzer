from typing import Tuple, Optional

import torch

class NodeFeatureAsymmetricTransformer(torch.nn.Module):
    def __init__(self,
                 feature_dim: int,
                 feature_out: Optional[int] = None,
                 twice_trans: bool = True):
        super().__init__()
        if feature_out is None:
            feature_out = feature_dim

        self.twice_trans = twice_trans
        self.pos_dir_transformer = torch.nn.Linear(feature_dim, feature_out, bias=False)
        self.neg_dir_transformer = torch.nn.Linear(feature_dim, feature_out, bias=False)
        if twice_trans:
            self.pos_dir_transformer_2 = torch.nn.Linear(feature_dim, feature_out, bias=False)
            self.neg_dir_transformer_2 = torch.nn.Linear(feature_dim, feature_out, bias=False)

    def forward(self, node_features: torch.Tensor) -> Tuple:
        if self.twice_trans:
            return self.pos_dir_transformer(node_features), \
                   self.neg_dir_transformer(node_features), \
                   self.pos_dir_transformer_2(node_features), \
                   self.neg_dir_transformer_2(node_features),
        else:
            return self.pos_dir_transformer(node_features), \
                   self.neg_dir_transformer(node_features),
