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


class DecoderOutputActivationAdapter(torch.nn.Module):
    def __init__(self, activation: str):
        super().__init__()
        if activation == 'sigmoid':
            self.forward_func = self._sigmoid_forward
        elif activation == 'softmax':
            self.forward_func = self._softmax_forward
        else:
            raise NotImplementedError(f'Activation: {activation}')

    def _sigmoid_forward(self, output_features: torch.Tensor) -> torch.Tensor:
        pred_scores = torch.sigmoid(output_features).squeeze(-1)
        return pred_scores

    def _softmax_forward(self, output_features: torch.Tensor) -> torch.Tensor:
        pred_scores = torch.softmax(output_features, dim=1)[:, :, :, 1]
        return pred_scores

    def forward(self, output_features: torch.Tensor) -> torch.Tensor:
        return self.forward_func(output_features)