from typing import Dict, Tuple

import torch

from pretrain.comp.nn.struct_decoder.directed.directed_decoder_helper import DecoderOutputActivationAdapter
from pretrain.comp.nn.struct_decoder.struct_decoder import StructDecoder


@StructDecoder.register('bilinear_separated')
class BilinearSeparatedStructDecoder(StructDecoder):
    def __init__(self,
                 input_dim: int,
                 **kwargs):
        super().__init__()
        # self.node_feature_transformer = NodeFeatureAsymmetricTransformer(input_dim, twice_trans=False)
        self.data_bilinear = torch.nn.Bilinear(input_dim, input_dim, 1)
        self.ctrl_bilinear = torch.nn.Bilinear(input_dim, input_dim, 1)

    def forward(self,
                node_features: torch.Tensor,
                extra_features: Dict[str, torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: Probability logits and predicted labels,
                 shape: [batch, 2, vn, vn]
        """
        # Shape: [batch, vertice, dim] -> [batch, vertice, vertice, dim]
        v_num = node_features.size(1)
        node_features_exp = node_features.unsqueeze(2).repeat(1,1,v_num,1)
        node_features_exp_t = node_features_exp.transpose(1,2).contiguous()

        data_pred_scores = torch.sigmoid(self.data_bilinear(node_features_exp, node_features_exp_t)).squeeze(-1)
        ctrl_pred_scores = torch.sigmoid(self.ctrl_bilinear(node_features_exp, node_features_exp_t)).squeeze(-1)

        pred_scores = torch.stack((data_pred_scores, ctrl_pred_scores), dim=1)
        return pred_scores, (pred_scores > 0.5).int()


@StructDecoder.register('bilinear_single')
class BilinearSingleStructDecoder(StructDecoder):
    """
    This decoder do not unify the decoding of ctrl and data edges,
    but only decode one kind of edges.
    """
    def __init__(self,
                 input_dim: int,
                 output_activation: str = 'sigmoid',
                 **kwargs):
        super().__init__()
        output_dim = 2 if output_activation == 'softmax' else 1
        self.bilinear = torch.nn.Bilinear(input_dim, input_dim, output_dim)
        self.output_activation_adapter = DecoderOutputActivationAdapter(output_activation)

    def forward(self,
                node_features: torch.Tensor,
                extra_features: Dict[str, torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape: [batch, vertice, dim] -> [batch, vertice, vertice, dim]
        v_num = node_features.size(1)
        node_features_exp = node_features.unsqueeze(2).repeat(1,1,v_num,1)
        node_features_exp_t = node_features_exp.transpose(1,2).contiguous()

        bilinear_output = self.bilinear(node_features_exp, node_features_exp_t)
        pred_scores = self.output_activation_adapter(bilinear_output)

        # pred shape: [batch, vertice, vertice]
        return pred_scores, (pred_scores > 0.5).int()


@StructDecoder.register('concat_linear_single')
class ConcatLinearSingleStructDecoder(StructDecoder):
    """
    This decoder do not unify the decoding of ctrl and data edges,
    but only decode one kind of edges.
    """
    def __init__(self,
                 input_dim: int,
                 output_activation: str = 'sigmoid',
                 **kwargs):
        super().__init__()
        output_dim = 2 if output_activation == 'softmax' else 1
        self.output_linear = torch.nn.Linear(input_dim*2, output_dim)
        self.output_activation_adapter = DecoderOutputActivationAdapter(output_activation)

    def forward(self,
                node_features: torch.Tensor,
                extra_features: Dict[str, torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape: [batch, vertice, dim] -> [batch, vertice, vertice, dim]
        v_num = node_features.size(1)
        node_features_exp = node_features.unsqueeze(2).repeat(1,1,v_num,1)
        node_features_exp_t = node_features_exp.transpose(1,2).contiguous()

        node_pair_features = torch.cat((node_features_exp, node_features_exp_t), dim=-1)
        linear_output = self.output_linear(node_pair_features)
        pred_scores = self.output_activation_adapter(linear_output)

        # pred shape: [batch, vertice, vertice]
        return pred_scores, (pred_scores > 0.5).int()


