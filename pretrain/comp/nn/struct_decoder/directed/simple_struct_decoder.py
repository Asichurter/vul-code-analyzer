from typing import Dict, Tuple

import torch

from common.nn.classifier import Classifier
from pretrain.comp.nn.struct_decoder.struct_decoder import StructDecoder
from pretrain.comp.nn.struct_decoder.directed.directed_decoder_helper import NodeFeatureAsymmetricTransformer
from common.nn.simple_merge_methods import get_merge_method


@StructDecoder.register('node_merge_unified')
class NodeMergeUnifiedStructDecoder(StructDecoder):
    def __init__(self,
                 input_dim: int,
                 connect_node_merge_method: str,
                 classifier: Classifier,
                 **kwargs):
        # Set 'twice_trans'=False to uniformly predict two edge types
        self.node_feature_transformer = NodeFeatureAsymmetricTransformer(input_dim, twice_trans=False)
        self.node_merge_method = get_merge_method(connect_node_merge_method)
        self.classifier = classifier

    def forward(self,
                node_features: torch.Tensor,
                extra_features: Dict[str, torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: probability logits and predicted labels.
        """
        # Shape: [batch, vertice, dim] -> [batch, vertice, vertice, dim]
        v_num = node_features.size(1)
        node_features_exp = node_features.unsqueeze(2).repeat(1,1,v_num)
        node_features_exp_t = node_features_exp.transpose(1,2).contiguous()
        node_features_merged = self.node_merge_method(node_features_exp, node_features_exp_t)

        # For directed edges, use different projector to distinguish opposed directions
        pos_dir_features, neg_dir_features = self.node_feature_transformer(node_features_merged)
        # Take triangular part for directed edges.
        # Set offset=1/-1 to exclude diagonal.
        pos_dir_features, neg_dir_features = pos_dir_features.triu(1), neg_dir_features.tril(-1)
        unified_node_wise_features = pos_dir_features + neg_dir_features

        return self.classifier(unified_node_wise_features)




