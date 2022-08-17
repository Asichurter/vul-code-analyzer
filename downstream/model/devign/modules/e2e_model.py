from typing import Iterator, Tuple

import torch
from dgl.nn.pytorch import GatedGraphConv
from torch import nn
import torch.nn.functional as f
from torch.nn import Parameter
from torch.nn.parameter import Parameter

from downstream.model.devign.modules.classfier import LinearSigmoidClassifier
from downstream.model.devign.devign_utils import make_mask_from_lens, mask_mean
from downstream.model.feature_extraction.mlm_line_feature_extractor import MLMLineExtractor
from downstream.model.feature_extraction.mlm_line_feature_extractor_v2 import MLMLineExtractorV2
from downstream.model.feature_extraction.cls_feature_extractor import ClsExtractorV2

class GGNNMeanEnd2End(nn.Module):
    def __init__(self,
                 line_extractor: MLMLineExtractor,
                 input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNMeanEnd2End, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.feature_extractor = line_extractor
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = LinearSigmoidClassifier(output_dim,
                                                  hidden_dims=[256],
                                                  activations=['relu'],
                                                  dropouts=[0.3],
                                                  ahead_feature_dropout=0.3,
                                                  out_dim=1) # nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def select_graph_node_features(self, features, node_counts):
        graph_features = []
        for i, node_count in enumerate(node_counts):
            nc = node_count.item()
            graph_features.append(features[i, :nc])
        return torch.cat(graph_features, dim=0)

    def forward(self, batch, cuda=False):
        graph, features, edge_types, codes = batch.get_network_inputs(cuda=cuda, ret_code=True)
        graph_features = self.feature_extractor.predict_batch_with_grad(codes, separate_instances=False)
        graph_node_features = self.select_graph_node_features(graph_features['line_features'],
                                                              graph_features['vertice_num'])
        assert graph_node_features.size(0) == features.size(0), \
            f'graph feature node size ({graph_node_features.size(0)}) ' \
            f'not equal to raw node feature size ({features.size(0)}) \n' \
            f'Codes: {codes}'
        outputs = self.ggnn(graph, graph_node_features, edge_types)
        h_i, lens = batch.de_batchify_graphs(outputs)
        mask = make_mask_from_lens(lens)
        classification_features = mask_mean(h_i, mask, dim=1)
        result, result_labels = self.classifier(classification_features)
        # result, result_labels = self.classifier(h_i.mean(dim=1))
        # ggnn_sum = self.classifier(h_i.mean(dim=1))         # Note: Here we modify 'sum' to 'mean'
        # result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for p in super().parameters(recurse):
            yield p
        for p in self.feature_extractor._model.parameters(recurse):
            yield p

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        for n,p in super().named_parameters(prefix, recurse):
            yield n,p
        for n,p in self.feature_extractor._model.named_parameters(prefix, recurse):
            yield n,p


class GGNNMeanEnd2EndV2(nn.Module):
    def __init__(self,
                 line_extractor: MLMLineExtractorV2,
                 input_dim, output_dim, max_edge_types, num_steps=8,
                 residual_forward: bool = False,
                 dynamic_node_features: bool = True):
        super(GGNNMeanEnd2EndV2, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.feature_extractor = line_extractor
        self.residual_forward = residual_forward
        self.dynamic_node_features = dynamic_node_features
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        if residual_forward:
            feature_dim = input_dim + output_dim
        else:
            feature_dim = output_dim
        self.classifier = LinearSigmoidClassifier(feature_dim,
                                                  hidden_dims=[256],
                                                  activations=['relu'],
                                                  dropouts=[0.3],
                                                  ahead_feature_dropout=0.3,
                                                  out_dim=1) # nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def select_graph_node_features(self, features, node_counts):
        graph_features = []
        for i, node_count in enumerate(node_counts):
            nc = node_count.item()
            graph_features.append(features[i, :nc])
        return torch.cat(graph_features, dim=0)

    def forward(self, batch, cuda=False):
        graph, raw_node_features, edge_types, codes = batch.get_network_inputs(cuda=cuda, ret_code=True)
        node_feature_outputs = self.feature_extractor.predict_batch_with_grad(codes, separate_instances=False)
        dynamic_node_features = self.select_graph_node_features(node_feature_outputs['line_features'],
                                                                node_feature_outputs['vertice_num'])
        assert dynamic_node_features.size(0) == raw_node_features.size(0), \
            f'graph feature node size ({dynamic_node_features.size(0)}) ' \
            f'not equal to raw node feature size ({raw_node_features.size(0)}) \n' \
            f'Codes: {codes}'

        if self.dynamic_node_features:
            outputs = self.ggnn(graph, dynamic_node_features, edge_types)
        else:
            outputs = self.ggnn(graph, raw_node_features, edge_types)

        if self.residual_forward:
            # Residual feature is always dynamic node featurs.
            outputs = torch.cat((outputs, dynamic_node_features), dim=1)

        h_i, lens = batch.de_batchify_graphs(outputs)
        mask = make_mask_from_lens(lens)
        classification_features = mask_mean(h_i, mask, dim=1)
        result, result_labels = self.classifier(classification_features)
        return result


class NodeMean(nn.Module):
    def __init__(self,
                 line_extractor: MLMLineExtractorV2,
                 input_dim, output_dim, max_edge_types, num_steps=8):
        super(NodeMean, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.feature_extractor = line_extractor
        feature_dim = input_dim
        self.classifier = LinearSigmoidClassifier(feature_dim,
                                                  hidden_dims=[256],
                                                  activations=['relu'],
                                                  dropouts=[0.3],
                                                  ahead_feature_dropout=0.3,
                                                  out_dim=1) # nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def select_graph_node_features(self, features, node_counts):
        graph_features = []
        for i, node_count in enumerate(node_counts):
            nc = node_count.item()
            graph_features.append(features[i, :nc])
        return torch.cat(graph_features, dim=0)

    def forward(self, batch, cuda=False):
        graph, raw_node_features, edge_types, codes = batch.get_network_inputs(cuda=cuda, ret_code=True)
        node_feature_outputs = self.feature_extractor.predict_batch_with_grad(codes, separate_instances=False)
        dynamic_node_features = self.select_graph_node_features(node_feature_outputs['line_features'],
                                                                node_feature_outputs['vertice_num'])
        h_i, lens = batch.de_batchify_graphs(dynamic_node_features)
        mask = make_mask_from_lens(lens)
        classification_features = mask_mean(h_i, mask, dim=1)
        result, result_labels = self.classifier(classification_features)
        return result


class ClsPooler(nn.Module):
    def __init__(self,
                 cls_extractor: ClsExtractorV2,
                 input_dim, output_dim, max_edge_types, num_steps=8):
        super(ClsPooler, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.feature_extractor = cls_extractor
        feature_dim = input_dim
        self.classifier = LinearSigmoidClassifier(feature_dim,
                                                  hidden_dims=[256],
                                                  activations=['relu'],
                                                  dropouts=[0.3],
                                                  ahead_feature_dropout=0.3,
                                                  out_dim=1) # nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        graph, raw_node_features, edge_types, codes = batch.get_network_inputs(cuda=cuda, ret_code=True)
        cls_feature_outputs = self.feature_extractor.predict_batch_with_grad(codes, separate_instances=False)
        cls_features = cls_feature_outputs['cls_features']
        result, result_labels = self.classifier(cls_features)
        return result


