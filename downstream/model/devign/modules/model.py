import torch
from dgl.nn.pytorch import GatedGraphConv
from torch import nn
import torch.nn.functional as f

from downstream.model.devign.modules.classfier import LinearSigmoidClassifier
from downstream.model.devign.devign_utils import make_mask_from_lens, mask_mean
from downstream.model.mlm_line_feature_extractor import MLMLineExtractor


class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        outputs = self.ggnn(graph, features, edge_types)
        x_i, _ = batch.de_batchify_graphs(features)
        h_i, _ = batch.de_batchify_graphs(outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(
            f.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            f.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg).squeeze(dim=-1)
        return result


class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        outputs = self.ggnn(graph, features, edge_types)
        h_i, _ = batch.de_batchify_graphs(outputs)
        ggnn_sum = self.classifier(h_i.mean(dim=1))         # Note: Here we modify 'sum' to 'mean'
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result

class GGNNSumNew(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSumNew, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = LinearSigmoidClassifier(output_dim,
                                                  hidden_dims=[256],
                                                  activations=['relu'],
                                                  dropouts=[0.3],
                                                  ahead_feature_dropout=0.3,
                                                  out_dim=1) # nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        outputs = self.ggnn(graph, features, edge_types)
        h_i, lens = batch.de_batchify_graphs(outputs)
        mask = make_mask_from_lens(lens)
        classification_features = mask_mean(h_i, mask, dim=1)
        result, result_labels = self.classifier(classification_features)
        # result, result_labels = self.classifier(h_i.mean(dim=1))
        # ggnn_sum = self.classifier(h_i.mean(dim=1))         # Note: Here we modify 'sum' to 'mean'
        # result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result

class GGNNMeanResidual(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNMeanResidual, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = LinearSigmoidClassifier(output_dim+input_dim,
                                                  hidden_dims=[256],
                                                  activations=['relu'],
                                                  dropouts=[0.3],
                                                  ahead_feature_dropout=0.3,
                                                  out_dim=1) # nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        outputs = self.ggnn.forward(graph, features, edge_types)
        output_res_features = torch.cat((outputs, features), dim=1)
        h_i, lens = batch.de_batchify_graphs(output_res_features)
        mask = make_mask_from_lens(lens)
        classification_features = mask_mean(h_i, mask, dim=1)
        result, result_labels = self.classifier(classification_features)
        # result, result_labels = self.classifier(h_i.mean(dim=1))
        # ggnn_sum = self.classifier(h_i.mean(dim=1))         # Note: Here we modify 'sum' to 'mean'
        # result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result

class GGNNMeanMixedResidual(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8, res_feature_dim=768):
        super(GGNNMeanMixedResidual, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.res_feature_dim = res_feature_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = LinearSigmoidClassifier(output_dim+res_feature_dim,
                                                  hidden_dims=[256],
                                                  activations=['relu'],
                                                  dropouts=[0.3],
                                                  ahead_feature_dropout=0.3,
                                                  out_dim=1) # nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph_features, mixed_res_features = features[:, :self.inp_dim], features[:, self.inp_dim:]
        assert mixed_res_features.size(1) == self.res_feature_dim

        outputs = self.ggnn.forward(graph, graph_features, edge_types)
        output_res_features = torch.cat((outputs, mixed_res_features), dim=1)

        h_i, lens = batch.de_batchify_graphs(output_res_features)
        mask = make_mask_from_lens(lens)
        classification_features = mask_mean(h_i, mask, dim=1)
        result, result_labels = self.classifier(classification_features)
        # result, result_labels = self.classifier(h_i.mean(dim=1))
        # ggnn_sum = self.classifier(h_i.mean(dim=1))         # Note: Here we modify 'sum' to 'mean'
        # result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result

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