from typing import Dict, Tuple, Optional, List, Iterable, Callable

import torch
from allennlp.data import Vocabulary
from allennlp.training.metrics import Metric

from pretrain.comp.metric.objective_pretrain_metric import ObjectiveLoss
from pretrain.comp.nn.loss_sampler.loss_sampler import LossSampler
from pretrain.comp.nn.struct_decoder.struct_decoder import StructDecoder
from pretrain.comp.nn.utils import construct_matrix_from_opt_edge_idxes
from pretrain.model.modules.code_analysis import CodeAnalysis

@CodeAnalysis.register('hybrid_pdg_analyzer')
class HybridPDGAnalyzer(CodeAnalysis):
    def __init__(
        self,
        vocab: Vocabulary,
        line_ctrl_decoder: StructDecoder,
        token_data_decoder: StructDecoder,
        ctrl_loss_sampler: LossSampler,
        data_loss_sampler: LossSampler,
        ctrl_metric: Optional[Metric] = None,
        data_metric: Optional[Metric] = None,
        pdg_ctrl_loss_coeff: float = 1.,
        pdg_data_loss_coeff: float = 1.,
        pdg_ctrl_loss_range: List[int] = [-1,-1],
        pdg_data_loss_range: List[int] = [-1,-1],
        token_edge_input_being_optimized: bool = False,     # For independent-forward model, this should be True to avoid size mismatch.
        line_edge_input_being_optimized: bool = False,      # For independent-forward model, this should be True to avoid size mismatch.
        add_pdg_loss_metric: bool = True,
        name: str = "m_pdg",
        **kwargs
    ):
        super().__init__(vocab, name=name)     # here just give a placeholder value
        self.line_ctrl_decoder = line_ctrl_decoder
        self.token_data_decoder = token_data_decoder
        self.ctrl_loss_sampler = ctrl_loss_sampler
        self.data_loss_sampler = data_loss_sampler
        self.ctrl_metric = ctrl_metric
        self.data_metric = data_metric

        self.pdg_ctrl_loss_coeff = pdg_ctrl_loss_coeff
        self.pdg_data_loss_coeff = pdg_data_loss_coeff
        self.pdg_ctrl_loss_range = pdg_ctrl_loss_range
        self.pdg_data_loss_range = pdg_data_loss_range
        self.token_edge_input_being_optimized = token_edge_input_being_optimized
        self.line_edge_input_being_optimized = line_edge_input_being_optimized

        self.add_pdg_loss_metric = add_pdg_loss_metric
        if add_pdg_loss_metric:
            self.pdg_ctrl_metric = ObjectiveLoss(name+'_ctrl')
            self.pdg_data_metric = ObjectiveLoss(name+'_data')

        self.test = 0

    def check_pdg_loss_in_range(self, range_to_check: List[int]):
        # Default behavior: Always in range.
        if range_to_check[0] == range_to_check[1] == -1:
            return True
        return range_to_check[0] <= self.cur_epoch <= range_to_check[1]


    def forward(self,
                code_features: Dict[str, torch.Tensor],
                code_labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
            Using the embedded token & line features to complete code analysis tasks.
        """
        encoded_line_node_features = code_features["code_line_features"]
        encoded_token_node_features = code_features["code_token_features"]
        code_token_mask = code_features["code_token_mask"]
        code_line_mask = code_features.get("code_line_mask")
        token_elem_mask = code_features.get("token_data_token_mask", None)
        line_ctrl_edges = code_labels.get("pdg_line_ctrl_edges", None) if code_labels is not None else None
        token_data_edges = code_labels.get("pdg_token_data_edges", None) if code_labels is not None else None

        # Shape: [batch, vn, vn, 4]
        pred_ctrl_edge_probs, pred_ctrl_edge_labels = self.line_ctrl_decoder(encoded_line_node_features)
        pred_data_edge_probs, pred_data_edge_labels = self.token_data_decoder(encoded_token_node_features)

        final_loss = 0.
        returned_dict = {}
        if line_ctrl_edges is None:
            returned_dict.update({
                'ctrl_edge_logits': pred_ctrl_edge_probs,
                'ctrl_edge_labels': pred_ctrl_edge_labels,
            })
        # Check pdg loss is in range.
        elif self.check_loss_in_range(self.pdg_ctrl_loss_range):
            if self.line_edge_input_being_optimized:
                line_ctrl_edges = construct_matrix_from_opt_edge_idxes(line_ctrl_edges, code_line_mask)
            pdg_ctrl_loss, pdg_ctrl_loss_mask = self.ctrl_loss_sampler.get_loss(line_ctrl_edges, pred_ctrl_edge_probs)
            pdg_ctrl_loss *= self.pdg_ctrl_loss_coeff

            final_loss += pdg_ctrl_loss
            if self.ctrl_metric is not None:
                self.ctrl_metric(pred_ctrl_edge_labels, line_ctrl_edges, pdg_ctrl_loss_mask)
            if self.add_pdg_loss_metric:
                self.pdg_ctrl_metric(pdg_ctrl_loss)
            returned_dict.update({
                'ctrl_edge_logits': pred_ctrl_edge_probs,
                'ctrl_edge_labels': pred_ctrl_edge_labels,
            })

        if token_data_edges is None:
            returned_dict.update({
                'data_edge_logits': pred_data_edge_probs,
                'data_edge_labels': pred_data_edge_labels,
            })
        # Check pdg loss is in range.
        elif self.check_loss_in_range(self.pdg_data_loss_range):
            if self.token_edge_input_being_optimized:
                token_data_edges = construct_matrix_from_opt_edge_idxes(token_data_edges, code_token_mask)
            pdg_data_loss, pdg_data_loss_mask = self.data_loss_sampler.get_loss(token_data_edges, pred_data_edge_probs,
                                                                                elem_mask=token_elem_mask)
            pdg_data_loss *= self.pdg_data_loss_coeff

            final_loss += pdg_data_loss
            if self.data_metric is not None:
                self.data_metric(pred_data_edge_labels, token_data_edges, pdg_data_loss_mask)
            if self.add_pdg_loss_metric:
                self.pdg_data_metric(pdg_data_loss)
            returned_dict.update({
                'data_edge_logits': pred_data_edge_probs,
                'data_edge_labels': pred_data_edge_labels,
            })

        returned_dict['loss'] = final_loss
        return returned_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.ctrl_metric is not None:
            ctrl_metric = self.ctrl_metric.get_metric(reset)
            # no metric name returned, use its class name instead
            if type(ctrl_metric) != dict:
                metric_name = self.ctrl_metric.__class__.__name__
                ctrl_metric = {metric_name: ctrl_metric}
            metrics.update(ctrl_metric)
        if self.data_metric is not None:
            data_metric = self.data_metric.get_metric(reset)
            # no metric name returned, use its class name instead
            if type(data_metric) != dict:
                metric_name = self.data_metric.__class__.__name__
                data_metric = {metric_name: data_metric}
            metrics.update(data_metric)

        if self.add_pdg_loss_metric:
            pdg_ctrl_loss_metric = self.pdg_ctrl_metric.get_metric(reset)
            pdg_data_loss_metric = self.pdg_data_metric.get_metric(reset)
            metrics.update(pdg_ctrl_loss_metric)
            metrics.update(pdg_data_loss_metric)

        return metrics
