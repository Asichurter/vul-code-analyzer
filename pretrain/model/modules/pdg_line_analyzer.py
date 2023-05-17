from typing import Dict, Tuple, Optional, List, Iterable, Callable

import torch
from allennlp.data import Vocabulary
from allennlp.training.metrics import Metric

from pretrain.comp.metric.objective_pretrain_metric import ObjectiveLoss
from pretrain.comp.nn.loss_sampler.loss_sampler import LossSampler
from pretrain.comp.nn.struct_decoder.struct_decoder import StructDecoder
from pretrain.comp.nn.utils import construct_matrix_from_opt_edge_idxes
from pretrain.model.modules.code_analysis import CodeAnalysis

@CodeAnalysis.register('line_pdg_analyzer')
class LinePDGAnalyzer(CodeAnalysis):
    def __init__(
        self,
        vocab: Vocabulary,
        pdg_ctrl_decoder: StructDecoder,
        pdg_data_decoder: StructDecoder,
        ctrl_loss_sampler: LossSampler,
        data_loss_sampler: LossSampler,
        ctrl_metric: Optional[Metric] = None,
        data_metric: Optional[Metric] = None,
        pdg_ctrl_loss_coeff: float = 1.,
        pdg_data_loss_coeff: float = 1.,
        pdg_ctrl_loss_range: List[int] = [-1,-1],
        pdg_data_loss_range: List[int] = [-1,-1],
        ctrl_edges_being_optimized: bool = False,     # For independent-forward model, this should be True to avoid size mismatch.
        data_edges_being_optimized: bool = False,      # For independent-forward model, this should be True to avoid size mismatch.
        add_pdg_loss_metric: bool = True,
        name: str = "m_pdg",
        **kwargs
    ):
        super().__init__(vocab, name=name)     # here just give a placeholder value
        self.pdg_ctrl_decoder = pdg_ctrl_decoder
        self.pdg_data_decoder = pdg_data_decoder
        self.ctrl_loss_sampler = ctrl_loss_sampler
        self.data_loss_sampler = data_loss_sampler
        self.ctrl_metric = ctrl_metric
        self.data_metric = data_metric

        self.pdg_ctrl_loss_coeff = pdg_ctrl_loss_coeff
        self.pdg_data_loss_coeff = pdg_data_loss_coeff
        self.pdg_ctrl_loss_range = pdg_ctrl_loss_range
        self.pdg_data_loss_range = pdg_data_loss_range
        self.data_edges_being_optimized = data_edges_being_optimized
        self.ctrl_edges_being_optimized = ctrl_edges_being_optimized

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
        code_line_mask = code_features.get("code_line_mask")
        pdg_ctrl_edge_labels = code_labels.get("pdg_ctrl_edges", None) if code_labels is not None else None
        pdg_data_edge_labels = code_labels.get("pdg_data_edges", None) if code_labels is not None else None

        # Shape: [batch, vn, vn, 4]
        pred_ctrl_edge_probs, pred_ctrl_edge_labels = self.pdg_ctrl_decoder(encoded_line_node_features)
        pred_data_edge_probs, pred_data_edge_labels = self.pdg_data_decoder(encoded_line_node_features)

        final_loss = 0.
        returned_dict = {}
        if pdg_ctrl_edge_labels is None:
            returned_dict.update({
                'ctrl_edge_logits': pred_ctrl_edge_probs,
                'ctrl_edge_preds': pred_ctrl_edge_labels,
            })
        # Check pdg loss is in range.
        elif self.check_loss_in_range(self.pdg_ctrl_loss_range):
            if self.ctrl_edges_being_optimized:
                pdg_ctrl_edge_labels = construct_matrix_from_opt_edge_idxes(pdg_ctrl_edge_labels, code_line_mask)
            pdg_ctrl_loss, pdg_ctrl_loss_mask = self.ctrl_loss_sampler.get_loss(pdg_ctrl_edge_labels, pred_ctrl_edge_probs)
            pdg_ctrl_loss *= self.pdg_ctrl_loss_coeff

            final_loss += pdg_ctrl_loss
            if self.ctrl_metric is not None:
                self.ctrl_metric(pred_ctrl_edge_labels, pdg_ctrl_edge_labels, pdg_ctrl_loss_mask)
            if self.add_pdg_loss_metric:
                self.pdg_ctrl_metric(pdg_ctrl_loss)
            returned_dict.update({
                'ctrl_edge_logits': pred_ctrl_edge_probs,
                'ctrl_edge_preds': pred_ctrl_edge_labels,
                'ctrl_edge_labels': pdg_ctrl_edge_labels,
                'ctrl_edge_mask': pdg_ctrl_loss_mask,
            })

        if pdg_data_edge_labels is None:
            returned_dict.update({
                'data_edge_logits': pred_data_edge_probs,
                'data_edge_preds': pred_data_edge_labels,
            })
        # Check pdg loss is in range.
        elif self.check_loss_in_range(self.pdg_data_loss_range):
            if self.data_edges_being_optimized:
                pdg_data_edge_labels = construct_matrix_from_opt_edge_idxes(pdg_data_edge_labels, code_line_mask)
            pdg_data_loss, pdg_data_loss_mask = self.data_loss_sampler.get_loss(pdg_data_edge_labels, pred_data_edge_probs)
            pdg_data_loss *= self.pdg_data_loss_coeff

            final_loss += pdg_data_loss
            if self.data_metric is not None:
                self.data_metric(pred_data_edge_labels, pdg_data_edge_labels, pdg_data_loss_mask)
            if self.add_pdg_loss_metric:
                self.pdg_data_metric(pdg_data_loss)
            returned_dict.update({
                'data_edge_logits': pred_data_edge_probs,
                'data_edge_preds': pred_data_edge_labels,
                'data_edge_labels': pdg_data_edge_labels,
                'data_edge_mask': pdg_data_loss_mask,
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
