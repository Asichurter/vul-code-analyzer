from typing import Dict, Tuple, Optional, List, Iterable, Callable

import torch
from allennlp.data import Vocabulary
from allennlp.training.metrics import Metric

from pretrain.comp.metric.objective_pretrain_metric import ObjectiveLoss
from pretrain.comp.nn.loss_sampler.loss_sampler import LossSampler
from pretrain.comp.nn.struct_decoder.struct_decoder import StructDecoder
from pretrain.model.modules.code_analysis import CodeAnalysis

@CodeAnalysis.register('cfg_analyzer')
class CFGAnalyzer(CodeAnalysis):
    def __init__(
        self,
        vocab: Vocabulary,
        cfg_decoder: StructDecoder,
        loss_sampler: LossSampler,
        cfg_metric: Optional[Metric] = None,
        cfg_loss_coeff: float = 1.,
        cfg_loss_range: List[int] = [-1,-1],
        # token_edge_input_being_optimized: bool = False,
        add_loss_metric: bool = True,
        **kwargs
    ):
        super().__init__(vocab, "cfg_analyzer")     # here just give a placeholder value
        self.cfg_decoder = cfg_decoder
        self.loss_sampler = loss_sampler
        self.cfg_metric = cfg_metric
        self.cfg_loss_coeff = cfg_loss_coeff
        self.cfg_loss_range = cfg_loss_range
        # self.token_edge_input_being_optimized = token_edge_input_being_optimized

        self.add_loss_metric = add_loss_metric
        if add_loss_metric:
            self.loss_metric = ObjectiveLoss('cfg')

        self.test = 0


    def forward(self,
                code_features: Dict[str, torch.Tensor],
                code_labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
            Using the embedded token & line features to complete CFG prediction.
        """
        encoded_line_node_features = code_features["code_line_features"]
        cfg_line_edges = code_labels.get("cfg_line_edges", None) if code_labels is not None else None

        # Shape: [batch, vn, vn, 4]
        pred_cfg_edge_probs, pred_cfg_edge_labels = self.cfg_decoder(encoded_line_node_features)

        returned_dict = {}
        if cfg_line_edges is None:
            returned_dict.update({
                'cfg_edge_logits': pred_cfg_edge_probs,
                'cfg_edge_labels': pred_cfg_edge_labels,
            })
        # Check pdg loss is in range.
        elif self.check_loss_in_range(self.cfg_loss_range):
            cfg_loss, cfg_loss_mask = self.loss_sampler.get_loss(cfg_line_edges, pred_cfg_edge_probs)
            cfg_loss *= self.cfg_loss_coeff
            returned_dict['loss'] = cfg_loss

            if self.cfg_metric is not None:
                self.cfg_metric(pred_cfg_edge_labels, cfg_line_edges, cfg_loss_mask)
            if self.add_loss_metric:
                self.loss_metric(cfg_loss)
            returned_dict.update({
                'cfg_edge_logits': pred_cfg_edge_probs,
                'cfg_edge_labels': pred_cfg_edge_labels,
            })

        return returned_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.cfg_metric is not None:
            cfg_metric = self.cfg_metric.get_metric(reset)
            # no metric name returned, use its class name instead
            if type(cfg_metric) != dict:
                metric_name = self.cfg_metric.__class__.__name__
                cfg_metric = {metric_name: cfg_metric}
            metrics.update(cfg_metric)

        if self.add_loss_metric:
            loss_metric = self.loss_metric.get_metric(reset)
            metrics.update(loss_metric)

        return metrics
