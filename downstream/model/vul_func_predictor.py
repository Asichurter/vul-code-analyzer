from typing import Dict, Optional

import torch
from allennlp.data import Vocabulary, TextFieldTensors

from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric

from common.nn.classifier import Classifier
from common.nn.loss_func import LossFunc
from utils import GlobalLogger as mylogger
from utils.allennlp_utils.metric_update import update_metric


@Model.register('vul_func_predictor')
class VulFuncPredictor(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        code_embedder: TextFieldEmbedder,
        code_encoder: Seq2SeqEncoder,
        code_feature_squeezer: Seq2VecEncoder,
        loss_func: LossFunc,
        classifier: Classifier,
        pretrained_state_dict_path: Optional[str] = None,
        load_prefix_remap: Dict[str, str] = {},  # Note this map is "mapping name of model parameter to match state dict"
        metric: Optional[Metric] = None,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.code_embedder = code_embedder
        self.code_encoder = code_encoder
        self.code_feature_squeezer = code_feature_squeezer
        self.loss_func = loss_func
        self.classifier = classifier
        self.metric = metric

        # Load partial remapped parameters from pre-trained
        if pretrained_state_dict_path is not None:
            # Note we map the loaded weights to cpu to align with other parameters in the model,
            # and trainer will help us to move them to GPU device together before training.
            state_dict = torch.load(pretrained_state_dict_path, map_location='cpu')
            self.partial_load_state_dict(state_dict, load_prefix_remap)


    def partial_load_state_dict(self,
                                state_dict: Dict[str, torch.Tensor],
                                prefix_remap: Dict[str, str]):
        """
        Load parameters from a state dict according to the given mapping.
        Note we try to remap the name of parameters of this model to match the
        keys of the given state dict in a prefix-matching manner.

        Also to note, unmapped parameters will be ignored, thus even the key is identical,
        it is also necessary to place this item in the map.
        """
        partial_state_dict = {}
        for name, par in self.named_parameters():
            load_name = None
            for prefix in prefix_remap:
                if name.startswith(prefix):
                    load_prefix = prefix_remap[prefix]
                    load_name = load_prefix + name[len(prefix):]
                    # Always match first
                    break
            # Only load mapped parameters
            if load_name is None:
                continue
            if name in state_dict:
                partial_state_dict[name] = state_dict[load_name]

        load_res = self.load_state_dict(partial_state_dict, strict=False)
        mylogger.info('partial_load_state_dict',
                      f'State dict loading result: {load_res}')


    def embed_encode_code(self, code: TextFieldTensors):
        # num_wrapping_dim = dim_num - 2
        num_wrapping_dim = 0

        # shape: (batch_size, max_input_sequence_length)
        mask = get_text_field_mask(code, num_wrapping_dims=num_wrapping_dim)
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_features = self.code_embedder(code, num_wrapping_dims=num_wrapping_dim)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self.code_encoder(embedded_features, mask)
        code_feature = self.code_feature_squeezer(encoder_outputs, mask)
        return {
            "outputs": code_feature
        }

    def forward(self,
                code: TextFieldTensors,
                label: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        # Shape: [batch, seq, dim]
        encoded_code_outputs = self.embed_encode_code(code)
        code_features = encoded_code_outputs['outputs']

        pred_logits, pred_labels = self.classifier(code_features)
        label = label.squeeze(-1)
        loss = self.loss_func(pred_logits, label)

        if self.metric is not None:
            update_metric(self.metric, pred_labels, pred_logits, label)

        return {
            'logits': pred_logits,
            'pred': pred_labels,
            'loss': loss
        }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.metric is not None:
            metric = self.metric.get_metric(reset)
            # no metric name returned, use its class name instead
            if type(metric) != dict:
                metric_name = self.metric.__class__.__name__
                metric = {metric_name: metric}
            metrics.update(metric)
        return metrics
