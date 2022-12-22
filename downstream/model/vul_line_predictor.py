from typing import Dict, Optional, Tuple
import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric

from common.nn.classifier import Classifier
from common.nn.loss_func import LossFunc
from common.comp.nn.line_extractor import LineExtractor
from utils.allennlp_utils.metric_update import update_metric
from utils.allennlp_utils.tokenizer_vocab_sensitive_utils import drop_tokenizer_special_tokens


@Model.register('vul_line_predictor')
class VulLinePredictor(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        code_embedder: TextFieldEmbedder,
        code_encoder: Seq2SeqEncoder,
        line_extractor: LineExtractor,
        loss_func: LossFunc,
        classifier: Classifier,
        metric: Optional[Metric] = None,
        wrapping_dim_for_code: int = 0,
        drop_tokenizer_special_token_type: str = 'codebert',
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.code_embedder = code_embedder
        self.code_encoder = code_encoder
        self.line_extractor = line_extractor
        self.loss_func = loss_func
        self.classifier = classifier
        self.metric = metric
        self.drop_tokenizer_special_token_type = drop_tokenizer_special_token_type

        self.wrapping_dim_for_code = wrapping_dim_for_code


    def embed_encode_code(self, code: TextFieldTensors):
        num_wrapping_dim = self.wrapping_dim_for_code

        # shape: (batch_size, max_input_sequence_length)
        mask = get_text_field_mask(code, num_wrapping_dims=num_wrapping_dim)
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_features = self.code_embedder(code, num_wrapping_dims=num_wrapping_dim)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self.code_encoder(embedded_features, mask)
        return {
            "outputs": encoder_outputs,
            "mask": mask
        }

    def get_line_features(self,
                          code_features: torch.Tensor,
                          code_mask: torch.Tensor,
                          line_indices: torch.Tensor,
                          vertice_num: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Move the dropping of tokenizer special tokens here, since it only
        # has influence on line-feature extraction.
        code_features, code_mask = drop_tokenizer_special_tokens(self.drop_tokenizer_special_token_type, code_features, code_mask)
        line_features, line_mask = self.line_extractor(code_features, code_mask, line_indices, vertice_num)
        return line_features, line_mask

    def get_line_vul_loss(self, line_vul_logits: torch.Tensor,
                          line_vul_labels: torch.Tensor,
                          line_mask: torch.BoolTensor):
        selected_logits = torch.masked_select(line_vul_logits, line_mask)
        selected_labels = torch.masked_select(line_vul_labels, line_mask)
        return self.loss_func(selected_logits, selected_labels)

    def forward(self,
                code: TextFieldTensors,
                line_indices: torch.Tensor,
                line_counts: torch.Tensor,
                line_vul_labels: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        # Shape: [batch, seq, dim]
        encoded_code_outputs = self.embed_encode_code(code)
        code_features = encoded_code_outputs['outputs']
        code_masks = encoded_code_outputs['mask']

        # Shape: [bsz, lines, dim]
        line_features, line_mask = self.get_line_features(code_features, code_masks, line_indices, line_counts)
        # Shape: [bsz, lines]
        line_vul_pred_logits, line_vul_pred_labels = self.classifier(line_features)
        line_vul_pred_labels = line_vul_pred_labels.squeeze(-1)
        # NOTE: Here we assume the classifier and the loss_func are coupled in configuration,
        #       such as "Binary Sigmoid Classifier" + "BCELoss", no extra check will be done.
        loss = self.get_line_vul_loss(line_vul_pred_logits, line_vul_labels, line_mask)

        if self.metric is not None:
            update_metric(self.metric, line_vul_pred_labels, line_vul_pred_logits, line_vul_labels, line_mask,
                          flatten_labels=False, ignore_shape=True)

        return {
            'logits': line_vul_pred_logits,
            'pred': line_vul_pred_labels,
            'mask': line_mask,
            'loss': loss,
            'labels': line_vul_labels,
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
