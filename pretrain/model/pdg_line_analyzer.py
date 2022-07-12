from typing import Dict, Tuple, Optional, List

import torch
from allennlp.common import Lazy
from allennlp.data import Vocabulary, TextFieldTensors

from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric

from pretrain.comp.nn.line_extractor import LineExtractor
from pretrain.comp.nn.loss_sampler.loss_sampler import LossSampler
from pretrain.comp.nn.node_encoder.node_encoder import NodeEncoder
from pretrain.comp.nn.code_objective.code_objective import CodeObjective
from pretrain.comp.nn.struct_decoder.struct_decoder import StructDecoder


@Model.register('code_line_pdg_analyzer')
class CodeLinePDGAnalyzer(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        code_embedder: TextFieldEmbedder,
        code_encoder: Seq2SeqEncoder,
        node_encoder: NodeEncoder,
        line_extractor: LineExtractor,
        struct_decoder: StructDecoder,
        loss_sampler: LossSampler,
        drop_tokenizer_special_token_type: str = 'codebert',
        metric: Optional[Metric] = None,
        code_objectives: List[Lazy[CodeObjective]] = [],
        pdg_loss_coeff: float = 1.,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.code_embedder = code_embedder
        self.code_encoder = code_encoder
        self.node_encoder = node_encoder
        self.line_extractor = line_extractor
        self.struct_decoder = struct_decoder
        self.loss_sampler = loss_sampler
        self.metric = metric
        self.drop_tokenizer_special_token_type = drop_tokenizer_special_token_type

        self.from_token_code_objectives = torch.nn.ModuleList()
        self.from_embedding_code_objectives = torch.nn.ModuleList()
        self.any_as_code_embedder = False
        self.preprocess_pretrain_objectives(code_objectives, vocab)
        self.pdg_loss_coeff = pdg_loss_coeff

        self.test = 0

    def preprocess_pretrain_objectives(self, objectives: List[Lazy[CodeObjective]], vocab: Vocabulary):
        as_code_embedder_count = 0
        for obj in objectives:
            obj = obj.construct(vocab=vocab)
            if obj.forward_from_where == 'token':
                self.from_token_code_objectives.append(obj)
                if obj.as_code_embedder:
                    as_code_embedder_count += 1
            else:
                self.from_embedding_code_objectives.append(obj)

        assert as_code_embedder_count < 2, f'Found {as_code_embedder_count} objectives as code embedder (as most 1 allowed)'
        self.any_as_code_embedder = as_code_embedder_count > 0

    def drop_tokenizer_special_tokens(self, embedded_code, code_mask):
        # For CodeBERT, drop <s> and </s> (first and last token)
        if self.drop_tokenizer_special_token_type.lower() == 'codebert':
            return embedded_code[:,1:-1], code_mask[:,1:-1]
        else:
            return embedded_code, code_mask

    def embed_encode_code(self, code: TextFieldTensors):
        # num_wrapping_dim = dim_num - 2
        num_wrapping_dim = 0

        # shape: (batch_size, max_input_sequence_length)
        mask = get_text_field_mask(code, num_wrapping_dims=num_wrapping_dim)
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_features = self.code_embedder(code, num_wrapping_dims=num_wrapping_dim)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self.code_encoder(embedded_features, mask)
        return {
            "mask": mask,
            "outputs": encoder_outputs
        }

    def get_line_node_features(self,
                               code_features: torch.Tensor,
                               code_mask: torch.Tensor,
                               line_idxes: torch.Tensor,
                               vertice_num: torch.Tensor
                               ) -> Tuple[torch.Tensor,torch.Tensor]:
        # Move the dropping of tokenizer special tokens here, since it only
        # has influence on line-feature extraction.
        code_features, code_mask = self.drop_tokenizer_special_tokens(code_features, code_mask)
        line_features, line_mask = self.line_extractor(code_features, code_mask,
                                                       line_idxes, vertice_num)
        return line_features, line_mask

    def encode_node_features(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Encode line feature into node-level attribute features using node encoder.
        It mostly includes dimension reduction operation.
        """
        return self.node_encoder(node_features)

    def pretrain_forward_from_token(self, device, code: TextFieldTensors, **kwargs):
        loss = torch.zeros((1,), device=device)
        embed_output = None
        for obj in self.from_token_code_objectives:
            obj_output = obj(code=code,
                             code_embed_func=self.embed_encode_code,
                             **kwargs)
            # Update loss.
            obj_loss = obj_output['loss'] # / len(self.from_token_code_objectives)
            loss += obj_loss
            obj.update_metric(loss)

            if obj.as_code_embedder:
                embed_output = obj_output

        return loss, embed_output

    def pretrain_forward_from_embedding(self,
                                        device,
                                        token_embedding: torch.Tensor,
                                        token_mask: torch.Tensor,
                                        tensor_dict: Dict[str, torch.Tensor] = {},
                                        **kwargs):
        loss = torch.zeros((1,), device=device)
        for obj in self.from_embedding_code_objectives:
            obj_output = obj(token_embedding=token_embedding,
                             token_mask=token_mask,
                             tensor_dict=tensor_dict,
                             **kwargs)
            # Update loss.
            obj_loss = obj_output['loss'] # / len(self.from_embedding_code_objectives)
            loss += obj_loss
            obj.update_metric(loss)

        return loss

    def forward(self,
                code: TextFieldTensors,
                line_idxes: torch.Tensor,
                edges: torch.Tensor,
                vertice_num: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        # [Outline]
        # 1. Token-based Pretrain Objective Computing.
        # 2. (Embed & Encode) ->
        # 3. Embedding-based Pretrain Objective Computing.
        # 4. Get Line Features ->
        # 5. Get Node Features ->
        # 6. Structure Prediction ->
        # 7. Final Loss Calculating
        # token_ids_cpu = code['code_tokens']['token_ids'].detach().cpu()
        from_token_pretrain_loss, encoded_code_outputs = self.pretrain_forward_from_token(line_idxes.device, code)
        if not self.any_as_code_embedder:
            # Shape: [batch, seq, dim]
            encoded_code_outputs = self.embed_encode_code(code)

        code_token_features, code_token_mask = encoded_code_outputs['outputs'], encoded_code_outputs['mask']
        from_embedding_pretrain_loss = self.pretrain_forward_from_embedding(line_idxes.device,
                                                                            code_token_features,
                                                                            code_token_mask)

        # Shape: [batch, max_lines, dim]
        code_line_features, code_line_mask = self.get_line_node_features(code_token_features, code_token_mask,
                                                                         line_idxes, vertice_num)
        # Shape: [batch, vn(max_lines), dim]
        node_features = self.encode_node_features(code_line_features)

        # Shape: [batch, vn, vn, 4]
        pred_edge_probs, pred_edge_labels = self.struct_decoder(node_features)
        loss, loss_mask = self.loss_sampler.get_loss(edges, pred_edge_probs, vertice_num)
        loss *= self.pdg_loss_coeff
        loss += (from_token_pretrain_loss + from_embedding_pretrain_loss).squeeze()

        if self.metric is not None:
            self.metric(pred_edge_labels, edges, loss_mask)

        return {
            'edge_logits': pred_edge_probs,
            'edge_labels': pred_edge_labels,
            'loss': loss
        }

    def pdg_predict(self,
                    code: TextFieldTensors,
                    line_idxes: torch.Tensor,
                    vertice_num: torch.Tensor,
                    meta_data: List,
                    return_node_features: bool = False,
                    ) -> Dict[str, torch.Tensor]:
        encoded_code_outputs = self.embed_encode_code(code)
        code_token_features, code_token_mask = encoded_code_outputs['outputs'], encoded_code_outputs['mask']

        # Shape: [batch, max_lines, dim]
        code_line_features, code_line_mask = self.get_line_node_features(code_token_features, code_token_mask,
                                                                         line_idxes, vertice_num)
        # Shape: [batch, vn(max_lines), dim]
        node_features = self.encode_node_features(code_line_features)

        # Shape: [batch, vn, vn, 4]
        pred_edge_probs, pred_edge_labels = self.struct_decoder(node_features)

        ret_dict = {
            'meta_data': meta_data,
            'edge_logits': pred_edge_probs,
            'edge_labels': pred_edge_labels,
        }
        if return_node_features:
            ret_dict['node_features'] = node_features

        return ret_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.metric is not None:
            metric = self.metric.get_metric(reset)
            # no metric name returned, use its class name instead
            if type(metric) != dict:
                metric_name = self.metric.__class__.__name__
                metric = {metric_name: metric}
            metrics.update(metric)

        for obj_list in [self.from_token_code_objectives, self.from_embedding_code_objectives]:
            for obj in obj_list:
                metric = obj.get_metric(reset)
                metrics.update(metric)

        return metrics
