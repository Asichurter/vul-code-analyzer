from typing import Dict, Tuple

import torch
from allennlp.data import Vocabulary, TextFieldTensors

from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask

from pretrain.comp.nn.line_extractor import LineExtractor
from pretrain.comp.nn.loss_sampler.loss_sampler import LossSampler
from pretrain.comp.nn.node_encoder.node_encoder import NodeEncoder
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
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.code_embedder = code_embedder
        self.code_encoder = code_encoder
        self.node_encoder = node_encoder
        self.line_extractor = line_extractor
        self.struct_decoder = struct_decoder
        self.loss_sampler = loss_sampler
        self.drop_tokenizer_special_token_type = drop_tokenizer_special_token_type

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
        encoder_outputs, mask = self.drop_tokenizer_special_tokens(encoder_outputs, mask)

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
        line_features, line_mask = self.line_extractor(code_features, code_mask,
                                                       line_idxes, vertice_num)
        return line_features, line_mask

    def encode_node_features(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Encode line feature into node-level attribute features using node encoder.
        It mostly includes dimension reduction operation.
        """
        return self.node_encoder(node_features)

    def forward(self,
                code: TextFieldTensors,
                line_idxes: torch.Tensor,
                edges: torch.Tensor,
                vertice_num: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        # [Outline]
        # 1.Embed & Encode ->
        # 2.Get Line Features ->
        # 3.Get Node Features ->
        # 4.Structure Prediction ->
        # 5.Loss Calculating

        # Shape: [batch, seq, dim]
        encoded_code_outputs = self.embed_encode_code(code)
        code_token_features, code_token_mask = encoded_code_outputs['outputs'], encoded_code_outputs['mask']
        # Shape: [batch, max_lines, dim]
        code_line_features, code_line_mask = self.get_line_node_features(code_token_features, code_token_mask,
                                                                         line_idxes, vertice_num)
        # Shape: [batch, vn(max_lines), dim]
        node_features = self.encode_node_features(code_line_features)

        # Shape: [batch, vn, vn, 4]
        pred_edge_probs, pred_edge_labels = self.struct_decoder(node_features)
        loss = self.loss_sampler.get_loss(edges, pred_edge_probs, vertice_num)

        return {
            'edge_logits': pred_edge_probs,
            'loss': loss
        }