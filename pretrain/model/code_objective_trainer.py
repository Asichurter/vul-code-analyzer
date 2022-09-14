from typing import Dict, List, Optional

import torch
from allennlp.common import Lazy
from allennlp.data import Vocabulary, TextFieldTensors

from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask

from pretrain.comp.nn.code_objective.code_objective import CodeObjective
from pretrain.comp.nn.line_extractor import LineExtractor, AvgLineExtractor


@Model.register('code_objective_trainer')
class CodeObjectiveTrainer(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        code_embedder: TextFieldEmbedder,
        code_encoder: Seq2SeqEncoder,
        drop_tokenizer_special_token_type: str = 'codebert',
        # metric: Optional[Metric] = None,
        code_objectives: List[Lazy[CodeObjective]] = [],
        line_extractor: Optional[LineExtractor] = None,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.code_embedder = code_embedder
        self.code_encoder = code_encoder
        self.drop_tokenizer_special_token_type = drop_tokenizer_special_token_type

        self.from_token_code_objectives = torch.nn.ModuleList()
        self.from_embedding_code_objectives = torch.nn.ModuleList()
        self.any_as_code_embedder = False
        self.preprocess_pretrain_objectives(code_objectives, vocab)
        self.line_extractor = line_extractor
        self.cur_epoch = 0

        self.test = 0

        assert self.any_as_code_embedder

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

    def pretrain_forward_from_token(self, device, code: TextFieldTensors, **kwargs):
        loss = torch.zeros((1,), device=device)
        embed_output = None
        for obj in self.from_token_code_objectives:
            obj_output = obj(code=code,
                             code_embed_func=self.embed_encode_code,
                             epoch=self.cur_epoch,
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
                             epoch=self.cur_epoch,
                             **kwargs)
            # Update loss.
            obj_loss = obj_output['loss'] # / len(self.from_embedding_code_objectives)
            loss += obj_loss
            obj.update_metric(loss)

        return loss

    def forward(self,
                code: TextFieldTensors,
                line_idxes: torch.Tensor,
                vertice_num: torch.Tensor,
                edges: Optional[torch.Tensor] = None,
                mlm_sampling_weights: Optional[torch.Tensor] = None,
                meta_data: Optional[List[Dict]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:

        # Since we can not inject kwarg into forward method call,
        # we can only fetch some information from meta_data field.
        if meta_data is None:
            forward_type = 'mlm'
        else:
            # Assert meta_data is dict type.
            forward_type = meta_data[0].get('forward_type', 'mlm')

        if forward_type == 'mlm':
            from_token_pretrain_loss, encoded_code_outputs = self.pretrain_forward_from_token(line_idxes.device, code,
                                                                                              mlm_sampling_weights=mlm_sampling_weights)
            if not self.any_as_code_embedder:
                # Shape: [batch, seq, dim]
                encoded_code_outputs = self.embed_encode_code(code)

            code_token_features, code_token_mask = encoded_code_outputs['outputs'], encoded_code_outputs['mask']
            from_embedding_pretrain_loss = self.pretrain_forward_from_embedding(line_idxes.device,
                                                                                code_token_features,
                                                                                code_token_mask)
            loss = (from_token_pretrain_loss + from_embedding_pretrain_loss).squeeze(-1)
            return {
                'loss': loss
            }
        elif forward_type == 'line_features':
            encoded_code_outputs = self.embed_encode_code(code)
            # 7.22 Fix grad mismatch bug: Forget to drop tokenizer tokens.
            code_token_features, code_token_mask = self.drop_tokenizer_special_tokens(
                encoded_code_outputs['outputs'],
                encoded_code_outputs['mask']
            )
            line_features, line_mask = self.line_extractor(code_token_features, code_token_mask, line_idxes, vertice_num)
            return {
                'line_features': line_features,
                'vertice_num': vertice_num,
                'meta_data': meta_data
            }
        elif forward_type == 'cls_features':
            encoded_code_outputs = self.embed_encode_code(code)
            # 7.22 Fix grad mismatch bug: Forget to drop tokenizer tokens.
            code_token_features = encoded_code_outputs['outputs']
            code_token_mask = encoded_code_outputs['mask']
            return {
                'cls_features': code_token_features[:,0],
                'meta_data': meta_data
            }
        else:
            raise ValueError(f'Unsupported forward type: {forward_type}')


    def extract_line_features(self,
                              code: TextFieldTensors,
                              line_idxes: torch.Tensor,
                              line_extractor: LineExtractor,
                              meta_data: Optional[List] = None,
                              vertice_num: Optional[torch.Tensor] = None,
                              **kwargs):
        encoded_code_outputs = self.embed_encode_code(code)
        # 7.22 Fix grad mismatch bug: Forget to drop tokenizer tokens.
        code_token_features, code_token_mask = self.drop_tokenizer_special_tokens(
            encoded_code_outputs['outputs'],
            encoded_code_outputs['mask']
        )
        line_features, line_mask = line_extractor(code_token_features, code_token_mask, line_idxes, vertice_num)
        return {
            'node_features': line_features,
            'vertice_num': vertice_num,
            'meta_data': meta_data
        }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for obj_list in [self.from_token_code_objectives, self.from_embedding_code_objectives]:
            for obj in obj_list:
                metric = obj.get_metric(reset)
                metrics.update(metric)

        return metrics
