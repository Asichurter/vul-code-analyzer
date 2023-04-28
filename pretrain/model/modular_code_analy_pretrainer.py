from typing import Dict, Tuple, Optional, List, Iterable

import torch
from allennlp.common import Lazy
from allennlp.data import Vocabulary, TextFieldTensors

from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask

from common.comp.nn.line_extractor import LineExtractor
from pretrain.comp.nn.node_encoder.node_encoder import NodeEncoder
from pretrain.comp.nn.code_objective.code_objective import CodeObjective
from pretrain.model.modules.code_analysis import CodeAnalysis
from utils.allennlp_utils.tokenizer_vocab_sensitive_utils import drop_tokenizer_special_tokens


@Model.register('modular_code_analy_pretrainer')
class ModularCodeAnalyPretrainer(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        code_embedder: TextFieldEmbedder,
        code_encoder: Seq2SeqEncoder,
        line_extractor: LineExtractor,
        line_node_encoder: NodeEncoder,
        token_node_encoder: NodeEncoder,
        drop_tokenizer_special_token_type: str = 'codebert',
        code_objectives: List[Lazy[CodeObjective]] = [],
        code_analysis_tasks: List[Lazy[CodeAnalysis]] = [],
        weight_file_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.code_embedder = code_embedder
        self.code_encoder = code_encoder
        self.line_extractor = line_extractor
        self.line_node_encoder = line_node_encoder
        self.token_node_encoder = token_node_encoder
        self.drop_tokenizer_special_token_type = drop_tokenizer_special_token_type

        self.from_token_code_objectives = torch.nn.ModuleList()
        self.from_embedding_code_objectives = torch.nn.ModuleList()
        self.code_analysis_tasks = torch.nn.ModuleList()
        self.any_as_code_embedder = False
        self.preprocess_pretrain_objectives(code_objectives, vocab)
        self.preprocess_code_analysis_tasks(code_analysis_tasks, vocab)

        self.cur_epoch = 0
        self.test = 0

        # Maybe pre-load weights
        if weight_file_path is not None:
            state_dict = torch.load(weight_file_path)
            self.load_state_dict(state_dict, strict=False)

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

    def preprocess_code_analysis_tasks(self, code_analysis_tasks: List[Lazy[CodeAnalysis]], vocab: Vocabulary):
        for task in code_analysis_tasks:
            task = task.construct(vocab=vocab)
            self.code_analysis_tasks.append(task)

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
        code_features, code_mask = drop_tokenizer_special_tokens(self.drop_tokenizer_special_token_type, code_features, code_mask)
        line_features, line_mask = self.line_extractor(code_features, code_mask,
                                                       line_idxes, vertice_num)
        return line_features, line_mask

    def _encode_line_node_features(self,
                                   node_features: torch.Tensor,
                                   node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode line feature into node-level attribute features using node encoder.
        It mostly includes dimension reduction operation.
        """
        return self.line_node_encoder(node_features, node_mask)

    def _encode_token_node_features(self,
                                    node_features: torch.Tensor,
                                    node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode token feature into node-level attribute features using node encoder.
        It mostly includes dimension reduction operation.
        """
        return self.token_node_encoder(node_features, node_mask)

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
            obj.update_metric(obj_loss)

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
            obj.update_metric(obj_loss)

        return loss

    def check_pdg_loss_in_range(self, range_to_check: List[int]):
        # Default behavior: Always in range.
        if range_to_check[0] == range_to_check[1] == -1:
            return True
        return range_to_check[0] <= self.cur_epoch <= range_to_check[1]

    def forward(self,
                code: TextFieldTensors,
                line_idxes: torch.Tensor,
                vertice_num: torch.Tensor,
                pdg_ctrl_edges: Optional[torch.Tensor] = None,
                pdg_data_edges: Optional[torch.Tensor] = None,
                cfg_line_edges: Optional[torch.Tensor] = None,
                mlm_sampling_weights: Optional[torch.Tensor] = None,
                mlm_span_tags: Optional[torch.Tensor] = None,
                token_data_token_mask: Optional[torch.Tensor] = None,
                meta_data: Optional[List] = [],
                **kwargs) -> Dict[str, torch.Tensor]:
        """
            Forward with full parameters.
        """
        if pdg_ctrl_edges is not None or pdg_data_edges is not None:
            from_token_pretrain_loss, encoded_code_outputs = self.pretrain_forward_from_token(line_idxes.device, code,
                                                                                              mlm_sampling_weights=mlm_sampling_weights,
                                                                                              mlm_span_tags=mlm_span_tags)
            if not self.any_as_code_embedder:
                # Shape: [batch, seq, dim]
                encoded_code_outputs = self.embed_encode_code(code)

            code_token_features, code_token_mask = encoded_code_outputs['outputs'], encoded_code_outputs['mask']
            from_embedding_pretrain_loss = self.pretrain_forward_from_embedding(line_idxes.device,
                                                                                code_token_features,
                                                                                code_token_mask)
        # No edges fed, just embed and predict edges
        else:
            from_token_pretrain_loss = torch.zeros((1,), device=line_idxes.device)
            from_embedding_pretrain_loss = torch.zeros((1,), device=line_idxes.device)
            encoded_code_outputs = self.embed_encode_code(code)
            code_token_features, code_token_mask = encoded_code_outputs['outputs'], encoded_code_outputs['mask']

        # Shape: [batch, max_lines, dim]
        code_line_features, code_line_mask = self.get_line_node_features(code_token_features, code_token_mask,
                                                                         line_idxes, vertice_num)
        # Shape: [batch, vn(max_lines), dim]
        encoded_line_node_features = self._encode_line_node_features(code_line_features, code_line_mask)
        encoded_token_node_features = self._encode_token_node_features(code_token_features, code_token_mask)

        code_features = {
            'code_line_features': encoded_line_node_features,
            'code_token_features': encoded_token_node_features,
            'code_token_mask': code_token_mask,
            'token_data_token_mask': token_data_token_mask,
        }
        code_labels = {
            'pdg_ctrl_edges': pdg_ctrl_edges,
            'pdg_data_edges': pdg_data_edges,
            'cfg_line_edges': cfg_line_edges,
        }

        returned_dict = {'meta_data': meta_data}
        final_loss = (from_token_pretrain_loss + from_embedding_pretrain_loss).squeeze()
        for task_model in self.code_analysis_tasks:
            task_forward_out = task_model(code_features, code_labels)
            final_loss += task_forward_out['loss']
            del task_forward_out['loss']
            returned_dict.update(task_forward_out)

        returned_dict['loss'] = final_loss
        return returned_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for obj_list in [self.from_token_code_objectives, self.from_embedding_code_objectives]:
            for obj in obj_list:
                ctrl_metric = obj.get_metric(reset)
                metrics.update(ctrl_metric)

        for analy_task in self.code_analysis_tasks:
            task_metrics = analy_task.get_metrics(reset)
            metrics.update(task_metrics)

        return metrics
