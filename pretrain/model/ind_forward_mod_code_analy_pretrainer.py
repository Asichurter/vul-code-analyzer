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


@Model.register('ind_forward_modular_code_analy_pretrainer')
class IndependentForwardModularCodeAnalyPretrainer(Model):
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

        self.code_objectives = torch.nn.ModuleList()
        self.code_analysis_tasks = torch.nn.ModuleList()
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
            self.code_objectives.append(obj)

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

    def code_obj_forward(self, code_obj: CodeObjective, code: TextFieldTensors, **kwargs):
        obj_output = code_obj(code=code,
                              code_embed_func=self.embed_encode_code,
                              epoch=self.cur_epoch,
                              **kwargs)
        obj_loss = obj_output['loss']
        code_obj.update_metric(obj_loss)

        return obj_loss, obj_output

    def forward(self,
                code: TextFieldTensors,
                line_idxes: torch.Tensor,
                vertice_num: torch.Tensor,
                forward_step_name: str,
                pdg_line_ctrl_edges: Optional[torch.Tensor] = None,
                pdg_token_data_edges: Optional[torch.Tensor] = None,
                cfg_line_edges: Optional[torch.Tensor] = None,
                mlm_sampling_weights: Optional[torch.Tensor] = None,
                mlm_span_tags: Optional[torch.Tensor] = None,
                token_data_token_mask: Optional[torch.Tensor] = None,
                ast_tokens: Optional[TextFieldTensors] = None,
                meta_data: Optional[List] = [],
                **kwargs) -> Dict[str, torch.Tensor]:
        """
            Forward with full parameters.
        """
        if forward_step_name != "full":
            fwd_type, mod_idx = forward_step_name.split('-')
            mod_idx = int(mod_idx)
        else:
            fwd_type, mod_idx = "full", 0
        assert fwd_type in ['full', 'code_obj', 'code_analy']

        final_loss: torch.Tensor = 0.
        output = {'meta_data': meta_data}

        code_obj_params = {
            "mlm_sampling_weights": mlm_sampling_weights,       # For MLM
            "mlm_span_tags": mlm_span_tags,                     # For MLM
            "ast_tokens":ast_tokens,                            # For AST Contras
        }
        # Code objective forward: all the objs are from tokens.
        if fwd_type == 'code_obj' or fwd_type == 'full':
            if fwd_type == 'code_obj':
                mod_indices = [mod_idx]
            else:
                mod_indices = list(range(len(self.code_objectives)))
            for idx in mod_indices:
                loss, code_obj_output = self.code_obj_forward(self.code_objectives[idx],
                                                              code,
                                                              **code_obj_params)
                final_loss += loss
                output.update(code_obj_output)

        # Code analysis forward
        if fwd_type == 'code_analy' or fwd_type == 'full':
            encoded_code_outputs = self.embed_encode_code(code)
            code_token_features, code_token_mask = encoded_code_outputs['outputs'], encoded_code_outputs['mask']
            # Shape: [batch, max_lines, dim]
            code_line_features, code_line_mask = self.get_line_node_features(code_token_features, code_token_mask, line_idxes, vertice_num)
            # Shape: [batch, vn(max_lines), dim]
            encoded_line_node_features = self._encode_line_node_features(code_line_features, code_line_mask)
            encoded_token_node_features = self._encode_token_node_features(code_token_features, code_token_mask)

            code_features = {
                'code_line_features': encoded_line_node_features,
                'code_token_features': encoded_token_node_features,
                'code_token_mask': code_token_mask,
                'code_line_mask': code_line_mask,
                'token_data_token_mask': token_data_token_mask,
            }
            code_labels = {
                'pdg_line_ctrl_edges': pdg_line_ctrl_edges,
                'pdg_token_data_edges': pdg_token_data_edges,
                'cfg_line_edges': cfg_line_edges,
            }

            if fwd_type == 'code_analy':
                mod_indices = [mod_idx]
            else:
                mod_indices = list(range(len(self.code_analysis_tasks)))
            for idx in mod_indices:
                task_model = self.code_analysis_tasks[idx]
                task_forward_out = task_model(code_features, code_labels)
                task_loss = task_forward_out.pop("loss")
                final_loss += task_loss
                output.update(task_forward_out)

        output["loss"] = final_loss
        return output


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for obj in self.code_objectives:
            ctrl_metric = obj.get_metric(reset)
            metrics.update(ctrl_metric)
        for analy_task in self.code_analysis_tasks:
            task_metrics = analy_task.get_metrics(reset)
            metrics.update(task_metrics)

        return metrics
