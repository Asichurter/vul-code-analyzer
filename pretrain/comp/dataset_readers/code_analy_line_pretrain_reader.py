from typing import Tuple, Iterable, Dict, List, Optional
import os
import re
from tqdm import tqdm

import torch
from allennlp.data import Tokenizer, TokenIndexer, Instance, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField

from common.modules.code_cleaner import CodeCleaner, TrivialCodeCleaner
from common.modules.tokenizer.ast_serial_tokenizer import ASTSerialTokenizer

from utils.allennlp_utils.tokenizer_vocab_sensitive_utils import pre_handle_special_tokenizer_tokens, \
    post_handle_special_tokenizer_tokens
from utils.file import read_dumped
from utils.pretrain_utils.check import check_pretrain_code_field_correctness
from utils.pretrain_utils.line_idx_util import truncate_tokens_and_make_line_index
from utils.pretrain_utils.mlm_mask_weight_gen import dispatch_mlm_weight_gen_method
from utils.pretrain_utils.mlm_span_mask_utils import dispatch_mlm_span_mask_tag_method
from utils.pretrain_utils.edge_matrix_utils import make_pdg_ctrl_edge_matrix_v1, make_pdg_ctrl_edge_matrix_v2, \
                                                   make_pdg_data_edge_matrix, make_pdg_data_edge_matrix_from_processed, \
                                                   make_pdg_ctrl_edge_matrix_v3
from utils.pretrain_utils.token_pdg_matrix_mask_utils import dispatch_token_mask_method
from utils.pretrain_utils.ast_utils import dispatch_ast_serial_tokenize_method
from utils.data_utils.changed_func_extraction import parse_tree


@DatasetReader.register('code_analy_line_pretrain_reader')
class CodeAnalyLinePretrainReader(DatasetReader):
    """
        Note: This dataset reader  is aligned with "ModularCodeAnalyPretrainer" model.
              To use original "CodeLineTokenHybridPDGAnalyzer" model, use
              "PackedHybridLineTokenPDGDatasetReader" instead.

              Line-level PDG only.
    """
    def __init__(self,
                 ################## Basic Module Config ##################
                 code_tokenizer: Tokenizer,
                 code_indexer: TokenIndexer,
                 code_cleaner: CodeCleaner = TrivialCodeCleaner(),  # Do not set this to keep consistent with char span of token level joern-parse
                 code_namespace: str = "code_tokens",
                 special_tokenizer_token_handler_type: str = 'codebert',
                 model_mode: Optional[str] = None,
                 ################## Preprocess Config ##################
                 max_lines: int = 50,
                 code_max_tokens: int = 512,
                 # tokenized_newline_char: str = 'Ċ',  # \n after tokenization by CodeBERT
                 only_keep_complete_lines: bool = True,
                 # hybrid_data_is_processed: bool = False,
                 # optimize_data_edge_input_memory: bool = True,
                 # optimize_line_edge_input_memory: bool = False,
                 ################## MLM Config ##################
                 mlm_sampling_weight_strategy: str = 'uniform',
                 mlm_span_mask_strategy: str = 'none',
                 ################## PDG Config ##################
                 # ----------------------- Data -----------
                 data_edge_version: str = 'v2',
                 # ----------------------- Ctrl -----------
                 ctrl_edge_version: str = 'v2',
                 ################## CFG Config ##################
                 ################## AST Contras Config ##################
                 ast_serial_tokenizer: Optional[ASTSerialTokenizer] = None,     # Set None to disable ast serialization, otherwise AstSerialTokenizer
                 ################## Extra Config ##################
                 debug: bool = False,
                 is_train: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.code_tokenizer = code_tokenizer
        self.code_token_indexers = {code_namespace: code_indexer}  # or {"tokens": SingleIdTokenIndexer()}
        self.max_lines = max_lines
        self.code_max_tokens = code_max_tokens
        # self.tokenized_newline_char = tokenized_newline_char
        self.code_cleaner = code_cleaner
        self.special_tokenizer_token_handler_type = special_tokenizer_token_handler_type
        self.only_keep_complete_lines = only_keep_complete_lines
        # self.hybrid_data_is_processed = hybrid_data_is_processed
        # self.pdg_token_data_processed_tokenizer_name = pdg_token_data_processed_tokenizer_name
        # self.optimize_data_edge_input_memory = optimize_data_edge_input_memory
        # self.optimize_line_edge_input_memory = optimize_line_edge_input_memory
        self.mlm_sampling_weight_method = dispatch_mlm_weight_gen_method(mlm_sampling_weight_strategy)
        self.mlm_span_mask_tag_gen_method = dispatch_mlm_span_mask_tag_method(mlm_span_mask_strategy)
        # self.multi_vs_multi_strategy = multi_vs_multi_strategy

        edge_matrix_func_map = {
            'v1': make_pdg_ctrl_edge_matrix_v1,
            'v2': make_pdg_ctrl_edge_matrix_v2,
            'v3': make_pdg_ctrl_edge_matrix_v3,
        }
        self.ctrl_edge_matrix_func = edge_matrix_func_map[ctrl_edge_version]
        self.data_edge_matrix_func = edge_matrix_func_map[data_edge_version]

        # self.token_data_edge_mask_func = dispatch_token_mask_method(token_data_edge_mask_strategy)
        # self.token_data_edge_mask_kwargs = token_data_edge_mask_kwargs
        self.model_mode = model_mode
        self.ast_serial_tokenizer = ast_serial_tokenizer

        self.is_train = is_train
        self.actual_read_samples = 0
        self.debug = debug


    def truncate_and_make_line_index(self, raw_code: str, tokens: List[Token]) -> Tuple[List[Token],torch.Tensor,int]:
        """
            A util caller wrapper.
        """
        return truncate_tokens_and_make_line_index(raw_code,
                                                   self.max_lines, self.code_max_tokens,
                                                   tokens,
                                                   self.special_tokenizer_token_handler_type,
                                                   self.model_mode,
                                                   self.only_keep_complete_lines)

    def _test_text_to_instance(self, packed_data: Dict) -> Tuple[bool, Instance]:
        raw_code = packed_data['raw_code']
        raw_code = self.code_cleaner.clean_code(raw_code)
        tokenized_code = self.code_tokenizer.tokenize(raw_code)
        tokenized_code, token_line_idxes, line_count = self.truncate_and_make_line_index(raw_code, tokenized_code)
        check_pretrain_code_field_correctness(self.special_tokenizer_token_handler_type, raw_code, tokenized_code, token_line_idxes, None, input_mode_count=1)

        # Ignore single-line code samples.
        if line_count <= 1:
            return False, Instance({})

        fields = {
            'code': TextField(tokenized_code, self.code_token_indexers),
            'line_idxes': TensorField(token_line_idxes),
            'vertice_num': TensorField(torch.Tensor([line_count])), # num. of line is vertice num.
        }
        return True, Instance(fields)

    def text_to_instance(self, packed_pdg: Dict) -> Tuple[bool, Instance]:
        if not self.is_train:
            return self._test_text_to_instance(packed_pdg)

        raw_code = packed_pdg['raw_code']
        pdg_line_ctrl_edges = packed_pdg['pdg_ctrl_edges']
        pdg_line_data_edges = packed_pdg['pdg_data_edges']
        cfg_line_edges = packed_pdg.get('cfg_edges')

        raw_code = self.code_cleaner.clean_code(raw_code)
        tokenized_code = self.code_tokenizer.tokenize(raw_code)
        tokenized_code, token_line_idxes, line_count = self.truncate_and_make_line_index(raw_code, tokenized_code)

        pdg_ctrl_edge_matrix = self.ctrl_edge_matrix_func(pdg_line_ctrl_edges, line_count)
        pdg_data_edge_matrix = self.data_edge_matrix_func(pdg_line_data_edges, line_count)

        # Ignore single-line code samples.
        if line_count == 1:
            return False, Instance({})

        mlm_sampling_weights, _ = self.mlm_sampling_weight_method(raw_code, tokenized_code)

        check_pretrain_code_field_correctness(self.special_tokenizer_token_handler_type, raw_code, tokenized_code, token_line_idxes, pdg_line_ctrl_edges)
        fields = {
            'code': TextField(tokenized_code, self.code_token_indexers),
            'line_idxes': TensorField(token_line_idxes),
            'pdg_ctrl_edges': TensorField(pdg_ctrl_edge_matrix),
            'pdg_data_edges': TensorField(pdg_data_edge_matrix),
            'vertice_num': TensorField(torch.Tensor([line_count])), # num. of line is vertice num.
            'mlm_sampling_weights': TensorField(mlm_sampling_weights),
        }

        # Make CFG matrix.
        if cfg_line_edges is not None:
            # This is a hack, since cfg edge is almost similar to pdg line ctrl edges
            cfg_line_edge_matrix = self.ctrl_edge_matrix_func(cfg_line_edges, line_count)
            fields['cfg_line_edges'] = TensorField(cfg_line_edge_matrix)

        span_tags = self.mlm_span_mask_tag_gen_method(raw_code, tokenized_code)
        if span_tags is not None:
            fields['mlm_span_tags'] = TensorField(span_tags)

        if self.ast_serial_tokenizer is not None:
            ast_serial_tokens = self.ast_serial_tokenizer.tokenize(raw_code)
            # AST serial tokens use the same indexer as code
            fields['ast_tokens'] = TextField(ast_serial_tokens, self.code_token_indexers)

        return True, Instance(fields)


    def _read(self, dataset_config: Dict) -> Iterable[Instance]:
        from utils import GlobalLogger as logger
        data_base_path = dataset_config['data_base_path']
        volume_range = dataset_config['volume_range']

        for vol in range(volume_range[0], volume_range[1]+1):   # closed interval
            logger.info('reader', f'Reading Vol. {vol}')

            vol_path = os.path.join(data_base_path, f'packed_line_vol_{vol}.pkl')
            packed_vol_data_items = read_dumped(vol_path)
            packed_vol_data_items = packed_vol_data_items[:100] if self.debug else packed_vol_data_items
            for pdg_data_item in packed_vol_data_items:
                try:
                    ok, instance = self.text_to_instance(pdg_data_item)
                    if ok:
                        self.actual_read_samples += 1
                        yield instance
                except FileNotFoundError as e:
                    logger.error('read', f'error: {e}. \npdg-item content: {pdg_data_item}')

        logger.info('reader', f'Total instances loaded by reader: {self.actual_read_samples}')