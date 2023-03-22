from typing import Tuple, Iterable, Dict, List, Optional
import os
import re
from tqdm import tqdm

import torch
from allennlp.data import Tokenizer, TokenIndexer, Instance, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField

from common.modules.code_cleaner import CodeCleaner, TrivialCodeCleaner
from utils.allennlp_utils.tokenizer_vocab_sensitive_utils import pre_handle_special_tokenizer_tokens, \
    post_handle_special_tokenizer_tokens
from utils.file import read_dumped
from utils.pretrain_utils.check import check_pretrain_code_field_correctness
from utils.pretrain_utils.mlm_mask_weight_gen import dispatch_mlm_weight_gen_method
from utils.pretrain_utils.mlm_span_mask_utils import dispatch_mlm_span_mask_tag_method
from utils.pretrain_utils.edge_matrix_utils import make_pdg_ctrl_edge_matrix_v1, make_pdg_ctrl_edge_matrix_v2, \
    make_pdg_data_edge_matrix, make_pdg_data_edge_matrix_from_processed, \
    make_pdg_data_edge_matrix_from_processed_optimized, make_line_edge_matrix_from_processed_optimized
from utils.pretrain_utils.token_pdg_matrix_mask_utils import dispatch_token_mask_method


@DatasetReader.register('code_analy_pretrain_reader')
class CodeAnalyPretrainReader(DatasetReader):
    """
        Note: This dataset reader  is aligned with "ModularCodeAnalyPretrainer" model.
              To use original "CodeLineTokenHybridPDGAnalyzer" model, use
              "PackedHybridLineTokenPDGDatasetReader" instead.
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
                 tokenized_newline_char: str = 'Ċ',  # \n after tokenization by CodeBERT
                 only_keep_complete_lines: bool = True,
                 hybrid_data_is_processed: bool = False,
                 optimize_data_edge_input_memory: bool = True,
                 optimize_line_edge_input_memory: bool = False,
                 ################## MLM Config ##################
                 mlm_sampling_weight_strategy: str = 'uniform',
                 mlm_span_mask_strategy: str = 'none',
                 ################## PDG Config ##################
                 # ----------------------- Token-Data -----------
                 multi_vs_multi_strategy: str = 'first',
                 pdg_token_data_processed_tokenizer_name: str = 'microsoft/codebert-base',
                 token_data_edge_mask_strategy: str = 'none',       # To exclude some token-pairs when calculating loss of token-data prediction, set this param
                 token_data_edge_mask_kwargs: Dict = {},  # Param of token_mask_method
                 # ----------------------- Ctrl -----------
                 ctrl_edge_version: str = 'v1',  # To adapt new version of ctrl edges input, only line-level ctrl edges but not data edges
                 ################## CFG Config ##################
                 ################## Extra Config ##################
                 debug: bool = False,
                 is_train: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.code_tokenizer = code_tokenizer
        self.code_token_indexers = {code_namespace: code_indexer}  # or {"tokens": SingleIdTokenIndexer()}
        self.max_lines = max_lines
        self.code_max_tokens = code_max_tokens
        self.tokenized_newline_char = tokenized_newline_char
        self.code_cleaner = code_cleaner
        self.special_tokenizer_token_handler_type = special_tokenizer_token_handler_type
        self.only_keep_complete_lines = only_keep_complete_lines
        self.hybrid_data_is_processed = hybrid_data_is_processed
        self.pdg_token_data_processed_tokenizer_name = pdg_token_data_processed_tokenizer_name
        self.optimize_data_edge_input_memory = optimize_data_edge_input_memory
        self.optimize_line_edge_input_memory = optimize_line_edge_input_memory
        self.mlm_sampling_weight_method = dispatch_mlm_weight_gen_method(mlm_sampling_weight_strategy)
        self.mlm_span_mask_tag_gen_method = dispatch_mlm_span_mask_tag_method(mlm_span_mask_strategy)
        self.multi_vs_multi_strategy = multi_vs_multi_strategy
        self.ctrl_edge_matrix_func = {
            'v1': make_pdg_ctrl_edge_matrix_v1,
            'v2': make_pdg_ctrl_edge_matrix_v2,
        }[ctrl_edge_version]
        self.token_data_edge_mask_func = dispatch_token_mask_method(token_data_edge_mask_strategy)
        self.token_data_edge_mask_kwargs = token_data_edge_mask_kwargs
        self.model_mode = model_mode

        self.is_train = is_train
        self.actual_read_samples = 0
        self.debug = debug


    def truncate_and_make_line_index(self, tokens: List[Token]) -> Tuple[List[Token],torch.Tensor,int]:
        """
        Truncate code tokens based on max_lines and max_tokens and determine line index for each token after tokenization.
        Line indexes (2D) will be used to aggregate line representation from token representations.
        Indexes and tokens are matched one-by-one.
        """
        line_idxes = []
        line_tokens = []
        current_line = 1        # line_index start from 1, to distinguish from padded zeros
        current_column = 0
        tokens = pre_handle_special_tokenizer_tokens(self.special_tokenizer_token_handler_type, tokens)

        for i, token in enumerate(tokens):
            line_idxes.append([current_line, current_column])   # 2D line-column index
            line_tokens.append(token)
            current_column += 1
            if token.text == self.tokenized_newline_char:
                current_line += 1
                current_column = 0
            # truncate code tokens if exceeding max_lines or max_tokens
            # NOTE: Since post-handle may not be the invert operation of pre-handle, the number of
            #       max tokens here may be slightly different from the given number.
            if current_line > self.max_lines or len(line_tokens) == self.code_max_tokens:
                break

        if self.only_keep_complete_lines:
            # FixBug: Empty tokens when 'current_column' is 0 or 'current_line' is 1.
            if current_column > 0 and current_line > 1:
                line_tokens = line_tokens[:-current_column]
                line_idxes = line_idxes[:-current_column]

        line_tokens, line_idxes = post_handle_special_tokenizer_tokens(self.special_tokenizer_token_handler_type, (line_tokens,), line_idxes,
                                                                       mode=self.model_mode)
        return line_tokens, torch.LongTensor(line_idxes), current_line-1

    def _test_text_to_instance(self, packed_data: Dict) -> Tuple[bool, Instance]:
        raw_code = packed_data['raw_code']
        raw_code = self.code_cleaner.clean_code(raw_code)
        tokenized_code = self.code_tokenizer.tokenize(raw_code)
        tokenized_code, token_line_idxes, line_count = self.truncate_and_make_line_index(tokenized_code)
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
        pdg_line_ctrl_edges = packed_pdg['line_edges']
        cfg_line_edges = packed_pdg.get('cfg_edges')

        raw_code = self.code_cleaner.clean_code(raw_code)
        tokenized_code = self.code_tokenizer.tokenize(raw_code)
        tokenized_code, token_line_idxes, line_count = self.truncate_and_make_line_index(tokenized_code)

        # Make PDG-ctrl matrix.
        if not self.optimize_line_edge_input_memory:
            pdg_ctrl_edge_matrix = self.ctrl_edge_matrix_func(pdg_line_ctrl_edges, line_count)
        else:
            pdg_ctrl_edge_matrix = make_line_edge_matrix_from_processed_optimized(line_count, pdg_line_ctrl_edges, skip_self_loop=True, shift=1)

        # Make CFG matrix.
        if cfg_line_edges is not None:
            if not self.optimize_line_edge_input_memory:
                # This is a hack, since cfg edge is almost similar to pdg line ctrl edges
                cfg_line_edge_matrix = self.ctrl_edge_matrix_func(cfg_line_edges, line_count)
            else:
                cfg_line_edge_matrix = make_line_edge_matrix_from_processed_optimized(line_count, cfg_line_edges, skip_self_loop=True, shift=1)
        else:
            cfg_line_edge_matrix = None

        # Make PDG-data matrix.
        # Check whether we need to process the joern-parse token data edges
        if self.hybrid_data_is_processed:
            if self.optimize_data_edge_input_memory:
                pdg_data_edge_matrix = make_pdg_data_edge_matrix_from_processed_optimized(tokenized_code,
                                                                                          packed_pdg['processed_token_data_edges'][self.pdg_token_data_processed_tokenizer_name])
            else:
                pdg_data_edge_matrix = make_pdg_data_edge_matrix_from_processed(tokenized_code,
                                                                                packed_pdg['processed_token_data_edges'][self.pdg_token_data_processed_tokenizer_name])
        else:
            if self.optimize_data_edge_input_memory:
                raise NotImplementedError
            token_nodes = packed_pdg['token_nodes']
            token_edges = packed_pdg['token_edges']
            pdg_data_edge_matrix = make_pdg_data_edge_matrix(raw_code, token_nodes, token_edges, tokenized_code, self.multi_vs_multi_strategy)

        # Ignore single-line code samples.
        if line_count == 1:
            return False, Instance({})

        mlm_sampling_weights, _ = self.mlm_sampling_weight_method(raw_code, tokenized_code)
        token_data_token_mask = self.token_data_edge_mask_func(raw_code, tokenized_code, **self.token_data_edge_mask_kwargs)

        check_pretrain_code_field_correctness(self.special_tokenizer_token_handler_type, raw_code, tokenized_code, token_line_idxes, pdg_line_ctrl_edges)
        fields = {
            'code': TextField(tokenized_code, self.code_token_indexers),
            'line_idxes': TensorField(token_line_idxes),
            'pdg_line_ctrl_edges': TensorField(pdg_ctrl_edge_matrix),
            'pdg_token_data_edges': TensorField(pdg_data_edge_matrix),
            'cfg_line_edges': TensorField(cfg_line_edge_matrix),
            'vertice_num': TensorField(torch.Tensor([line_count])), # num. of line is vertice num.
            'mlm_sampling_weights': TensorField(mlm_sampling_weights),
            'token_data_token_mask': TensorField(token_data_token_mask),
        }

        span_tags = self.mlm_span_mask_tag_gen_method(raw_code, tokenized_code)
        if span_tags is not None:
            fields['mlm_span_tags'] = TensorField(span_tags)

        return True, Instance(fields)


    def _read(self, dataset_config: Dict) -> Iterable[Instance]:
        from utils import GlobalLogger as logger
        data_base_path = dataset_config['data_base_path']
        volume_range = dataset_config['volume_range']

        for vol in range(volume_range[0], volume_range[1]+1):   # closed interval
            logger.info('PackedHybridTokenLineReader.read', f'Reading Vol. {vol}')

            vol_path = os.path.join(data_base_path, f'packed_hybrid_vol_{vol}.pkl')
            packed_vol_data_items = read_dumped(vol_path)
            packed_vol_data_items = packed_vol_data_items[:100] if self.debug else packed_vol_data_items
            for pdg_data_item in packed_vol_data_items:
                try:
                    ok, instance = self.text_to_instance(pdg_data_item)
                    if ok:
                        self.actual_read_samples += 1
                        yield instance
                # todo: revert here to include all exceptions
                except FileNotFoundError as e:
                    logger.error('read', f'error: {e}. \npdg-item content: {pdg_data_item}')

        logger.info('reader', f'Total instances loaded by reader: {self.actual_read_samples}')