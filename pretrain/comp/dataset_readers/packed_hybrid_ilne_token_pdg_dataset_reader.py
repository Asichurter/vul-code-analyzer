from typing import Tuple, Iterable, Dict, List, Optional
import os
import re
from tqdm import tqdm

import torch
from allennlp.data import Tokenizer, TokenIndexer, Instance, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField

from common.modules.code_cleaner import CodeCleaner, TrivialCodeCleaner
from utils.file import read_dumped
from utils.pretrain_utils.mlm_mask_weight_gen import dispatch_mlm_weight_gen_method
from utils.pretrain_utils.mlm_span_mask_utils import dispatch_mlm_span_mask_tag_method
from utils.joern_utils.joern_dev_pdg_parse_utils import build_token_level_pdg_struct


@DatasetReader.register('packed_hybrid_line_token_pdg')
class PackedHybridLineTokenPDGDatasetReader(DatasetReader):
    def __init__(self,
                 code_tokenizer: Tokenizer,
                 code_indexer: TokenIndexer,
                 # volume_range: Tuple[int,int],  # closed interval: [a,b]
                 pdg_max_vertice: int,  # For line-level approach, this should be equal to "max_lines"
                 max_lines: int,
                 code_max_tokens: int,
                 code_namespace: str = "code_tokens",
                 code_cleaner: CodeCleaner = TrivialCodeCleaner(),  # Do not set this to keep consistent with char span of token level joern-parse
                 tokenized_newline_char: str = 'Ċ',  # \n after tokenization by CodeBERT
                 special_tokenizer_token_handler_type: str = 'codebert',
                 only_keep_complete_lines: bool = True,
                 mlm_sampling_weight_strategy: str = 'uniform',
                 mlm_span_mask_strategy: str = 'none',
                 multi_vs_multi_strategy: str = 'first',
                 hybrid_data_is_processed: bool = False,
                 processed_tokenizer_name: str = 'microsoft/codebert-base',
                 optimize_data_edge_input_memory: bool = True,
                 ctrl_edge_version: str = 'v1',
                 debug: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.code_tokenizer = code_tokenizer
        self.code_token_indexers = {code_namespace: code_indexer}  # or {"tokens": SingleIdTokenIndexer()}
        # self.volume_range = volume_range
        self.pdg_max_vertice = pdg_max_vertice
        self.max_lines = max_lines
        self.code_max_tokens = code_max_tokens
        self.tokenized_newline_char = tokenized_newline_char
        self.code_cleaner = code_cleaner
        # self.add_EOS_token = add_EOS_token
        # self.skip_first_token_line_index = skip_first_token_line_index
        self.special_tokenizer_token_handler_type = special_tokenizer_token_handler_type
        self.only_keep_complete_lines = only_keep_complete_lines
        self.hybrid_data_is_processed = hybrid_data_is_processed
        self.processed_tokenizer_name = processed_tokenizer_name
        self.optimize_data_edge_input_memory = optimize_data_edge_input_memory
        self.mlm_sampling_weight_method = dispatch_mlm_weight_gen_method(mlm_sampling_weight_strategy)
        self.mlm_span_mask_tag_gen_method = dispatch_mlm_span_mask_tag_method(mlm_span_mask_strategy)
        self.multi_vs_multi_strategy = multi_vs_multi_strategy
        self.ctrl_edge_matrix_func = {
            'v1': self.make_ctrl_edge_matrix_v1,
            'v2': self.make_ctrl_edge_matrix_v2,
        }[ctrl_edge_version]

        self.actual_read_samples = 0
        self.debug = debug


    def make_ctrl_edge_matrix_v1(self,
                                 line_edges: List[str],
                                 line_count: int) -> torch.LongTensor:
        """
        Make line-level ctrl dependency matrix from edge data.
        V1: Results from latest version of joern, including line-level data edges.

        """
        # To cover the last line (line_count-th), we have to allocate one more line here.
        # Set 1 as default to distinguish from padded positions
        matrix = torch.ones((line_count+1, line_count+1))

        for edge in line_edges:
            tail, head, etype = re.split(',| ', edge)   # tail/head vertice index start from 1 instead of 0
            tail, head, etype = int(tail), int(head), int(etype)
            # Ignore uncovered vertices (lines)
            if tail > line_count or head > line_count:
                continue
            if etype == 3 or etype == 2:
                matrix[tail, head] = 2

        # Drop 0-th row and column, since line index starts from 1.
        return matrix[1:, 1:]

    def make_ctrl_edge_matrix_v2(self,
                                 line_edges: List[str],
                                 line_count: int) -> torch.LongTensor:
        """
        Make line-level ctrl dependency matrix from edge data.
        V2: Results from 0.3.1 version of joern, only line-level ctrl edges.

        """
        # To cover the last line (line_count-th), we have to allocate one more line here.
        # Set 1 as default to distinguish from padded positions
        matrix = torch.ones((line_count+1, line_count+1))

        for edge in line_edges:
            tail, head = edge
            # Ignore uncovered vertices (lines)
            if tail > line_count or head > line_count:
                continue
            matrix[tail, head] = 2

        # Drop 0-th row and column, since line index starts from 1.
        # Now line index start from 0.
        return matrix[1:, 1:]

    def make_data_edge_matrix(self,
                              raw_code: str,
                              token_nodes: List[List[str]],
                              token_edges: List[List[str]],
                              tokenized_tokens: List[Token],
                              multi_vs_multi_strategy: str) -> torch.Tensor:
        # Parse nodes and edges from joern-parse, to build token-level data edges
        _, token_data_edges = build_token_level_pdg_struct(raw_code, tokenized_tokens,
                                                           token_nodes, token_edges,
                                                           multi_vs_multi_strategy,
                                                           to_build_token_ctrl_edges=False)
        token_len = len(tokenized_tokens)

        # Set 1 as default to distinguish from padded positions
        matrix = torch.ones((token_len, token_len))

        for edge in token_data_edges:
            s_token_idx, e_token_idx = edge.split()
            s_token_idx, e_token_idx = int(s_token_idx), int(e_token_idx)
            if s_token_idx >= token_len or e_token_idx >= token_len:
                continue
            if s_token_idx == e_token_idx:
                continue
            # Set edge value to 2
            matrix[s_token_idx, e_token_idx] = 2

        return matrix

    def make_data_edge_matrix_from_processed(self,
                                             tokens: List[Token],
                                             processed_data_edges: List[Tuple[int,int]]) -> torch.Tensor:
        token_len = len(tokens)
        # Set 1 as default to distinguish from padded positions
        matrix = torch.ones((token_len, token_len))

        for edge in processed_data_edges:
            s_token_idx, e_token_idx = edge
            if s_token_idx >= token_len or e_token_idx >= token_len:
                continue
            if s_token_idx == e_token_idx:
                continue
            # Set edge value to 2
            matrix[s_token_idx, e_token_idx] = 2

        return matrix

    def make_data_edge_matrix_from_processed_optimized(self,
                                                       tokens: List[Token],
                                                       processed_data_edges: List[Tuple[int,int]]) -> torch.Tensor:
        """
        Compared to making matrix at loading time, here we only give the edges idxes,
        to allow construct token matrix at run-time to avoid unaffordable memory consumption.
        """
        token_len = len(tokens)
        idxes = []
        for edge in processed_data_edges:
            s_token_idx, e_token_idx = edge
            if s_token_idx >= token_len or e_token_idx >= token_len:
                continue
            if s_token_idx == e_token_idx:
                continue
            idxes.append([s_token_idx, e_token_idx])

        # Append a placeholder idx, to avoid key missing error when calling "batch_tensor"
        if len(idxes) == 0:
            idxes.append([0,0])

        return torch.Tensor(idxes)

    def pre_handle_special_tokenizer_tokens(self, tokens: List[Token]) -> List[Token]:
        if self.special_tokenizer_token_handler_type == 'codebert':
            return tokens[1:-1]
        else:
            return tokens


    def post_handle_special_tokenizer_tokens(self,
                                             tokens: List[Token],
                                             line_idxes: List[int]
                                             ) -> Tuple:
        if self.special_tokenizer_token_handler_type == 'codebert':
            tokens.insert(0, Token('<s>'))
            tokens.append(Token('</s>'))
        else:
            pass
        return tokens, line_idxes


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
        tokens = self.pre_handle_special_tokenizer_tokens(tokens)

        for i, token in enumerate(tokens):
            line_idxes.append([current_line, current_column])   # 2D line-column index
            line_tokens.append(token)
            current_column += 1
            if token.text == self.tokenized_newline_char:
                current_line += 1
                current_column = 0
            # truncate code tokens if exceeding max_lines or max_tokens
            if current_line > self.max_lines or len(line_tokens) == self.code_max_tokens:
                break

        if self.only_keep_complete_lines:
            # FixBug: Empty tokens when 'current_column' is 0 or 'current_line' is 1.
            if current_column > 0 and current_line > 1:
                line_tokens = line_tokens[:-current_column]
                line_idxes = line_idxes[:-current_column]

        line_tokens, line_idxes = self.post_handle_special_tokenizer_tokens(line_tokens, line_idxes)
        return line_tokens, torch.LongTensor(line_idxes), current_line-1

    def _check_data_correctness(self,
                                original_code: str,
                                tokenized_code: List[Token],
                                line_indexes: torch.Tensor,
                                edge_matrix: torch.Tensor):
        from utils import GlobalLogger as mylogger
        # 1. Check tokenized code
        if self.special_tokenizer_token_handler_type == 'codebert':
            if len(tokenized_code) == 2:
                mylogger.error('_check_data_correctness',
                               f'Found empty tokenized code, original code: {original_code}')
        else:
            mylogger.warning('_check_data_correctness', f'Unhandled tokenized type: {self.special_tokenizer_token_handler_type}')

        # 2. Check consistency between code and line index
        if self.special_tokenizer_token_handler_type == 'codebert':
            if len(tokenized_code) - 2 != len(line_indexes):
                mylogger.error('_check_data_correctness',
                               f'Inconsistency found between line index and tokenized code: ' +
                               f'code_len({len(tokenized_code)}) - 2 != index_len({len(line_indexes)}). '
                               f'\noriginal code: {original_code}')

    def text_to_instance(self, packed_pdg: Dict) -> Tuple[bool, Instance]:
        raw_code = packed_pdg['raw_code']
        line_edges = packed_pdg['line_edges']
        # original_total_line = packed_pdg['total_line']

        raw_code = self.code_cleaner.clean_code(raw_code)
        tokenized_code = self.code_tokenizer.tokenize(raw_code)
        tokenized_code, token_line_idxes, line_count = self.truncate_and_make_line_index(tokenized_code)
        edge_matrix = self.ctrl_edge_matrix_func(line_edges, line_count)
        # Check whether we need to process the joern-parse token data edges
        if self.hybrid_data_is_processed:
            if self.optimize_data_edge_input_memory:
                data_matrix = self.make_data_edge_matrix_from_processed_optimized(tokenized_code,
                                                                                  packed_pdg['processed_token_data_edges'][self.processed_tokenizer_name])
            else:
                data_matrix = self.make_data_edge_matrix_from_processed(tokenized_code,
                                                                        packed_pdg['processed_token_data_edges'][self.processed_tokenizer_name])
        else:
            if self.optimize_data_edge_input_memory:
                raise NotImplementedError
            token_nodes = packed_pdg['token_nodes']
            token_edges = packed_pdg['token_edges']
            data_matrix = self.make_data_edge_matrix(raw_code, token_nodes, token_edges, tokenized_code, self.multi_vs_multi_strategy)

        # Ignore single-line code samples.
        if line_count == 1:
            return False, Instance({})

        mlm_sampling_weights, _ = self.mlm_sampling_weight_method(raw_code, tokenized_code)

        self._check_data_correctness(raw_code, tokenized_code, token_line_idxes, line_edges)
        fields = {
            'code': TextField(tokenized_code, self.code_token_indexers),
            'line_idxes': TensorField(token_line_idxes),
            'line_ctrl_edges': TensorField(edge_matrix),
            'token_data_edges': TensorField(data_matrix),
            'vertice_num': TensorField(torch.Tensor([line_count])), # num. of line is vertice num.
            'mlm_sampling_weights': TensorField(mlm_sampling_weights),
        }

        span_tags = self.mlm_span_mask_tag_gen_method(raw_code, tokenized_code)
        if span_tags is not None:
            fields['mlm_span_tags'] = TensorField(span_tags)

        return True, Instance(fields)


    def _read(self, dataset_config: Dict) -> Iterable[Instance]:
        from utils import GlobalLogger as logger
        data_base_path = dataset_config['data_base_path']
        volume_range = dataset_config['volume_range']   # close interval

        for vol in range(volume_range[0], volume_range[1]+1):
            logger.info('PackedHybridTokenLineReader.read', f'Reading Vol. {vol}')

            vol_path = os.path.join(data_base_path, f'packed_hybrid_vol_{vol}.pkl')
            packed_vol_data_items = read_dumped(vol_path)
            packed_vol_data_items = packed_vol_data_items[:100] if self.debug else packed_vol_data_items
            for pdg_data_item in tqdm(packed_vol_data_items, desc='from_vol_packed_data'):
                try:
                    ok, instance = self.text_to_instance(pdg_data_item)
                    if ok:
                        self.actual_read_samples += 1
                        yield instance
                # TODO: revert
                except FileNotFoundError as e:
                    logger.error('read', f'error: {e}. \npdg-item content: {pdg_data_item}')

        logger.info('reader', f'Total samples loaded: {self.actual_read_samples}')

if __name__ == '__main__':
    import json
    import _jsonnet
    from utils.dict import overwrite_dict, delete_dict_items
    from utils.allennlp_utils.build_utils import build_dataset_reader_from_dict

    base_config_path = '/data1/zhijietang/temp/vul_temp/config/pretrain_ver_26.jsonnet'
    data_vol_base_path = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_line_token_hybrid_data'

    reader_config = json.loads(_jsonnet.evaluate_file(base_config_path))['dataset_reader']
    reader_config = overwrite_dict(reader_config,
                                   {'type': 'packed_hybrid_line_token_pdg',
                                    'multi_vs_multi_strategy': 'first',
                                    'code_cleaner': {'type': 'trivial'}})
    reader = build_dataset_reader_from_dict(reader_config)
    instances = list(reader.read({
        "data_base_path": data_vol_base_path,
        "volume_range": [0, 1]
    }))
