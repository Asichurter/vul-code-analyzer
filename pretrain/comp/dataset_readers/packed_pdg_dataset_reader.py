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


@DatasetReader.register('packed_line_pdg')
class PackedLinePDGDatasetReader(DatasetReader):
    def __init__(self,
                 code_tokenizer: Tokenizer,
                 code_indexer: TokenIndexer,
                 # volume_range: Tuple[int,int],  # closed interval: [a,b]
                 pdg_max_vertice: int,  # For line-level approach, this should be equal to "max_lines"
                 max_lines: int,
                 code_max_tokens: int,
                 code_namespace: str = "code_tokens",
                 code_cleaner: CodeCleaner = TrivialCodeCleaner(),
                 tokenized_newline_char: str = 'Ċ',  # \n after tokenization by CodeBERT
                 special_tokenizer_token_handler_type: str = 'codebert',
                 only_keep_complete_lines: bool = True,
                 unified_label: bool = True,
                 from_raw_data: bool = True,
                 mlm_sampling_weight_strategy: str = 'uniform',
                 mlm_span_mask_strategy: str = 'none',
                 packed_mode: str = 'normal',
                 packed_file_temp: str = 'packed_vol_{}.pkl',
                 model_mode: Optional[str] = None,
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
        self.unified_label = unified_label
        self.from_raw_data = from_raw_data
        self.mlm_sampling_weight_method = dispatch_mlm_weight_gen_method(mlm_sampling_weight_strategy)
        self.mlm_span_mask_tag_gen_method = dispatch_mlm_span_mask_tag_method(mlm_span_mask_strategy)

        # For later packed dataset capability
        assert packed_mode in ['normal', 'hybrid_processed']
        self.packed_mode = packed_mode
        self.packed_file_temp = packed_file_temp
        self.model_mode = model_mode

        self.actual_read_samples = 0


    def make_edge_matrix(self,
                         edges: List[str],
                         line_count: int) -> torch.LongTensor:
        """
        Make line-level dependency matrix from edge data.
        Edge type id:
            # 1: Data dependency
            # 2: Control dependency
            # 3: Both data and control dependency
            # 4: No dependency
        :param edges:
        :param max_vertice:
        :return:
        """
        # To cover the last line (line_count-th), we have to allocate one more line here.
        if self.unified_label:
            matrix = torch.zeros((line_count+1, line_count+1), dtype=torch.long)
            matrix += 4  # default no dependency
        else:
            matrix = torch.zeros((2, line_count+1, line_count+1))
            matrix += 1  # default no dependency

        for edge in edges:
            tail, head, etype = re.split(',| ', edge)   # tail/head vertice index start from 1 instead of 0
            tail, head, etype = int(tail), int(head), int(etype)
            # Ignore uncovered vertices (lines)
            if tail > line_count or head > line_count:
                continue
            if self.unified_label:
                matrix[tail, head] = etype
            else:
                if etype == 3 or etype == 1:
                    matrix[0, tail, head] = 2   # 2 refers to edge
                if etype == 3 or etype == 2:
                    matrix[1, tail, head] = 2

        # Drop 0-th row and column, since line index starts from 1.
        if self.unified_label:
            return matrix[1:, 1:]
        else:
            return matrix[:, 1:, 1:]


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


    def text_to_instance(self, packed_pdg: Dict) -> Tuple[bool, Instance]:
        if self.packed_mode == 'normal':
            code = packed_pdg['code']
            edges = packed_pdg['edges']
            original_total_line = packed_pdg['total_line']
        elif self.packed_mode == 'hybrid_processed':
            code = packed_pdg['raw_code']
            edges = packed_pdg['line_edges']
        else:
            raise ValueError

        code = self.code_cleaner.clean_code(code)
        tokenized_code = self.code_tokenizer.tokenize(code)
        tokenized_code, token_line_idxes, line_count = self.truncate_and_make_line_index(tokenized_code)
        edge_matrix = self.make_edge_matrix(edges, line_count)
        # Ignore single-line code samples.
        if line_count == 1:
            return False, Instance({})

        mlm_sampling_weights, _ = self.mlm_sampling_weight_method(code, tokenized_code)

        check_pretrain_code_field_correctness(self.special_tokenizer_token_handler_type, code, tokenized_code, token_line_idxes, edges)
        fields = {
            'code': TextField(tokenized_code, self.code_token_indexers),
            'line_idxes': TensorField(token_line_idxes),
            'edges': TensorField(edge_matrix),
            'vertice_num': TensorField(torch.Tensor([line_count])), # num. of line is vertice num.
            'mlm_sampling_weights': TensorField(mlm_sampling_weights),
        }

        span_tags = self.mlm_span_mask_tag_gen_method(code, tokenized_code)
        if span_tags is not None:
            fields['mlm_span_tags'] = TensorField(span_tags)

        return True, Instance(fields)


    def _read(self, dataset_config: Dict) -> Iterable[Instance]:
        from utils import GlobalLogger as logger
        data_base_path = dataset_config['data_base_path']
        volume_range = dataset_config['volume_range']   # close interval

        for vol in range(volume_range[0], volume_range[1]+1):
            logger.info('PackedLinePDGDatasetReader.read', f'Reading Vol. {vol}')

            # From raw data, use "self.text_to_instance"
            if self.from_raw_data:
                vol_path = os.path.join(data_base_path, f'vol{vol}')
                for item in tqdm(os.listdir(vol_path), desc='from_raw_data'):
                    try:
                        pdg_data_item = read_dumped(os.path.join(vol_path, item))
                        ok, instance = self.text_to_instance(pdg_data_item)
                        if ok:
                            self.actual_read_samples += 1
                            yield instance
                    except Exception as e:
                        logger.error('read', f'file path: {os.path.join(vol_path, item)}, error: {str(e)}')

            # From packed volume data pkl, also use "self.text_to_instance"
            else:
                vol_path = os.path.join(data_base_path, self.packed_file_temp.format(vol))
                packed_vol_data_items = read_dumped(vol_path)
                for pdg_data_item in tqdm(packed_vol_data_items, desc='from_vol_packed_data'):
                    try:
                        ok, instance = self.text_to_instance(pdg_data_item)
                        if ok:
                            self.actual_read_samples += 1
                            yield instance
                    except Exception as e:
                        logger.error('read', f'Error: {e}, pdg-item content: {pdg_data_item}')

        logger.info('reader', f'Total {self.actual_read_samples} instances loaded')

    def text_to_dict(self, packed_pdg: Dict) -> Tuple[bool, Dict]:
        code = packed_pdg['code']
        edges = packed_pdg['edges']
        original_total_line = packed_pdg['total_line']

        code = self.code_cleaner.clean_code(code)
        tokenized_code = self.code_tokenizer.tokenize(code)
        tokenized_code, token_line_idxes, line_count = self.truncate_and_make_line_index(tokenized_code)
        edge_matrix = self.make_edge_matrix(edges, line_count)
        # Ignore single-line code samples.
        if line_count == 1:
            return False, {}

        check_pretrain_code_field_correctness(self.special_tokenizer_token_handler_type, code, tokenized_code, token_line_idxes, edges)
        data_dict = {
            'code': tokenized_code,
            'line_idxes': token_line_idxes,
            'edges': edge_matrix,
            'vertice_num': line_count
        }
        return True, data_dict


    def read_as_json(self, dataset_config: Dict) -> Iterable[Dict]:
        from utils import GlobalLogger as logger
        data_base_path = dataset_config['data_base_path']
        volume_range = dataset_config['volume_range']   # close interval

        for vol in range(volume_range[0], volume_range[1]+1):
            vol_path = os.path.join(data_base_path, f'vol{vol}')
            logger.info('PackedLinePDGDatasetReader.read_as_json', f'Reading Vol. {vol}')
            for item in tqdm(os.listdir(vol_path)):
                try:
                    # logger.info('read_as_json', os.path.join(vol_path, item))
                    pdg_data_item = read_dumped(os.path.join(vol_path, item))
                    ok, instance = self.text_to_dict(pdg_data_item)
                    if ok:
                        self.actual_read_samples += 1
                        yield instance
                except Exception as e:
                    logger.error('read', f'file path: {os.path.join(vol_path, item)}, error: {str(e)}')


    def dict_to_instance(self, packed_data: Dict) -> Instance:
        tokenized_code = packed_data['code']
        token_line_idxes = packed_data['line_idxes']
        edge_matrix = packed_data['edges']
        line_count = packed_data['vertice_num']
        fields = {
            'code': TextField(tokenized_code, self.code_token_indexers),
            'line_idxes': TensorField(token_line_idxes),
            'edges': TensorField(edge_matrix),
            'vertice_num': TensorField(torch.Tensor([line_count])), # num. of line is vertice num.
        }
        return Instance(fields)