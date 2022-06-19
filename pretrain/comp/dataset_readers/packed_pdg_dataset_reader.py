from typing import Tuple, Iterable, Dict, List, Optional
import os
import re
from tqdm import tqdm

import torch
from allennlp.data import Tokenizer, TokenIndexer, Instance, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField

from common.code_cleaner import CodeCleaner, TrivialCodeCleaner
from utils.file import read_dumped


@DatasetReader.register('packed_line_pdg')
class PackedLinePDGDatasetReader(DatasetReader):
    def __init__(self,
                 code_tokenizer: Tokenizer,
                 code_indexer: TokenIndexer,
                 volume_range: Tuple[int,int],  # closed interval: [a,b]
                 pdg_max_vertice: int,  # For line-level approach, this should be equal to "max_lines"
                 max_lines: int,
                 code_max_tokens: int,
                 code_cleaner: CodeCleaner = TrivialCodeCleaner(),
                 tokenized_newline_char: str = 'Ċ',  # \n after tokenization by CodeBERT
                 special_tokenizer_token_handler_type: str = 'codebert',
                 only_keep_complete_lines: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.code_tokenizer = code_tokenizer
        self.code_token_indexers = {"code_tokens": code_indexer}  # or {"tokens": SingleIdTokenIndexer()}
        self.volume_range = volume_range
        self.pdg_max_vertice = pdg_max_vertice
        self.max_lines = max_lines
        self.code_max_tokens = code_max_tokens
        self.tokenized_newline_char = tokenized_newline_char
        self.code_cleaner = code_cleaner
        # self.add_EOS_token = add_EOS_token
        # self.skip_first_token_line_index = skip_first_token_line_index
        self.special_tokenizer_token_handler_type = special_tokenizer_token_handler_type
        self.only_keep_complete_lines = only_keep_complete_lines


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
        matrix = torch.zeros((line_count, line_count), dtype=torch.long)
        matrix += 4       # default no dependency
        for edge in edges:
            tail, head, etype = re.split(',| ', edge)   # tail/head vertice index start from 1 instead of 0
            tail, head, etype = int(tail), int(head), int(etype)
            # Ignore uncovered vertices (lines)
            if tail >= line_count or head >= line_count:
                continue
            matrix[tail, head] = etype
        return matrix


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
            line_tokens = line_tokens[:-current_column]
            line_idxes = line_idxes[:-current_column]

        line_tokens, line_idxes = self.post_handle_special_tokenizer_tokens(line_tokens, line_idxes)
        return line_tokens, torch.LongTensor(line_idxes), current_line-1


    def text_to_instance(self, packed_pdg: Dict) -> Instance:
        code = packed_pdg['code']
        edges = packed_pdg['edges']
        original_total_line = packed_pdg['total_line']

        code = self.code_cleaner.clean_code(code)
        tokenized_code = self.code_tokenizer.tokenize(code)
        # tokenized_code = self.drop_tokenizer_special_tokens(tokenized_code)
        tokenized_code, token_line_idxes, line_count = self.truncate_and_make_line_index(tokenized_code)
        edge_matrix = self.make_edge_matrix(edges, line_count)

        fields = {
            'code': TextField(tokenized_code, self.code_token_indexers),
            'line_idxes': TensorField(token_line_idxes),
            'edges': TensorField(edge_matrix),
            'vertice_num': TensorField(torch.Tensor([line_count])), # num. of line is vertice num.
        }
        return Instance(fields)


    def _read(self, data_base_path: str) -> Iterable[Instance]:
        from utils import GlobalLogger as logger

        for vol in range(self.volume_range[0], self.volume_range[1]+1):
            vol_path = os.path.join(data_base_path, f'vol{vol}')
            logger.info('PackedLinePDGDatasetReader.read', f'Reading Vol. {vol}')
            for item in tqdm(os.listdir(vol_path)):
                pdg_data_item = read_dumped(os.path.join(vol_path, item))
                yield self.text_to_instance(pdg_data_item)