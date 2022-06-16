from typing import Tuple, Iterable, Dict, List
import os
import re

import torch
from allennlp.data import Tokenizer, TokenIndexer, Instance, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField

from utils.file import read_dumped

@DatasetReader.register('packed_pdg')
class PackedPDGDatasetReader(DatasetReader):
    def __init__(self,
                 code_tokenizer: Tokenizer,
                 code_indexer: TokenIndexer,
                 volume_range: Tuple[int,int],  # closed interval: [a,b]
                 pdg_max_vertice: int,
                 code_max_tokens: int,
                 tokenized_newline_char: str = 'Ċ', # \n after tokenization by CodeBERT
                 **kwargs):
        super().__init__(**kwargs)
        self.code_tokenizer = code_tokenizer
        self.code_token_indexers = {"code_tokens": code_indexer}  # or {"tokens": SingleIdTokenIndexer()}
        self.volume_range = volume_range
        self.pdg_max_vertice = pdg_max_vertice
        self.code_max_tokens = code_max_tokens
        self.tokenized_newline_char = tokenized_newline_char


    def make_edge_matrix(self, edges: List[str], max_vertice: int) -> torch.LongTensor:
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
        matrix = torch.zeros((max_vertice, max_vertice), dtype=torch.long)
        matrix += 4       # default no dependency
        for edge in edges:
            tail, head, etype = re.split(',| ', edge)   # tail/head vertice index start from 1 instead of 0
            matrix[tail, head] = int(etype)
        return matrix


    def get_code_token_line_idx(self, tokens: List[Token]) -> torch.Tensor:
        """
        Determine line index for each token after tokenization.
        Line indexes will be used to aggregate line representation from token representations.
        Indexes and tokens are matched one-by-one.
        """
        line_idxes = []
        current_line = 1        # line_index start from 1, to distinguish from padded zeros
        for token in tokens:
            line_idxes.append(current_line)
            if token.text == self.tokenized_newline_char:
                current_line += 1
        return torch.Tensor(line_idxes)


    def text_to_instance(self, packed_pdg: Dict) -> Instance:
        code = packed_pdg['code']
        edges = packed_pdg['edges']
        vertice_num = packed_pdg['total_line']

        tokenized_code = self.code_tokenizer.tokenize(code)
        token_line_idxes = self.get_code_token_line_idx(tokenized_code)
        edge_matrix = self.make_edge_matrix(edges)

        fields = {
            'code': TextField(tokenized_code, self.code_token_indexers),
            'line_idxes': TensorField(token_line_idxes),
            'edges': TensorField(edge_matrix),
            'vertice_num': TensorField(torch.Tensor([vertice_num])),
        }
        return Instance(fields)


    def _read(self, data_base_path: str) -> Iterable[Instance]:
        for vol in range(self.volume_range[0], self.volume_range[1]+1):
            vol_path = os.path.join(data_base_path, f'vol{vol}')
            for item in os.listdir(vol_path):
                pdg_data_item = read_dumped(os.path.join(vol_path, item))
                yield self.text_to_instance(pdg_data_item)