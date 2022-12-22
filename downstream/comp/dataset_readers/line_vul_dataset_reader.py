import torch
from typing import Iterable, Dict, Optional, List, Tuple
from tqdm import tqdm
import logging

from allennlp.data import Instance, Tokenizer, TokenIndexer, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField

from common.modules.code_cleaner import CodeCleaner, PreLineTruncateCodeCleaner
from utils.allennlp_utils.tokenizer_vocab_sensitive_utils import pre_handle_special_tokenizer_tokens, \
    post_handle_special_tokenizer_tokens
from utils.downstream_utils.tokenize_utils import downstream_tokenize
from utils.file import read_dumped

logger = logging.getLogger(__name__)

@DatasetReader.register('line_vul_detect_base')
class LineVulDetectBaseDatasetReader(DatasetReader):
    def __init__(self,
                 code_tokenizer: Tokenizer,
                 code_indexer: TokenIndexer,
                 code_max_tokens: int,
                 max_lines: int,
                 code_namespace: str = "code_tokens",
                 code_cleaner: CodeCleaner = PreLineTruncateCodeCleaner(200),  # Pre-truncate lines to prevent long time waited
                 tokenizer_type: str = 'codebert',
                 model_mode: Optional[str] = None,
                 tokenized_newline_char: str = 'Ċ',  # \n after tokenization by CodeBERT
                 only_keep_complete_lines: bool = True,
                 debug: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.code_tokenizer = code_tokenizer
        self.code_indexers = {code_namespace: code_indexer}
        self.code_max_tokens = code_max_tokens
        self.max_lines = max_lines
        self.code_cleaner = code_cleaner
        self.tokenizer_type = tokenizer_type
        self.model_mode = model_mode
        self.tokenized_newline_char = tokenized_newline_char
        self.only_keep_complete_lines = only_keep_complete_lines

        self.total_loaded_instance = 0
        self.debug = debug

    def truncate_and_make_line_index(self, tokens: List[Token]) -> Tuple[List[Token],torch.Tensor,int]:
        """
        Truncate code tokens based on max_lines and max_tokens and determine line index for each token after tokenization.
        Line indexes (2D) will be used to aggregate line representation from token representations.
        Indexes and tokens are matched one-by-one.

        This method is borrowed from "pretrain.packed_hybrid_line_token_pdg_dataset_reader".
        """
        line_idxes = []
        line_tokens = []
        current_line = 1        # line_index start from 1, to distinguish from padded zeros
        current_column = 0
        tokens = pre_handle_special_tokenizer_tokens(self.tokenizer_type, tokens)

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

        line_tokens, line_idxes = post_handle_special_tokenizer_tokens(self.tokenizer_type, (line_tokens,), line_idxes,
                                                                       mode=self.model_mode)
        return line_tokens, torch.LongTensor(line_idxes), current_line-1

    def _make_flaw_line_ground_truth(self, flaw_lines: List[int], line_count: int):
        # todo: Check the boundary of line count below
        vul_line_ground_truth_tensor = [1 if i in flaw_lines else 0 for i in range(line_count)]
        # No vul lines are included
        if sum(vul_line_ground_truth_tensor) == 0:
            return None
        else:
            return torch.LongTensor(vul_line_ground_truth_tensor)

    def text_to_instance(self, data_item: Dict) -> Tuple[bool,Optional[Instance]]:
        code = data_item['code']
        vul = data_item['target']
        flaw_line_index = data_item['flaw_line_index']
        # Skip non-vul or empty flaw_line instances
        if vul != 1 or flaw_line_index is None:
            return False, None
        else:
            flaw_lines = [int(idx) for idx in flaw_line_index.split(',')]

        code = self.code_cleaner.clean_code(code)
        code_tokens = self.code_tokenizer.tokenize(code)
        tokenized_code, line_indices, line_count = self.truncate_and_make_line_index(code_tokens)
        # tokenized_code = downstream_tokenize(self.code_tokenizer, code, self.tokenizer_type, self.model_mode)

        # Make and check vul lines, fail if no vul lines are included
        vul_line_ground_truth_tensor = self._make_flaw_line_ground_truth(flaw_lines, line_count)
        if vul_line_ground_truth_tensor is None:
            return False, None

        fields = {
            'code': TextField(tokenized_code, self.code_indexers),
            'line_vul_labels': TensorField(vul_line_ground_truth_tensor),
            'line_indices': TensorField(torch.LongTensor(line_indices)),
            'line_counts': TensorField(torch.LongTensor([line_count])),
        }
        return True, Instance(fields)


    def _read(self, file_path) -> Iterable[Instance]:
        cur_loaded_count = self.total_loaded_instance
        data = read_dumped(file_path)
        data = data[:150] if self.debug else data
        for item in tqdm(data):
            ok, instance =  self.text_to_instance(item)
            if ok:
                self.total_loaded_instance += 1
                yield instance

        logger.info(f"Total {self.total_loaded_instance - cur_loaded_count} loaded by reader")