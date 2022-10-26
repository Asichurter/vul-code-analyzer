from typing import Optional, List, Tuple

from allennlp.data import Token
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder, PretrainedTransformerMismatchedEmbedder

from utils import GlobalLogger as mylogger

@Tokenizer.register('pretrained_mismatch_truncate')
class PretrainedMismatchTruncateTokenizer(Tokenizer):
    def __init__(self,
                 pretrained_tokenizer: PretrainedTransformerTokenizer,
                 max_word_piece_len: int = 512,
                 raw_token_tokenizer: Optional[Tokenizer] = None,
                 start_token: Optional[str] = None,
                 binsearch_check_max_step: int = 40,
                 **kwargs):
        self.pretrained_tokenizer = pretrained_tokenizer
        self.max_word_piece_len = max_word_piece_len
        self.raw_token_tokenizer = raw_token_tokenizer
        self.start_token = start_token
        self.binsearch_check_max_step = binsearch_check_max_step

    def tokenize_raw_tokens(self, text: str) -> List[str]:
        if self.raw_token_tokenizer is None:
            return text.split()
        else:
            tokens = self.raw_token_tokenizer.tokenize(text)
            return [t.text for t in tokens]

    def maybe_add_start_token(self, raw_tokens: List[str]):
        if self.start_token is None:
            return raw_tokens
        else:
            return [self.start_token] + raw_tokens

    def binsearch_max_len_offset_range(self, intra_offsets: List[Tuple[int,int]]) -> int:
        # Empty token list
        if len(intra_offsets) == 0:
            return 2
        max_len = self.max_word_piece_len
        # Check if out of range
        if intra_offsets[-1][-1] < max_len:
            return len(intra_offsets) + 1

        offset_len = len(intra_offsets)
        left_index, right_index = 0, offset_len-1
        # No need to search
        if right_index == 0:
            return 2

        binsearch_step = 0
        while True:
            if left_index == right_index:
                return left_index   # todo: Check this returned index

            mid_index = (left_index + right_index) // 2
            binsearch_step += 1
            if binsearch_step >= self.binsearch_check_max_step:
                mylogger.warning('binsearch_offset',
                                 f'binsearch step={binsearch_step}, index_range={intra_offsets[mid_index]}, target={max_len} and offsets={intra_offsets}')

            # print(f'index={mid_index}, index_range={intra_offsets[mid_index]}, target={max_len} and offsets={intra_offsets}')
            left_bound, right_bound = intra_offsets[mid_index]
            left_check = max_len >= left_bound
            right_check = max_len <= right_bound
            if left_check and right_check:
                return mid_index
            elif left_check and not right_check:    # Look rightward
                if left_index == mid_index:
                    left_index += 1             # Take care when index and len are adjacent to avoid infinite loop
                else:
                    left_index = mid_index
            elif not left_check and right_check:    # Look leftward
                right_index = mid_index
            else:
                assert False, f"bin range search failed with target={max_len} and offsets={intra_offsets}"


    def tokenize(self, text: str) -> List[Token]:
        raw_tokens = self.tokenize_raw_tokens(text)
        raw_tokens = self.maybe_add_start_token(raw_tokens)
        intra_word_pieces, intra_offsets = self.pretrained_tokenizer.intra_word_tokenize(raw_tokens)
        boundary_token_index = self.binsearch_max_len_offset_range(intra_offsets)
        tokenized_raw_tokens = raw_tokens[:boundary_token_index-1]
        return [Token(t) for t in tokenized_raw_tokens]


