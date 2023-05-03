import re
from typing import List, Tuple, Optional

import torch
from allennlp.data import Token

from utils.allennlp_utils.tokenizer_vocab_sensitive_utils import pre_handle_special_tokenizer_tokens, \
                                                                 post_handle_special_tokenizer_tokens


def truncate_tokens_and_make_line_index(raw_code: str,
                                        max_lines: int,
                                        max_tokens: int,
                                        tokens: List[Token],
                                        tokenizer_type: str,
                                        mode: Optional[str] = None,
                                        only_keep_complete_lines: bool = True) -> Tuple[List[Token], torch.Tensor, int]:
    """
    Truncate code tokens based on max_lines and max_tokens and determine line index for each token after tokenization.
    Line indexes (2D) will be used to aggregate line representation from token representations.
    Indexes and tokens are matched one-by-one.

    This version has fixed the problem of sticky multiple new line token, which incurs shifted line indices.

    Return:
        - Line indices.
        - truncated tokens.
        - Line count.
    """
    line_idxes = []
    line_tokens = []
    current_line = 1  # line_index start from 1, to distinguish from padded zeros
    current_column = -1
    current_nl_idx = 0
    tokens = pre_handle_special_tokenizer_tokens(tokenizer_type, tokens)

    # Find the char indices of new-lines
    new_line_indices = []
    for m in re.finditer('\n', raw_code):
        new_line_indices.append(m.start())
    # Add a dummy nl at last to avoid out-of-bound
    new_line_indices.append(1e10)

    early_break = False
    for i, token in enumerate(tokens):
        # Update current line based on char span interleaving
        # while token.idx is not None and token.idx_end-1 > new_line_indices[current_nl_idx]:
        while token.idx is not None and token.idx > new_line_indices[current_nl_idx]:
            current_line += 1
            current_column = -1     # Add one hereafter to be zero
            current_nl_idx += 1

        current_column += 1
        line_idxes.append([current_line, current_column])  # 2D line-column index
        line_tokens.append(token)

        # truncate code tokens if exceeding max_lines or max_tokens
        # NOTE: Since post-handle may not be the invert operation of pre-handle, the number of
        #       max tokens here may be slightly different from the given number.
        if current_line > max_lines or len(line_tokens) == max_tokens:
            break

    deleted_line = 0
    if only_keep_complete_lines and early_break:
        # FixBug: Empty tokens when 'current_column' is 0 or 'current_line' is 1.
        if current_column > 0 and current_line > 1:
            line_tokens = line_tokens[:-current_column]
            line_idxes = line_idxes[:-current_column]
            deleted_line = 1

    line_tokens, line_idxes = post_handle_special_tokenizer_tokens(tokenizer_type,
                                                                   (line_tokens,), line_idxes,
                                                                   mode=mode)
    return line_tokens, torch.LongTensor(line_idxes), current_line - deleted_line


def truncate_tokens_and_make_line_index_fixed(raw_code: str,
                                              max_lines: int,
                                              max_tokens: int,
                                              tokens: List[Token],
                                              tokenizer_type: str,
                                              mode: Optional[str] = None,
                                              only_keep_complete_lines: bool = True) -> Tuple[List[Token], torch.Tensor, int]:
    """
    Fix the issue where line indices are not correct meeting multiple sticky nl.
    """
    line_idxes = []
    line_tokens = []
    current_line = 1  # line_index start from 1, to distinguish from padded zeros
    current_column = -1
    current_nl_idx = 0
    tokens = pre_handle_special_tokenizer_tokens(tokenizer_type, tokens)

    # Find the char indices of new-lines
    new_line_indices = []
    for m in re.finditer('\n', raw_code):
        new_line_indices.append(m.start())
    # Add a dummy nl at last to avoid out-of-bound
    new_line_indices.append(1e10)

    early_break = False
    for i, token in enumerate(tokens):
        # Update current line based on char span interleaving
        # [Fix] Use idx_end to check.
        while token.idx is not None and token.idx_end-1 > new_line_indices[current_nl_idx]:
            current_line += 1
            current_column = -1     # Add one hereafter to be zero
            current_nl_idx += 1

        current_column += 1
        line_idxes.append([current_line, current_column])  # 2D line-column index
        line_tokens.append(token)

        # truncate code tokens if exceeding max_lines or max_tokens
        # NOTE: Since post-handle may not be the invert operation of pre-handle, the number of
        #       max tokens here may be slightly different from the given number.
        if current_line > max_lines or len(line_tokens) == max_tokens:
            break

    deleted_line = 0
    if only_keep_complete_lines and early_break:
        # FixBug: Empty tokens when 'current_column' is 0 or 'current_line' is 1.
        if current_column > 0 and current_line > 1:
            line_tokens = line_tokens[:-current_column]
            line_idxes = line_idxes[:-current_column]
            deleted_line = 1

    line_tokens, line_idxes = post_handle_special_tokenizer_tokens(tokenizer_type,
                                                                   (line_tokens,), line_idxes,
                                                                   mode=mode)
    return line_tokens, torch.LongTensor(line_idxes), current_line - deleted_line

if __name__ == '__main__':
    code = 'rand();\nint a = 1;\nprint(a);\n\n'
