from typing import List, Tuple, Optional, Iterable

import torch
from allennlp.data import Token, Vocabulary
from itertools import chain

codebert_families = ['codebert']
unixcoder_families = ['unixcoder_base', 'unixcoder_base_nine']
microsoft_pretrain_families = codebert_families + unixcoder_families
no_operation_types = ['pretrained_bpe']

def pre_handle_special_tokenizer_tokens(special_tokenizer_token_handler_type: str,
                                        tokens: List[Token],
                                        line_idxes: Optional[List] = None) -> List[Token]:
    """
        Truncate special tokens, leave valid tokens only.
    """
    if special_tokenizer_token_handler_type in microsoft_pretrain_families:
        return tokens[1:-1]
    else:
        return tokens

_unixcoder_modes = ["<encoder-decoder>", "<encoder-only>", "<decoder-only>"]

def make_unixcoder_input(token_input_list: Iterable[List[Token]], mode: str):
    assert mode is not None and mode in _unixcoder_modes
    # Format reference: https://github.com/microsoft/CodeBERT/issues/178
    tokens = [Token('<s>'), Token(mode), Token('</s>')]
    for token_list in token_input_list:
        tokens.extend(token_list + [Token('</s>')])
    return tokens

def post_handle_special_tokenizer_tokens(special_tokenizer_token_handler_type: str,
                                         token_inputs: Iterable[List[Token]],
                                         line_idxes: Optional[List] = None,
                                         mode: Optional[str] = None) -> Tuple:
    """
        Revert special tokens to make input for pre-trained model.
    """
    if special_tokenizer_token_handler_type in codebert_families:
        tokens = [Token('<s>')]
        for token_list in token_inputs:
            tokens.extend(token_list + [Token('</s>')])
    elif special_tokenizer_token_handler_type in unixcoder_families:
        tokens = make_unixcoder_input(token_inputs, mode)
    elif special_tokenizer_token_handler_type in no_operation_types:
        tokens = []
        for token_list in token_inputs:
            tokens.extend(token_list)
    else:
        raise NotImplementedError

    return tokens, line_idxes

def drop_tokenizer_special_tokens(drop_tokenizer_special_token_type: str, embedded_code, code_mask):
    tokenizer_type = drop_tokenizer_special_token_type.lower()
    # For CodeBERT, drop <s> and </s> (first and last token)
    if tokenizer_type in codebert_families:
        # TODO: Here exists a potential bug:
        #       When code is not full of the max len and padded with <pad>,
        #       the last token may be <pad> but not </s> we want to drop.
        #       However, avg line extraction works fine with this since </s> will be
        #       regarded as <pad> and scattered to 0-th row to drop.
        return embedded_code[:,1:-1], code_mask[:,1:-1]
    # For UnixCoder, drop <s>, mode, </s> and last </s> (first 3 + last 1)
    elif tokenizer_type in unixcoder_families:
        return embedded_code[:, 3:-1], code_mask[:, 3:-1]
    else:
        raise NotImplementedError(f"Not supported type for drop_tokenizer_special_tokens: {drop_tokenizer_special_token_type}")


def sample_replace_tokens(tokenizer_type: str,
                          vocab: Vocabulary,
                          code_namespace: str,
                          size: int,
                          device) -> torch.LongTensor:
    """
        Used for generating replacing tokens of part of sampled tokens during MLM task.
    """
    if tokenizer_type == 'codebert':
        # Exclude <s>, </s>, <pad> and <MLM> tokens.
        low, high = 3, vocab.get_vocab_size(code_namespace)
    elif tokenizer_type in unixcoder_families:
        # For unixcoder, first 119 items are special tokens, tokens after #49999 are ast tokens.
        low, high = 119, 49999
    else:
        raise NotImplementedError

    return torch.randint(low, high, (size,), device=device, dtype=torch.long)

def get_mask_of_token_to_be_masked(tokenizer_type: str, token_ids: torch.Tensor) -> torch.Tensor:
    if tokenizer_type in codebert_families:
        return token_ids.gt(2)
    elif tokenizer_type in unixcoder_families:
        return token_ids.gt(118) & token_ids.lt(50000)
    else:
        raise NotImplementedError(f"Tokenizer type: {tokenizer_type}")