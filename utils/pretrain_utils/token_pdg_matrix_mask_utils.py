from typing import List
import torch
from allennlp.data import Token

from utils.pretrain_utils.lexer_based_token_analyse_utils import lexer_match_tokens_and_intersect_allennlp_tokens

def _none_token_mask(raw_code: str, tokens: List[Token], **kwargs) -> torch.Tensor:
    mask = torch.ones(len(tokens),)
    return mask

_token_name_types = ['Token.Name']

def _token_name_token_mask(raw_code: str, tokens: List[Token], **kwargs) -> torch.Tensor:
    mask = torch.zeros(len(tokens),)
    unmasked_indices = lexer_match_tokens_and_intersect_allennlp_tokens(raw_code, tokens, _token_name_types, is_filtered_list=False)
    mask[unmasked_indices] = 1
    return mask

def noise_sampling_wrapper(raw_func):
    def _wrapped_func(raw_code: str, tokens: List[Token], **kwargs):
        mask = raw_func(raw_code, tokens, **kwargs)
        # Sample noise tokens from masked tokens.
        # Implemented in a dynamic noise ratio feeding manner.
        if 'noise_ratio' in kwargs:
            noise_ratio = kwargs['noise_ratio']
            masked_indices = (~(mask.bool())).nonzero().squeeze(-1)
            rand_probs = torch.rand(len(masked_indices), )
            masked_noise_indices = torch.masked_select(masked_indices, rand_probs.lt(noise_ratio))
            mask[masked_noise_indices] = 1
        return mask
    return _wrapped_func

_token_mask_method_dispatcher = {
    'token_name': _token_name_token_mask,
    'token_name_noise': noise_sampling_wrapper(_token_name_token_mask),

    'none': _none_token_mask,
    None: _none_token_mask,
}

from utils import GlobalLogger as mylogger

def dispatch_token_mask_method(method_name: str):
    if method_name not in _token_mask_method_dispatcher:
        mylogger.warning('dispatch_token_mask_method',
                         f'method {method_name} not in accepted list: {_token_mask_method_dispatcher.keys()}')
    return _token_mask_method_dispatcher.get(method_name, _none_token_mask)
