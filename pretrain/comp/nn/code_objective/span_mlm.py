from typing import Callable, Tuple, Dict, Optional, List

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors

from common.nn.activation_builder import build_activation
from pretrain.comp.nn.code_objective.code_objective import CodeObjective
from pretrain.comp.nn.utils import stat_true_count_in_batch_dim, sample_2D_mask_by_count_along_batch_dim, \
    multinomial_sample_2D_mask_by_count_along_batch_dim
from pretrain.comp.nn.code_objective.mlm import MlmObjective
from utils.pretrain_utils.mlm_span_mask_utils import get_span_mask_from_token_mask


class SpanMlmObjective(MlmObjective):
    def __init__(self,
                 vocab: Vocabulary,
                 code_namespace: str,
                 token_dimension: int,
                 name: str,
                 vocab_size: int = 50265,  # default to "CodeBERT + 1"
                 as_code_embedder: bool = True,
                 token_id_key: str = 'token_ids',
                 tokenizer_type: str = 'codebert',
                 mask_token: str = '<MLM>',
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 loss_coeff: float = 1.,
                 sample_ratio: float = 0.15,  # how many tokens to sample
                 mask_ratio: float = 0.8,  # how many sampled tokens to mask
                 replace_ratio: float = 0.1,  # how many sampled tokens to replace with a random token
                 negative_sampling_k: Optional[int] = None,
                 loss_epoch_range: List[int] = [-1, -1],
                 **kwargs):
        super().__init__(vocab, code_namespace, token_dimension, name,
                         vocab_size, as_code_embedder, token_id_key, tokenizer_type,
                         mask_token, dropout, activation, loss_coeff, sample_ratio,
                         mask_ratio, replace_ratio, negative_sampling_k, loss_epoch_range,
                         **kwargs)

    def random_mask(self,
                    code: TextFieldTensors,
                    mlm_sampling_weights: Optional[torch.Tensor] = None,
                    **kwargs) -> Tuple[TextFieldTensors, torch.Tensor, torch.Tensor]:
        """
        Mask & replace the code input, this is core function of the mlm.
        Differently, this function will sample the whole span of tokens
        if any of the tokens within the span are sampled.
        NOTE: "mlm_span_tags" should appear in "kwargs".

        Return:
        - code: Masked input, with token_ids changed.
        - sampled_mask: Mask to indicate which tokens are masked(replaced).
        - original_sampled_token_ids: Real token id labels of these masked(replaced) tokens, for producing mlm loss.
        """
        mlm_span_tags: torch.Tensor = kwargs['mlm_span_tags']
        assert mlm_span_tags is not None, "MLM span tags are not given in params. Check reader output."

        token_ids = code[self.code_namespace][self.token_id_key]
        candidate_mask = self.get_mask_of_token_to_be_masked(token_ids)

        # Sample token mask
        token_count = stat_true_count_in_batch_dim(candidate_mask)
        sampled_count = (token_count * self.sample_ratio).int()
        sampled_mask = multinomial_sample_2D_mask_by_count_along_batch_dim(candidate_mask, sampled_count, weight=mlm_sampling_weights)

        # Make up span mask after sampling token mask
        sampled_mask = get_span_mask_from_token_mask(mlm_span_tags, sampled_mask)
        original_sampled_token_ids = self._clone_sampled_original_token_ids(token_ids, sampled_mask)

        mask_action_mask, replace_action_mask = self._random_select_mask_action(sampled_mask)
        token_ids = self._mask_tokens(token_ids, mask_action_mask)
        token_ids = self._replace_tokens(token_ids, replace_action_mask)
        code[self.code_namespace][self.token_id_key] = token_ids

        return code, sampled_mask, original_sampled_token_ids
