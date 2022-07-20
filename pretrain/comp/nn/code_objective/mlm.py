from typing import Callable, Tuple, Dict, Optional, List

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors

from common.nn.activation_builder import build_activation
from pretrain.comp.nn.code_objective.code_objective import CodeObjective
from pretrain.comp.nn.utils import stat_true_count_in_batch_dim, sample_2D_mask_by_count_in_batch_dim


@CodeObjective.register('mlm')
class MlmObjective(CodeObjective):
    def __init__(self,
                 vocab: Vocabulary,
                 code_namespace: str,
                 token_dimension: int,
                 name: str,
                 vocab_size: int = 50265,       # default to "CodeBERT + 1"
                 as_code_embedder: bool = True,
                 token_id_key: str = 'token_ids',
                 tokenizer_type: str = 'codebert',
                 mask_token: str = '<MLM>',
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 loss_coeff: float = 1.,
                 sample_ratio: float = 0.15,    # how many tokens to sample
                 mask_ratio: float = 0.8,       # how many sampled tokens to mask
                 replace_ratio: float = 0.1,    # how many sampled tokens to replace with a random token
                 negative_sampling_k: Optional[int] = None,
                 loss_epoch_range: List[int] = [-1, -1],
                 **kwargs):
        super().__init__(vocab,
                         name=name,
                         loss_coeff=loss_coeff,
                         as_code_embedder=as_code_embedder,
                         forward_from_where='token',
                         loss_epoch_range = loss_epoch_range,
                         **kwargs)
        self.vocab = vocab
        self.code_namespace = code_namespace
        self.dense = torch.nn.Linear(token_dimension, token_dimension)
        self.output_weight = torch.nn.Linear(token_dimension, vocab_size, bias=False)

        self.dropout = dropout
        self.activation = build_activation(activation)
        self.sample_ratio = sample_ratio
        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio
        self.token_id_key = token_id_key
        self.tokenizer_type = tokenizer_type
        self.mask_token = mask_token
        self.negative_sampling_k = negative_sampling_k

        # self.mask_token_id = self.vocab.get_token_index(mask_token, self.code_namespace)


    def get_mask_of_token_to_be_masked(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.tokenizer_type == 'codebert':
            return token_ids.gt(2)
        else:
            raise NotImplementedError(f"Tokenizer type: {self.tokenizer_type}")

    def random_select_mask_action(self, sampled_mask: torch.Tensor):
        sampled_indexes = sampled_mask.nonzero()
        sampled_size = sampled_indexes.size(0)
        action_dist = torch.rand((sampled_size,), device=sampled_mask.device) #.unsqueeze(-1).repeat(2)

        mask_action_positions = action_dist.lt(self.mask_ratio)
        replace_action_positions = action_dist.gt(self.mask_ratio) & \
                                   action_dist.lt(self.mask_ratio + self.replace_ratio)

        # Here we assume that mask is in 2D shape, namely: [batch, len].
        mask_action_indexes = torch.masked_select(sampled_indexes, mask_action_positions.unsqueeze(-1)).view(-1, 2)
        replace_action_indexes = torch.masked_select(sampled_indexes, replace_action_positions.unsqueeze(-1)).view(-1, 2)

        # Refill the mask matrix of mask and replace action.
        mask_action_mask = torch.zeros_like(sampled_mask, device=sampled_mask.device, dtype=torch.bool)
        mask_action_mask[mask_action_indexes[:,0],mask_action_indexes[:,1]] = True
        replace_action_mask = torch.zeros_like(sampled_mask, device=sampled_mask.device, dtype=torch.bool)
        replace_action_mask[replace_action_indexes[:,0], replace_action_indexes[:,1]] = True

        return mask_action_mask, replace_action_mask


    def mask_tokens(self, token_ids: torch.Tensor, mask: torch.Tensor):
        mask_token_id = self.vocab.get_token_index(self.mask_token, self.code_namespace)
        token_ids = token_ids.masked_fill(mask, mask_token_id)
        return token_ids

    def replace_tokens(self, token_ids: torch.Tensor, mask: torch.Tensor):
        replace_idxes = mask.nonzero()
        replaced_token_ids = self._sample_tokens(size=replace_idxes.size(0),
                                                 device=token_ids.device)
        token_ids[replace_idxes[:,0], replace_idxes[:,1]] = replaced_token_ids
        return token_ids

    def _sample_tokens(self, size, device):
        if self.tokenizer_type == 'codebert':
            # Exclude <s>, </s>, <pad> and <MLM> tokens.
            low, high = 3, self.vocab.get_vocab_size(self.code_namespace)
        else:
            raise NotImplementedError

        return torch.randint(low, high, (size,), device=device, dtype=torch.long)


    def clone_sampled_original_token_ids(self, token_ids, sampled_mask):
        sampled_idxes = sampled_mask.nonzero()
        cloned = token_ids[sampled_idxes[:,0], sampled_idxes[:,1]].clone()
        return cloned


    def random_mask(self, code: TextFieldTensors) -> Tuple[TextFieldTensors, torch.Tensor, torch.Tensor]:
        token_ids = code[self.code_namespace][self.token_id_key]
        candidate_mask = self.get_mask_of_token_to_be_masked(token_ids)

        token_count = stat_true_count_in_batch_dim(candidate_mask)
        sampled_count = (token_count * self.sample_ratio).int()
        sampled_mask = sample_2D_mask_by_count_in_batch_dim(candidate_mask, sampled_count)
        original_sampled_token_ids = self.clone_sampled_original_token_ids(token_ids, sampled_mask)


        mask_action_mask, replace_action_mask = self.random_select_mask_action(sampled_mask)
        token_ids = self.mask_tokens(token_ids, mask_action_mask)
        token_ids = self.replace_tokens(token_ids, replace_action_mask)
        code[self.code_namespace][self.token_id_key] = token_ids

        return code, sampled_mask, original_sampled_token_ids


    def mlm_loss(self, pred_logits, label, epoch):
        """
        Produce MLM loss based on predicted logits and real token ids of masked tokens.
        Negative sampling can be done here to mask some negative items when computing
        probability with softmax().
        :param pred_logits: Logits of masked tokens. Shape: [sampled_size, vocab_size]
        :param label: Original token ids of masked tokens. Shape: [sampled_size,]
        """
        device = pred_logits.device
        sampled_size, vocab_size = pred_logits.size(0), pred_logits.size(1)

        # Maybe do negative sampling.
        if self.negative_sampling_k is not None:
            mask = torch.zeros_like(pred_logits, dtype=torch.bool, device=device)
            first_dim_idx = torch.arange(0,sampled_size, device=device)
            # First, we should ensure the label indexes are not masked.
            mask[first_dim_idx, label] = True

            # Then, for each row (predicted word), we uniformly sample K negative word.
            # We use uniform negative sampling in default, but it can be improved using a
            # frequency-related distribution instead.
            # [NOTE]: Since the multinomial sampling does not exclude the index of label,
            #         the actual sampled negative samples may be K-1 sometimes!
            neg_sampled_indexes = torch.multinomial(torch.ones_like(pred_logits, device=device), self.negative_sampling_k, replacement=False)
            mask = mask.scatter(1, neg_sampled_indexes, True)

            # -----------------------------------------------------------------------------
            # WARN: THE IMPLEMENTATION BELOW MAY BE VERY SLOW (3~4 s/it).
            # -----------------------------------------------------------------------------
            # count = torch.ones((sampled_size,), dtype=torch.long) * self.negative_sampling_k
            # neg_sampled_mask = sample_2D_mask_by_count_in_batch_dim(~mask, count)
            # mask = mask | neg_sampled_mask
            # -----------------------------------------------------------------------------

            # Finally, we set the logits of masked positions to be -INF.
            pred_logits = pred_logits.masked_fill(~mask, float('-inf'))

        # [BugFix 6.28]
        # F.cross_entropy will call softmax itself to compute probability distribution,
        # thus it is of no need to call it again to scale logits smaller.
        # pred_probs = pred_logits.softmax(dim=-1)
        loss  = F.cross_entropy(pred_logits, label)
        loss = loss * self.loss_coeff
        loss = self.rectify_loss_based_on_range(loss, epoch)
        return loss


    def forward(self, **kwargs) -> Dict:
        return self.forward_from_token(**kwargs)


    def forward_from_token(self,
                           code: TextFieldTensors,
                           code_embed_func: Callable,
                           epoch: int,
                           **kwargs) -> Dict:
        code, sampled_mask, original_sampled_token_ids = self.random_mask(code)
        code_embed_outputs = code_embed_func(code)
        code_embeddings = code_embed_outputs['outputs']

        # Forward MLM inner layers.
        sampled_idxes = sampled_mask.nonzero()
        sampled_code_embeddings = code_embeddings[sampled_idxes[:, 0], sampled_idxes[:, 1], :]
        sampled_code_embeddings = F.dropout(sampled_code_embeddings, self.dropout)
        sampled_code_embeddings = self.activation(self.dense(sampled_code_embeddings))
        sampled_code_embeddings = F.dropout(sampled_code_embeddings, self.dropout)
        sampled_code_outputs = self.output_weight(sampled_code_embeddings)

        mlm_loss = self.mlm_loss(sampled_code_outputs, original_sampled_token_ids, epoch)
        output_dict =  {'loss': mlm_loss}
        output_dict.update(code_embed_outputs)
        return output_dict

