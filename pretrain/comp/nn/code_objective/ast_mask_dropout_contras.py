from typing import Dict, Optional, List, Callable

import torch
from allennlp.data import Vocabulary, TextFieldTensors

from common.nn.pooler import Pooler
from pretrain.comp.nn.code_objective.code_objective import CodeObjective
from utils.allennlp_utils.tokenizer_vocab_sensitive_utils import drop_tokenizer_special_tokens

from utils import GlobalLogger as mylogger

@CodeObjective.register("ast_mask_dropout_contras")
class ASTMaskDropoutContrastiveLearning(CodeObjective):
    """
    The MCL of UniXCoder, using AST sequence as input.

    Token embeddings will first be pooled to a fixed-length code representation, different dropouts
    will be used to generate positive pairs, while other codes will be used as negative pairs.

    Note the dropout is working during feature forwarding.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 name: str,
                 token_pooler: Pooler,
                 tokenizer_type: str = 'codebert',
                 as_code_embedder: bool = True,
                 drop_tokenizer_special_tokens: bool = True,
                 loss_coeff: float = 1.,
                 loss_epoch_range: List[int] = [-1, -1],
                 log_epsilon: float = 1e-10,
                 similarity_temperature: float = 10.,
                 negative_pair_num: int = -1,
                 **kwargs):
        super().__init__(vocab,
                         name=name,
                         loss_coeff=loss_coeff,
                         as_code_embedder=as_code_embedder,
                         forward_from_where='token',
                         loss_epoch_range = loss_epoch_range,
                         **kwargs)
        self.vocab = vocab

        self.token_pooler = token_pooler
        self.tokenizer_type = tokenizer_type
        self.drop_tokenizer_special_tokens = drop_tokenizer_special_tokens
        self.log_epsilon = log_epsilon
        self.similarity_temperature = similarity_temperature
        self.negative_pair_num = negative_pair_num

        self.sim = torch.nn.CosineSimilarity(-1)


    def mask_dropout_constras_loss(self,
                                   epoch: int,
                                   token_embeddings: torch.Tensor,
                                   token_embeddings_extra: torch.Tensor,
                                   token_mask: Optional[torch.Tensor] = None,
                                   token_mask_extra: Optional[torch.Tensor] = None,
                                   **kwargs) -> torch.Tensor:
        """
        Produce dropout-constrastive learning loss.

        Token embeddings will be first pooled into code representations.
        Positive pairs are constructed by different dropouts.
        Negative pairs are constructed from other code representations within the batch
        (note that batch_size may account for performance).

        Similarity is set as cosine-similarity in default, while similarity temperature is also enabled.
        """
        if not self.training:
            return torch.zeros((1,), device=token_embeddings.device)

        # mylogger.debug('dim_dropout_contras', f'token_embedding size: {token_embeddings.size()}')
        if self.drop_tokenizer_special_tokens:
            # TODO: Check if here gives a in-place operation ?
            token_embeddings, token_mask = drop_tokenizer_special_tokens(self.tokenizer_type, token_embeddings, token_mask)
            token_embeddings_extra, token_mask_extra = drop_tokenizer_special_tokens(self.tokenizer_type, token_embeddings_extra, token_mask_extra)

        # code repre shape: [bsz, dim]
        code_repres = self.token_pooler.forward(token_embeddings, token_mask)
        code_repres_extra = self.token_pooler.forward(token_embeddings_extra, token_mask_extra)

        # pos_sims shape: [bsz, 1]
        pos_sims = self.sim(code_repres, code_repres_extra).unsqueeze(1) * self.similarity_temperature
        bsz = code_repres.size(0)
        code_repres_expand = code_repres.unsqueeze(1).repeat(1,bsz,1)
        code_repres_expand_t = code_repres_extra.unsqueeze(0).repeat(bsz,1,1)
        # neg_sims shape: [bsz, bsz]
        neg_sims = self.sim(code_repres_expand, code_repres_expand_t) * self.similarity_temperature

        # Mask diagonal to remove one duplicate self-vs-self similarity
        neg_sims.fill_diagonal_(float("-inf"))

        # We concat the positive pair similarity at 0-th position of the sim matrix,
        # while other elements are all negative pairs (selves have been excluded before).
        # concat_sims shape: [bsz, bsz+1]
        concat_sims = torch.concat((pos_sims, neg_sims), dim=1)
        probs = torch.softmax(concat_sims, 1)
        # Compute loss from positive pair.
        loss = torch.log(probs[:,0] + self.log_epsilon)
        loss = -1 * loss.mean() * self.loss_coeff
        # loss = self.rectify_loss_based_on_range(loss, epoch)
        return loss


    def forward(self, **kwargs) -> Dict:
        return self.forward_from_token(**kwargs)


    def forward_from_token(self,
                           code: TextFieldTensors,
                           code_embed_func: Callable,
                           epoch: int,
                           **kwargs) -> Dict:
        if self.check_obj_in_range(epoch):
            # Forward ast tokens
            ast_tokens = kwargs['ast_tokens']

            # Forward the same code embedder twice.
            code_embed_outputs = code_embed_func(ast_tokens)
            code_embeddings, code_mask = code_embed_outputs['outputs'], code_embed_outputs['mask']
            code_embed_outputs_extra = code_embed_func(ast_tokens)
            code_embeddings_extra, code_mask_extra = code_embed_outputs_extra['outputs'], code_embed_outputs_extra['mask']

            # Compute contrastive loss from two forward representations.
            dropout_constras_loss = self.mask_dropout_constras_loss(epoch, code_embeddings, code_embeddings_extra, code_mask, code_mask_extra, **kwargs)
            output_dict =  {'loss': dropout_constras_loss}
            output_dict.update(code_embed_outputs)
            return output_dict
        else:
            return self.get_obj_not_in_range_result()

