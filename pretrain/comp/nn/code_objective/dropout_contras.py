from typing import Dict, Optional, List

import torch
from allennlp.data import Vocabulary

from common.nn.pooler import Pooler
from pretrain.comp.nn.code_objective.code_objective import CodeObjective
from utils.allennlp_utils.tokenizer_vocab_sensitive_utils import drop_tokenizer_special_tokens

from utils import GlobalLogger as mylogger

@CodeObjective.register("dim_dropout_contras")
class DimDropoutContrastiveLearning(CodeObjective):
    """
    The MCL of UniXCoder, except that our input is unimodel of code but not multimodal.

    Token embeddings will first be pooled to a fixed-length code representation, different dropouts
    will be used to generate positive pairs, while other codes will be used as negative pairs.

    Note the dropout is working the feature dim of code representations.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 name: str,
                 token_pooler: Pooler,
                 tokenizer_type: str = 'codebert',
                 drop_tokenizer_special_tokens: bool = True,
                 dropout: float = 0.1,
                 loss_coeff: float = 1.,
                 loss_epoch_range: List[int] = [-1, -1],
                 log_epsilon: float = 1e-10,
                 similarity_temperature: float = 10.,
                 **kwargs):
        super().__init__(vocab,
                         name=name,
                         loss_coeff=loss_coeff,
                         as_code_embedder=False,
                         forward_from_where='embedding',
                         loss_epoch_range = loss_epoch_range,
                         **kwargs)
        self.vocab = vocab

        self.token_pooler = token_pooler
        self.dropout = dropout
        self.tokenizer_type = tokenizer_type
        self.drop_tokenizer_special_tokens = drop_tokenizer_special_tokens
        self.log_epsilon = log_epsilon
        self.similarity_temperature = similarity_temperature

        self.dropout_src = torch.nn.Dropout(self.dropout)
        self.dropout_pos = torch.nn.Dropout(self.dropout)
        self.sim = torch.nn.CosineSimilarity(-1)


    def dropout_constras_loss(self,
                              epoch: int,
                              token_embeddings: torch.Tensor,
                              token_mask: Optional[torch.Tensor] = None,
                              **kwargs) -> torch.Tensor:
        """
        Produce dropout-constrastive learning loss.

        Token embeddings will be first pooled into code representations.
        Positive pairs are constructed by different dropouts.
        Negative pairs are constructed from other code representations within the batch
        (note that batch_size may account for performance).

        Similarity is set as cosine-similarity in default, while similarity temperature is also enabled.
        """
        mylogger.debug('dim_dropout_contras', f'token_embedding size: {token_embeddings.size()}')
        if self.drop_tokenizer_special_tokens:
            token_embeddings, token_mask = drop_tokenizer_special_tokens(self.tokenizer_type, token_embeddings, token_mask)

        # code repre shape: [bsz, dim]
        code_repres = self.token_pooler.forward(token_embeddings, token_mask)
        code_repres_src = self.dropout_src(code_repres)
        code_repres_pos = self.dropout_pos(code_repres)

        # pos_sims shape: [bsz, 1]
        pos_sims = self.sim(code_repres_src, code_repres_pos).unsqueeze(1) * self.similarity_temperature
        bsz = code_repres.size(0)
        code_repres_expand = code_repres_src.unsqueeze(1).repeat(1,bsz,1)
        code_repres_expand_t = code_repres_pos.unsqueeze(0).repeat(bsz,1,1)
        # neg_sims shape: [bsz, bsz]
        neg_sims = self.sim(code_repres_expand, code_repres_expand_t) * self.similarity_temperature

        # Mask diagonal to avoid self-vs-self similarity are considered
        neg_sims.fill_diagonal_(float("-inf"))

        # We concat the positive pair similarity at 0-th position of the sim matrix,
        # while other elements are all negative pairs (selves have been excluded before).
        # concat_sims shape: [bsz, bsz+1]
        concat_sims = torch.concat((pos_sims, neg_sims), dim=1)
        probs = torch.softmax(concat_sims, 1)
        # Compute loss from positive pair.
        loss = torch.log(probs[:,0] + self.log_epsilon)
        loss = -1 * loss.mean() * self.loss_coeff
        loss = self.rectify_loss_based_on_range(loss, epoch)

        return loss


    def forward(self, **kwargs) -> Dict:
        return self.forward_from_embedding(**kwargs)

    def forward_from_embedding(self,
                               token_embedding: torch.Tensor,
                               token_mask: torch.Tensor,
                               epoch: int,
                               **kwargs) -> Dict:
        dropout_constras_loss = self.dropout_constras_loss(epoch, token_embedding, token_mask, **kwargs)
        output_dict =  {'loss': dropout_constras_loss}
        return output_dict

