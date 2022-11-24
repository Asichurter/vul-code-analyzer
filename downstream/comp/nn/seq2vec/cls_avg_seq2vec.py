from overrides import overrides

import torch.nn

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder


@Seq2VecEncoder.register("cls_avg")
class ClsAvgSeq2Vec(Seq2VecEncoder):
    def __init__(self,
                 embedding_dim: int):
        super().__init__()
        self._embedding_dim = embedding_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor = None):
        cls_tokens_out = torch.select(tokens, -2, 0)
        cls_mask_out = mask.sum(-1).bool()

        avg_count = cls_mask_out.sum(-1)
        avg_count = torch.max(avg_count, avg_count.new_ones(1))     # Avoid zero-division
        avg_out = (cls_tokens_out * cls_mask_out.unsqueeze(-1)).sum(-2) / avg_count[:, None]
        return avg_out
