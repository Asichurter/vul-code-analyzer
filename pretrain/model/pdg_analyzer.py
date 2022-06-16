import torch
from allennlp.data import Vocabulary

from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder

from pretrain.comp.nn.node_encoder.node_encoder import NodeEncoder
from pretrain.comp.nn.struct_decoder.struct_decoder import StructDecoder


@Model.register('code_pdg_analyzer')
class CodePDGAnalyzer(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        code_embedder: TextFieldEmbedder,
        code_encoder: Seq2SeqEncoder,
        node_encoder: NodeEncoder,
        struct_decoder: StructDecoder,
        # edge_sampler: EdgeSampler,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.code_embedder = code_embedder
        self.code_encoder = code_encoder
        self.node_encoder = node_encoder
        self.struct_decoder = struct_decoder
        # self.edge_sampler = edge_sampler

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        pass