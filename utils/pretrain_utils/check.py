import torch
from allennlp.data import TextFieldTensors, Vocabulary


def check_identifier_matching(edges: torch.Tensor,
                              code: TextFieldTensors,
                              vocab: Vocabulary,
                              namespace: str = 'code_tokens',
                              token_id_key: str = 'token_ids',
                              batch_i: int = 0):
    for edge in edges[batch_i]:
        if edge.sum().item() == 0:
            continue
        s_id = code[namespace][token_id_key][batch_i][edge[0].int().item()].item()
        e_id = code[namespace][token_id_key][batch_i][edge[1].int().item()].item()
        print(vocab.get_token_from_index(s_id, namespace), end=' ')
        print(vocab.get_token_from_index(e_id, namespace))