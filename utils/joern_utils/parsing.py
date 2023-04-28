from typing import List, Union, Tuple
import re

import torch
from allennlp.data import Token
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


def parse_token_level_graph_as_line_level(raw_code: str,
                                          data_edges: List[List[int]],
                                          tokens: List[Token],
                                          num_line_elem: int,
                                          start_line: int = 1):
    """
        Parse the token-level dependency edges using char spans and map them
        into line-level, by checking span sequential relation between tokens and
        new line tokens.
    """

    def _make_edge_repr(_sid, _eid):
        return f"{_sid} {_eid}"

    # Find the char indices of new-lines
    new_line_indices = []
    for m in re.finditer('\n', raw_code):
        new_line_indices.append(m.start())

    def _get_line_num(_token: Token):
        _line_num = start_line
        # Here we assume a token will not span across two lines, thus no \n will in
        for _i, _nl_idx in enumerate(new_line_indices):
            if _token.idx_end < _nl_idx:
                break
            _line_num += 1
        return _line_num

    edge_list = torch.Tensor(data_edges).nonzero().tolist()
    edge_set = set()
    edges = []
    for edge in edge_list:
        sid, eid = edge
        # Check token in truncated range
        if sid >= len(tokens) or eid >= len(tokens):
            continue
        # Check duplicate
        s_line_num = _get_line_num(tokens[sid])
        e_line_num = _get_line_num(tokens[eid])
        edge_repr = _make_edge_repr(s_line_num, e_line_num)
        if edge_repr in edge_set:
            continue
        else:
            edges.append([s_line_num, e_line_num])
            edge_set.add(edge_repr)

    # Refill the line-level matrix elements.
    # Here start_line may make the real matrix shift.
    matrix = torch.zeros((num_line_elem + start_line, num_line_elem + start_line))
    if len(edges) > 0:
        edges = torch.LongTensor(edges)
        matrix[edges[:, 0], edges[:, 1]] = 1

    return matrix[start_line:, start_line:]

if __name__ == "__main__":
    code = 'void RenderWidgetHostViewGtk::SetAccessibilityFocus(int acc_obj_id) {\n  if (!host_)\n    return;\n\n  host_->AccessibilitySetFocus(acc_obj_id);\n}'
    num_line = 6
    tokenizer = PretrainedTransformerTokenizer('microsoft/codebert-base')
    tokens = tokenizer.tokenize(code)
    data_edges = torch.zeros((len(tokens), len(tokens)))
    data_edges[15][45] = 1
    matrix = parse_token_level_graph_as_line_level(code, data_edges, tokens, num_line)