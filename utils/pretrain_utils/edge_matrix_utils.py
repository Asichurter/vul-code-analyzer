import re
from typing import List, Tuple

import torch
from allennlp.data import Token

from utils.joern_utils.joern_dev_pdg_parse_utils import build_token_level_pdg_struct


def make_pdg_ctrl_edge_matrix_v1(line_edges: List[str],
                                 line_count: int) -> torch.LongTensor:
    """
    Make line-level ctrl dependency matrix from edge data.
    V1: Results from latest version of joern, including line-level data edges.

    """
    # To cover the last line (line_count-th), we have to allocate one more line here.
    # Set 1 as default to distinguish from padded positions
    matrix = torch.ones((line_count + 1, line_count + 1))

    for edge in line_edges:
        tail, head, etype = re.split(',| ', edge)  # tail/head vertice index start from 1 instead of 0
        tail, head, etype = int(tail), int(head), int(etype)
        # Ignore uncovered vertices (lines)
        if tail > line_count or head > line_count:
            continue
        if etype == 3 or etype == 2:
            matrix[tail, head] = 2

    # Drop 0-th row and column, since line index starts from 1.
    return matrix[1:, 1:]


def make_pdg_ctrl_edge_matrix_v2(line_edges: List[Tuple[int,int]],
                                 line_count: int) -> torch.LongTensor:
    """
    Make line-level ctrl dependency matrix from edge data.
    V2: Results from 0.3.1 version of joern, only line-level ctrl edges.

    """
    # To cover the last line (line_count-th), we have to allocate one more line here.
    # Set 1 as default to distinguish from padded positions
    matrix = torch.ones((line_count + 1, line_count + 1))

    for edge in line_edges:
        tail, head = edge
        # Ignore uncovered vertices (lines)
        if tail > line_count or head > line_count:
            continue
        matrix[tail, head] = 2

    # Drop 0-th row and column, since line index starts from 1.
    # Now line index start from 0.
    return matrix[1:, 1:]

def make_pdg_data_edge_matrix(raw_code: str,
                              token_nodes: List[List[str]],
                              token_edges: List[List[str]],
                              tokenized_tokens: List[Token],
                              multi_vs_multi_strategy: str) -> torch.Tensor:
    # Parse nodes and edges from joern-parse, to build token-level data edges
    _, token_data_edges = build_token_level_pdg_struct(raw_code, tokenized_tokens,
                                                       token_nodes, token_edges,
                                                       multi_vs_multi_strategy,
                                                       to_build_token_ctrl_edges=False)
    token_len = len(tokenized_tokens)

    # Set 1 as default to distinguish from padded positions
    matrix = torch.ones((token_len, token_len))

    for edge in token_data_edges:
        s_token_idx, e_token_idx = edge.split()
        s_token_idx, e_token_idx = int(s_token_idx), int(e_token_idx)
        if s_token_idx >= token_len or e_token_idx >= token_len:
            continue
        if s_token_idx == e_token_idx:
            continue
        # Set edge value to 2
        matrix[s_token_idx, e_token_idx] = 2

    return matrix

def make_pdg_data_edge_matrix_from_processed(tokens: List[Token],
                                             processed_data_edges: List[Tuple[int,int]]) -> torch.Tensor:
    token_len = len(tokens)
    # Set 1 as default to distinguish from padded positions
    matrix = torch.ones((token_len, token_len))

    for edge in processed_data_edges:
        s_token_idx, e_token_idx = edge
        if s_token_idx >= token_len or e_token_idx >= token_len:
            continue
        if s_token_idx == e_token_idx:
            continue
        # Set edge value to 2
        matrix[s_token_idx, e_token_idx] = 2

    return matrix

def make_pdg_data_edge_matrix_from_processed_optimized(tokens: List[Token],
                                                       processed_data_edges: List[Tuple[int,int]]) -> torch.Tensor:
    """
    Compared to making matrix at loading time, here we only give the edges idxes,
    to allow construct token matrix at run-time to avoid unaffordable memory consumption.
    """
    token_len = len(tokens)
    idxes = []
    for edge in processed_data_edges:
        s_token_idx, e_token_idx = edge
        if s_token_idx >= token_len or e_token_idx >= token_len:
            continue
        if s_token_idx == e_token_idx:
            continue
        idxes.append([s_token_idx, e_token_idx])

    # Append a placeholder idx, to avoid key missing error when calling "batch_tensor"
    if len(idxes) == 0:
        idxes.append([0,0])

    return torch.Tensor(idxes)