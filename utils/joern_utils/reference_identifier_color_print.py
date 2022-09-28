from typing import List

from allennlp.data import Token
from termcolor import cprint

_colors_ = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'grey']

def paired_token_colored_print(raw_text, token_a: Token, token_b: Token, color_idx=0):
    a_range = (token_a.idx, token_a.idx_end)
    b_range = (token_b.idx, token_b.idx_end)
    sorted_ranges = (a_range, b_range) if a_range[0] < b_range[0] else (b_range, a_range)
    print(raw_text[:sorted_ranges[0][0]], end='')
    cprint(raw_text[sorted_ranges[0][0]:sorted_ranges[0][1]], color=_colors_[color_idx], end='')
    print(raw_text[sorted_ranges[0][1]:sorted_ranges[1][0]], end='')
    cprint(raw_text[sorted_ranges[1][0]:sorted_ranges[1][1]], color=_colors_[color_idx], end='')
    print(raw_text[sorted_ranges[1][1]:], end='')

def multi_paired_token_colored_print(raw_text: str, data_edges: List[str], tokens: List[Token]):
    tokens_a, tokens_b = [], []
    for edge in data_edges:
        sid, eid = edge.split()
        tokens_a.append(tokens[int(sid)])
        tokens_b.append(tokens[int(eid)])

    assert len(tokens_a) == len(tokens_b)
    spans_a = [(t.idx, t.idx_end, i%8) for i,t in enumerate(tokens_a)]
    spans_b = [(t.idx, t.idx_end, i%8) for i,t in enumerate(tokens_b)]
    spans = spans_a + spans_b
    spans = sorted(spans, key=lambda x: x[0])
    print(raw_text[:spans[0][0]], end='')
    for i in range(len(spans)):
        cprint(raw_text[spans[i][0]:spans[i][1]], color=_colors_[spans[i][2]%8] ,end='')
        if i != len(spans) - 1:
            print(raw_text[spans[i][1]:spans[i+1][0]], end='')
        else:
            print(raw_text[spans[i][1]:], end='')