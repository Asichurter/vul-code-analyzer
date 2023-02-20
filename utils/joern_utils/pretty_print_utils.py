from typing import List, Union, Tuple

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

def multi_paired_token_colored_print(raw_text: str, data_edges: List[Union[str,Tuple[int,int]]], tokens: List[Token], processed: bool = False):
    if len(data_edges) == 0:
        print(raw_text)
        return

    tokens_a, tokens_b = [], []
    for edge in data_edges:
        if not processed:
            sid, eid = edge.split()
            sid, eid = int(sid), int(eid)
        else:
            sid, eid = edge
        # Check token in truncated range
        if sid >= len(tokens) or eid >= len(tokens):
            continue
        else:
            tokens_a.append(tokens[sid])
            tokens_b.append(tokens[eid])

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

def multi_paired_token_tagged_print(raw_text: str, data_edges: List[Union[str, Tuple[int, int]]], tokens: List[Token], processed: bool = False):
    tokens_a, tokens_b = [], []
    for edge in data_edges:
        if not processed:
            sid, eid = edge.split()
            sid, eid = int(sid), int(eid)
        else:
            sid, eid = edge
        # Check token in truncated range
        if sid >= len(tokens) or eid >= len(tokens):
            continue
        else:
            tokens_a.append(tokens[sid])
            tokens_b.append(tokens[eid])

    assert len(tokens_a) == len(tokens_b)
    spans_a = [(t.idx, t.idx_end, i) for i,t in enumerate(tokens_a)]
    spans_b = [(t.idx, t.idx_end, i) for i,t in enumerate(tokens_b)]
    spans = spans_a + spans_b
    spans = sorted(spans, key=lambda x: x[0])
    print(raw_text[:spans[0][0]], end='')
    for i in range(len(spans)):
        print(raw_text[spans[i][0]:spans[i][1]]+f'([{spans[i][2]}])', end='')
        if i != len(spans) - 1:
            print(raw_text[spans[i][1]:spans[i+1][0]], end='')
        else:
            print(raw_text[spans[i][1]:], end='')

    print("\n\nLegends:")
    legend = {i: raw_text[spans_a[i][0]:spans_a[i][1]] for i in range(len(spans_a))}
    print(legend)

def print_code_with_line_num(code: str, start_line_num: int = 0):
    newline_idx = code.find('\n')
    new_code = ''
    line_count = start_line_num
    while newline_idx != -1:
        new_code += f'#{line_count}\t'
        new_code += code[:newline_idx+1]
        line_count += 1
        code = code[newline_idx+1:]
        newline_idx = code.find('\n')
    new_code += code
    print(new_code)


if __name__ == '__main__':
    code = '''seat_set_active_session (Seat *seat, Session *session)
{
    GList *link;
    g_return_if_fail (seat != NULL);
    SEAT_GET_CLASS (seat)->set_active_session (seat, session);
    for (link = seat->priv->sessions; link; link = link->next)
    {
        Session *s = link->data;
        if (s == session || session_get_is_stopping (s))
            continue;
        if (IS_GREETER (s))
        {
            l_debug (seat, "Stopping greeter");
            session_stop (s);
        }
    }
    if (seat->priv->active_session && 
        seat->priv->active_session != session)
    {
        if (session != seat->priv->active_session && !IS_GREETER (seat->priv->active_session))
            session_lock (seat->priv->active_session);
        g_object_unref (seat->priv->active_session);
    }
    session_activate (session);
    seat->priv->active_session = g_object_ref (session);
}
    '''
    print_code_with_line_num(code)