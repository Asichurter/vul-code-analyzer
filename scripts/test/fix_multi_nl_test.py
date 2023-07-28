import re
import torch

def find_consecutive_lines(code):
    # Del line indices start from 0
    del_line_indices = []  # 0
    new_line_indices = []  # 1
    # 2
    for m in re.finditer('\n', code):  # 3
        new_line_indices.append(m.start())
    if len(new_line_indices) > 0:
        for i in range(len(new_line_indices) - 1):
            # Consecutive lines
            # if new_line_indices[i] == new_line_indices[i + 1] + 1:
            if code[new_line_indices[i]:new_line_indices[i + 1] + 1].strip() == '':
                del_line_indices.append(i + 1)

    while '\n\n' in code:
        code = code.replace('\n\n', '\n')
    return code, del_line_indices

def shift_graph_matrix(mat, del_lines, shift=0):
    assert len(mat.shape) == 2 and mat.size(0) == mat.size(1)
    line_spans = [-1] + [n-shift for n in del_lines] + [len(mat)]
    print(line_spans)

    new_rows = []
    for start, end in zip(line_spans[:-1],line_spans[1:]):
        if start < end:
            new_rows.append(mat[start+1:end])
    new_mat = torch.cat(new_rows, dim=0)

    new_cols = []
    for start, end in zip(line_spans[:-1],line_spans[1:]):
        if start < end:
            new_cols.append(new_mat[:, start+1:end])
    new_mat = torch.cat(new_cols, dim=1)
    return new_mat

m = torch.Tensor(list(range(25))).view(5,5)
code = '1\n\n2\n3\n4\n'
code, d_lines = find_consecutive_lines(code)
a_m = shift_graph_matrix(m, d_lines)

