import csv
from typing import List, Dict, Optional, Tuple, Iterable
from allennlp.data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer

from utils.file import load_text

ast_edge_type = 'IS_AST_PARENT'
ctrl_dependency_edge_type = 'CONTROLS'
data_use_dependency_edge_type = 'USE'
data_def_dependency_edge_type = 'DEF'
symbol_node_type = 'Symbol'
identifier_node_type = 'Identifier'

def read_csv_as_list(path):
    rows = list(csv.reader(open(path, 'r')))
    fields = ''.join(rows[0]).split('\t')
    list_rows = []
    for row in rows[1:]:
        values = ''.join(row).split('\t')
        assert len(values) == len(fields)
        list_rows.append(values)

    # return dict_rows
    return list_rows

def read_csv_as_dict(path):
    rows = list(csv.reader(open(path, 'r')))
    fields = ''.join(rows[0]).split('\t')
    dict_rows = []
    for row in rows[1:]:
        values = ''.join(row).split('\t')
        assert len(values) == len(fields)
        dict_row = {k:v for k,v in zip(fields, values)}
        dict_rows.append(dict_row)

    return dict_rows

class Node:
    def __init__(self, cmd, nid, node_type, node_code, node_loc, function_id, child_num, is_cfg_node, operator, base_type, complete_type, identifier):
        self.cmd = cmd
        self.nid = int(nid)
        self.node_type = node_type
        self.node_code = node_code
        self.node_loc = node_loc
        self.function_id = function_id
        self.child_num = child_num
        self.is_cfg_node = is_cfg_node
        self.operator = operator
        self.base_type = base_type
        self.complete_type = complete_type
        self.identifier = identifier

        self.ast_childs = []
        self.ast_parent = None

    def _handle_null_val(self, val):
        return None if val == '' else val

    def add_ast_child(self, child_nid):
        self.ast_childs.append(child_nid)

    def set_ast_parent(self, parent_nid):
        self.ast_parent = parent_nid

    def is_symbol(self):
        return self.node_type == symbol_node_type

    def is_identifier(self):
        return self.node_type == identifier_node_type

    def __str__(self):
        return f'cmd: {self.cmd}, key: {self.nid}, type: {self.node_type}, code: {self.node_code}, loc: {self.node_loc}, ' \
               f'func_id: {self.function_id}, child_num: {self.child_num}, is_cfg_node: {self.is_cfg_node}, ' \
               f'operator: {self.operator}, base_type: {self.base_type}, identifier: {self.identifier}'

def find_child_identifier_node_ids(root_node_id: int, nodes: List[Node]) -> List[int]:
    node_id_stack = [root_node_id]
    identifier_node_ids = []
    while len(node_id_stack) > 0:
        cur_node_id = node_id_stack.pop(-1)
        cur_node = nodes[cur_node_id]
        if cur_node.is_identifier():
            identifier_node_ids.append(cur_node_id)

        for child_node_id in cur_node.ast_childs:
            node_id_stack.append(child_node_id)

    return identifier_node_ids

def parse_char_range(loc: str) -> Tuple[int,int]:
    _, _, char_start, char_end = loc.split(':')
    return int(char_start), int(char_end)

def parse_and_sort_char_spans_from_nids(node_ids: Iterable[int],
                                        nodes: List[Node]):
    char_spans = [parse_char_range(nodes[_id].node_loc) for _id in node_ids]
    char_spans = sorted(char_spans, key=lambda e: e[0], reverse=False)
    return char_spans

def intersect_char_spans_with_allennlp_tokens(tokens: List[Token],
                                              char_spans: List[Tuple[int,int]]):
    """
    Note:
        To ensure this function works well,
        "char_spans" are assumed to be sorted and non-overlapped.
    """
    span_i = 0
    allennlp_target_token_indices = []
    for token_i, token in enumerate(tokens):
        idx, idx_end = token.idx, token.idx_end
        # Skip speicial tokens
        if idx is None or idx_end is None:
            continue
        # No more spans to check, break
        if span_i >= len(char_spans):
            break

        cur_span = char_spans[span_i]
        # Check span intersection
        if (idx - cur_span[1]) * (idx_end - cur_span[0]) <= 0:
            allennlp_target_token_indices.append(token_i)
        # Check if current token span ends and move to next span
        if idx_end >= cur_span[1]:
            span_i += 1

    return allennlp_target_token_indices

def build_token_level_pdg_struct(raw_code: str,
                                 allennlp_tokenizer: Tokenizer,
                                 node_rows: List[List],
                                 edge_dicts: List[Dict],
                                 multi_vs_multi_strategy: str = 'all'):
    assert multi_vs_multi_strategy in ['all', 'first']
    tokens = allennlp_tokenizer.tokenize(raw_code)
    pdg_ctrl_edges = set()
    pdg_data_edges = set()

    pdg_nodes: List[Optional[Node]] = [None] * (len(node_rows)+1)
    for node_row in node_rows:
        node = Node(*node_row)
        nid = node.nid
        pdg_nodes[nid] = node

    # Build AST struct
    for edge in edge_dicts:
        if edge['type'] == ast_edge_type:
            ast_p_id, ast_c_id = int(edge['start']), int(edge['end'])
            pdg_nodes[ast_p_id].add_ast_child(ast_c_id)
            pdg_nodes[ast_c_id].set_ast_parent(ast_p_id)

    # Process control dependencies
    for edge in edge_dicts:
        if edge['type'] == ctrl_dependency_edge_type:
            ctrl_sid, ctrl_eid = int(edge['start']), int(edge['end'])
            # Do not handle symbol control edge
            if pdg_nodes[ctrl_sid].is_symbol() or pdg_nodes[ctrl_eid].is_symbol():
                continue

            # Find identifier nodes
            ctrl_start_identifier_nids = find_child_identifier_node_ids(ctrl_sid, pdg_nodes)
            ctrl_end_identifier_nids = find_child_identifier_node_ids(ctrl_eid, pdg_nodes)
            # Parse char spans of selected identifiers
            ctrl_start_identifier_char_spans = parse_and_sort_char_spans_from_nids(ctrl_start_identifier_nids, pdg_nodes)
            ctrl_end_identifier_char_spans = parse_and_sort_char_spans_from_nids(ctrl_end_identifier_nids, pdg_nodes)
            # Intersect char spans with allennlp tokens
            ctrl_start_token_indices = intersect_char_spans_with_allennlp_tokens(tokens, ctrl_start_identifier_char_spans)
            ctrl_end_token_indices = intersect_char_spans_with_allennlp_tokens(tokens, ctrl_end_identifier_char_spans)

            if multi_vs_multi_strategy == 'all':
                pass
            elif multi_vs_multi_strategy == 'first':
                ctrl_start_token_indices = ctrl_start_token_indices[0]
                ctrl_end_token_indices = ctrl_end_token_indices[0]

            # Add token-level ctrl edges
            for sid in ctrl_start_token_indices:
                for eid in ctrl_end_token_indices:
                    pdg_ctrl_edges.add(f'{sid} {eid}')

    # Todo: Process data dependency

if __name__ == '__main__':
    node_csv_path = '/data1/zhijietang/dockers/joern-dev/tests/parsed/testCode2/test2.cpp/nodes.csv'
    edge_csv_path = '/data1/zhijietang/dockers/joern-dev/tests/parsed/testCode2/test2.cpp/edges.csv'
    raw_code = load_text('/data1/zhijietang/dockers/joern-dev/tests/testCode2/test2.cpp')
    tokenizer = PretrainedTransformerTokenizer('microsoft/codebert-base')

    nodes = read_csv_as_list(node_csv_path)
    edges = read_csv_as_dict(edge_csv_path)
    build_pdg_struct(raw_code, tokenizer, nodes, edges)
