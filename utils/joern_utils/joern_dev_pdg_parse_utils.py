import csv
from typing import List, Dict, Optional, Tuple, Iterable
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
import re

from utils.file import load_text

ast_edge_type = 'IS_AST_PARENT'
ctrl_dependency_edge_type = 'CONTROLS'
data_use_dependency_edge_type = 'USE'
data_def_dependency_edge_type = 'DEF'
reaches_dependenct_edge_type = 'REACHES'
symbol_node_type = 'Symbol'
identifier_node_type = 'Identifier'


def clean_signature_line(code: str) -> str:
    code = re.sub(r'( |\t|\n)+', ' ', code)
    return code

def preprocess_rows(rows):
    processed_rows = []
    for row in rows:
        processed_row = ''.join(row).split('\n')
        processed_rows.extend(processed_row)
    return processed_rows

def read_csv_as_list(path):
    rows = list(csv.reader(open(path, 'r')))
    # BugFix: Handle some rows span across multiple line items
    rows = preprocess_rows(rows)

    fields = ''.join(rows[0]).split('\t')
    list_rows = []
    for row in rows[1:]:
        values = ''.join(row).split('\t')
        assert len(values) == len(fields), f'len(values) != len(fields), value: {row}({len(values)}), field_len: {len(fields)}'
        list_rows.append(values)

    # return dict_rows
    return list_rows

def read_csv_as_dict(path):
    rows = list(csv.reader(open(path, 'r')))
    # BugFix: Handle some rows span across multiple line items
    rows = preprocess_rows(rows)

    fields = ''.join(rows[0]).split('\t')
    dict_rows = []
    for row in rows[1:]:
        values = ''.join(row).split('\t')
        assert len(values) == len(fields)
        dict_row = {k:v for k,v in zip(fields, values)}
        dict_rows.append(dict_row)

    return dict_rows


##############################################################################
# Apply multi v.s. multi strategy
# ---------------------------------------------------------------------------
#       That is to say, if identifier A depends on B and both identifiers
#       are tokenized into multiple wordpieces, how we handle this actually
#       "multi-to-multi" case and generate wordpiece-level edge connections.
##############################################################################
def apply_multi_vs_multi_strategy(obj_list: List, strategy: str) -> List:
    if strategy == 'all':
        return obj_list
    elif strategy == 'first':
        return obj_list[0:1]
    else:
        raise NotImplementedError(f'No such multi_vs_multi strategy: {strategy}')

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

    def is_empty_loc(self):
        return self.node_loc == ''

    def __str__(self):
        return f'cmd: {self.cmd}, key: {self.nid}, type: {self.node_type}, code: {self.node_code}, loc: {self.node_loc}, ' \
               f'func_id: {self.function_id}, child_num: {self.child_num}, is_cfg_node: {self.is_cfg_node}, ' \
               f'operator: {self.operator}, base_type: {self.base_type}, identifier: {self.identifier}'

# Change this list if other types of edges are of interest
accepted_edge_types = [ast_edge_type, ctrl_dependency_edge_type,
                       data_use_dependency_edge_type, data_def_dependency_edge_type,
                       reaches_dependenct_edge_type]

class Edge:
    def __init__(self, start_id, end_id, edge_type, edge_var):
        self.sid = int(start_id)
        self.eid = int(end_id)
        self.type = edge_type
        self.var = edge_var

    def is_ast_parent_edge(self):
        return self.type == ast_edge_type

    def is_ctrl_edge(self):
        return self.type == ctrl_dependency_edge_type

    def is_def_edge(self):
        return self.type == data_def_dependency_edge_type

    def is_use_edge(self):
        return self.type == data_use_dependency_edge_type

    def is_data_edge(self):
        return self.is_use_edge() or self.is_def_edge()

    def is_reaches_edge(self):
        return self.type == reaches_dependenct_edge_type

    def __str__(self):
        return f'start: {self.sid}, end: {self.eid}, type: {self.type}, var: {self.var}'

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def build_edge(edge_list: List):
        edge_type = edge_list[2] # edge_dict['type']
        if edge_type in accepted_edge_types:
            edge = Edge(*edge_list)
            return edge
        else:
            return None

def find_child_identifier_node_ids(root_node_id: int, nodes: List[Node]) -> List[int]:
    """
    Depth-first search identifier child nodes, using a simulated stack.
    """
    node_id_stack = [root_node_id]
    identifier_node_ids = []
    while len(node_id_stack) > 0:
        cur_node_id = node_id_stack.pop(-1)
        if cur_node_id < 0:
            continue
        cur_node = nodes[cur_node_id]
        if cur_node.is_identifier():
            identifier_node_ids.append(cur_node_id)

        for child_node_id in cur_node.ast_childs:
            node_id_stack.append(child_node_id)

    return identifier_node_ids

def parse_char_span(loc: str, signature_shift_len: int = 0) -> Tuple[int, int]:
    line_num, _, _, char_start, char_end = loc.split(':')
    # FixBug: end idx need to increase 1
    char_start, char_end = int(char_start), int(char_end) + 1
    # FixBug: Fix char shift for signature line
    if line_num == '2':
        char_start -= signature_shift_len
        char_end -= signature_shift_len
    return char_start, char_end


def parse_and_sort_char_spans_from_nids(node_ids: Iterable[int],
                                        nodes: List[Node],
                                        signature_shift_len: int = 0) -> List[Tuple[int, int]]:
    char_spans = [parse_char_span(nodes[_id].node_loc, signature_shift_len) for _id in node_ids]
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
        token_idx, token_idx_end = token.idx, token.idx_end
        # Skip speicial tokens
        if token_idx is None or token_idx_end is None:
            continue
        # No more spans to check, break
        if span_i >= len(char_spans):
            break

        cur_span = char_spans[span_i]
        # Convert left-close-right-open range to double-close
        span_idx, span_idx_end = cur_span[0], cur_span[1] - 1
        token_idx_end = token_idx_end - 1
        # Check span intersection
        if (token_idx - span_idx_end) * (token_idx_end - span_idx) <= 0:
            allennlp_target_token_indices.append(token_i)
        # Check if current token span ends and move to next span
        while span_i < len(char_spans) and token_idx_end >= char_spans[span_i][1]:
            span_i += 1

    return allennlp_target_token_indices

def build_token_level_data_pdg_old(data_edges: List[Edge],
                                   pdg_nodes: List[Node],
                                   tokens: List[Token],
                                   signature_len: int,
                                   multi_vs_multi_strategy: str) -> Iterable[str]:
    ################################################################################################
    # DEPRECATED: NOT CONSIDER CONTROL FLOW. Use "build_token_level_data_pdg" instead.
    # ------------------------------------------------------------------------------------------
    # We use hacks to process data dependencies:
    # ------------------------------------------------------------------------------------------
    # 1. Since symbols can be defined multiple times, we must determine which definition a use edge
    #    matches. We note the def-use edges are somehow in order, where a use edge only matches
    #    the latest definition of a symbol. Thus we can sequentially process def&use edges.
    # ------------------------------------------------------------------------------------------
    # 2. Some symbols, like member visiting of pointers and undefined functions, can be properly
    #    filtered by checking if a used symbol has been defined previously.
    ################################################################################################
    symbol_def_nid_map = {}
    existed_use_nid_and_symbol_nid_pairs = set()
    pdg_data_edges = set()
    for edge in data_edges:
        if edge.is_def_edge():
            def_identifier_nid, sym_nid = edge.sid, edge.eid
            symbol_code = pdg_nodes[sym_nid].node_code
            def_node_identifier_nids = find_child_identifier_node_ids(def_identifier_nid, pdg_nodes)
            update_count = 0
            for identifier_nid in def_node_identifier_nids:
                # Find the exactly matched identifier for symbol
                if pdg_nodes[identifier_nid].node_code == symbol_code:
                    # TODO: Should we check if only one unique identifier exactly matches?
                    # NOTE: If symbol has been defined previously, we shall update it here
                    symbol_def_nid_map[sym_nid] = identifier_nid
                    update_count += 1
            if update_count != 1:
                print(f'[Warning] DEF edge has update_count={update_count}, edge: {str(edge)}')

        elif edge.is_use_edge():
            use_nid, sym_nid = edge.sid, edge.eid
            # Skip these symbols without explicit definition
            if sym_nid not in symbol_def_nid_map:
                continue
            else:
                def_identifier_nid = symbol_def_nid_map[sym_nid]

            symbol_code = pdg_nodes[sym_nid].node_code
            use_node_identifier_nids = find_child_identifier_node_ids(use_nid, pdg_nodes)
            for use_identifier_nid in use_node_identifier_nids:
                if pdg_nodes[use_identifier_nid].node_code == symbol_code:
                    # Pre-check if use-symbol pair has been processed before
                    use_symbol_nid_pair = f'{use_identifier_nid} {sym_nid}'
                    if use_symbol_nid_pair in existed_use_nid_and_symbol_nid_pairs:
                        continue
                    else:
                        existed_use_nid_and_symbol_nid_pairs.add(use_symbol_nid_pair)

                    def_use_token_edges = intersect_tokens_from_nids([def_identifier_nid],
                                                                     [use_identifier_nid],
                                                                     pdg_nodes,
                                                                     tokens,
                                                                     signature_len,
                                                                     multi_vs_multi_strategy)
                    for token_edge in def_use_token_edges:
                        pdg_data_edges.add(token_edge)

    return pdg_data_edges


def build_token_level_ctrl_pdg(ctrl_edges: List[Edge],
                               pdg_nodes: List[Node],
                               tokens: List[Token],
                               signature_len: int,
                               multi_vs_multi_strategy: str) -> Iterable[str]:
    pdg_ctrl_edges = set()
    # Process control dependencies
    for edge in ctrl_edges:
        if edge.is_ctrl_edge():
            ctrl_sid, ctrl_eid = edge.sid, edge.eid
            # Do not handle node with empty location info
            if pdg_nodes[ctrl_sid].is_empty_loc() or pdg_nodes[ctrl_eid].is_empty_loc():
                continue

            # Find identifier nodes
            # TODO: CTRL edges are built on identifiers, is this logically right?
            ctrl_start_identifier_nids = find_child_identifier_node_ids(ctrl_sid, pdg_nodes)
            ctrl_end_identifier_nids = find_child_identifier_node_ids(ctrl_eid, pdg_nodes)
            ctrl_token_edges = intersect_tokens_from_nids(ctrl_start_identifier_nids,
                                                          ctrl_end_identifier_nids,
                                                          pdg_nodes,
                                                          tokens,
                                                          signature_len,
                                                          multi_vs_multi_strategy)
            for ctrl_token_edge in ctrl_token_edges:
                pdg_ctrl_edges.add(ctrl_token_edge)

    return pdg_ctrl_edges

def build_token_level_data_pdg(reaches_edges: List[Edge],
                               pdg_nodes: List[Node],
                               tokens: List[Token],
                               signature_len: int,
                               multi_vs_multi_strategy: str) -> Iterable[str]:
    """
    Basic logic is using REACHES edge to mine the identifier usage dependencies.
    Thus we still only focus on the identifiers from the nodes of the REACHES edge, and
    every REACHES edge will trigger an attempt of building identifier data dependencies
    between two REACHES nodes.
    """
    all_reaches_token_edges = set()
    for edge in reaches_edges:
        src_reaches_nid, tgt_reaches_nid, identifier_var = edge.sid, edge.eid, edge.var
        src_reaches_identifier_ids = find_child_identifier_node_ids(src_reaches_nid, pdg_nodes)
        tgt_reaches_identifier_ids = find_child_identifier_node_ids(tgt_reaches_nid, pdg_nodes)
        # Filter identifiers not match the value given by REACHES edge
        src_reaches_identifier_ids = list(filter(lambda _id: pdg_nodes[_id].node_code == identifier_var, src_reaches_identifier_ids))
        tgt_reaches_identifier_ids = list(filter(lambda _id: pdg_nodes[_id].node_code == identifier_var, tgt_reaches_identifier_ids))
        reaches_token_edges = intersect_tokens_from_nids(src_reaches_identifier_ids,
                                                         tgt_reaches_identifier_ids,
                                                         pdg_nodes,
                                                         tokens,
                                                         signature_len,
                                                         multi_vs_multi_strategy)
        for token_edge in reaches_token_edges:
            all_reaches_token_edges.add(token_edge)

    return all_reaches_token_edges

def build_token_level_pdg_struct(raw_code: str,
                                 tokens: List[Token],
                                 node_rows: List[List],
                                 edge_lists: List[List],
                                 multi_vs_multi_strategy: str = 'all',
                                 to_build_token_ctrl_edges: bool = False) -> Tuple[Iterable[str], Iterable[str]]:
    assert multi_vs_multi_strategy in ['all', 'first']

    # Build nodes
    pdg_nodes: List[Optional[Node]] = [None] * (len(node_rows)+1)
    for node_row in node_rows:
        node = Node(*node_row)
        nid = node.nid
        pdg_nodes[nid] = node

    # Build edges
    ast_edges: List[Edge] = []
    ctrl_edges: List[Edge] = []
    data_edges: List[Edge] = []
    reaches_edges: List[Edge] = []
    for edge_list in edge_lists:
        edge = Edge.build_edge(edge_list)
        if edge is not None:
            if edge.is_ast_parent_edge():
                ast_edges.append(edge)
            elif edge.is_ctrl_edge():
                ctrl_edges.append(edge)
            elif edge.is_data_edge():
                data_edges.append(edge)
            elif edge.is_reaches_edge():
                reaches_edges.append(edge)

    # Build AST struct
    for edge in ast_edges:
        if edge.is_ast_parent_edge():
            ast_p_id, ast_c_id = edge.sid, edge.eid
            pdg_nodes[ast_p_id].add_ast_child(ast_c_id)
            pdg_nodes[ast_c_id].set_ast_parent(ast_p_id)

    signature_len = raw_code.find('{') + 1

    # Build token edges
    common_params = [pdg_nodes, tokens, signature_len, multi_vs_multi_strategy]
    if to_build_token_ctrl_edges:
        token_ctrl_pdg_edges = build_token_level_ctrl_pdg(ctrl_edges, *common_params)
    else:
        token_ctrl_pdg_edges = set()
    # token_data_pdg_edges_old = build_token_level_data_pdg_old(data_edges, *common_params)
    token_data_pdg_edges = build_token_level_data_pdg(reaches_edges, *common_params)

    return token_ctrl_pdg_edges, token_data_pdg_edges

def intersect_tokens_from_nids(src_nids: Iterable[int],
                               tgt_nids: Iterable[int],
                               nodes: List[Node],
                               tokens: List[Token],
                               signature_len: int,
                               multi_vs_multi_strategy: str) -> Iterable[str]:
    # Parse char spans of selected identifiers
    src_char_spans = parse_and_sort_char_spans_from_nids(src_nids, nodes, signature_len)
    tgt_chat_spans = parse_and_sort_char_spans_from_nids(tgt_nids, nodes, signature_len)
    # Intersect char spans with allennlp tokens
    src_token_indices = intersect_char_spans_with_allennlp_tokens(tokens, src_char_spans)
    tgt_token_indices = intersect_char_spans_with_allennlp_tokens(tokens, tgt_chat_spans)
    # Apply multi v.s. multi handling strategy
    src_token_indices = apply_multi_vs_multi_strategy(src_token_indices, multi_vs_multi_strategy)
    tgt_token_indices = apply_multi_vs_multi_strategy(tgt_token_indices, multi_vs_multi_strategy)

    # Add token-level data edges
    token_to_token_edges = set()
    for sid in src_token_indices:
        for eid in tgt_token_indices:
            # Filter self-loop connection
            if sid != eid:
                token_to_token_edges.add(f'{sid} {eid}')

    return token_to_token_edges

# TODO: Write preprocess into joern_parse code

def convert_func_signature_to_one_line(code_path):
    """
    This function aims to convert the signature of a c/cpp function
    into the uniform one line form, with left brace in a new line.
    This function should be called before calling "joern-parse" to
    help fix the location shift bug of function signature identifiers.

    Example:
        seat_set_active_session (Seat *seat, Session *session)
        {
            ...
        }
    """
    with open(code_path, 'r') as f:
        text = f.read()
        left_bracket_first_idx = text.find('{')
        signature_text = text[:left_bracket_first_idx]
        signature_text = clean_signature_line(signature_text).strip()
        text = signature_text + '\n' + text[left_bracket_first_idx:]

    with open(code_path, 'w') as f:
        f.write(text)

if __name__ == '__main__':
    import time
    raw_code_path = '/data1/zhijietang/dockers/joern-dev/tests/testCode2/test2.cpp'
    node_csv_path = '/data1/zhijietang/dockers/joern-dev/tests/parsed_testCode2/testCode2/test2.cpp/nodes.csv'
    edge_csv_path = '/data1/zhijietang/dockers/joern-dev/tests/parsed_testCode2/testCode2/test2.cpp/edges.csv'
    convert_func_signature_to_one_line(raw_code_path)
    raw_code = load_text(raw_code_path)
    tokenizer = PretrainedTransformerTokenizer('microsoft/codebert-base')

    nodes = read_csv_as_list(node_csv_path)
    edges = read_csv_as_list(edge_csv_path)
    tokens = tokenizer.tokenize(raw_code)
    start_time = time.time()
    ctrl_edges, data_edges = build_token_level_pdg_struct(raw_code, tokens, nodes, edges, multi_vs_multi_strategy='first')
    end_time = time.time()
    print(f'Time: {(end_time - start_time) * 1000} ms')
    # convert_func_signature_to_one_line(raw_code_path)