####################################################################################################
# Name-based heuristic baseline for PDG prediction (data).

# 【Impl】: Reuse token-level PDG building utils for AllenNLP tokens to build ground-truth.
# 【Note】: Prediction design should be considered, such as edge connection strategy, type matching and etc.
####################################################################################################

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.file import load_pickle
from utils.pretrain_utils.lexer_based_token_analyse_utils import lexer_analyze_and_make_allennlp_tokens
from utils.joern_utils.joern_dev_pdg_parse_utils import build_token_level_pdg_struct
from utils.joern_utils.joern_dev_parse import convert_func_signature_to_one_line
from utils.pretrain_utils.const import identifier_token_types

target_identifier_types = identifier_token_types

def extract_data_id(file_path: str) -> str:
    file_name = file_path.split('/')[-1]
    data_id = file_name.split('.')[0]
    return data_id

def predict_file(token_data, only_connect_last: bool = True):
    raw_code = convert_func_signature_to_one_line(code=token_data['code'], redump=False)
    # Lexer tokenization & type parsing
    tokens, lex_token_types = lexer_analyze_and_make_allennlp_tokens(raw_code)
    _, token_data_edges = build_token_level_pdg_struct(raw_code, tokens,
                                                       token_data['nodes'], token_data['edges'],
                                                       multi_vs_multi_strategy="first",
                                                       to_build_token_ctrl_edges=False)
    # Stage 1: Init ground-truth edges.
    # Key: identifier src-tgt format string
    # Value: [0]: predict, [1]: ground-truth
    identifier_edges = {}
    for edge in token_data_edges:
        s_token_idx, e_token_idx = edge.split()
        s_token_idx, e_token_idx = int(s_token_idx), int(e_token_idx)
        # print(f"{tokens[s_token_idx]} -> {tokens[e_token_idx]}")
        identifier_edges[edge] = [0,1]

    # Stage 2: Update predicted edges.
    # Note: predictions & ground-truth set should be unioned, not only either one
    pred_data_edges = gen_name_based_data_edges(tokens, lex_token_types, only_connect_last=only_connect_last)
    for edge in pred_data_edges:
        if edge in identifier_edges:
            identifier_edges[edge][0] = 1
        else:
            identifier_edges[edge] = [1,0]

    # Stage 3: Flatten
    flat_pred_list = [v[0] for k,v in identifier_edges.items()]
    flat_gt_list = [v[1] for k,v in identifier_edges.items()]
    return flat_pred_list, flat_gt_list

def gen_name_based_data_edges(lex_tokens, lex_token_types, only_connect_last: bool = False):
    """
        Params:
        - only_connect_last: If an identifier has appeared once more, only generate edge between current and last appeared one, else all.
    """
    token2idxlist = {}
    edges = []

    for i, token in enumerate(lex_tokens):
        # Identifier type filter
        token_text = token.text
        token_type = str(lex_token_types[i])
        if token_type not in target_identifier_types:
            continue

        # Look backward, implicit order constraint
        if token_text in token2idxlist:
            # print(f'Connect {token_text}, {len(token2idxlist[token_text])} items')
            for src_idx in reversed(token2idxlist[token_text]):
                if check_token_match(lex_tokens, lex_token_types, src_idx, i):
                    edges.append(make_edge_str(src_idx, i))
                    if only_connect_last:
                        break
        else:
            token2idxlist[token_text] = []

        # Update
        token2idxlist[token_text].append(i)

    # print(token2idxlist)
    return edges

def make_edge_str(src_idx, tgt_idx):
    return f"{src_idx} {tgt_idx}"

def check_token_match(lex_tokens, lex_token_types, src_token_idx, tgt_token_idx, mode=0):
    """
        Check mode:
        - 0: All
        - 1: Extra Type Match
    """
    if mode == 0:
        return True
    elif mode == 1:
        return str(lex_token_types[src_token_idx]) == str(lex_token_types[tgt_token_idx])
    else:
        raise ValueError(f'Not support match mode: {mode}')



def name_based_data_dep_predict():
    """
        Params:
        - joern_raw_token_data: Raw data from Joern parsing, including nodes and edges.
        - packed_data: Packed pre-training data, including raw code and token-level data edges from pre-trained tokenizers.
    """
    joern_raw_token_data_path_temp = '/data2/zhijietang/vul_data/datasets/joern_vulberta/joern_parsed_raw/joern_parsed_raw_vol{}.pkl'
    packed_data_path_temp = '/data2/zhijietang/vul_data/datasets/pre_trained_vulberta/packed_process_hybrid_data/packed_hybrid_vol_{}.pkl'

    vols = list(range(221, 229))
    all_pred_list = []
    all_groundtruth_list = []

    for vol in vols:
        print(f"· Vol {vol}")
        joern_token_vol_data_path = joern_raw_token_data_path_temp.format(vol)
        packed_vol_data_path = packed_data_path_temp.format(vol)
        joern_token_vol_data = load_pickle(joern_token_vol_data_path)
        packed_token_vol_data = load_pickle(packed_vol_data_path)
        packed_vol_data_id_map = {o['id']: o for o in packed_token_vol_data}

        vol_missed_cnt = 0
        for i, token_data in tqdm(enumerate(joern_token_vol_data)):
            data_id = extract_data_id(token_data['file_path'])
            if data_id not in packed_vol_data_id_map:
                print(f'[Warn] Missing line-level item for vol {vol}, id {data_id}, cnt: {vol_missed_cnt}')
                vol_missed_cnt += 1
                continue
            else:
                token_data['code'] = packed_vol_data_id_map[data_id]['raw_code']

            preds, gts = predict_file(token_data, only_connect_last=False)
            file_recall = recall_score(gts, preds)
            # if file_recall < 0.99:
            #     print(f"[Debug] Vol {vol} id {data_id} (idx={i}) recall != 1: {file_recall}")
            all_pred_list.extend(preds)
            all_groundtruth_list.extend(gts)

    print('\n\n' + "*"*75)
    print(f'Total edges: {len(all_groundtruth_list)}')
    print(f'Accuracy: {accuracy_score(all_groundtruth_list, all_pred_list)}')
    print(f'Precision: {precision_score(all_groundtruth_list, all_pred_list)}')
    print(f'Recall: {recall_score(all_groundtruth_list, all_pred_list)}')
    print(f'F1: {f1_score(all_groundtruth_list, all_pred_list)}')


if __name__ == '__main__':
    name_based_data_dep_predict()

