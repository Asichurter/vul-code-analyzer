####################################################################################################
# Name-based heuristic baseline for PDG prediction (data).

# 【Impl】: Reuse token-level PDG building utils for AllenNLP tokens to build ground-truth.
# 【Note】: Prediction design should be considered, such as edge connection strategy, type matching and etc.
####################################################################################################
import sys
from typing import List

from allennlp.data import Token
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import re
import pandas

sys.path.append("../../../")

from utils.file import load_pickle, dump_json, read_dumped
from utils.pretrain_utils.lexer_based_token_analyse_utils import lexer_analyze_and_make_allennlp_tokens
from utils.joern_utils.joern_dev_pdg_parse_utils import build_token_level_pdg_struct
from utils.joern_utils.joern_dev_parse import convert_func_signature_to_one_line
from utils.pretrain_utils.const import identifier_token_types

target_identifier_types = identifier_token_types

def extract_data_id(file_path: str) -> str:
    file_name = file_path.split('/')[-1]
    data_id = file_name.split('.')[0]
    return data_id

def get_line_count_from_tokens(raw_code: str, tokens: List[Token]):
    """
        Compute the line count from tokens, where raw_code is complete but tokens are partial.
    """
    # Find the char indices of new-lines
    new_line_indices = []
    for m in re.finditer('\n', raw_code):
        new_line_indices.append(m.start())
    # Add a dummy nl at last to avoid out-of-bound
    new_line_indices.append(1e10)

    cur_line = 0
    cur_nl_idx = 0
    for i, t in enumerate(tokens):
        if t.idx is None:
            continue
        while t.idx <= new_line_indices[cur_nl_idx] <= t.idx_end:
            cur_line += 1
            cur_nl_idx += 1
    return cur_line + 1

def predict_file(token_data, only_connect_last: bool = True):
    """
        【Deprecated, not use】
    """
    raw_code = convert_func_signature_to_one_line(code=token_data['code'], redump=False)
    # Lexer tokenization & type parsing
    tokens, lex_token_types = lexer_analyze_and_make_allennlp_tokens(raw_code, max_len=None)
    _, token_data_edges = build_token_level_pdg_struct(raw_code, tokens,
                                                       token_data['nodes'], token_data['edges'],
                                                       multi_vs_multi_strategy="first",
                                                       to_build_token_ctrl_edges=False)
    predictions = torch.zeros((len(tokens), len(tokens)))
    labels = torch.zeros((len(tokens), len(tokens)))

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

def predict_file_v2(token_data, only_connect_last: bool = True):
    """
        Include all the pairs of tokens from lexer, not identifiers only.

        Complete version.
    """
    raw_code = convert_func_signature_to_one_line(code=token_data['code'], redump=False)
    # Lexer tokenization & type parsing
    tokens, lex_token_types = lexer_analyze_and_make_allennlp_tokens(raw_code, max_len=512)
    tokenized_line_cnt = get_line_count_from_tokens(raw_code, tokens)
    full_line_cnt = get_line_count_from_raw_code(raw_code, tokens)
    # Skip instances not full covered
    if tokenized_line_cnt < full_line_cnt:
        # print(f"tokenized_line_cnt({tokenized_line_cnt}) < full_line_cnt({full_line_cnt}), skipped")
        return {}

    # Stage 1: Generate ground-truth labels.
    _, token_data_edges = build_token_level_pdg_struct(raw_code, tokens,
                                                       token_data['nodes'], token_data['edges'],
                                                       multi_vs_multi_strategy="first",
                                                       to_build_token_ctrl_edges=False)
    predictions = torch.zeros((len(tokens), len(tokens)), dtype=torch.long)
    labels = torch.zeros((len(tokens), len(tokens)), dtype=torch.long)
    for edge in token_data_edges:
        s_token_idx, e_token_idx = edge.split()
        s_token_idx, e_token_idx = int(s_token_idx), int(e_token_idx)
        # print(f"{tokens[s_token_idx]} -> {tokens[e_token_idx]}")
        labels[s_token_idx][e_token_idx] = 1

    # Stage 2: Generate predictions based on name-based heuristics.
    pred_data_edges = gen_name_based_data_edges(tokens, lex_token_types, only_connect_last=only_connect_last)
    for edge in pred_data_edges:
        s_token_idx, e_token_idx = edge.split()
        s_token_idx, e_token_idx = int(s_token_idx), int(e_token_idx)
        predictions[s_token_idx][e_token_idx] = 1

    # Stage 3: Flatten
    flat_pred_list = torch.flatten(predictions) # .tolist()
    flat_gt_list = torch.flatten(labels) # .tolist()

    # Split point hash: Map line count to interval for stat
    line_cnt_level = min(30, tokenized_line_cnt // 10 * 10)
    return {
        line_cnt_level: (flat_pred_list, flat_gt_list)
    }

def predict_file_v2_partial(token_data, Ns: List[int], only_connect_last: bool = True):
    """
        Include all the pairs of tokens from lexer, not identifiers only.

        Partial code version.
    """
    raw_code = convert_func_signature_to_one_line(code=token_data['code'], redump=False)
    # Lexer tokenization & type parsing
    full_tokens, _ = lexer_analyze_and_make_allennlp_tokens(raw_code, max_len=512)
    tokenized_line_cnt = get_line_count_from_tokens(raw_code, full_tokens)
    results = {}

    for N in Ns:
        if N > tokenized_line_cnt:
            continue
        partial_raw_code = split_partial_code(raw_code, N)
        partial_tokens, partial_token_types = lexer_analyze_and_make_allennlp_tokens(partial_raw_code, max_len=512)

        # Stage 1: Generate ground-truth labels.
        _, token_data_edges = build_token_level_pdg_struct(partial_raw_code, partial_tokens,
                                                           token_data['nodes'], token_data['edges'],
                                                           multi_vs_multi_strategy="first",
                                                           to_build_token_ctrl_edges=False)
        predictions = torch.zeros((len(partial_tokens), len(partial_tokens)), dtype=torch.long)
        labels = torch.zeros((len(partial_tokens), len(partial_tokens)), dtype=torch.long)
        for edge in token_data_edges:
            s_token_idx, e_token_idx = edge.split()
            s_token_idx, e_token_idx = int(s_token_idx), int(e_token_idx)
            # print(f"{tokens[s_token_idx]} -> {tokens[e_token_idx]}")
            labels[s_token_idx][e_token_idx] = 1

        # Stage 2: Generate predictions based on name-based heuristics.
        pred_data_edges = gen_name_based_data_edges(partial_tokens, partial_token_types, only_connect_last=only_connect_last)
        for edge in pred_data_edges:
            s_token_idx, e_token_idx = edge.split()
            s_token_idx, e_token_idx = int(s_token_idx), int(e_token_idx)
            predictions[s_token_idx][e_token_idx] = 1

        # Stage 3: Flatten
        flat_pred_list = torch.flatten(predictions) # .tolist()
        flat_gt_list = torch.flatten(labels) # .tolist()
        results[N] = (flat_pred_list, flat_gt_list)
    return results

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


def cal_micro_sub_metrics(predictions, labels):
    TP = torch.sum((predictions == 1) * (labels == 1)).item()
    FP = torch.sum((predictions == 1) * (labels == 0)).item()
    TN = torch.sum((predictions == 0) * (labels == 0)).item()
    FN = torch.sum((predictions == 0) * (labels == 1)).item()
    return TP, FP, TN, FN, len(predictions)

def cal_detailed_metrics(TP, FP, TN, FN, ill_fill_val=1):
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if TP + FP != 0 else ill_fill_val
    recall = TP / (TP + FN)    if TP + FN != 0 else ill_fill_val
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return accuracy, precision, recall, f1

def post_process_results(results, reduce, macro_ill_default=1):
    reduced = {}
    for key in results:
        preds, labels = results[key]
        if reduce == 'micro':
            micros = cal_micro_sub_metrics(preds, labels)
            reduced[key] = micros
        elif reduce == 'macro':
            micros = cal_micro_sub_metrics(preds, labels)
            macros = cal_detailed_metrics(*micros[:4], ill_fill_val=macro_ill_default)
            reduced[key] = (*macros, 1)     # instance count
        else:
            raise ValueError(f"reduced={reduced}")
    return reduced

def vol_accumulate_metrics(metrics, reduce):
    vol_metric = {}
    if reduce == 'micro':
        for key in metrics:
            vol_metric[key] = {}
            metric_names = ["TP", "FP", "TN", "FN", "lens"]
            for metric_name, val in zip(metric_names, metrics[key]):
                vol_metric[key][metric_name] = val
    elif reduce == 'macro':
        for key in metrics:
            vol_metric[key] = {}
            metric_names = ["accuracy", "precision", "recall", "f1", "lens"]
            count = metrics[key][-1]
            for metric_name, val in zip(metric_names, metrics[key]):
                vol_metric[key][metric_name] = val / count
            # recover count val
            vol_metric[key]["lens"] *= count
    return vol_metric


def name_based_data_dep_predict(predict_version, vols,
                                Ns=None,
                                reduce='micro',
                                only_connect_last=False,
                                dump_base_path: str = "/data2/zhijietang/temp/icse2024_mater/revision/"):
    """
        Params:
        - joern_raw_token_data: Raw data from Joern parsing, including nodes and edges.
        - packed_data: Packed pre-training data, including raw code and token-level data edges from pre-trained tokenizers.
        - predict_version: Elements to be predicted.
            - v1: Only identifer pairs, union predictions and labels.
            - v2: All the lexer token pairs, aligned with PDBERT.
    """
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # vlis7
    joern_raw_token_data_path_temp = '/data2/zhijietang/vul_data/datasets/joern_vulberta/joern_parsed_raw/joern_parsed_raw_vol{}.pkl'
    packed_data_path_temp = '/data2/zhijietang/vul_data/datasets/pre_trained_vulberta/packed_process_hybrid_data/packed_hybrid_vol_{}.pkl'
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # vlis6
    # joern_raw_token_data_path_temp = '/data1/zhijietang/vul_data/datasets/docker/joern_dev_analysis_results/joern_parsed_raw_vol{}.pkl'
    # packed_data_path_temp = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_process_hybrid_data/packed_hybrid_vol_{}.pkl'
    # ----------------------------------------------------------------------------------------------------------------------------------------

    # vols = list(range(221, 229))
    # all_pred_list = []
    # all_groundtruth_list = []
    # 【TP, FP, TN, FN, len】

    def update_sub_metric(_sub_metrics, _res):
        for key in _res:
            for i in range(len(_res[key])):
                _sub_metrics[key][i] += _res[key][i]
            # # incr len
            # _sub_metrics[key][-1] += 1

    for vol in vols:
        print(f"· Vol {vol}")
        vol_instance_cnt = 0
        if reduce == 'micro':
            sub_metric_len = 5      # TP, FP, TN, FN, len
        elif reduce == 'macro':
            sub_metric_len = 5      # acc, pre, rec, f1, len
        else:
            raise ValueError(f"reduce={reduce}")
        if Ns is None:
            sub_metrics = {i*10: [0]*sub_metric_len for i in range(4)}
        else:
            sub_metrics = {N: [0]*sub_metric_len for N in Ns}

        joern_token_vol_data_path = joern_raw_token_data_path_temp.format(vol)
        packed_vol_data_path = packed_data_path_temp.format(vol)
        joern_token_vol_data = load_pickle(joern_token_vol_data_path)
        packed_token_vol_data = load_pickle(packed_vol_data_path)
        packed_vol_data_id_map = {o['id']: o for o in packed_token_vol_data}

        vol_missed_cnt = 0
        for i, token_data in tqdm(enumerate(joern_token_vol_data), total=len(joern_token_vol_data)):
            data_id = extract_data_id(token_data['file_path'])
            if data_id not in packed_vol_data_id_map:
                print(f'[Warn] Missing line-level item for vol {vol}, id {data_id}, cnt: {vol_missed_cnt}')
                vol_missed_cnt += 1
                continue
            else:
                token_data['code'] = packed_vol_data_id_map[data_id]['raw_code']

            if predict_version == 'full_v1':
                results = predict_file(token_data, only_connect_last=only_connect_last)
            elif predict_version == 'full_v2':
                results = predict_file_v2(token_data, only_connect_last=only_connect_last)
            elif predict_version == 'partial':
                results = predict_file_v2_partial(token_data, Ns=Ns, only_connect_last=only_connect_last)
            else:
                raise ValueError
            # file_recall = recall_score(gts, preds)
            # if file_recall < 0.99:
            #     print(f"[Debug] Vol {vol} id {data_id} (idx={i}) recall != 1: {file_recall}")
            # all_pred_list.extend(preds)
            # all_groundtruth_list.extend(gts)
            processed_results = post_process_results(results, reduce, macro_ill_default=1.)
            update_sub_metric(sub_metrics, processed_results)
            if len(processed_results) > 0:
                vol_instance_cnt += 1

        edge_type = 'edgel' if only_connect_last else 'edgef'
        file_name = f"heuristic_results_{predict_version}_{reduce}_{edge_type}_vol{vol}.json"
        dump_json(vol_accumulate_metrics(sub_metrics, reduce),
        path=f"{dump_base_path}{file_name}")
        print(f"\nVol instance count: {vol_instance_cnt}, dump to {dump_base_path}{file_name}\n")

    # TP, FP, TN, FN, lens = sub_metrics
    # print(f'Accuracy: {accuracy_score(all_groundtruth_list, all_pred_list)}')
    # print(f'Precision: {precision_score(all_groundtruth_list, all_pred_list)}')
    # print(f'Recall: {recall_score(all_groundtruth_list, all_pred_list)}')
    # print(f'F1: {f1_score(all_groundtruth_list, all_pred_list)}')


def get_line_count_from_raw_code(raw_code: str, tokens: List[Token]):
    if raw_code[-1] != '\n':
        res = 1
    else:
        res = 0
    return raw_code.count('\n') + res


def split_partial_code(code: str, n_line: int):
    code_lines = code.split("\n")
    return '\n'.join(code_lines[:n_line])


def yield_predict_instances(raw_data_item, Ns=None):
    """
        "Ns=None" means "not partial", otherwise gives split points.
    """
    if Ns is None:
        yield "full", raw_data_item
    else:
        for N in Ns:
            raw_code = raw_data_item["code"]
            raw_data_item["code"] = split_partial_code(raw_code, N)


def cal_final_micro_results_from_sub_metrics(path_temp, vols, keys):
    metric_names = ['TP', 'FP', 'TN', 'FN', 'lens']
    metrics = {k:{m:0 for m in metric_names} for k in keys}
    # print(metrics)
    for vol in vols:
        vol_res = read_dumped(path_temp.format(vol))
        # print(vol_res)
        for key in keys:
            for i, metric_name in enumerate(metric_names):
                metrics[key][metric_name] += vol_res[key][metric_name]

    df = pandas.DataFrame(dtype=float)
    for key in keys:
        ms = metrics[key]
        TP, FP, TN, FN, lens = ms['TP'], ms['FP'], ms['TN'], ms['FN'], ms['lens']
        print(f'\n\nN={key}')
        print("*"*75)
        print('【meta-info】')
        print(f'Total edges: {lens}')
        print('-' * 75)
        print('【sub-metrics】')
        print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}')
        print('-'*75)
        print('【detailed metrics】')
        accuracy, precision, recall, f1 = cal_detailed_metrics(TP, FP, TN, FN)
        print(f'Accuracy: {round(accuracy * 100, 2)}')
        print(f'Precision: {round(precision * 100, 2)}')
        print(f'recall: {round(recall * 100, 2)}')
        print(f'F1: {round(f1 * 100, 2)}')
        df = df.append([
            [float(key), round(accuracy * 100, 2), round(precision * 100, 2), round(recall * 100, 2), round(f1 * 100, 2)]],
            ignore_index=True)
    df.to_csv(f"/data2/zhijietang/temp/icse2024_mater/baseline/revision/partial/agg_res_micro.csv")

def cal_final_macro_results_from_sub_metrics(path_temp, vols, keys):
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'lens']
    metrics = {k:{m:0 for m in metric_names} for k in keys}
    # print(metrics)
    for vol in vols:
        vol_res = read_dumped(path_temp.format(vol))
        # print(vol_res)
        for key in keys:
            for i, metric_name in enumerate(metric_names):
                metrics[key][metric_name] += vol_res[key][metric_name]

    df = pandas.DataFrame(dtype=float)
    for key in keys:
        ms = metrics[key]
        vol_cnt = len(vols)
        accuracy, precision, recall, f1, lens = ms['accuracy']/vol_cnt, ms['precision']/vol_cnt, \
                                                ms['recall']/vol_cnt, ms['f1']/vol_cnt, ms['lens']
        print(f'\n\nN={key}')
        print("*"*75)
        print('【meta-info】')
        print(f'Total edges: {lens}')
        print('-'*75)
        print('【detailed metrics】')
        print(f'Accuracy: {round(accuracy * 100, 2)}')
        print(f'Precision: {round(precision * 100, 2)}')
        print(f'recall: {round(recall * 100, 2)}')
        print(f'F1: {round(f1 * 100, 2)}')

        df = df.append([
            [float(key), round(accuracy * 100, 2), round(precision * 100, 2), round(recall * 100, 2),
             round(f1 * 100, 2)]],
            ignore_index=True)
    df.to_csv(f"/data2/zhijietang/temp/icse2024_mater/baseline/revision/partial/agg_res_macro.csv")

if __name__ == '__main__':
    # Set "Ns=None" to run full-version, elsewise run partial-version
    # Set "reduce='micro'" to run original flatten metric calculation, elsewise run "function-level" calculation
    # name_based_data_dep_predict(predict_version='full_v2', vols=list(range(221,229)),
    #                             Ns=None,
    #                             only_connect_last=True, reduce="micro")
    # name_based_data_dep_predict(predict_version='full_v2', vols=list(range(221,229)),
    #                             Ns=None,
    #                             only_connect_last=True, reduce="macro")
    # name_based_data_dep_predict(predict_version='partial', vols=list(range(221,229)),
    #                             Ns=[5,10,15,20,25,30],
    #                             only_connect_last=True, reduce="micro")
    # name_based_data_dep_predict(predict_version='partial', vols=list(range(221,229)),
    #                             Ns=[5,10,15,20,25,30],
    #                             only_connect_last=True, reduce="macro",
    #                             dump_base_path="/data2/zhijietang/temp/icse2024_mater/revision/partial/")
    # name_based_data_dep_predict(predict_version='full_v2', vols=list(range(221,229)),
    #                             Ns=None,
    #                             only_connect_last=False, reduce="micro",
    #                             dump_base_path="/data2/zhijietang/temp/icse2024_mater/baseline/revision/full/")
    # name_based_data_dep_predict(predict_version='full_v2', vols=list(range(221,229)),
    #                             Ns=None,
    #                             only_connect_last=False, reduce="macro",
    #                             dump_base_path="/data2/zhijietang/temp/icse2024_mater/baseline/revision/full/")
    # name_based_data_dep_predict(predict_version='partial', vols=list(range(221,229)),
    #                             Ns=[5,10,15,20,25,30],
    #                             only_connect_last=False, reduce="micro",
    #                             dump_base_path="/data2/zhijietang/temp/icse2024_mater/baseline/revision/partial/")
    # name_based_data_dep_predict(predict_version='partial', vols=list(range(221,229)),
    #                             Ns=[5,10,15,20,25,30],
    #                             only_connect_last=False, reduce="macro",
    #                             dump_base_path="/data2/zhijietang/temp/icse2024_mater/baseline/revision/partial/")


    # cal_final_micro_results_from_sub_metrics("/data2/zhijietang/temp/icse2024_mater/revision/full/heuristic_results_full_v2_micro_edgel_vol{}.json",
    #                                    list(range(221,229)),
    #                                    keys = ['0', '10', '20', '30'])
    # cal_final_macro_results_from_sub_metrics("/data2/zhijietang/temp/icse2024_mater/revision/full/heuristic_results_full_v2_macro_edgel_vol{}.json",
    #                                          list(range(221, 229)),
    #                                          keys=['0', '10', '20', '30'])
    # cal_final_micro_results_from_sub_metrics("/data2/zhijietang/temp/icse2024_mater/revision/partial/heuristic_results_partial_micro_edgel_vol{}.json",
    #                                          list(range(221, 229)),
    #                                          keys=['5','10','15','20','25','30'])
    # cal_final_macro_results_from_sub_metrics("/data2/zhijietang/temp/icse2024_mater/revision/partial/heuristic_results_partial_macro_edgel_vol{}.json",
    #                                          list(range(221, 229)),
    #                                          keys=['5','10','15','20','25','30'])
    cal_final_micro_results_from_sub_metrics("/data2/zhijietang/temp/icse2024_mater/baseline/revision/full/heuristic_results_full_v2_micro_edgef_vol{}.json",
                                       list(range(221,229)),
                                       keys = ['0', '10', '20', '30'])
    cal_final_macro_results_from_sub_metrics("/data2/zhijietang/temp/icse2024_mater/baseline/revision/full/heuristic_results_full_v2_macro_edgef_vol{}.json",
                                             list(range(221, 229)),
                                             keys=['0', '10', '20', '30'])
    cal_final_micro_results_from_sub_metrics("/data2/zhijietang/temp/icse2024_mater/baseline/revision/partial/heuristic_results_partial_micro_edgef_vol{}.json",
                                             list(range(221, 229)),
                                             keys=['5','10','15','20','25','30'])
    cal_final_macro_results_from_sub_metrics("/data2/zhijietang/temp/icse2024_mater/baseline/revision/partial/heuristic_results_partial_macro_edgef_vol{}.json",
                                             list(range(221, 229)),
                                             keys=['5','10','15','20','25','30'])

    # print('* Full *')
    # cal_final_results_from_sub_metrics("/data2/zhijietang/temp/icse2024_mater/heuristic_results_v2_full_vol{}.json", list(range(221,229)))
    # print('* Last *')
    # cal_final_results_from_sub_metrics("/data2/zhijietang/temp/icse2024_mater/heuristic_results_v2_last_vol{}.json", list(range(221,229)))

