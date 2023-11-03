import re
import time
from typing import List, Dict
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy
import sys

sys.path.append("/data2/zhijietang/projects/vul-code-analyzer")

from allennlp.common import JsonDict
from allennlp.data import Instance, Token
from allennlp.predictors import Predictor
from allennlp.models import Model
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from common.modules.code_cleaner import TrivialCodeCleaner
from pretrain import *
from common import *
from utils.file import read_dumped, dump_json
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config
from utils.joern_utils.pretty_print_utils import print_code_with_line_num
from utils.joern_utils.joern_dev_parse import convert_func_signature_to_one_line
from utils.pretrain_utils.mat import remove_consecutive_lines, shift_graph_matrix, shift_edges_in_matrix

cuda_device = 0

class PDGPredictor(Predictor):
    def predict_pdg(self, code: str):
        instance = self._json_to_instance({
            'raw_code': code
        })
        return self.predict_instance(instance)

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        ok, instance = self._dataset_reader.text_to_instance(json_dict)
        if not ok:
            raise ValueError
        else:
            return instance


def set_reader(_reader: CodeAnalyLinePretrainReader, _max_lines):
    _reader.is_train = False
    _reader.code_cleaner = TrivialCodeCleaner()     # To avoid multiple nl elimination, may affect performance
    # _reader.pdg_max_vertice = _max_lines
    # _reader.max_lines = _max_lines
    return _reader

def build_my_partial_ground_truth_pdg(data_item: Dict,
                                      raw_code: str,
                                      tokens: List[Token],
                                      data_dep_label_key: str,
                                      total_line: int,
                                      Ns: List[int]):
    total_token = len(tokens)
    full_total_line = raw_code.count('\n') + 2

    pdg_ctrl_graph = torch.zeros((full_total_line, full_total_line))
    pdg_data_graph = torch.zeros((10000, 10000))        # No need of placeholder for tokens
    if len(data_item['line_edges']) != 0:
        pdg_ctrl_edge_list = torch.LongTensor(data_item['line_edges'])
        pdg_ctrl_graph[pdg_ctrl_edge_list[:,0],pdg_ctrl_edge_list[:,1]] = 1
    # Drop first line
    pdg_ctrl_graph = pdg_ctrl_graph[1:total_line+1, 1:total_line+1]

    pdg_data_edge_list = data_item['processed_token_data_edges'][data_dep_label_key]
    if len(pdg_data_edge_list) != 0:
        pdg_data_edge_list = torch.LongTensor(pdg_data_edge_list)
        pdg_data_graph[pdg_data_edge_list[:,0],pdg_data_edge_list[:,1]] = 1
    pdg_data_graph = pdg_data_graph[:total_token, :total_token]

    # Find the char indices of new-lines
    new_line_indices = []
    for m in re.finditer('\n', raw_code):
        new_line_indices.append(m.start())
    # Add a dummy nl at last to avoid out-of-bound
    new_line_indices.append(1e10)

    line_limit_to_token_idx = {}
    cur_line = 1
    cur_nl_idx = 0
    for i, t in enumerate(tokens):
        if t.idx is None:
            continue
        while t.idx <= new_line_indices[cur_nl_idx] < t.idx_end:
            line_limit_to_token_idx[cur_line] = i
            cur_line += 1
            cur_nl_idx += 1
    # Compensate for missing last nl
    line_limit_to_token_idx[total_line] = len(tokens) - 1

    for n in Ns:
        if n > total_line:
            yield [], []
        else:
            token_n = line_limit_to_token_idx[n]
            yield pdg_ctrl_graph[:n, :n].flatten().tolist(), \
                  pdg_data_graph[:token_n+1, :token_n+1].flatten().tolist()

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
    return cur_line

def split_partial_code(code: str, n_line: int):
    code_lines = code.split("\n")
    return '\n'.join(code_lines[:n_line])

def predict_one_file(code, ctrl_edges, data_edges, n):
    code_snippet = split_partial_code(code, n)
    pdg_output = predictor.predict_pdg(code_snippet)
    ctrl_pred, data_pred = pdg_output['ctrl_edge_labels'], pdg_output['data_edge_labels']
    ctrl_pred = torch.IntTensor(ctrl_pred).flatten().tolist()
    data_pred = torch.IntTensor(data_pred).flatten().tolist()

    ctrl_label, data_label, line_count = dataset_reader.process_test_labels(code_snippet, ctrl_edges, data_edges)
    # Minus one to revert the real matrix.
    ctrl_label = (ctrl_label-1).flatten().tolist()
    data_label = (data_label-1).flatten().tolist()

    if line_count <= n:
        return ctrl_pred, data_pred, ctrl_label, data_label
    else:
        return [], [], [], []

def process_my_code(code: str):
    """
        Since in the original tokenization system, multiple new lines were not properly
        handled causing line count error.
        Here we insert a space into them to prevent from this special case.
    """
    if code[-1] != '\n':
        code = code + '\n'
    return code

def process_result_as_conf_matrix(predicts, labels):
    """
    Return: TN, FN, FP, TP
    """
    p_tensor = numpy.array(predicts, dtype=int)
    l_tensor = numpy.array(labels, dtype=int)
    indices = numpy.arange(len(predicts))
    m = numpy.zeros((len(predicts), 2, 2), dtype=int)
    m[indices, p_tensor, l_tensor] = 1
    return m.sum(0)

def cal_f1_from_conf_matrix(conf_m):
    TN, FN, FP, TP = conf_m.flatten()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return 2*precision*recall / (precision + recall)

def cal_metrics_from_results(preds, labels, ill_fill_val=1, as_np_array=True):
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=ill_fill_val).item()
    recall = recall_score(labels, preds, zero_division=ill_fill_val).item()
    f1 = f1_score(labels, preds, zero_division=ill_fill_val).item()
    return numpy.array([acc, precision, recall, f1]) if as_np_array else (acc, precision, recall, f1)

def cal_metrics_from_conf_mat(conf_m, ill_fill_val=1, as_np_array=True):
    accuracy = (conf_m[0][0] + conf_m[1][1]).item() / conf_m.sum().item()
    p1_cnt = conf_m[1].sum().item()
    l1_cnt = conf_m[:,1].sum().item()
    precision = conf_m[1][1].item() / p1_cnt if p1_cnt > 0 else ill_fill_val
    recall = conf_m[1][1].item() / l1_cnt if l1_cnt > 0 else ill_fill_val
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else ill_fill_val
    return numpy.array([accuracy, precision, recall, f1]) if as_np_array else (accuracy, precision, recall, f1)

PATH_PREFIX = 'data2'

# model_path = f'/{PATH_PREFIX}/zhijietang/vul_data/run_logs/pretrain/' + '57/' + 'model.tar.gz'
# config_path = f'/{PATH_PREFIX}/zhijietang/vul_data/run_logs/pretrain/' + '57/' + 'config.json'
model_path = f'/{PATH_PREFIX}/zhijietang/temp/local_archived_pdbert_base.tar.gz'
config_path = f'/{PATH_PREFIX}/zhijietang/temp/pdbert_archived/raw_config.json'
tokenizer_name = 'microsoft/codebert-base'

max_lines = 50
Ns = [5,10,15,20,25,30]

# f_output = open("/data1/zhijietang/temp/joern_failed_cases/joern_failed_cases_summary", "w")
# sys.stdout = f_output

torch.cuda.set_device(cuda_device)
print('\n\n')
# print(f'[main] Building tokenizer: {tokenizer_name}\n')
# tokenizer = PretrainedTransformerTokenizer(tokenizer_name)
print(f'[main] Building model from: {model_path}\n')
model = Model.from_archive(model_path)
model = model.cuda(cuda_device)
print(f'[main] Building reader from: {config_path}\n')
dataset_reader = build_dataset_reader_from_config(config_path)
dataset_reader = set_reader(dataset_reader, max_lines)
predictor = PDGPredictor(model, dataset_reader, frozen=True)

# partial_ctrl_preds = {k:numpy.zeros((2,2), dtype=int) for k in Ns}
# partial_data_preds = {k:[] for k in Ns}
# partial_ctrl_labels = {k:[] for k in Ns}
# partial_data_labels = {k:[] for k in Ns}

# micro sub-metrics: TP, FP, TN, FN
partial_ctrl_micro_results = {k:numpy.zeros((2, 2), dtype=int) for k in Ns}
partial_data_micro_results = {k:numpy.zeros((2, 2), dtype=int) for k in Ns}
# macro sub-metrics: acc., prec., rec., f1
partial_ctrl_macro_results = {k:numpy.zeros((4,)) for k in Ns}
partial_data_macro_results = {k:numpy.zeros((4,)) for k in Ns}
partial_overall_macro_results = {k:numpy.zeros((4,)) for k in Ns}
partial_counts = {n:0 for n in Ns}

def eval_partial_for_my_data(data_base_path,
                             file_name_temp,
                             dump_path,
                             vol_range,
                             max_tokens=512,
                             rm_consecutive_nls=False,
                             reduce: str = 'micro'):
    print("Building components...")
    svol, evol = vol_range
    vols = list(range(svol, evol+1))
    pretrained_model = 'microsoft/codebert-base'
    # tokenizer = PretrainedTransformerTokenizer(pretrained_model, max_length=max_tokens)
    tokenizer = PretrainedTransformerTokenizer("/data2/zhijietang/temp/codebert-base",
                                               max_length=max_tokens)
    results = {}
    total_instance = 0

    for vol in vols:
        test_file_path = data_base_path + file_name_temp.format(vol)
        print(f'Eval on Vol.{vol} ...')
        data_items = read_dumped(test_file_path)    # [:100]
        total_instance += len(data_items)
        for i, data_item in tqdm(enumerate(data_items), total=len(data_items)):
            raw_code = data_item['raw_code']
            raw_code = process_my_code(convert_func_signature_to_one_line(code=raw_code, redump=False))
            if rm_consecutive_nls:
                raw_code, del_line_indices = remove_consecutive_lines(raw_code)
            else:
                del_line_indices = None
            tokens = tokenizer.tokenize(raw_code)
            max_lines = get_line_count_from_tokens(raw_code, tokens)

            cdg_edges = data_item['line_edges']
            if rm_consecutive_nls:
                cdg_edges = shift_edges_in_matrix(cdg_edges, del_line_indices)
            ddg_edges = data_item['processed_token_data_edges'][pretrained_model]

            # labels_generator = build_my_partial_ground_truth_pdg(data_item, raw_code, tokens, pretrained_model, max_lines, Ns)
            for n in Ns:
                if n <= max_lines:
                    try:
                        cdg_preds, ddg_preds, cdg_labels, ddg_labels = predict_one_file(raw_code,
                                                                                        cdg_edges,
                                                                                        ddg_edges,
                                                                                        n)
                    except Exception as e:
                        print(f"Error when predicting #{i}, n={n}, err: {e}, skipped")
                        continue

                    assert len(cdg_preds) == len(cdg_labels), \
                           f"CDG: pred ({len(cdg_preds)}) != label ({len(cdg_labels)}). \n- Code: {raw_code}"
                    assert len(ddg_preds) == len(ddg_labels), \
                           f"DDG: pred ({len(ddg_preds)}) != label ({len(ddg_labels)}). \n- Code: {raw_code}"

                    if reduce == 'micro':
                        ctrl_res_m = process_result_as_conf_matrix(cdg_preds, cdg_labels)
                        data_res_m = process_result_as_conf_matrix(ddg_preds, ddg_labels)
                        partial_ctrl_micro_results[n] += ctrl_res_m
                        partial_data_micro_results[n] += data_res_m
                    elif reduce == 'macro':
                        # ctrl_res = cal_metrics_from_results(cdg_preds, cdg_labels, ill_fill_val=1)
                        # data_res = cal_metrics_from_results(ddg_preds, ddg_labels, ill_fill_val=1)
                        # overall_res = cal_metrics_from_results(cdg_preds+ddg_preds, cdg_labels+ddg_labels, ill_fill_val=1)
                        ctrl_res_m = process_result_as_conf_matrix(cdg_preds, cdg_labels)
                        data_res_m = process_result_as_conf_matrix(ddg_preds, ddg_labels)
                        overall_res_m = process_result_as_conf_matrix(cdg_preds+ddg_preds, cdg_labels+ddg_labels)
                        ctrl_res = cal_metrics_from_conf_mat(ctrl_res_m, ill_fill_val=1)
                        data_res = cal_metrics_from_conf_mat(data_res_m, ill_fill_val=1)
                        overall_res = cal_metrics_from_conf_mat(overall_res_m, ill_fill_val=1)
                        partial_ctrl_macro_results[n] += ctrl_res
                        partial_data_macro_results[n] += data_res
                        partial_overall_macro_results[n] += overall_res
                    else:
                        raise ValueError

                    ctrl_res_m = process_result_as_conf_matrix(cdg_preds, cdg_labels)
                    data_res_m = process_result_as_conf_matrix(ddg_preds, ddg_labels)

                    # c_pairs = int(partial_ctrl_micro_results[n].sum())
                    # if c_pairs > 0:
                    #     ctrl_f1_item = cal_f1_from_conf_matrix(ctrl_res_m)
                    #     real_n = int(len(cdg_preds) ** 0.5)
                    #     pred_edges = torch.LongTensor(cdg_preds).reshape((real_n,real_n)).nonzero()
                    #     label_edges = torch.LongTensor(cdg_labels).reshape((real_n,real_n)).nonzero()

                    partial_ctrl_micro_results[n] += ctrl_res_m
                    partial_data_micro_results[n] += data_res_m
                    partial_counts[n] += 1

    for n in Ns:
        c_pairs = int(partial_ctrl_micro_results[n].sum())
        d_pairs = int(partial_data_micro_results[n].sum())
        c_f1 = cal_f1_from_conf_matrix(partial_ctrl_micro_results[n]) if c_pairs > 0 else None
        d_f1 = cal_f1_from_conf_matrix(partial_data_micro_results[n]) if d_pairs > 0 else None
        overall_f1 = cal_f1_from_conf_matrix(partial_ctrl_micro_results[n] + partial_data_micro_results[n]) if c_pairs+d_pairs > 0 else None
        n_result = {
            'total_instance': total_instance,
            'total_valid_instance': partial_counts[n],
            'c_pairs': c_pairs,
            'd_pairs': d_pairs,
            'ctrl_f1': c_f1 ,
            'data_f1': d_f1,
            'overall_f1': overall_f1
        }
        results[n] = n_result
        print("\n" + '*' * 50)
        print(f"N = {n}")
        print(n_result)

    dump_json(results, dump_path)

if __name__ == '__main__':
    # eval_partial_for_my_data(data_base_path="/data2/zhijietang/vul_data/datasets/joern_vulberta/packed_process_hybrid_data/",
    #                          file_name_temp="packed_hybrid_vol_{}.pkl",
    #                          dump_path="/data2/zhijietang/temp/icse2024_intrin/pdbert_intrin_partial_256_results.json",
    #                          vol_range=(9999,9999),
    #                          max_tokens=256)
    eval_partial_for_my_data(data_base_path="/data2/zhijietang/vul_data/datasets/docker/fan_dedup/tokenized_packed_fixed/",
                             file_name_temp="packed_hybrid_vol_{}.pkl",
                             dump_path="/data2/zhijietang/temp/icse2024_intrin/bigvul_fixed_partial_512.json",
                             vol_range=(0,1),
                             max_tokens=512,
                             rm_consecutive_nls=False)