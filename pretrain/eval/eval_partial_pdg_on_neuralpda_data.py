import re
import time
from typing import List, Dict
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import os

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
from utils.joern_utils.parsing import parse_token_level_graph_as_line_level

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

def build_neuralpda_partial_ground_truth_pdg(nodes, pdg_edges: List[Dict], n_line: int, sliding: bool = False):
    node_num = len(nodes)
    # Since node start from 1, here we use 1 row * column as placeholder
    pdg_ctrl = torch.zeros((node_num+1, node_num+1))
    pdg_data = torch.zeros((node_num+1, node_num+1))
    pdg_ctrl_edges = []
    pdg_data_edges = []
    for edge in pdg_edges:
        start, end, etype = edge['node_out'], edge['node_in'], edge['edge_type']
        start, end = int(start), int(end)
        if etype == 'data_dependency':
            pdg_data_edges.append([start, end])
        else:
            pdg_ctrl_edges.append([start, end])
    pdg_ctrl_edges = torch.LongTensor(pdg_ctrl_edges)
    pdg_data_edges = torch.LongTensor(pdg_data_edges)
    if len(pdg_ctrl_edges) > 0:
        pdg_ctrl[pdg_ctrl_edges[:, 0], pdg_ctrl_edges[:, 1]] = 1
    if len(pdg_data_edges) > 0:
        pdg_data[pdg_data_edges[:, 0], pdg_data_edges[:, 1]] = 1

    # Drop the 1-st placeholder row & column
    pdg_ctrl, pdg_data = pdg_ctrl[1:, 1:], pdg_data[1:, 1:]

    ctrl_labels, data_labels = [], []
    if sliding:
        for i in range(node_num-n_line):
            ctrl_labels.extend(pdg_ctrl[i:i+n_line, i:i+n_line].flatten().tolist())
            data_labels.extend(pdg_data[i:i+n_line, i:i+n_line].flatten().tolist())
    elif len(pdg_ctrl) >= n_line:
        ctrl_labels.extend(pdg_ctrl[:n_line, :n_line].flatten().tolist())
        data_labels.extend(pdg_data[:n_line, :n_line].flatten().tolist())
    return ctrl_labels, data_labels

def build_my_partial_ground_truth_pdg(data_item: Dict,
                                      raw_code: str,
                                      tokens: List[Token],
                                      data_dep_label_key: str,
                                      Ns: List[int]):
    # Since node start from 1, here we use 1 row * column as placeholder
    total_line = raw_code.count('\n')
    if raw_code[-1] != '\n':
        total_line += 1
    total_token = len(tokens)

    pdg_ctrl_graph = torch.zeros((total_line+1, total_line+1))
    pdg_data_graph = torch.zeros((total_token, total_token))        # No need of placeholder for tokens
    if len(data_item['line_edges']) != 0:
        pdg_ctrl_edge_list = torch.LongTensor(data_item['line_edges'])
        pdg_ctrl_graph[pdg_ctrl_edge_list[:,0],pdg_ctrl_edge_list[:,1]] = 1
    # Drop first line
    pdg_ctrl_graph = pdg_ctrl_graph[1:]
    pdg_data_edge_list = data_item['processed_token_data_edges'][data_dep_label_key]
    if len(pdg_data_edge_list) != 0:
        pdg_data_edge_list = torch.LongTensor(pdg_data_edge_list)
        pdg_data_graph[pdg_data_edge_list[:,0],pdg_data_edge_list[:,1]] = 1

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
        while t.idx <= new_line_indices[cur_nl_idx] <= t.idx_end:
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

def split_code_lines(code: str, n_line: int):
    code_lines = code.split("\n")
    code_snippets = []
    for i in range(len(code_lines)-n_line):
        snippet = '\n'.join(code_lines[i:i + n_line])
        # Fix the case where last line of snippet is empty, making line count going wrong
        if code_lines[i+n_line-1] == '':
            snippet += '\n'
        code_snippets.append(snippet)
    return code_snippets

def split_partial_code(code: str, n_line: int):
    code_lines = code.split("\n")
    if len(code_lines) < n_line:
        return []
    else:
        return ['\n'.join(code_lines[:n_line])]

def predict_one_file(code, n):
    # code_snippets = split_code_lines(code, n)
    code_snippets = split_partial_code(code, n)
    ctrl_preds, data_preds = [], []
    for snippet in code_snippets:
        pdg_output = predictor.predict_pdg(snippet)
        c_pred, d_pred = pdg_output['ctrl_edge_preds'], pdg_output['data_edge_preds']
        c_pred = torch.IntTensor(c_pred).flatten().tolist()
        d_pred = torch.IntTensor(d_pred).flatten().tolist()
        ctrl_preds.extend(c_pred)
        data_preds.extend(d_pred)
    return ctrl_preds, data_preds

def process_code(code: str):
    """
        Since in the original tokenization system, multiple new lines were not properly
        handled causing line count error.
        Here we insert a space into them to prevent from this special case.
    """
    while code.find('\n\n') != -1:
        code = code.replace('\n\n', '\n \n')
    # code = re.sub(r' +|\t+', ' ', code)
    return code

def process_my_code(code: str):
    """
        Since in the original tokenization system, multiple new lines were not properly
        handled causing line count error.
        Here we insert a space into them to prevent from this special case.
    """
    if code[-1] != '\n':
        return code + '\n'

cuda_device = 0
model_path = '/data2/zhijietang/vul_data/run_logs/pretrain/' + '57/' + 'model.tar.gz'
config_path = '/data2/zhijietang/vul_data/run_logs/pretrain/' + '57/' + 'config.json'
tokenizer_name = 'microsoft/codebert-base'

max_lines = 8
Ns = [3,4,5,6,7,8]

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

partial_ctrl_preds = {k:[] for k in Ns}
partial_data_preds = {k:[] for k in Ns}
partial_ctrl_labels = {k:[] for k in Ns}
partial_data_labels = {k:[] for k in Ns}

def eval_partial_for_neuralpda_data():
    data_path = '/data2/zhijietang/projects/NeuralPDA-main/datasets/c_8/functions_test.json'

    print("Predicting...")
    data_items = read_dumped(data_path)
    for i, data_item in tqdm(enumerate(data_items), total=len(data_items)):
        raw_code = data_item['func_code']
        raw_code = process_code(raw_code)
        for n in Ns:
            cdg_preds, ddg_preds = predict_one_file(raw_code, n)
            cdg_labels, ddg_labels = build_neuralpda_partial_ground_truth_pdg(data_item['nodes'], data_item['pdg_edges'], n, sliding=False)
            assert len(cdg_preds) == len(cdg_labels), f"CDG: pred ({len(cdg_preds)}) != label ({len(cdg_labels)}). \n- Code: {raw_code}"
            assert len(ddg_preds) == len(ddg_labels), f"DDG: pred ({len(ddg_preds)}) != label ({len(ddg_labels)}). \n- Code: {raw_code}"
            partial_ctrl_labels[n].extend(cdg_labels)
            partial_ctrl_preds[n].extend(cdg_preds)
            partial_data_labels[n].extend(ddg_labels)
            partial_data_preds[n].extend(ddg_preds)

    for n in Ns:
        print("\n" + '*'*50)
        print(f"N = {n}")
        print(f'Pairs: {len(partial_ctrl_labels[n])} each.')
        print(f'Ctrl F1: {f1_score(partial_ctrl_labels[n], partial_ctrl_preds[n])}')
        print(f'Data F1: {f1_score(partial_data_labels[n], partial_data_preds[n])}')
        print(f'Overall F1: {f1_score(partial_ctrl_labels[n]+partial_data_labels[n], partial_ctrl_preds[n]+partial_data_preds[n])}')


def eval_partial_for_my_data():
    print("Building components...")
    data_base_path = "/data2/zhijietang/vul_data/datasets/joern_vulberta/packed_process_hybrid_data/"
    vols = list(range(220, 229))
    pretrained_model = 'microsoft/codebert-base'

    tokenizer = PretrainedTransformerTokenizer(pretrained_model, max_length=256)
    results = {}

    for vol in vols:
        test_file_path = data_base_path + f"packed_hybrid_vol_{vol}.pkl"
        print(f'Eval on Vol.{vol} ...')
        data_items = read_dumped(test_file_path)
        for i, data_item in tqdm(enumerate(data_items), total=len(data_items)):
            raw_code = data_item['raw_code']
            raw_code = process_my_code(raw_code)
            tokens = tokenizer.tokenize(raw_code)
            labels_generator = build_my_partial_ground_truth_pdg(data_item, raw_code, tokens, pretrained_model, Ns)
            max_lines = get_line_count_from_tokens(raw_code, tokens)
            for n in Ns:
                if n <= max_lines:
                    cdg_preds, ddg_preds = predict_one_file(raw_code, n)
                    cdg_labels, ddg_labels = next(labels_generator)
                else:
                    cdg_preds, ddg_preds = [], []
                    cdg_labels, ddg_labels = [], []
                assert len(cdg_preds) == len(cdg_labels), \
                       f"CDG: pred ({len(cdg_preds)}) != label ({len(cdg_labels)}). \n- Code: {raw_code}"
                assert len(ddg_preds) == len(ddg_labels), \
                       f"DDG: pred ({len(ddg_preds)}) != label ({len(ddg_labels)}). \n- Code: {raw_code}"
                partial_ctrl_labels[n].extend(cdg_labels)
                partial_ctrl_preds[n].extend(cdg_preds)
                partial_data_labels[n].extend(ddg_labels)
                partial_data_preds[n].extend(ddg_preds)

    for n in Ns:
        n_result = {
            'pairs': len(partial_ctrl_labels[n]),
            'ctrl_f1': f1_score(partial_ctrl_labels[n], partial_ctrl_preds[n]),
            'data_f1': f1_score(partial_data_labels[n], partial_data_preds[n]),
            'overall_f1': f1_score(partial_ctrl_labels[n] + partial_data_labels[n], partial_ctrl_preds[n] + partial_data_preds[n])
        }
        results[n] = n_result
        print("\n" + '*' * 50)
        print(f"N = {n}")
        print(n_result)

    dump_json(results, "/data2/zhijietang/temp/pdbert_intrin_partial_results.json")

if __name__ == '__main__':
    eval_partial_for_my_data()