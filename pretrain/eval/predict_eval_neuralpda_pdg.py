import time
from typing import List, Dict
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import re

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from allennlp.models import Model
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

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


def set_reader(_reader, _max_lines):
    _reader.is_train = False
    # _reader.pdg_max_vertice = _max_lines
    # _reader.max_lines = _max_lines
    return _reader

def build_ground_truth_pdg(nodes, pdg_edges: List[Dict]):
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
    return pdg_ctrl[1:, 1:], pdg_data[1:, 1:]

def process_code(code: str):
    """
        Since in the original tokenization system, multiple new lines were not properly
        handled causing line count error.
        Here we insert a space into them to prevent from this special case.
    """
    while code.find('\n\n') != -1:
        code = code.replace('\n\n', '\n \n')
    code = re.sub(r' +|\t+', ' ', code)
    return code

def align_two_matrix_size(a_mat, b_mat):
    if len(a_mat) == len(b_mat):
        return a_mat, b_mat
    elif len(a_mat) < len(b_mat):
        size = len(a_mat)
        return a_mat, b_mat[:size, :size]
    else:
        size = len(b_mat)
        return a_mat[:size, :size], b_mat

cuda_device = 0
model_path = '/data2/zhijietang/vul_data/run_logs/pretrain/' + '57/' + 'model_epoch_9.tar.gz'
config_path = '/data2/zhijietang/vul_data/run_logs/pretrain/' + '57/' + 'config.json'
# code_path = '/data1/zhijietang/temp/joern_failed_case_1.cpp'
tokenizer_name = 'microsoft/codebert-base'

data_path = '/data2/zhijietang/projects/NeuralPDA-main/datasets/c_8/functions_test.json'

max_lines = 8

# f_output = open("/data1/zhijietang/temp/joern_failed_cases/joern_failed_cases_summary", "w")
# sys.stdout = f_output

torch.cuda.set_device(cuda_device)
print('\n\n')
print(f'[main] Building tokenizer: {tokenizer_name}\n')
tokenizer = PretrainedTransformerTokenizer(tokenizer_name)
print(f'[main] Building model from: {model_path}\n')
model = Model.from_archive(model_path)
model = model.cuda(cuda_device)
print(f'[main] Building reader from: {config_path}\n')
dataset_reader = build_dataset_reader_from_config(config_path)
dataset_reader = set_reader(dataset_reader, max_lines)
predictor = PDGPredictor(model, dataset_reader, frozen=True)


def predict_one_file(code, num_line_elem):
    pdg_output = predictor.predict_pdg(code)
    tokens = tokenizer.tokenize(code)

    cdg_pred = torch.LongTensor(pdg_output['ctrl_edge_labels'])
    ddg_pred = parse_token_level_graph_as_line_level(raw_code, pdg_output['data_edge_labels'], tokens, num_line_elem, start_line=1)

    return cdg_pred, ddg_pred

pdg_ctrl_labels = []
pdg_ctrl_preds = []
pdg_data_labels = []
pdg_data_preds = []

full_records = []

data_items = read_dumped(data_path)
for i, data_item in tqdm(enumerate(data_items)):
    raw_code = data_item['func_code']
    raw_code = process_code(raw_code)
    cdg_preds, ddg_preds = predict_one_file(raw_code, len(data_item['nodes']))
    cdg_labels, ddg_labels = build_ground_truth_pdg(data_item['nodes'], data_item['pdg_edges'])

    full_records.append({
        'index': i,
        'code': raw_code,
        'cdg_preds': cdg_preds.tolist(),
        'cdg_labels': cdg_labels.tolist(),
        'ddg_preds': ddg_preds.tolist(),
        'ddg_labels': ddg_labels.tolist(),
    })

    cdg_preds, cdg_labels = align_two_matrix_size(cdg_preds, cdg_labels)
    ddg_preds, ddg_labels = align_two_matrix_size(ddg_preds, ddg_labels)

    cdg_labels = cdg_labels.flatten().tolist()
    ddg_labels = ddg_labels.flatten().tolist()
    cdg_preds = cdg_preds.flatten().tolist()
    ddg_preds = ddg_preds.flatten().tolist()
    pdg_ctrl_labels.extend(cdg_labels)
    pdg_ctrl_preds.extend(cdg_preds)
    pdg_data_labels.extend(ddg_labels)
    pdg_data_preds.extend(ddg_preds)

    cdg_f1 = f1_score(cdg_labels, cdg_preds) if sum(cdg_labels) != 0 else None
    ddg_f1 = f1_score(ddg_labels, ddg_preds) if sum(ddg_labels) != 0 else None
    full_records[-1].update({
        'cdg_f1': cdg_f1,
        'ddg_f1': ddg_f1,
    })

    assert len(pdg_ctrl_labels) == len(pdg_ctrl_preds), f"\ncode: \n{raw_code}\n\n lens: {len(cdg_labels), len(cdg_preds)}\n"
    assert len(pdg_data_labels) == len(pdg_data_preds), f"\ncode: \n{raw_code}\n\n lens: {len(ddg_labels), len(ddg_preds)}\n"

overall_labels = pdg_ctrl_labels + pdg_data_labels
overall_preds = pdg_ctrl_preds + pdg_data_preds

print(f'Pairs: {len(pdg_ctrl_labels)}')
print(f'Overall Precision: {precision_score(overall_labels, overall_preds)}')
print(f'Overall Recall: {recall_score(overall_labels, overall_preds)}')
print(f'Overall F1: {f1_score(overall_labels, overall_preds)}')

dump_json(full_records, "/data2/zhijietang/temp/pdbert_neuraldpa_data_intri_results.json")