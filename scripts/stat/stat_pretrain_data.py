import sys
from tqdm import tqdm
import numpy
import lizard

sys.path.extend([f'/data2/zhijietang/projects/vul-code-analyzer'])

from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from common.modules.code_cleaner import TrivialCodeCleaner
from pretrain import *
from common import *
from utils.file import read_dumped, dump_json
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config


model_path = '/data2/zhijietang/vul_data/run_logs/pretrain/' + '57/' + 'model.tar.gz'
config_path = '/data2/zhijietang/vul_data/run_logs/pretrain/' + '57/' + 'config.json'
tokenizer_name = 'microsoft/codebert-base'
data_base_path = "/data2/zhijietang/vul_data/datasets/joern_vulberta/packed_process_hybrid_data/"

print(f'[main] Building reader from: {config_path}\n')
dataset_reader: PackedHybridLineTokenPDGDatasetReader = build_dataset_reader_from_config(config_path)

tokenizer = PretrainedTransformerTokenizer(tokenizer_name)

def cal_raw_stat(data_item):
    raw_code = data_item['raw_code']
    raw_tokens = tokenizer.tokenize(raw_code)
    line_count = raw_code.count('\n')
    token_count = len(raw_tokens)
    cc = cal_cc(raw_code)  # todo
    ctrl_edges = len(data_item['line_edges'])
    data_edges = len(data_item['processed_token_data_edges'][tokenizer_name])
    return {
        '#token': token_count,
        '#line': line_count,
        '#cc': cc,
        '#cdg_node': line_count,
        '#cdg_edge': ctrl_edges,
        '#ddg_node': token_count,
        '#ddg_edge': data_edges,
    }

def cal_truncate_stat(data_item):
    raw_code = data_item['raw_code']
    ctrl_edges = data_item['line_edges']
    data_edges = data_item['processed_token_data_edges'][tokenizer_name]
    ctrl_matrix, data_matrix, line_count = dataset_reader.process_test_labels(raw_code, ctrl_edges, data_edges)
    # Revert boolean matrix
    ctrl_matrix -= 1
    data_matrix -= 1
    cc = cal_cc(raw_code)
    return {
        '#token': len(data_matrix),
        '#line': len(ctrl_matrix),
        '#cc': cc,
        '#cdg_node': len(ctrl_matrix),
        '#cdg_edge': ctrl_matrix.sum().item(),
        '#ddg_node': len(data_matrix),
        '#ddg_edge': data_matrix.sum().item(),
    }

def update_list_dict(src, new_data, exclude_none=True):
    for k, v in new_data.items():
        if v is None and exclude_none:
            continue
        src[k].append(v)

def cal_cc(raw_code: str):
    res = lizard.analyze_file.analyze_source_code("AllTests.cpp", raw_code)
    if len(res.__dict__['function_list']) > 0:
        function_dict = res.__dict__['function_list'][0].__dict__
        return function_dict['cyclomatic_complexity']
    else:
        return None

raw_stat_results = {
    '#token': [],
    '#line': [],
    '#cc': [],
    '#cdg_node': [],
    '#cdg_edge': [],
    '#ddg_node': [],
    '#ddg_edge': [],
}
truncate_stat_results = {
    '#token': [],
    '#line': [],
    '#cc': [],
    '#cdg_node': [],
    '#cdg_edge': [],
    '#ddg_node': [],
    '#ddg_edge': [],
}

svol, evol = 221, 228
for vol in range(svol, evol+1):
    print(f'Processing Vol.{vol}...')
    vol_file_path = data_base_path + f"packed_hybrid_vol_{vol}.pkl"
    vol_data = read_dumped(vol_file_path)
    for i, data_item in tqdm(enumerate(vol_data), total=len(vol_data)):
        raw_stat = cal_raw_stat(data_item)
        trun_stat = cal_truncate_stat(data_item)
        update_list_dict(raw_stat_results, raw_stat, exclude_none=True)
        update_list_dict(truncate_stat_results, trun_stat, exclude_none=True)

raw_reduced_results = {}
trun_reduced_results = {}

for to_write, to_read in zip([raw_reduced_results, trun_reduced_results], [raw_stat_results, truncate_stat_results]):
    for key, val_list in to_read.items():
        mean, median = numpy.mean(val_list), numpy.median(val_list)
        to_write[key] = {
            'mean': mean,
            'median': median,
            'count': len(val_list)
        }

print(f'\nRaw: {raw_reduced_results}\n')
print(f'\nTruncated: {trun_reduced_results}\n')
dump_json(raw_reduced_results, "/data2/zhijietang/temp/pdbert_data_stat/pretrain_test_raw_stat.json")
dump_json(trun_reduced_results, "/data2/zhijietang/temp/pdbert_data_stat/pretrain_test_trun_stat.json")