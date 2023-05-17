import sys
from tqdm import tqdm
import numpy
import lizard

sys.path.extend([f'/data2/zhijietang/projects/vul-code-analyzer'])

from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from common.modules.code_cleaner import TrivialCodeCleaner
from pretrain import *
from common import *
from downstream import *
from utils.file import read_dumped, dump_json
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config


config_path = '/data2/zhijietang/vul_data/run_logs/cvss_metric/' + '25/' + 'config.json'
tokenizer_name = 'microsoft/codebert-base'
data_base_path = "/data2/zhijietang/vul_data/datasets/Fan_et_al/cvss_metric_pred/split_1/"
code_key = 'code'

print(f'[main] Building reader from: {config_path}\n')
dataset_reader: PackedHybridLineTokenPDGDatasetReader = build_dataset_reader_from_config(config_path)

tokenizer = PretrainedTransformerTokenizer(tokenizer_name)

def cal_raw_stat(data_item, raw_code_key: str):
    raw_code = data_item[raw_code_key]
    raw_tokens = tokenizer.tokenize(raw_code)
    line_count = raw_code.count('\n')
    token_count = len(raw_tokens)
    cc = cal_cc(raw_code)
    return {
        '#token': token_count,
        '#line': line_count,
        '#cc': cc,
    }

def cal_truncate_stat(data_item, raw_code_key: str):
    raw_code = data_item[raw_code_key]
    line_count = raw_code.count('\n')
    tokens, label = dataset_reader.test_process_data(data_item)
    cc = cal_cc(raw_code)
    return {
        '#token': len(tokens),
        '#line': line_count,
        '#cc': cc,
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
}
truncate_stat_results = {
    '#token': [],
    '#line': [],
    '#cc': [],
}

for split in ['train', 'validate', 'test']:
    print(f'Processing Split {split}...')
    split_file_path = data_base_path + f"{split}.json"
    split_data = read_dumped(split_file_path)
    for i, data_item in tqdm(enumerate(split_data), total=len(split_data)):
        raw_stat = cal_raw_stat(data_item, code_key)
        trun_stat = cal_truncate_stat(data_item, code_key)
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
dump_json(raw_reduced_results, "/data2/zhijietang/temp/pdbert_data_stat/downstream_vul_assess_raw_stat.json")
dump_json(trun_reduced_results, "/data2/zhijietang/temp/pdbert_data_stat/downstream_vul_assess_trun_stat.json")