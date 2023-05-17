import sys
from tqdm import tqdm
import numpy
import lizard
import re

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

def process_code(raw_code: str):
    raw_code = re.sub('\n+', '\n', raw_code)
    return raw_code

def cal_raw_stat(data_item):
    raw_code = data_item['raw_code']
    raw_code = process_code(raw_code)
    line_count = raw_code.count('\n')
    return {
        '#line': line_count,
    }

def update_list_dict(src, new_data, exclude_none=True):
    for k, v in new_data.items():
        if v is None and exclude_none:
            continue
        src[k].append(v)

raw_stat_results = {
    '#line': [],
}

svol, evol = 221, 228
for vol in range(svol, evol+1):
    print(f'Processing Vol.{vol}...')
    vol_file_path = data_base_path + f"packed_hybrid_vol_{vol}.pkl"
    vol_data = read_dumped(vol_file_path)
    for i, data_item in tqdm(enumerate(vol_data), total=len(vol_data)):
        raw_stat = cal_raw_stat(data_item)
        update_list_dict(raw_stat_results, raw_stat, exclude_none=True)

raw_reduced_results = {}

for to_write, to_read in zip([raw_reduced_results], [raw_stat_results]):
    for key, val_list in to_read.items():
        mean, median = numpy.mean(val_list), numpy.median(val_list)
        to_write[key] = {
            'mean': mean,
            'median': median,
            'count': len(val_list)
        }

print(f'\nRaw: {raw_reduced_results}\n')
dump_json(raw_reduced_results, "/data2/zhijietang/temp/pdbert_data_stat_rm_eptline/pretrain_test_rm_empline_stat.json")