import sys
from tqdm import tqdm
import numpy
import re

sys.path.extend([f'/data2/zhijietang/projects/vul-code-analyzer'])

from utils.file import read_dumped, dump_json

data_base_path = "/data2/zhijietang/vul_data/datasets/Fan_et_al/cvss_metric_pred/split_1/"
code_key = 'code'
dump_path = "/data2/zhijietang/temp/pdbert_data_stat_rm_eptline/downstream_vul_assess_empline_stat.json"

def process_code(raw_code: str):
    raw_code = re.sub('\n+', '\n', raw_code)
    return raw_code

def cal_raw_stat(data_item):
    raw_code = data_item[code_key]
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

for split in ['train', 'validate', 'test']:
    print(f'Processing Split {split}...')
    split_file_path = data_base_path + f"{split}.json"
    split_data = read_dumped(split_file_path)
    for i, data_item in tqdm(enumerate(split_data), total=len(split_data)):
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
dump_json(raw_reduced_results, dump_path)