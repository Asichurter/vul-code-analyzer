################################################################################
# This script is used to reformat the code of reveal dataset to
# make it more compact (remove useless spaces)
# and closer to the style of pre-training.
################################################################################

import subprocess
import os
from tqdm import tqdm

from utils.file import load_json, load_text, dump_text, dump_json

reformat_cmd = '/data1/zhijietang/miniconda3/lib/python3.8/site-packages/clang_format/data/bin/clang-format -i -style=file '
temp_file_path = '/data1/zhijietang/temp/temp_reformat_code.cpp'

src_packed_data_path = '/data1/zhijietang/vul_data/datasets/reveal/non-vulnerables.json'
tgt_packed_data_path = '/data1/zhijietang/vul_data/datasets/reveal/non-vulnerables_reformat.json'

# datas = load_json(src_packed_data_path)
#
# for data in tqdm(datas):
#     code = data['code']
#     dump_text(code, temp_file_path)
#     subprocess.run(reformat_cmd+temp_file_path, shell=True, check=True)
#     new_code = load_text(temp_file_path)
#     data['code'] = new_code
#
# dump_json(datas, tgt_packed_data_path)

# Convert all jsons under a directory
root_dir = '/data1/zhijietang/vul_data/datasets/reveal/common/reformat_random_split'
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.split('.')[-1] == 'json':
            file_path = os.path.join(root, file)
            datas = load_json(file_path)
            print(f'Processing {file_path}')
            for data in tqdm(datas):
                code = data['code']
                dump_text(code, temp_file_path)
                subprocess.run(reformat_cmd + temp_file_path, shell=True, check=True)
                new_code = load_text(temp_file_path)
                data['code'] = new_code
            dump_json(datas, file_path)