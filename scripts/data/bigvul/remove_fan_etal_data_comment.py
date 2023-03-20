import os
from tqdm import tqdm
import subprocess

from utils.file import load_text, read_dumped, dump_text, dump_json

rm_comment_cmd = 'gcc -fpreprocessed -dD -E -P {} > {} && mv {} {}'

# base_pkl_data_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/vul_detection_datas.pkl'
# base_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/files'
base_pkl_data_path = '/data1/zhijietang/vul_data/datasets/devign/function.json'
base_path = '/data1/zhijietang/vul_data/datasets/devign/files'
tgt_pkl_data_path = '/data1/zhijietang/vul_data/datasets/devign/cleaned_function.json'

#------------------------------------------------------------------
# Stage 1:
# Dump the code of each data item as code file
#------------------------------------------------------------------
print(f'# Stage 1: Dump files')
datas = read_dumped(base_pkl_data_path)
for i, data in enumerate(datas):
    # vul = data['vul']
    vul = data['target']
    tgt_path = os.path.join(base_path, f'{vul}_{i}.cpp')
    # dump_text(data['code'], tgt_path)
    dump_text(data['func'], tgt_path)

#------------------------------------------------------------------
# Stage 2:
# Use gcc to preprocess and clean the comments among code in-place
#------------------------------------------------------------------
print(f'# Stage 2: Remove comments')
for item in tqdm(os.listdir(base_path)):
    print(item, end=' ')
    src_file_path = os.path.join(base_path, item)
    temp_file_path = os.path.join(base_path, 'temp.cpp')
    try:
        subprocess.run(rm_comment_cmd.format(src_file_path, temp_file_path, temp_file_path, src_file_path), shell=True, check=True)
    except Exception as e:
        print(f'\n[Error] Err for item({item}): {e}')
        continue
    print('Done')

#------------------------------------------------------------------
# Stage 3:
# Re-collect the code and vul label from files as pkl file
#------------------------------------------------------------------
print(f'# Stage 3: Re-collect code from files')
items = []
for item in os.listdir(base_path):
    src_file_path = os.path.join(base_path, item)
    code = load_text(src_file_path)
    vul = int(item.split('_')[0])
    items.append({
        'code': code,
        'vul': vul
    })
dump_json(items, tgt_pkl_data_path)
