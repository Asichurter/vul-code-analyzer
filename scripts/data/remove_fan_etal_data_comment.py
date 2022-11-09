import os
from tqdm import tqdm
import subprocess

from utils.file import load_text

rm_comment_cmd = 'gcc -fpreprocessed -dD -E -P {} > {} && mv {} {}'

base_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/files'
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
