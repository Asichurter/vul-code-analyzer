import os
from tqdm import tqdm

from utils.file import load_json

data_base_path = '/data1/zhijietang/vul_data/datasets/docker/packed_reveal'
code_len_file_path = '/data1/zhijietang/vul_data/datasets/docker/reveal_tokenized_code_len.json'

code_lens = load_json(code_len_file_path)
for folder in ['vulnerables', 'non-vulnerables', 'non-vulnerables-2']:
    folder_path = os.path.join(data_base_path, folder)
    for item in tqdm(os.listdir(folder_path)):
        item_path = os.path.join(folder_path, item)
        data_item = load_json(item_path)
        file_name = item.split('.')[0]

        if data_item['total_line'] > 50 or code_lens[folder][file_name] > 512:
            os.remove(item_path)

