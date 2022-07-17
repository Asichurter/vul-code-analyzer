
from utils.file import load_json, dump_json

data_base_path = '/data1/zhijietang/vul_data/datasets/reveal/random_split/split_2/'

total_count = 0
for split in ['train', 'validate', 'test']:
    data_path = data_base_path + f'{split}.json'
    datas = load_json(data_path)

    for i,data in enumerate(datas):
        data['id'] = i + total_count

    dump_json(datas, data_path)
    total_count += len(datas)

