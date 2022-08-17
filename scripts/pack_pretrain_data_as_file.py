import sys
import os
import time
from tqdm import tqdm

sys.path.extend(['/data1/zhijietang/projects/vul-code-analyzer'])

from utils.file import dump_pickle, load_json
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config

reader_config_path = '/data1/zhijietang/vul_data/run_logs/pretrain/12/config.json'
# reader = build_dataset_reader_from_config(reader_config_path)
packed_data = []

# for vol in range(69, 70):
#     packed_data.clear()
#     dump_packed_file_path = f'/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_vol_data/packed_vol_{vol}.pkl'
#
#     vol_base_path = f'/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_data/vol{vol}/'
#     print(f'Reading Vol.{vol}...')
#     for item in tqdm(os.listdir(vol_base_path)):
#         # start = time.time()
#         data = load_json(os.path.join(vol_base_path, item))
#         # end = time.time()
#         # if end-start > 1:
#         #     print(f'Item {item} consume {end-start} sec.')
#         packed_data.append(data)
#
#     # dataset_config = {
#     #     'data_base_path': '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_data/',
#     #     'volume_range': [ver, ver]
#     # }
#     # for instance in tqdm(reader.read_as_json(dataset_config)):
#     #     packed_data.append(instance)
#
#     dump_pickle(packed_data, dump_packed_file_path)

for vol in os.listdir('/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_data'):
    if 'debug' in vol:
        continue
    vol_num = vol[3:]
    if int(vol_num) < 20:
        continue
    packed_data.clear()
    dump_packed_file_path = f'/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_vol_data/packed_vol_{vol_num}.pkl'

    vol_base_path = f'/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_data/{vol}/'
    for item in tqdm(os.listdir(vol_base_path)):
        # start = time.time()
        data = load_json(os.path.join(vol_base_path, item))
        # end = time.time()
        # if end-start > 1:
        #     print(f'Item {item} consume {end-start} sec.')
        packed_data.append(data)

    # dataset_config = {
    #     'data_base_path': '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_data/',
    #     'volume_range': [ver, ver]
    # }
    # for instance in tqdm(reader.read_as_json(dataset_config)):
    #     packed_data.append(instance)

    print(f'Vol.{vol_num} dumped {len(packed_data)} items')
    dump_pickle(packed_data, dump_packed_file_path)