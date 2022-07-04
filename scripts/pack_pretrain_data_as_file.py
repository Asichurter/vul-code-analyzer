from tqdm import tqdm

from pretrain.comp.dataset_readers.packed_pdg_dataset_reader import PackedLinePDGDatasetReader
from utils.file import dump_pickle
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config

reader_config_path = '/data1/zhijietang/vul_data/run_logs/pretrain/12/config.json'
reader = build_dataset_reader_from_config(reader_config_path)
packed_data = []

for ver in range(69, 70):
    packed_data.clear()
    dump_packed_file_path = f'/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_vol_data/packed_vol_{ver}.pkl'
    dataset_config = {
        'data_base_path': '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_data/',
        'volume_range': [ver, ver]
    }
    for instance in tqdm(reader.read_as_json(dataset_config)):
        packed_data.append(instance)

    dump_pickle(packed_data, dump_packed_file_path)
    print(f'\nVol {ver} done.\n')