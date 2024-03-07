from tqdm import tqdm

from utils.file import load_pickle, dump_pickle
from utils.joern_utils.joern_dev_parse import convert_func_signature_to_one_line


def extract_data_id(file_path: str) -> str:
    file_name = file_path.split('/')[-1]
    data_id = file_name.split('.')[0]
    return data_id

def add_code_to_token_data_from_hybrid_data():
    line_level_data_path_temp = '/data2/zhijietang/vul_data/datasets/joern_vulberta/line_packed_vol_data/packed_vol_{}.pkl'
    token_level_data_path_temp = '/data2/zhijietang/vul_data/datasets/joern_vulberta/joern_parsed_raw/joern_parsed_raw_vol{}.pkl'

    vols = list(range(0, 229))

    for vol in vols:
        print(f"· Vol {vol}")
        line_vol_data_path = line_level_data_path_temp.format(vol)
        token_vol_data_path = token_level_data_path_temp.format(vol)
        line_vol_data = load_pickle(line_vol_data_path)
        token_vol_data = load_pickle(token_vol_data_path)

        line_vol_data_id_map = {o['id']: o for o in line_vol_data}
        vol_missed_cnt = 0
        for token_data in tqdm(token_vol_data):
            data_id = extract_data_id(token_data['file_path'])
            if data_id not in line_vol_data_id_map:
                print(f'[Warn] Missing line-level item for vol {vol}, id {data_id}, cnt: {vol_missed_cnt}')
                vol_missed_cnt += 1
                continue

            line_data = line_vol_data_id_map[data_id]
            raw_code = convert_func_signature_to_one_line(code=line_data['code'], redump=False)
            token_data['raw_code'] = raw_code

        dump_pickle(token_vol_data, token_vol_data_path)

if __name__ == "__main__":
    add_code_to_token_data_from_line_data()

