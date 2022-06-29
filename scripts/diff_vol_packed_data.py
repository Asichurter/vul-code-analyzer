import os

from utils.file import load_json

old_vol_data_path = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_data/vol30/'
new_vol_data_path = '/data1/zhijietang/vul_data/datasets/docker/packed_data_jsons_new/vol30/'

def diff_edges(old_packed_obj, new_packed_obj):
    old_edges = {*old_packed_obj['edges']}
    new_edges = {*new_packed_obj['edges']}
    return new_edges-old_edges, old_edges-new_edges

def main():
    from utils import GlobalLogger as mylogger

    mylogger.info('packed_diff', f'New vol size: {len(os.listdir(new_vol_data_path))}')
    mylogger.info('packed_diff', f'Old vol size: {len(os.listdir(old_vol_data_path))}')

    for vol_file in os.listdir(old_vol_data_path):
        mylogger.info('packed_diff', f'File: {vol_file}')
        old_file_path = os.path.join(old_vol_data_path, vol_file)
        new_file_path = os.path.join(new_vol_data_path, vol_file)
        if not os.path.exists(new_file_path):
            mylogger.warning('packed_diff', f'File not in new vol: {new_file_path}')
            continue

        old_packed_obj = load_json(old_file_path)
        new_packed_obj = load_json(new_file_path)
        add_edges, rem_edges = diff_edges(old_packed_obj, new_packed_obj)
        a = 0

if __name__ == '__main__':
    main()
