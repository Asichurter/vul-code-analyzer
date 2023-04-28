import os
from typing import List, Tuple, Iterable
from tqdm import tqdm

from utils.file import read_dumped, dump_pickle

def adapt_neuralpda_data(src_data_path, tgt_dump_path):
    new_adapted_data_items = []

    src_data = read_dumped(src_data_path)
    for item in tqdm(src_data):
        ctrl_edges, data_edges = [], []
        for edge in item['pdg_edges']:
            new_edge = (int(edge['node_out']), int(edge['node_in']))
            if edge['edge_type'] == 'data_dependency':
                data_edges.append(new_edge)
            else:
                ctrl_edges.append(new_edge)

        adapted_data_item = {
            'id': item['fid'],
            'raw_code': item['func_code'],
            'pdg_ctrl_edges': ctrl_edges,
            'pdg_data_edges': data_edges,
        }
        new_adapted_data_items.append(adapted_data_item)

    dump_pickle(new_adapted_data_items, tgt_dump_path)
    print(f'\nSave to {tgt_dump_path}\n')

if __name__ == '__main__':
    src_base_path = '/data2/zhijietang/vul_data/datasets/neuralpda/intrin_cpp/c_8/'
    tgt_base_path = '/data2/zhijietang/vul_data/datasets/neuralpda/pdbert_format_data'
    adapt_neuralpda_data(os.path.join(src_base_path, "functions_train.json"),
                         os.path.join(tgt_base_path, "packed_line_vol_0.pkl"))
    adapt_neuralpda_data(os.path.join(src_base_path, "functions_val.json"),
                         os.path.join(tgt_base_path, "packed_line_vol_1.pkl"))
    adapt_neuralpda_data(os.path.join(src_base_path, "functions_test.json"),
                         os.path.join(tgt_base_path, "packed_line_vol_2.pkl"))