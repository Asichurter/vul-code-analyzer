import os
import numpy
import torch

from utils.file import load_pickle, dump_pickle
from utils import GlobalLogger as mylogger

w2v_src_folder = '/data1/zhijietang/vul_data/datasets/reveal/small/devign/w2v/rs_1'
mlm_src_folder = '/data1/zhijietang/vul_data/datasets/reveal/small/devign/mlm_ver15_e9/rs_1'
tgt_dump_folder = '/data1/zhijietang/vul_data/datasets/reveal/small/devign/w2v_mlm_mixed/rs_1'

files = ['train.pkl', 'validate.pkl', 'test.pkl']

for file in files:
    print(file)
    w2v_file = load_pickle(os.path.join(w2v_src_folder, file))
    mlm_file = load_pickle(os.path.join(mlm_src_folder, file))

    mlm_idx_map = {}
    for i, mlm_item in enumerate(mlm_file):
        mlm_idx_map[mlm_item['id']] = i

    tgt_data_items = []
    for w2v_item in w2v_file:
        w2v_features = w2v_item['node_features']
        item_id = w2v_item['id']

        if item_id not in mlm_idx_map:
            raise ValueError

        mlm_features = mlm_file[mlm_idx_map[item_id]]['node_features']

        # Pad zeros if node count not match
        if w2v_features.shape[0] > mlm_features.size(0):
            mylogger.warning('main', f'File: {w2v_item["id"]} has mismatched mlm/w2v node_count: {(w2v_features.shape[0], mlm_features.size(0))}')
            pad_len = w2v_features.shape[0] - mlm_features.size(0)
            mlm_features = torch.cat((mlm_features, torch.zeros(1,mlm_features.size(1))), dim=0)
        elif w2v_features.shape[0] < mlm_features.size(0):
            raise ValueError
        mixed_features = numpy.concatenate((w2v_features, mlm_features.numpy()), axis=1)

        tgt_data_item = {
            'node_features': mixed_features,
            'node_count': w2v_item['node_count'],
            'graph': w2v_item['graph'],
            'target': w2v_item['target'],
            'code': w2v_item['code'],
            'id': w2v_item['id'],
        }
        tgt_data_items.append(tgt_data_item)

    dump_pickle(tgt_data_items, os.path.join(tgt_dump_folder, file))
