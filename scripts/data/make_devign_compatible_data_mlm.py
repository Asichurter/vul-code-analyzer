import os
import torch
from os.path import join as path_join
from tqdm import tqdm

from allennlp.models import Model
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader

from common import *
from pretrain import *

from utils.allennlp_utils.build_utils import build_dataset_reader_from_dict
from utils.file import dump_pickle, load_json, load_pickle

# Stage 1: Extract MLM node features

# model_dump_base_path = '/data1/zhijietang/vul_data/run_logs/pretrain/15'
# data_base_path = '/data1/zhijietang/vul_data/datasets/reveal/random_split/split_1'
# # data_base_path = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_vol_data/'
# data_file_name = 'test.json'
# # data_file_name = 'packed_vol_69.pkl'
# target_dump_path = '/data1/zhijietang/vul_data/datasets/reveal/predict/mlm_e9/rs_1/test.pkl'
# model_name = 'microsoft/codebert-base'
# load_model_name = 'model_epoch_9.tar.gz'
#
# max_len = 256
# max_lines = 50
# cuda_device = 0
# batch_size = 32
# label = 0
#
# overwrite_reader_config = {
#     'type': 'raw_pdg_predict',
#     'max_lines': max_lines,
#     'code_max_tokens': max_len,
#     'identifier_key': 'id',
# }
#
# reader_config = load_json(path_join(model_dump_base_path, 'config.json'))['dataset_reader']
# reader_config.update(overwrite_reader_config)
# del reader_config['from_raw_data']
# del reader_config['pdg_max_vertice']
# reader: RawPDGPredictDatasetReader = build_dataset_reader_from_dict(reader_config)
#
# model: CodeObjectiveTrainer = Model.from_archive(path_join(model_dump_base_path, load_model_name))
#
# data_loader = MultiProcessDataLoader(reader, path_join(data_base_path, data_file_name),
#                                      batch_size=batch_size, shuffle=True, cuda_device=cuda_device)
# data_loader.index_with(model.vocab)
#
# if cuda_device != -1:
#     model = model.cuda(cuda_device)
#     torch.cuda.set_device(cuda_device)
#
# line_extractor = AvgLineExtractor(max_lines)
#
# target_pdg_data = []
# with torch.no_grad():
#     model.eval()
#     for i, batch in enumerate(tqdm(data_loader)):
#         pdg_outputs = model.extract_line_features(line_extractor=line_extractor ,**batch)
#         batch_len = len(pdg_outputs['node_features'])
#         for j in range(batch_len):
#             line_count = batch['vertice_num'][j].item()
#             pdg_obj = {
#                 'node_features': pdg_outputs['node_features'][j,:line_count].detach().cpu(),
#                 'target': pdg_outputs['meta_data'][j]['label'], # label
#                 'node_count': line_count,
#                 'id': pdg_outputs['meta_data'][j]['id']
#             }
#             target_pdg_data.append(pdg_obj)
#
# dump_pickle(target_pdg_data, target_dump_path)


# Stage 2: Copy graph structure from predicted graph of full pre-trained model

from utils import GlobalLogger as mylogger

graph_base_pkl_path = '/data1/zhijietang/vul_data/datasets/reveal/predict/raw_50_line_256_token_e9/rs_1/test.pkl'
target_pkl_path = '/data1/zhijietang/vul_data/datasets/reveal/predict/mlm_e9/rs_1/test.pkl'
target_redump_path = '/data1/zhijietang/vul_data/datasets/reveal/predict/mlm_e9/rs_1/test.pkl'

base_graphs = load_pickle(graph_base_pkl_path)
target_graphs = load_pickle(target_pkl_path)

graph_id_map = {
    g['id']:i for i,g in enumerate(base_graphs)
}

for i, graph in tqdm(enumerate(target_graphs)):
    graph_id = graph['id']
    if graph_id not in graph_id_map:
        mylogger.error('main', f'graph=[{graph}] not found in map of base map')
        continue

    base_graph = base_graphs[graph_id_map[graph_id]]
    if graph['node_count'] != base_graph['node_count']:
        mylogger.warning('main', f'Base graph = [{base_graph}] \n' +
                                 f'Target graph=[{graph}]\n' +
                                 'with same id but different node_count')

    graph['graph'] = base_graph['graph']

dump_pickle(target_graphs, target_redump_path)





