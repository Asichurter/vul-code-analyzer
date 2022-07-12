import os
import torch
from os.path import join as path_join
from tqdm import tqdm

from allennlp.models import Model
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader

from common import *
from pretrain import *

from utils.allennlp_utils.build_utils import build_dataset_reader_from_config
from utils.file import dump_pickle, load_json

model_dump_base_path = '/data1/zhijietang/vul_data/run_logs/pretrain/12'
data_base_path = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_vol_data/'
data_file_name = 'packed_vol_69.pkl'
target_dump_path = '/data1/zhijietang/vul_data/reveal/predict/50_line_256_token/vulnerables.pkl'

code_namespace = 'code_tokens'
model_name = 'microsoft/codebert-base'

max_len = 256
max_lines = 50
cuda_device = 0
batch_size = 32
label = 0

data_path_config = {
    "data_base_path": data_base_path,
    "volume_range": [69,69]
}

def convert_graph_edges(edge_labels: torch.Tensor, meta_data):
    edge_labels = edge_labels.detach().cpu()
    converted_edges = []
    data_edge_idxes = torch.nonzero(edge_labels[0])
    for idxes in data_edge_idxes:
        converted_edges.append([idxes[0], 0, idxes[1]])

    ctrl_edge_idxes = torch.nonzero(edge_labels[1])
    for idxes in ctrl_edge_idxes:
        converted_edges.append([idxes[0], 1, idxes[1]])
    return converted_edges


reader_config = load_json(path_join(model_dump_base_path, 'config.json'))['dataset_reader']
reader = build_dataset_reader_from_config(path_join(model_dump_base_path, 'config.json'))

model: CodeLinePDGAnalyzer = Model.from_archive(path_join(model_dump_base_path, 'model.tar.gz'))

data_loader = MultiProcessDataLoader(reader, data_path_config,
                                     batch_size=batch_size, shuffle=True, cuda_device=cuda_device)
data_loader.index_with(model.vocab)

if cuda_device != -1:
    model = model.cuda(cuda_device)
    torch.cuda.set_device(cuda_device)

target_pdg_data = []
with torch.no_grad():
    model.eval()
    for i, batch in enumerate(tqdm(data_loader)):
        pdg_outputs = model(**batch)
        a = 0
        # batch_len = len(pdg_outputs['node_features'])
        # for j in range(batch_len):
        #     pdg_obj = {
        #         'node_features': pdg_outputs['node_features'][j].detach().cpu(),
        #         'graph': convert_graph_edges(pdg_outputs['edge_labels'][j], pdg_outputs['meta_data'][j]),
        #         'target': label,
        #         'id': pdg_outputs['meta_data'][j]['id']
        #     }
        #     target_pdg_data.append(pdg_obj)

dump_pickle(target_pdg_data, target_dump_path)





