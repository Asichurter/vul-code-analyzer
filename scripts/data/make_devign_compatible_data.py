import os
import torch
from os.path import join as path_join
from tqdm import tqdm

from allennlp.models import Model
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader

from common import *
from pretrain import *

from utils.allennlp_utils.build_utils import build_dataset_reader_from_dict
from utils.file import dump_pickle, load_json

model_dump_base_path = '/data1/zhijietang/vul_data/run_logs/pretrain/12'
data_base_path = '/data1/zhijietang/vul_data/datasets/reveal/random_split/split_1'
# data_base_path = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_vol_data/'
data_file_name = 'test.json'
# data_file_name = 'packed_vol_69.pkl'
target_dump_path = '/data1/zhijietang/vul_data/datasets/reveal/predict/raw_50_line_256_token_e9/rs_1/test.pkl'
model_name = 'microsoft/codebert-base'
load_model_name = 'model_epoch_9.tar.gz'

max_len = 256
max_lines = 50
cuda_device = 0
batch_size = 32
label = 0

overwrite_reader_config = {
    'type': 'raw_pdg_predict',
    'max_lines': max_lines,
    'code_max_tokens': max_len,
    'identifier_key': 'id',
}

def convert_graph_edges(edge_labels: torch.Tensor):
    converted_edges = []
    data_edge_idxes = torch.nonzero(edge_labels[0])
    for idxes in data_edge_idxes:
        converted_edges.append([idxes[0].item(), 0, idxes[1].item()])

    ctrl_edge_idxes = torch.nonzero(edge_labels[1])
    for idxes in ctrl_edge_idxes:
        converted_edges.append([idxes[0].item(), 1, idxes[1].item()])
    return converted_edges

reader_config = load_json(path_join(model_dump_base_path, 'config.json'))['dataset_reader']
reader_config.update(overwrite_reader_config)
del reader_config['from_raw_data']
del reader_config['pdg_max_vertice']
reader: RawPDGPredictDatasetReader = build_dataset_reader_from_dict(reader_config)

model: CodeLinePDGAnalyzer = Model.from_archive(path_join(model_dump_base_path, load_model_name))

data_loader = MultiProcessDataLoader(reader, path_join(data_base_path, data_file_name),
                                     batch_size=batch_size, shuffle=True, cuda_device=cuda_device)
data_loader.index_with(model.vocab)

if cuda_device != -1:
    model = model.cuda(cuda_device)
    torch.cuda.set_device(cuda_device)

target_pdg_data = []
with torch.no_grad():
    model.eval()
    for i, batch in enumerate(tqdm(data_loader)):
        pdg_outputs = model.pdg_predict(return_node_features=True, return_encoded_node=False, **batch)
        batch_pred_edges = pdg_outputs['edge_labels'].detach().cpu()
        batch_len = len(pdg_outputs['edge_labels'])
        for j in range(batch_len):
            line_count = batch['vertice_num'][j].item()
            pred_edges = batch_pred_edges[j,:,:line_count,:line_count]
            converted_edges = convert_graph_edges(pred_edges)
            pdg_obj = {
                'node_features': pdg_outputs['node_features'][j,:line_count].detach().cpu(),
                'graph': converted_edges,
                'target': pdg_outputs['meta_data'][j]['label'], # label
                'node_count': line_count,
                'id': pdg_outputs['meta_data'][j]['id']
            }
            target_pdg_data.append(pdg_obj)

dump_pickle(target_pdg_data, target_dump_path)





