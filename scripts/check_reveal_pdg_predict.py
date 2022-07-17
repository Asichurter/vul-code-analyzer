from os.path import join as join_path
import os
import torch
from allennlp.common import JsonDict
from allennlp.data import Instance
import numpy

from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from pretrain import *
from utils.file import load_json
from utils.allennlp_utils.build_utils import build_dataset_reader_from_dict


class PdgPredictor(Predictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text_fields = {
            'code': json_dict['code'],
            'hash': json_dict['name']
        }
        ok, instance = self._dataset_reader.text_to_instance(text_fields)
        if not ok:
            raise ValueError(f'Not ok instance: {instance.human_readable_dict()}')
        else:
            return instance

def score_pdg_predict(data_reader: RawPDGPredictDatasetReader, predicted_edges, label_edges, line_count):
    label_edge_matrices = data_reader.make_edge_matrix(label_edges, line_count) - 1
    preds = torch.Tensor(predicted_edges)
    pred_line_count = preds.shape[1]

    _data_preds, _ctrl_preds = preds[0].reshape(-1,).tolist(), preds[1].reshape(-1,).tolist()
    label_edge_matrices = label_edge_matrices[:, :pred_line_count, :pred_line_count]
    _data_labels = label_edge_matrices[0].reshape(-1,).tolist()
    _ctrl_labels = label_edge_matrices[1].reshape(-1,).tolist()

    return (_data_preds, _ctrl_preds), (_data_labels, _ctrl_labels)

    # preds = preds.reshape(-1,).numpy()
    # labels = label_edge_matrices[:,:pred_line_count,:pred_line_count].reshape(-1,).numpy()
    # return {
    #     'accuracy': accuracy_score(labels, preds),
    #     'precision': precision_score(labels, preds),
    #     'recall': recall_score(labels, preds),
    #     'f1': f1_score(labels, preds)
    # }

def print_metrics(preds, labels, title):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    print(f'{title} stat:')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')

load_model_base_path = '/data1/zhijietang/vul_data/run_logs/pretrain/12'
load_model_file_name = 'model_epoch_9.tar.gz'

data_base_path = '/data1/zhijietang/vul_data/datasets/docker/packed_reveal/'

max_lines = 50
# batch_size = 32
cuda_device = 2

overwrite_reader_config = {
    'type': 'raw_pdg_predict',
    'max_lines': max_lines,
    'code_max_tokens': 256,
    'identifier_key': 'hash',
}

reader_config = load_json(join_path(load_model_base_path, 'config.json'))['dataset_reader']
reader_config.update(overwrite_reader_config)
del reader_config['from_raw_data']
del reader_config['pdg_max_vertice']
reader: RawPDGPredictDatasetReader = build_dataset_reader_from_dict(reader_config)

model: CodeLinePDGAnalyzer = Model.from_archive(join_path(load_model_base_path, load_model_file_name))

predictor = PdgPredictor(model, reader)

# data_loader = MultiProcessDataLoader(reader, join_path(data_base_path, data_file_name),
#                                      batch_size=batch_size, shuffle=True, cuda_device=cuda_device)
# data_loader.index_with(model.vocab)

if cuda_device != -1:
    model = model.cuda(cuda_device)
    torch.cuda.set_device(cuda_device)

target_pdg_data = []
data_preds, data_labels, ctrl_preds, ctrl_labels = [], [], [], []
with torch.no_grad():
    model.eval()
    for folder in os.listdir(data_base_path):
        folder_base_path = join_path(data_base_path, folder)
        folder_count = 0
        for item in os.listdir(folder_base_path):
            item_path = join_path(folder_base_path, item)
            data_item = load_json(item_path)
            if data_item['total_line'] > max_lines or data_item['total_line'] <= 1:
                continue

            data_item['name'] = item_path
            predict_output = predictor.predict_json(data_item)
            pred_edges = predict_output['edge_labels']
            (data_pred, ctrl_pred), (data_label, ctrl_label) = score_pdg_predict(reader, pred_edges, data_item['edges'], data_item['total_line'])
            data_preds.extend(data_pred)
            data_labels.extend(data_label)
            ctrl_preds.extend(ctrl_pred)
            ctrl_labels.extend(ctrl_label)
            # print(f'Item: {item_path}\nMetrics:{metrics}')
            # accs.append(metrics['accuracy'])
            # pres.append(metrics['precision'])
            # recs.append(metrics['recall'])
            # f1s.append(metrics['f1'])
            folder_count += 1

        print(f'{folder}: {folder_count}')

print('\n\n' + '*'*70)
print_metrics(data_preds, data_labels, 'Data-dependecy')
print('\n\n' + '*'*70)
print_metrics(ctrl_preds, ctrl_labels, 'Control-dependecy')

# print('\n\n' + '*'*70)
# print('Statistics:')
# print(f'Total: {len(accs)}')
# print(f'Avg Accuracy: {numpy.mean(accs)}')
# print(f'Avg Precision: {numpy.mean(pres)}')
# print(f'Avg Recall: {numpy.mean(recs)}')
# print(f'Avg F1: {numpy.mean(f1s)}')

#     for i, batch in enumerate(tqdm(data_loader)):
#         pdg_outputs = model.pdg_predict(return_node_features=True, return_encoded_node=False, **batch)
#         batch_pred_edges = pdg_outputs['edge_labels'].detach().cpu()
#         batch_len = len(pdg_outputs['edge_labels'])
#         for j in range(batch_len):
#             line_count = batch['vertice_num'][j].item()
#             pred_edges = batch_pred_edges[j,:,:line_count,:line_count]
#             converted_edges = convert_graph_edges(pred_edges)
#             pdg_obj = {
#                 'node_features': pdg_outputs['node_features'][j,:line_count].detach().cpu(),
#                 'graph': converted_edges,
#                 'target': pdg_outputs['meta_data'][j]['label'], # label
#                 'node_count': line_count,
#                 'id': pdg_outputs['meta_data'][j]['id']
#             }
#             target_pdg_data.append(pdg_obj)
#
# dump_pickle(target_pdg_data, target_dump_path)