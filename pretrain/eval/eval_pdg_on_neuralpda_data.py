import sys
from pprint import pprint
from typing import Tuple, List
from tqdm import tqdm
from pprint import pprint

from sklearn.metrics import f1_score

import torch
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.models.model import Model

sys.path.extend(['/data2/zhijietang/projects/vul-code-analyzer'])

# Import modules
from pretrain import *
from common import *
# Import utils
from utils import GlobalLogger as mylogger
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config
from utils.file import save_evaluate_results
from utils.cmd_args import read_pdg_eval_args

# args = read_pdg_eval_args()
cuda_device = 0
batch_size = 32

vol_start, vol_end = 2,2
model_base_path = f'/data2/zhijietang/vul_data/run_logs/pretrain/200/'
model_path = model_base_path + 'model.tar.gz'
data_base_path = '/data2/zhijietang/vul_data/datasets/neuralpda/pdbert_format_data/'

def remove_padded(outputs, label_name, pred_name, mask_name):
    labels, preds, masks = outputs[label_name], outputs[pred_name], outputs[mask_name]
    return torch.masked_select(labels, masks).detach().cpu().tolist(), \
            torch.masked_select(preds, masks).detach().cpu().tolist()

def predict_on_dataloader(_model, _data_loader):
    pdg_data_preds = []
    pdg_ctrl_preds = []
    pdg_data_labels = []
    pdg_ctrl_labels = []
    with torch.no_grad():
        _model.eval()
        for i, batch in enumerate(tqdm(_data_loader)):
            outputs = _model(**batch)
            pdg_ctrl_label, pdg_ctrl_pred = remove_padded(outputs, 'ctrl_edge_labels', 'ctrl_edge_preds', 'ctrl_edge_mask')
            pdg_data_label, pdg_data_pred = remove_padded(outputs, 'data_edge_labels', 'data_edge_preds', 'data_edge_mask')
            pdg_ctrl_labels.extend(pdg_ctrl_label)
            pdg_ctrl_preds.extend(pdg_ctrl_pred)
            pdg_data_labels.extend(pdg_data_label)
            pdg_data_preds.extend(pdg_data_pred)
    return pdg_ctrl_labels, pdg_ctrl_preds, pdg_data_labels, pdg_data_preds

dataset_reader = build_dataset_reader_from_config(
    config_path=model_base_path + 'config.json',
    serialization_dir=model_base_path
)
model = Model.from_archive(model_path)
data_loader = MultiProcessDataLoader(dataset_reader,
                                     {'data_base_path': data_base_path,
                                      'volume_range': [vol_start, vol_end]},
                                     shuffle=False,
                                     batch_size=batch_size,
                                     # collate_fn=data_collector,
                                     cuda_device=cuda_device)
data_loader.index_with(model.vocab)

if cuda_device != -1:
    model = model.cuda(cuda_device)
    torch.cuda.set_device(cuda_device)

pdg_ctrl_labels, pdg_ctrl_preds, pdg_data_labels, pdg_data_preds = predict_on_dataloader(model, data_loader)

print('\n\n' + '*'*80 + '\n\n')
print(f'PDG-ctrl F1: {f1_score(pdg_ctrl_labels, pdg_ctrl_preds)}, Total: {len(pdg_data_labels)}')
print(f'PDG-data F1: {f1_score(pdg_data_labels, pdg_data_preds)}, Total: {len(pdg_ctrl_labels)}')
print(f'PDG-overall F1: {f1_score(pdg_ctrl_labels+pdg_data_labels, pdg_ctrl_preds+pdg_data_preds)}, Total: {len(pdg_ctrl_labels)+len(pdg_data_labels)}')
# sys.exit(0)