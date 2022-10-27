import sys
from pprint import pprint
from typing import Tuple, List
from tqdm import tqdm

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.models.model import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.extend(['/data1/zhijietang/projects/vul-code-analyzer'])

from downstream import *
from utils import GlobalLogger as mylogger
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config
from utils.file import save_evaluate_results, dump_pred_results
from utils.cmd_args import read_reveal_eval_args

args = read_reveal_eval_args()

version = args.version
data_file_name = args.data_file_name
model_name = args.model_name
cuda_device = args.cuda
subset = args.subset
subfolder = args.subfolder
run_log_dir = args.run_log_dir
split = args.split

data_base_path = f"/data1/zhijietang/vul_data/datasets/reveal/{subfolder}/{subset}/"
data_file_path = data_base_path + data_file_name
model_base_path = f'/data1/zhijietang/vul_data/run_logs/{run_log_dir}/{version}/rs_{split}/'
model_path = model_base_path + model_name

batch_size = 32
bared_model = False

def predict_on_dataloader(model, data_loader) -> Tuple[List, List, List]:
    all_pred = []
    all_ref = []
    all_score = []
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(data_loader)):
            outputs = model(**batch)
            all_pred.extend(outputs['pred'].cpu().detach().tolist())
            all_score.extend(outputs['logits'].cpu().detach().tolist())
            all_ref.extend(batch['label'].cpu().detach().squeeze().tolist())
    return all_ref, all_pred, all_score

mylogger.info('evaluate', f'version = {version}')
mylogger.info('evaluate', f'cv = {split}')
mylogger.info('evaluate', f'model = {model_name}')
mylogger.info('evaluate', f'data_file = {data_file_name}')
mylogger.info('evaluate', f'data_base_path = {data_base_path}')

dataset_reader = build_dataset_reader_from_config(
    config_path=model_base_path + 'config.json',
    serialization_dir=model_base_path
)

if bared_model:
    vocab = Vocabulary.from_files(model_base_path + 'vocabulary')
    model_params = Params.from_file(model_base_path + 'config.json')['model']
    model = Model.from_params(model_params, vocab=vocab)
else:
    model = Model.from_archive(model_path)

# from utils.allennlp_utils.load_utils import partial_load_state_dict
#
# mylogger.info('main', 'Partial loading parameters from pretrain/12...')
# state_dict = torch.load('/data1/zhijietang/vul_data/run_logs/pretrain/12/best.th', 'cpu')
# partial_load_state_dict(model, state_dict,
#                         prefix_remap={'code_embedder': 'code_embedder'})

if cuda_device != -1:
    model = model.cuda(cuda_device)
    torch.cuda.set_device(cuda_device)

data_loader = MultiProcessDataLoader(dataset_reader,
                                     data_file_path,
                                     shuffle=False,
                                     batch_size=batch_size,
                                     # collate_fn=data_collector,
                                     cuda_device=cuda_device)
data_loader.index_with(model.vocab)

if cuda_device != -1:
    model = model.cuda(cuda_device)
    torch.cuda.set_device(cuda_device)

all_ref, all_pred, all_score = predict_on_dataloader(model, data_loader)
# all_score = torch.Tensor(all_score).exp().softmax(-1)[:,1].tolist()
result_dict = {
    'Accuracy': accuracy_score(all_ref, all_pred),
    'Precision': precision_score(all_ref, all_pred),
    'Recall': recall_score(all_ref, all_pred),
    'F1-Score': f1_score(all_ref, all_pred),
    'AUC': roc_auc_score(all_ref, all_score)
}

print('*'*80)
pprint(result_dict)

save_evaluate_results(result_dict,
                      {
                          'test_file_name': data_file_name,
                          'test_model_name': model_name
                      },
                      model_base_path+'eval_results.json')
dump_pred_results(model_base_path, {"labels": all_ref, "scores": all_score,})
sys.exit(0)