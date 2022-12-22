import sys
from pprint import pprint
from typing import Tuple, List
from tqdm import tqdm
import numpy

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
from utils.cmd_args import read_multi_task_classification_eval_args

args = read_multi_task_classification_eval_args()

version = args.version
data_file_name = args.data_file_name
model_name = args.model_name
cuda_device = args.cuda
subset = args.subset
subfolder = args.subfolder
run_log_dir = args.run_log_dir
split = args.split
task_names = args.task_names.split(',')

data_base_path = f"/data1/zhijietang/vul_data/datasets/{args.dataset}/{subfolder}/{subset}/"
data_file_path = data_base_path + data_file_name
if split is not None:
    model_base_path = f'/data1/zhijietang/vul_data/run_logs/{run_log_dir}/{version}/rs_{split}/'
else:
    model_base_path = f'/data1/zhijietang/vul_data/run_logs/{run_log_dir}/{version}/'
model_path = model_base_path + model_name

batch_size = args.batch_size
bared_model = False

def predict_on_dataloader(_model, _data_loader, _task_num) -> Tuple[List, List, List]:
    all_pred = [[] for i in range(_task_num)]
    all_ref = [[] for i in range(_task_num)]
    all_score = [[] for i in range(_task_num)]
    with torch.no_grad():
        _model.eval()
        for i, batch in enumerate(tqdm(_data_loader)):
            outputs = _model(**batch)
            for j in range(_task_num):
                all_pred[j].extend(outputs['pred'][j].cpu().detach().tolist())
                all_score[j].extend(outputs['logits'][j].cpu().detach().tolist())
                all_ref[j].extend(batch['label'][:,j].cpu().detach().squeeze().tolist())        # Labels are collected by instances, not by tasks
    return all_ref, all_pred, all_score

mylogger.info('evaluate', f'run_log_dir = {run_log_dir}')
mylogger.info('evaluate', f'version = {version}')
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

all_ref, all_pred, all_score = predict_on_dataloader(model, data_loader, len(task_names))
# all_score = torch.Tensor(all_score).exp().softmax(-1)[:,1].tolist()
result_dict = {}
task_f1s = []
for i,task_name in enumerate(task_names):
    task_metrics = {f'{task_name}_f1_score': f1_score(all_ref[i], all_pred[i], average=args.average)}
    task_f1s.append(task_metrics[f'{task_name}_f1_score'])
    if args.all_metrics:
        task_metrics.update({
            f'{task_name}_accuracy': accuracy_score(all_ref[i], all_pred[i]),
            f'{task_name}_precision': precision_score(all_ref[i], all_pred[i], average=args.average),
            f'{task_name}_recall': recall_score(all_ref[i], all_pred[i], average=args.average),
        })
    result_dict.update(task_metrics)
result_dict['_f1_mean'] = numpy.mean(task_f1s)

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