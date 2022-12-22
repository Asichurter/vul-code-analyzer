import sys
from pprint import pprint
from typing import Tuple, List

from allennlp.predictors import Predictor
from tqdm import tqdm

import torch
from allennlp.common import Params, JsonDict
from allennlp.data import Vocabulary, Instance
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.models.model import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from utils.downstream_utils.rank_based_metric import LineVulMetric
from utils.file import read_dumped

sys.path.extend(['/data1/zhijietang/projects/vul-code-analyzer'])

from downstream import *
from utils import GlobalLogger as mylogger
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config
from utils.file import save_evaluate_results, dump_pred_results
from utils.cmd_args import read_classification_eval_args

args = read_classification_eval_args()

version = args.version
data_file_name = args.data_file_name
model_name = args.model_name
cuda_device = args.cuda
subset = args.subset
subfolder = args.subfolder
run_log_dir = args.run_log_dir
split = args.split

predict_on_dataloader = False

data_base_path = f"/data1/zhijietang/vul_data/datasets/{args.dataset}/{subfolder}/{subset}/"
data_file_path = data_base_path + data_file_name
if split is not None:
    model_base_path = f'/data1/zhijietang/vul_data/run_logs/{run_log_dir}/{version}/rs_{split}/'
else:
    model_base_path = f'/data1/zhijietang/vul_data/run_logs/{run_log_dir}/{version}/'
model_path = model_base_path + model_name

batch_size = args.batch_size
bared_model = False

class LineVulPredictor(Predictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        ok, instance = self._dataset_reader.text_to_instance(json_dict)
        return instance

def mask_select_tensors(batch, outputs):
    mask = outputs['mask']
    labels = batch['line_vul_labels']
    logits = outputs['logits']
    pred = outputs['pred']
    return torch.masked_select(logits, mask).cpu().detach().tolist(), \
           torch.masked_select(pred, mask).cpu().detach().tolist(), \
           torch.masked_select(labels, mask).cpu().detach().squeeze().tolist()

def predict_on_dataloader(_model, _data_loader) -> Tuple:
    all_pred = []
    all_ref = []
    all_score = []
    linevul_metric = LineVulMetric()
    with torch.no_grad():
        _model.eval()
        for i, batch in enumerate(tqdm(_data_loader)):
            outputs = _model(**batch)
            logits, preds, labels = mask_select_tensors(batch, outputs)
            all_pred.extend(preds)
            all_score.extend(logits)
            all_ref.extend(labels)
            linevul_metric.score(logits, [i for i in range(len(batch['line_vul_labels'][0])) if labels[i]==1])
    return all_ref, all_pred, all_score, linevul_metric

# def predict_on_instances(_predictor: Predictor, _dataset_reader, _data_path):
#     all_pred = []
#     all_ref = []
#     all_score = []
#     linevul_metric = LineVulMetric()
#     test_data_items = read_dumped(_data_path)
#     for item in tqdm(test_data_items):
#         outputs = _predictor.predict_json(item)


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

predictor = LineVulPredictor(model, dataset_reader)

data_loader = MultiProcessDataLoader(dataset_reader,
                                     data_file_path,
                                     shuffle=False,
                                     batch_size=batch_size,
                                     # collate_fn=data_collector,
                                     cuda_device=cuda_device)
data_loader.index_with(model.vocab)
all_ref, all_pred, all_score, linevul_metric = predict_on_dataloader(model, data_loader)
# all_score = torch.Tensor(all_score).exp().softmax(-1)[:,1].tolist()
result_dict = {
    'Precision': precision_score(all_ref, all_pred, average=args.average),
    'Recall': recall_score(all_ref, all_pred, average=args.average),
    'F1-Score': f1_score(all_ref, all_pred, average=args.average),
    # 'AUC': roc_auc_score(all_ref, all_score, average=args.average)
}
result_dict.update(linevul_metric.get_metric())

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

