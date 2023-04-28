import sys
from pprint import pprint
from typing import Tuple, List
from tqdm import tqdm
from pprint import pprint

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

args = read_pdg_eval_args()
mylogger.info('eval_pdg', f"Args: {args}")
cuda_device = args.cuda
batch_size = 32

vol_start, vol_end = args.vol_range.split(',')
vol_start, vol_end = int(vol_start), int(vol_end)
model_base_path = f'/data2/zhijietang/vul_data/run_logs/{args.run_log_dir}/{args.version}/'
model_path = f'/data2/zhijietang/vul_data/run_logs/{args.run_log_dir}/{args.version}/{args.model_name}'

def predict_on_dataloader(_model, _data_loader):
    # all_pred = []
    # all_ref = []
    # all_score = []
    with torch.no_grad():
        _model.eval()
        for i, batch in enumerate(tqdm(_data_loader)):
            batch['forward_step_name'] = 'code_analy-1'     # To adapt independent-forward model
            outputs = _model(**batch)
            # all_pred.extend(outputs['pred'].cpu().detach().tolist())
            # all_score.extend(outputs['logits'].cpu().detach().tolist())
            # all_ref.extend(batch['label'].cpu().detach().squeeze().tolist())
    return _model.get_metrics()

def get_detailed_metrics(_model):
    _ctrl_metrics = _model.ctrl_metric.get_detailed_metrics()
    ctp, ctn, cfp, cfn = _ctrl_metrics['tp'], _ctrl_metrics['tn'], _ctrl_metrics['fp'], _ctrl_metrics['fn']
    c_recall = ctp / (ctp + cfn)
    c_precision = ctp / (ctp + cfp)
    c_f1 = 2*c_recall*c_precision / (c_recall + c_precision)

    _data_metrics = _model.data_metric.get_detailed_metrics()
    dtp, dtn, dfp, dfn = _data_metrics['tp'], _data_metrics['tn'], _data_metrics['fp'], _data_metrics['fn']
    d_recall = dtp / (dtp + dfn)
    d_precision = dtp / (dtp + dfp)
    d_f1 = 2*d_recall*d_precision / (d_recall + d_precision)

    o_recall = (ctp + dtp) / (ctp + dtp + cfn + dfn)
    o_precision = (ctp + dtp) / (ctp + dtp + cfp + dfp)
    o_f1 = 2*o_recall*o_precision / (o_recall + o_precision)

    return {
        'ctrl': {
            'recall': c_recall,
            'precision': c_precision,
            'f1': c_f1
        },
        'data': {
            'recall': d_recall,
            'precision': d_precision,
            'f1': d_f1
        },
        'overall': {
            'recall': o_recall,
            'precision': o_precision,
            'f1': o_f1
        }
    }



dataset_reader = build_dataset_reader_from_config(
    config_path=model_base_path + 'config.json',
    serialization_dir=model_base_path
)
model = Model.from_archive(model_path)
data_loader = MultiProcessDataLoader(dataset_reader,
                                     {'data_base_path': args.data_base_path,
                                      'volume_range': [vol_start, vol_end]},
                                     shuffle=False,
                                     batch_size=batch_size,
                                     # collate_fn=data_collector,
                                     cuda_device=cuda_device)
data_loader.index_with(model.vocab)

if cuda_device != -1:
    model = model.cuda(cuda_device)
    torch.cuda.set_device(cuda_device)

metrics = predict_on_dataloader(model, data_loader)
print('\n\n' + '*'*80 + '\n\n')
pprint(metrics)

save_evaluate_results(metrics,
                      {
                          'test_vol_range': args.vol_range,
                          'test_model_path': model_path
                      },
                      model_base_path+'eval_results.json')
pprint(get_detailed_metrics(model))
# sys.exit(0)