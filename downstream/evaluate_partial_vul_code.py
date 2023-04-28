from tqdm import tqdm

import torch
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.models.model import Model
from sklearn.metrics import accuracy_score

from downstream import FuncVulDetectBaseDatasetReader
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config

model_base_path = '/data2/zhijietang/vul_data/run_logs/vul_func_pred/87/'
model_name = 'model.tar.gz'

data_base_path = '/data2/zhijietang/vul_data/datasets/neuralpda/vul_snippets/'
data_file_name = 'vul_code_snippets.json'

batch_size = 32
cuda_device = 0


def predict_on_dataloader(_model, _data_loader):
    all_pred = []
    with torch.no_grad():
        _model.eval()
        for i, batch in enumerate(tqdm(_data_loader)):
            outputs = _model(**batch)
            all_pred.extend(outputs['pred'].cpu().detach().tolist())
    return all_pred


print('\n Building dataset reader...')
dataset_reader = build_dataset_reader_from_config(
    config_path=model_base_path + 'config.json',
    serialization_dir=model_base_path
)

print('\nBuilding model...')
model = Model.from_archive(model_base_path + model_name)
print('\n Building components...')
data_loader = MultiProcessDataLoader(dataset_reader,
                                     data_base_path + data_file_name,
                                     shuffle=False,
                                     batch_size=batch_size,
                                     # collate_fn=data_collector,
                                     cuda_device=cuda_device)
data_loader.index_with(model.vocab)

if cuda_device != -1:
    model = model.cuda(cuda_device)
    torch.cuda.set_device(cuda_device)

print('\nPredicting...')
all_preds = predict_on_dataloader(model, data_loader)

acc = accuracy_score(y_true=[1]*len(all_preds),
                     y_pred=all_preds)
print(f'\n Accuracy = {round(acc, 4) * 100} ({len(all_preds)} in total)')

