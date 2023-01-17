import torch

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from allennlp.models import Model
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from pretrain import *
from common import *
from utils.file import load_text
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config

class PDGPredictor(Predictor):
    def predict_pdg(self, code: str):
        instance = self._json_to_instance({
            'raw_code': code
        })
        return self.predict_instance(instance)

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        ok, instance = self._dataset_reader.text_to_instance(json_dict)
        if not ok:
            raise ValueError
        else:
            return instance


def set_reader(_reader):
    _reader.is_train = False
    return _reader

cuda_device = 4
model_path = '/data1/zhijietang/vul_data/run_logs/pretrain/' + '57/' + 'model_epoch_9.tar.gz'
config_path = '/data1/zhijietang/vul_data/run_logs/pretrain/' + '57/' + 'config.json'
code_path = '/data1/zhijietang/temp/test.c'
tokenizer_name = 'microsoft/codebert-base'

torch.cuda.set_device(cuda_device)
print(f'[main] Building tokenizer: {tokenizer_name}')
tokenizer = PretrainedTransformerTokenizer(tokenizer_name)
print(f'[main] Building model from: {model_path}')
model = Model.from_archive(model_path)
model = model.cuda(cuda_device)
print(f'[main] Building reader from: {config_path}')
dataset_reader = build_dataset_reader_from_config(config_path)
dataset_reader = set_reader(dataset_reader)
predictor = PDGPredictor(model, dataset_reader, frozen=True)

print(f'[main] Predicting {code_path}')
code = load_text(code_path)
pdg_output = predictor.predict_pdg(code)

cdg = torch.Tensor(pdg_output['ctrl_edge_labels'])
ddg = torch.Tensor(pdg_output['data_edge_labels'])
