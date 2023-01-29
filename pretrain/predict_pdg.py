import time
import torch

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from allennlp.models import Model
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from pretrain import *
from common import *
from utils.file import load_text, read_dumped, dump_text
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config
from utils.joern_utils.pretty_print_utils import multi_paired_token_colored_print, print_code_with_line_num

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

cuda_device = 7
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

##################  For Pred Time Statistics ##################

code_pkl = read_dumped('/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_process_hybrid_data/packed_hybrid_vol_0.pkl')

pdg_pred_secs = []
for i in range(100):
    code = code_pkl[i]['raw_code']
    dump_text(code, code_path)

    start = time.time()
    code = load_text(code_path)
    tokens = tokenizer.tokenize(code)
    pdg_output = predictor.predict_pdg(code)
    end = time.time()
    pdg_pred_secs.append(end-start)

    cdg = torch.Tensor(pdg_output['ctrl_edge_labels'])
    ddg = torch.Tensor(pdg_output['data_edge_labels'])
    ddg_list = ddg.nonzero().tolist()

    if len(ddg_list) > 0:
        print('\n')
        multi_paired_token_colored_print(code, ddg_list, tokens, processed=True)

    print('\n')
    print_code_with_line_num(code, start_line_num=0)
    print(f'\n{cdg.nonzero().tolist()}')

    print('\n')
    print('#'*70 + '\n')

print(f'Avg PDG predict time: {sum(pdg_pred_secs) / len(pdg_pred_secs)}')


##################  For Single Code File ##################

# print(f'[main] Predicting {code_path}')
# code = load_text(code_path)
# tokens = tokenizer.tokenize(code)
# pdg_output = predictor.predict_pdg(code)
#
# cdg = torch.Tensor(pdg_output['ctrl_edge_labels'])
# ddg = torch.Tensor(pdg_output['data_edge_labels'])
#
# print('\n\n')
# multi_paired_token_colored_print(code, ddg.nonzero().tolist(), tokens, processed=True)
#
# print('\n\n')
# print_code_with_line_num(code, start_line_num=0)
# print(f'\n{cdg.nonzero().tolist()}')