import re
import time
import torch
import os
import sys

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

from utils.pretrain_utils.mat import remove_consecutive_lines, shift_graph_matrix

sys.path.append("/data2/zhijietang/projects/vul-code-analyzer")

from pretrain import *
from common import *
from utils.file import load_text, read_dumped, dump_text
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config
from utils.joern_utils.pretty_print_utils import multi_paired_token_colored_print, print_code_with_line_num, multi_paired_token_tagged_print
from utils.joern_utils.joern_dev_parse import convert_func_signature_to_one_line

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

cuda_device = 0
model_path = '/data2/zhijietang/temp/local_archived_pdbert_base.tar.gz'
config_path = '/data2/zhijietang/temp/pdbert_archived/raw_config.json'
code_path = '/data2/zhijietang/vul_data/datasets/docker/fan_dedup/raw_code/vol0/7065.cpp'
tokenizer_name = "/data2/zhijietang/temp/codebert-base"

# f_output = open("/data1/zhijietang/temp/joern_failed_cases/joern_failed_cases_summary", "w")
# sys.stdout = f_output

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

# code_pkl = read_dumped('/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_process_hybrid_data/packed_hybrid_vol_0.pkl')
#
# pdg_pred_secs = []
# for i in range(100):
#     code = code_pkl[i]['raw_code']
#     dump_text(code, code_path)
#
#     start = time.time()
#     code = load_text(code_path)
#     tokens = tokenizer.tokenize(code)
#     pdg_output = predictor.predict_pdg(code)
#     end = time.time()
#     pdg_pred_secs.append(end-start)
#
#     cdg = torch.Tensor(pdg_output['ctrl_edge_labels'])
#     ddg = torch.Tensor(pdg_output['data_edge_labels'])
#     ddg_list = ddg.nonzero().tolist()
#
#     if len(ddg_list) > 0:
#         print('\n')
#         multi_paired_token_colored_print(code, ddg_list, tokens, processed=True)
#
#     print('\n')
#     print_code_with_line_num(code, start_line_num=0)
#     print(f'\n{cdg.nonzero().tolist()}')
#
#     print('\n')
#     print('#'*70 + '\n')
#
# print(f'Avg PDG predict time: {sum(pdg_pred_secs) / len(pdg_pred_secs)}')


##################  For Single Code File ##################

temp_path = '/data2/zhijietang/temp/temp_code.txt'
code = '''os_erase_area (int top, int left, int bottom, int right)
{
    int y, x, i, j;
    if ((top == 1) && (bottom == h_screen_rows) &&
	(left == 1) && (right == h_screen_cols)) {
#ifdef COLOR_SUPPORT
      bkgdset(u_setup.current_color | ' ');
      erase();
      bkgdset(0);
#else
      erase();
#endif
    } else {
	int saved_style = u_setup.current_text_style;
	os_set_text_style(u_setup.current_color);
	getyx(stdscr, y, x);
	top--; left--; bottom--; right--;
	for (i = top; i <= bottom; i++) {
	  move(i, left);
	  for (j = left; j <= right; j++)
	    addch(' ');
	}
	move(y, x);
	os_set_text_style(saved_style);
    }
}'''
dump_text(code, temp_path)

def predict_one_file(code_file_path):
    print(f'[main] Predicting {code_file_path}')
    code = load_text(code_file_path)
    code = convert_func_signature_to_one_line(code=code, redump=False)
    code, del_line_indices = remove_consecutive_lines(code)
    tokens = tokenizer.tokenize(code)
    pdg_output = predictor.predict_pdg(code)

    cdg = torch.Tensor(pdg_output['ctrl_edge_labels'])
    ddg = torch.Tensor(pdg_output['data_edge_labels'])
    cdg = shift_graph_matrix(cdg, del_line_indices)

    print(f'Size: CDG: {cdg.size()}, DDG: {ddg.size()}')
    print(f"DDG: {ddg.nonzero().tolist()}")

    print('\n\n' + 'Data-Depedency:\n')
    multi_paired_token_colored_print(code, ddg.nonzero().tolist(), tokens, processed=True)
    print("\n")
    multi_paired_token_tagged_print(code, ddg.nonzero().tolist(), tokens, processed=True)

    print('\n\n')
    print_code_with_line_num(code, start_line_num=0)
    print(f'\nCtrl-Dependency: {cdg.nonzero().tolist()}')

# predict_one_file(code_path)
predict_one_file(temp_path)

##################  For Dumping Results of Batched Files ##################

# files_dir_path = '/data1/zhijietang/temp/joern_failed_cases'
# results_dump_path = ''
# results = []
#
# for file in os.listdir(files_dir_path):
#     if file.split('.')[-1] != 'cpp':
#         continue
#     file_path = os.path.join(files_dir_path, file)
#     print('\n\n\n' + '-'*75)
#     predict_one_file(file_path)
#
# f_output.close()