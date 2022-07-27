import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaConfig, RobertaTokenizer

RAW_MODEL_PATH = "/data1/zhijietang/vul_data/run_logs/pretrain/20/state_epoch_14.th"
MODEL_PATH = "F:/projects/transformer_repos/codebert-pdg-mlm/pytorch_model.bin" # "/data1/zhijietang/vul_data/transformers_repos/codebert-pdg-mlm/pytorch_model.bin"
model_type = 'microsoft/codebert-base'
config = AutoConfig.from_pretrained(model_type)
model = AutoModel.from_pretrained(MODEL_PATH, config=config)
tokenizer = AutoTokenizer.from_pretrained()

# state_dict = torch.load(RAW_MODEL_PATH, map_location='cpu')
# match_prefix = 'code_embedder.token_embedder_code_tokens.transformer_model.'
# new_state_dict = {}
#
# for n,p in state_dict.items():
#     if n.startswith(match_prefix):
#         real_name = n.replace(match_prefix, '')
#         if real_name == 'embeddings.word_embeddings.weight':
#             p = p[:-1]
#             print(f'Reshape embeddings to shape: {p.shape}')
#         new_state_dict[real_name] = p
#
# torch.save(new_state_dict, MODEL_PATH)


import os
from transformers import AutoModel, AutoTokenizer, AutoConfig

model_root_path = '/data1/zhijietang/vul_data/transformers_repos/codebert-pdg-mlm' # root of model package
config = AutoConfig.from_pretrained(model_root_path)
tokenizer = AutoTokenizer.from_pretrained(model_root_path)
model = AutoModel.from_pretrained(model_root_path, config=config)
