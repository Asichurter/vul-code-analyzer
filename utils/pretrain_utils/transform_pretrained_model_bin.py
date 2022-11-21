"""
    This script is used to transform the pre-trained model state dict into pytorch_model.bin
     of the Transformers model package, which can be directly used via "Transformers.Auto" or
     AllenNLP.
"""


import torch

weight_path = '/data1/zhijietang/vul_data/run_logs/pretrain/57/state_epoch_9.th'
target_dump_path = '/data1/zhijietang/vul_data/transformers_repos/codebert-hybridPDG-mlm-best/pytorch_model.bin'
real_vocab_size = 50265

extra_weight_path = '/data1/zhijietang/vul_data/transformers_repos/codebert-mlm/pytorch_model.bin'
pretrained_weights_prefix = 'code_embedder.token_embedder_code_tokens.transformer_model.'
vocab_embedding_key = 'embeddings.word_embeddings.weight'
extra_weights_need_to_be_add = []
# extra_weights_need_to_be_add = ['pooler.dense.weight', 'pooler.dense.bias'] # These two parameters should be added when doing parallel training

model_to_be_adapt = torch.load(weight_path)
extra_reference_model = torch.load(extra_weight_path)
new_state_dict = {}

# Step 1: Load weights by truncating model name prefix.
for key in model_to_be_adapt:
    if key.startswith(pretrained_weights_prefix):
        new_key = key[len(pretrained_weights_prefix):]
        new_state_dict[new_key] = model_to_be_adapt[key]

# Step 2: Truncating weight size of embeddings using real target vocab size.
new_state_dict[vocab_embedding_key] = new_state_dict[vocab_embedding_key][:real_vocab_size]

# Step 3: Load extra weights from reference model that are not appearing in the loaded model.
for key in extra_weights_need_to_be_add:
    new_state_dict[key] = extra_reference_model[key]

torch.save(new_state_dict, target_dump_path)

print(f'Done, save to {target_dump_path}')

