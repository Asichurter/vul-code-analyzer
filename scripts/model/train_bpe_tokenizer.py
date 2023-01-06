import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from utils.file import read_dumped

# data_base_path = '/data1/zhijietang/vul_data/datasets/devign/codex_glue_splits/split_0/'
# tokenizer_dump_path = '/data1/zhijietang/vul_data/tokenizers/devign_10000/tokenizer_config.json'
# data_base_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/splits/split_0/'
# tokenizer_dump_path = '/data1/zhijietang/vul_data/tokenizers/fan_vuldet_30000/tokenizer_config.json'
# data_base_path = '/data1/zhijietang/vul_data/datasets/reveal/common/random_split/split_0/'
# tokenizer_dump_path = '/data1/zhijietang/vul_data/tokenizers/reveal_10000/tokenizer_config.json'
# data_base_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/cvss_metric_pred/split_1'
# tokenizer_dump_path = '/data1/zhijietang/vul_data/tokenizers/fan_cvss_10000/tokenizer_config.json'
data_base_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/cwe_pred/split_0/'
tokenizer_dump_path = '/data1/zhijietang/vul_data/tokenizers/fan_cwe_10000/tokenizer_config.json'


trained_files = ['train.json', 'validate.json', 'test.json']
code_key = 'code'
vocab_size = 10000
min_frequency = 0

def read_instances():
    instances = []
    for item in trained_files:
        item_path = os.path.join(data_base_path, item)
        datas = read_dumped(item_path)
        for data in datas:
            instances.append(data[code_key])
    return instances

unk_word = '[UNK]'
pad_word = '[PAD]'
mask_word = '[MASK]'
special_words = [pad_word, unk_word, mask_word, "[CLS]", "[SEP]"]

tokenizer = Tokenizer(BPE(unk_token=unk_word))
trainer = BpeTrainer(special_tokens=special_words, vocab_size=vocab_size, min_frequency=min_frequency)
tokenizer.pre_tokenizer = Whitespace()

print('Reading instances...')
bpe_train_instances = read_instances()
print('Training...')
tokenizer.train_from_iterator(bpe_train_instances, trainer, len(bpe_train_instances))

print('Saving...')
tokenizer.save(tokenizer_dump_path, pretty=True)
