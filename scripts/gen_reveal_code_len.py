import os
from tqdm import tqdm

from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from utils.file import load_text, dump_json


tokenizer = PretrainedTransformerTokenizer('microsoft/codebert-base')
data_base_path = '/data1/zhijietang/vul_data/datasets/docker/reveal'
code_len_dump_path = '/data1/zhijietang/vul_data/datasets/docker/reveal_tokenized_code_len.json'

if __name__ == '__main__':
    sub_folders = ['vulnerables', 'non-vulnerables', 'non-vulnerables-2']
    code_lens = {s:{} for s in sub_folders}
    for subfoler in sub_folders:
        print(subfoler)
        subfolder_path = os.path.join(data_base_path, subfoler)
        for item in tqdm(os.listdir(subfolder_path)):
            item_path = os.path.join(subfolder_path, item)
            code = load_text(item_path)
            tokens = tokenizer.tokenize(code)
            file_name = item.split('.')[0]
            code_lens[subfoler][file_name] = len(tokens)

    dump_json(code_lens, code_len_dump_path)