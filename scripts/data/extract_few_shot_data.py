

from utils.file import read_dumped, dump_json
from utils.data_utils.data_split import class_sensitive_random_split

# src_file_path = '/data1/zhijietang/vul_data/datasets/reveal/common/random_split/split_2/train.json'
# src_file_path = '/data1/zhijietang/vul_data/datasets/devign/codex_glue_splits/split_0/train.json'
src_file_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/vul_det/cleaned_splits/split_0/train.json'
tgt_file_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/vul_det/cleaned_splits/split_0/train_few_shot_10%.json'
few_shot_ratio = 0.1
label_key = 'vul'

src_datas = read_dumped(src_file_path)

few_shot_datas, _, _ = class_sensitive_random_split(src_datas, few_shot_ratio, 0, label_key, True)
dump_json(few_shot_datas, tgt_file_path)

print(f'Dump {len(few_shot_datas)} instances to `{tgt_file_path}`')