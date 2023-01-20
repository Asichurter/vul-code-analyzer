from tqdm import tqdm

from pygments.lexers.c_cpp import CppLexer

from utils.file import read_dumped, dump_pickle, dump_json

from allennlp.training.metrics import Metric

cpp_lexer = CppLexer()

def cpp_lexer_parse(raw_code: str):
    return list(cpp_lexer.get_tokens_unprocessed(raw_code))

original_data_path = "/data1/zhijietang/vul_data/datasets/Fan_et_al/fan_full_data.pkl"
tgt_data_dump_path = "/data1/zhijietang/vul_data/datasets/Fan_et_al/fan_full_data_dedup.pkl"

# original_data_path = "/data1/zhijietang/vul_data/datasets/Fan_et_al/linevul_splits/linevul_vul_det_all_dedup.json"
# tgt_data_dump_path = "/data1/zhijietang/vul_data/datasets/Fan_et_al/linevul_splits/linevul_vul_det_all_dedup_2.json"


def extract_func_sig(func):
    return func.split('\n')[0].strip()

def extract_func_sig_by_lexer(func):
    tokens = cpp_lexer_parse(func)
    filtered_text = ' '.join([t[-1] for t in tokens])
    return filtered_text

def extract_data_items(_data):
    return {
        'code': _data['func_before'],
        'vul': int(data['vul'])
    }

print(f'Reading data...')
original_datas = read_dumped(original_data_path)
# Note: Since deduplicate results depend on order of items,
#       to keep as much  positive samples as possible,
#       we sort and check the positive samples first to make sure
#       they are included and only exclude these duplicated negative samples
original_datas = sorted(original_datas, key=lambda x: x['vul'], reverse=True)

sig_extractor = extract_func_sig
code_key = 'func_before' # code'

print(f'Extracting signatures...')
func_sigs = [sig_extractor(d[code_key]) for d in original_datas]
excluded_tags = [False] * len(original_datas)
deduplicated_datas = []
cur_sigs = set()

# print(f'Deduplicating...')
# for i, data in tqdm(enumerate(original_datas), total=len(original_datas)):
#     # print(f'{i} / {len(original_datas)}')
#     if excluded_tags[i]:
#         print(f'Item #{i} excluded, sig: {func_sigs[i]}')
#         continue
#     else:
#         func_sig = func_sigs[i]
#         for j in range(i+1, len(original_datas)):
#             if excluded_tags[j]:
#                 continue
#             func_sig2 = func_sigs[j]
#             if func_sig == func_sig2:
#                 excluded_tags[j] = True
#
#         deduplicated_datas.append(data)
# print(f'Excluded: {sum(excluded_tags)} / {len(original_datas)}')

print(f'Deduplicating...')
for i, data in tqdm(enumerate(original_datas), total=len(original_datas)):
    # print(f'{i} / {len(original_datas)}')
    if func_sigs[i] in cur_sigs:
        print(f'Item #{i} excluded, sig: {func_sigs[i]}')
        continue
    else:
        cur_sigs.add(func_sigs[i])
        deduplicated_datas.append(extract_data_items(data))

print(f'Excluded: {len(original_datas) - len(deduplicated_datas)} / {len(original_datas)}')


print(f'Dumping to {tgt_data_dump_path}')
dump_pickle(deduplicated_datas, tgt_data_dump_path)