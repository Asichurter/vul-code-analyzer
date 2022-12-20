from tqdm import tqdm

from utils.file import read_dumped, dump_pickle, dump_json

# original_data_path = "/data1/zhijietang/vul_data/datasets/Fan_et_al/fan_full_data.pkl"
# tgt_data_dump_path = "/data1/zhijietang/vul_data/datasets/Fan_et_al/fan_full_data_dedup.pkl"

original_data_path = "/data1/zhijietang/vul_data/datasets/Fan_et_al/linevul_splits/linevul_vul_det_all_dedup.json"
tgt_data_dump_path = "/data1/zhijietang/vul_data/datasets/Fan_et_al/linevul_splits/linevul_vul_det_all_dedup_2.json"


def extract_func_sig(func):
    return func.split('\n')[0].strip()

original_datas = read_dumped(original_data_path)
# Note: Since deduplicate results depend on order of items,
#       to keep as much  positive samples as possible,
#       we sort and check the positive samples first to make sure
#       they are included and only exclude these duplicated negative samples
original_datas = sorted(original_datas, key=lambda x: x['vul'], reverse=True)

func_sigs = [extract_func_sig(d['code']) for d in original_datas]
excluded_tags = [False] * len(original_datas)
deduplicated_datas = []

for i, data in tqdm(enumerate(original_datas), total=len(original_datas)):
    # print(f'{i} / {len(original_datas)}')
    if excluded_tags[i]:
        continue
    else:
        func_sig = func_sigs[i]
        for j in range(i+1, len(original_datas)):
            if excluded_tags[j]:
                continue
            func_sig2 = func_sigs[j]
            if func_sig == func_sig2:
                excluded_tags[j] = True

        deduplicated_datas.append(data)

print(f'Excluded: {sum(excluded_tags)} / {len(original_datas)}')
# dump_pickle(deduplicated_datas, tgt_data_dump_path)
dump_json(deduplicated_datas, tgt_data_dump_path)