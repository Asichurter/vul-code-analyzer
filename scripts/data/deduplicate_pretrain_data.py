from tqdm import tqdm

from utils.file import read_dumped, dump_pickle, load_pickle

pretrain_data_path_temp = '/data1/zhijietang/vul_data/datasets/joern_vulberta/packed_process_hybrid_data/packed_hybrid_vol_{}.pkl'

class PDGElem:
    def __init__(self, item):
        self.item = item

    def __hash__(self):
        return self.item['raw_code']

    def __eq__(self, other):
        return self.item['raw_code'] == other.item['raw_code']

    def __repr__(self):
        return str(self.item)

def read_vols_code(vols):
    items = []
    for vol in tqdm(vols):
        vol_data = read_dumped(pretrain_data_path_temp.format(vol))
        vol_codes = [d['raw_code'] for d in vol_data]
        items.extend(vol_codes)
    return items

def read_vols_pdg(vols):
    items = []
    for vol in tqdm(vols):
        vol_data = read_dumped(pretrain_data_path_temp.format(vol))
        # vol_pdgs = [PDGElem(d) for d in vol_data]
        items.extend(vol_data)
    return items

def inner_deduplicate(items, code_key='raw_code'):
    deduplicate_items = []
    code_set = set()
    for item in items:
        code = item[code_key]
        if code not in code_set:
            code_set.add(code)
            deduplicate_items.append(item)
    return deduplicate_items

def intra_deduplicate(items_base, items_to_dedup,
                      base_code_key='raw_code', tgt_code_key='raw_code'):
    deduplicated_items = []
    code_set = set([d[base_code_key] for d in items_base])
    for item in tqdm(items_to_dedup):
        code = item[tgt_code_key]
        if code not in code_set:
            deduplicated_items.append(item)
    return deduplicated_items

print('Reading vols...')
# test_set = read_vols_pdg(list(range(221, 229)))
# valid_set = read_vols_pdg(list(range(201, 221)))
train_set = read_vols_pdg(list(range(201)))
fan_dedup = load_pickle("/data1/zhijietang/vul_data/datasets/Fan_et_al/fan_full_data_dedup.pkl")

# print('Inner deduplicating...')
# test_set_dedup = inner_deduplicate(test_set)
# print(f'Deduplicated test set: {len(test_set_dedup)}/{len(test_set)}')
# valid_set_dedup = inner_deduplicate(valid_set)
# print(f'Deduplicated valid set: {len(valid_set_dedup)}/{len(valid_set)}')

# print('Init train set...')
# train_set_set = inner_deduplicate(train_set)
# print(f'Deduplicated train set: {len(train_set_set)}/{len(train_set)}')

print('Intra dedupcaliting...')
fan_dedup_from_train_set = intra_deduplicate(train_set, fan_dedup,
                                            base_code_key='raw_code',
                                             tgt_code_key='code')
print(f'Intra-dedup fan-dedup from train set: {len(fan_dedup_from_train_set)} / {len(fan_dedup)}')
# test_set_intra_dedup = intra_deduplicate(train_set, test_set_dedup)
# print(f'Intra-dedup test set: {len(test_set_intra_dedup)}')
# valid_set_intra_dedup = intra_deduplicate(train_set, valid_set_dedup)
# print(f'Intra-dedup valid set: {len(valid_set_intra_dedup)}')

# dump_pickle(valid_set_intra_dedup, pretrain_data_path_temp.format(999))
# dump_pickle(test_set_intra_dedup, pretrain_data_path_temp.format(9999))
dump_pickle(fan_dedup_from_train_set,
            "/data1/zhijietang/vul_data/datasets/Fan_et_al/fan_full_data_dedup_from_pretrain.pkl")

