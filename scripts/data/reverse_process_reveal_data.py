################################################################################
# This script is used for reverting the reformat code from ReVeal dataset to the
# original split-by-space tokens.

# The raw dataset is reformatted by gcc preprocessor, and here we also utilize
# gcc preprocessor to map the raw code tokens to match and find the index of
# reformatted instances in the original vul/non-vul list.
# Error checking is also enabled to monitor any miss or mismatch.
################################################################################
import os

from utils.data_utils.data_clean import reformat_c_cpp_code
from utils.file import read_dumped, dump_json

def extract_code_sig(code):
    sig = code.split('\n')[0]
    # sig = code
    sig = sig.replace(' ', '')
    sig = sig.split('(')[0]
    return sig

def extract_code_sig_fix(code):
    sig = code.split('{')[0]
    # sig = code
    sig = sig.replace(' ', '')
    sig = sig.replace('\n', '')
    sig = sig.replace('\t', '')
    return sig

def norm_func(code):
    sig = code
    sig = sig.replace(' ', '')
    sig = sig.replace('\n', '')
    sig = sig.replace('\t', '')
    return sig

def non(code):
    return code


raw_vul_data_path = '/data1/zhijietang/vul_data/datasets/reveal/vulnerables.json'
raw_non_vul_data_path = '/data1/zhijietang/vul_data/datasets/reveal/non-vulnerables.json'
processed_data_base_path = '/data1/zhijietang/vul_data/datasets/reveal/common/random_split/split_2/'
tgt_dump_data_base_path = '/data1/zhijietang/vul_data/datasets/reveal/common/raw_token_splits/split_2/'
sig_extrac_func = non

raw_vuls = read_dumped(raw_vul_data_path)
raw_non_vuls = read_dumped(raw_non_vul_data_path)
print('Reformatting vuls...')
raw_vul_sigs = [reformat_c_cpp_code(item['code'], '/data1/zhijietang/temp/temp_code.c') for item in raw_vuls]
print('Reformatting non_vuls...')
raw_non_vul_sigs = [reformat_c_cpp_code(item['code'], '/data1/zhijietang/temp/temp_code.c') for item in raw_non_vuls]
vul_taken_indices = set()
non_vul_taken_indices = set()


for split in ['train.json', 'validate.json', 'test.json']:
    split_datas = read_dumped(os.path.join(processed_data_base_path, split))
    split_raw_datas = []

    for data_i, data in enumerate(split_datas):
        data_hash = data['hash']
        if data['vul'] == 1:
            raw_datas = raw_vuls
            raw_sigs = raw_vul_sigs
            taken_indices = vul_taken_indices
        else:
            raw_datas = raw_non_vuls
            raw_sigs = raw_non_vul_sigs
            taken_indices = non_vul_taken_indices

        matched_item_indices = [idx for idx,item in enumerate(raw_datas) if item['hash']==data_hash]
        sig = sig_extrac_func(data['code'])
        exact_match_raw_indices = []
        for matched_item_index in matched_item_indices:
            if sig == raw_sigs[matched_item_index]:
                exact_match_raw_indices.append(matched_item_index)

        if len(exact_match_raw_indices) == 0:
            print(f'[Error] No matched item for {data_i}-th item of {split}, hash: {data_hash}.')
        elif len(exact_match_raw_indices) > 1:
            print(f'[Warning] More than one matched items for {data_i}-th item of {split}, hash: {data_hash}. Matched indices: {exact_match_raw_indices}')
        else:
            exact_match_raw_index = exact_match_raw_indices[0]
            if exact_match_raw_index in taken_indices:
                print(f'[Warning] Raw index {exact_match_raw_index} has been taken more than once.')
            else:
                taken_indices.add(exact_match_raw_index)
                matched_data_item = raw_datas[exact_match_raw_index]
                matched_data_item['vul'] = data['vul']
                matched_data_item['index'] = exact_match_raw_index
                split_raw_datas.append(matched_data_item)

    print(f'\n[Info] {split} —— Processed items: {len(split_datas)}, Found matched items: {len(split_raw_datas)}')

    if not os.path.exists(tgt_dump_data_base_path):
        os.mkdir(tgt_dump_data_base_path)
    dump_json(split_raw_datas, os.path.join(tgt_dump_data_base_path, split))
