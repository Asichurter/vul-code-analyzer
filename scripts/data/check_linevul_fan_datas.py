import numpy
import Levenshtein
import subprocess
import os
import random
from unidiff import PatchSet

from utils.file import load_json, dump_text, read_dumped, load_text


fan_original_data_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/fan_full_data.pkl'
# linevul_data_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/linevul_splits/linevul_vul_det_all.json'
linevul_data_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/linevul_splits/linevul_vul_det_before_func_all.json'
temp_files_a_path = '/data1/zhijietang/temp/diff_test/a'
temp_files_b_path = '/data1/zhijietang/temp/diff_test/b'
tgt_diff_path = '/data1/zhijietang/temp/diff_test/fan_linevul_comp_dedup_100.diff'

git_diff_dump_cmd_temp = 'git diff --no-index -w --unified=50000 --output={} {} {}'

original_data = read_dumped(fan_original_data_path)
linevul_data = read_dumped(linevul_data_path)
random.shuffle(original_data)
random.shuffle(linevul_data)

def extract_sig(code):
    return code.split('\n')[0]

def retrieve_topk_similar_items(items, keys, query, sim_func, k=5):
    sims = [sim_func(key, query) for key in keys]
    sims_index = numpy.argsort(sims)
    topk_items = []
    topk_sims = []

    for i,idx in enumerate(reversed(sims_index)):
        if i == k:
            break
        topk_items.append(items[idx])
        topk_sims.append(sims[idx])
    return topk_items, topk_sims

def corase_retrieve(keys, query, sim_func, threshold=-5):
    retrieved_items_indices = []
    for i, key in enumerate(keys):
        sim = sim_func(key, query)
        if sim > threshold:
            retrieved_items_indices.append(i)
    return retrieved_items_indices

def edit_distance_sim(n1, n2):
    return -1 * Levenshtein.distance(n1, n2)

def fetch_multi_list_by_indices(indices, *items_list):
    return [[items[i] for i in indices] for items in items_list]

original_keys = [extract_sig(f['func_before']) for f in original_data]
linevul_keys = [extract_sig(f['code']) for f in linevul_data]
original_values = [f['func_before'] for f in original_data]
linevul_values = [f['code'] for f in linevul_data]

os.system(f'rm {temp_files_a_path}/*')
os.system(f'rm {temp_files_b_path}/*')
# for i in range(len(linevul_data)):

corase_thresh = 5

for i in range(1000):
    print(f"{i} / {len(linevul_data)}")
    linevul_key = linevul_keys[i]
    # hit_item = retrieve_topk_similar_items(original_data, original_keys, linevul_key, edit_distance_sim, k=1)[0][0]

    # Coarse retrieve by func_sig
    corase_hit_item_indices = corase_retrieve(original_keys, linevul_key, edit_distance_sim, threshold=-corase_thresh)
    # Fine-grained localization by full text edit dist
    hit_items, item_sims = retrieve_topk_similar_items(*fetch_multi_list_by_indices(corase_hit_item_indices, original_data, original_values), query=linevul_values[i], sim_func=edit_distance_sim, k=1)

    if len(corase_hit_item_indices) == 0:
        print(f'Warning: #{i} with sig="{linevul_key}" (threshold={corase_thresh}) do not match any results.')
        continue

    hit_item = hit_items[0]
    dump_text(hit_item['func_before'], temp_files_a_path+f'/{i}')
    dump_text(linevul_data[i]['code'], temp_files_b_path+f'/{i}')

subprocess.run(git_diff_dump_cmd_temp.format(tgt_diff_path, temp_files_a_path, temp_files_b_path), shell=True)
ps = PatchSet(load_text(tgt_diff_path))