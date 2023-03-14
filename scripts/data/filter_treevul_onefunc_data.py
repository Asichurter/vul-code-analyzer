from utils.file import read_dumped
from utils.data_utils.data_split import random_split, dump_split_helper
from utils.data_utils.data_dist_check import DataLabelDistStat

filtered_data_path = '/data1/zhijietang/vul_data/datasets/treevul-CVE/changed_funcs/treevul_changed_cwe_funcs_full.json'
dump_split_path = '/data1/zhijietang/vul_data/datasets/treevul-CVE/changed_funcs/split_0'
treevul_filtered_data_path = read_dumped(filtered_data_path)

ok_items = []

for item in treevul_filtered_data_path:
    if len(item['changed_funcs']) != 1:
        continue
    if len(item['changed_funcs'][0]['changed_funcs']) != 1:
        continue
    new_item = {**item}
    # the 1-st changed function, 1-st element (before change function)
    new_item['code'] = item['changed_funcs'][0]['changed_funcs'][0][0]
    ok_items.append(new_item)

print(f'Size: {len(ok_items)}')
stat = DataLabelDistStat(ok_items, lambda x:x['cwe_path'][0][0])
stat.show_dist()

# dump_split_helper(dump_split_path, 'json', *random_split(ok_items, 0.7, 0.1))