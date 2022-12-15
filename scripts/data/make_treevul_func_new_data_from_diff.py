import os

from utils.file import load_json, load_text, dump_json
from utils.data_utils.changed_func_extraction import extract_changed_cpp_funcs_from_diff

diffs_base_path = '/data1/zhijietang/vul_data/datasets/treevul-CVE/changed_funcs/treevul_filtered_diffs_v2/all_diffs'
cwe_path_list_file_path = '/data1/zhijietang/vul_data/datasets/treevul-CVE/changed_funcs/treevul_filtered_diffs_v2/treevul_pathlist.json'
extracted_data_dump_path = '/data1/zhijietang/vul_data/datasets/treevul-CVE/changed_funcs/treevul_changed_cwe_funcs_full.json'

cwe_path_list_info = load_json(cwe_path_list_file_path)

extracted_datas = []
for i,file in enumerate(os.listdir(diffs_base_path)):
    print(i, file)
    # file = 'ImageMagick---ImageMagick---fd6144f89f33f3065b0a8436f9af81ab9561459f.diff'
    # file = 'the-tcpdump-group---tcpdump---e2256b4f2506102be2c6f7976f84f0d607c53d43.diff'
    # file = 'zephyrproject-rtos---zephyr---bdb53f244aaae6aecfd33d015804999e5e788ffc.diff'
    diff_file_path = os.path.join(diffs_base_path, file)
    diff = load_text(diff_file_path)
    changed_funcs, succeed = extract_changed_cpp_funcs_from_diff(diff, compare_direc=True)
    if succeed:
        cwe_path = cwe_path_list_info[file]['cwe_path']
        cve_list = cwe_path_list_info[file]['cve_list']
        data_item = {
            'file': file,
            'cwe_path': cwe_path,
            'cve_list': cve_list,
            'changed_funcs': changed_funcs
        }
        extracted_datas.append(data_item)
    print('\n')

print('\n')
print('*'*50)
print(f'Succeed: {len(extracted_datas)} / {len(os.listdir(diffs_base_path))}')
dump_json(extracted_datas, extracted_data_dump_path)