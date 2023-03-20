from utils.file import load_json, dump_pickle

L3_data_file_path = '/data1/zhijietang/vul_data/datasets/treevul-CVE/treevul_dataset_cleaned_full.json'
tgt_data_file_dump_path = '/data1/zhijietang/vul_data/datasets/treevul-CVE/filtered/c_cpp_filtered_full_commits_fixed.pkl'

datas = load_json(L3_data_file_path)

accepted_PLs = ['C', 'C++']

commit_files = {}
for i,data in enumerate(datas):
    key = f"{data['repo']}-{data['commit_id']}"
    if key not in commit_files:
        commit_files[key] = []
    commit_files[key].append(i)

filtered_commits = {}
for key, f_indices in commit_files.items():
    flag = True
    files = []
    for idx in f_indices:
        file = datas[idx]
        pl = file['PL']
        # Ensure all the files in the commit are accepted
        if pl not in accepted_PLs:
            flag = False
            break
        else:
            files.append(file)
    if flag:
        filtered_commits[key] = files

dump_pickle(filtered_commits, tgt_data_file_dump_path)