from utils.file import load_json, dump_pickle

L3_data_file_path = '/data1/zhijietang/vul_data/datasets/treevul-CVE/treevul_dataset_cleaned_level3.json'
tgt_data_file_dump_path = '/data1/zhijietang/vul_data/datasets/treevul-CVE/filtered/c_cpp_filtered_L3_commits.pkl'

datas = load_json(L3_data_file_path)

accepted_PLs = ['C', 'C++']

commit_files = {}
for i,data in enumerate(datas):
    commit_id = data['commit_id']
    if commit_id not in commit_files:
        commit_files[commit_id] = []
    commit_files[commit_id].append(i)

filtered_commits = {}
for commit_id, f_indices in commit_files.items():
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
        filtered_commits[commit_id] = files

dump_pickle(filtered_commits, tgt_data_file_dump_path)