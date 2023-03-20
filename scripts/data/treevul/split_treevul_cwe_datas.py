import os

from utils.data_utils.data_split import class_sensitive_random_split
from utils.file import load_pickle, dump_json, dump_pickle

commit_data_path = '/data1/zhijietang/vul_data/datasets/treevul-CVE/filtered/c_cpp_filtered_full_commits.pkl'
tgt_data_dump_path = '/data1/zhijietang/vul_data/datasets/treevul-CVE/filtered/filtered_full_splits/split_1'

train_ratio = 0.7
val_ratio = 0.1

commit_files = load_pickle(commit_data_path)
cwe_commits = []
train_commits, val_commits, test_commits = [], [], []

for cid, files in commit_files.items():
    assert len(files[0]['path_list']) == 1, f'commit={cid} has more than one path ({len(files[0]["path_list"])})'
    cwe_label = files[0]['path_list'][0][0] # Root of the CWE tree
    cwe_commits.append({
        'cid': cid,
        'cwe_label': cwe_label,
        'files': files
    })

(train_commits,
 val_commits,
 test_commits) = class_sensitive_random_split(cwe_commits, train_ratio, val_ratio, 'cwe_label', True)

for commits, file_name in zip((train_commits, val_commits, test_commits), ('train', 'validate', 'test')):
    dump_json(commits, os.path.join(tgt_data_dump_path, f'{file_name}.json'))

