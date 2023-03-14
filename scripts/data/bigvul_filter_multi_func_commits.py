from utils.file import read_dumped

commit_data_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/fan_full_data.pkl'
commits = read_dumped(commit_data_path)

table = {}
for i, commit in enumerate(commits):
    key = f'{commit["project"]}---{commit["commit_id"]}'
    if key not in table:
        table[key] = {'0': [], '1': []}
    table[key][commit['vul']].append(i)

one_func_commits = []
for k,v in table.items():
    if len(v['1']) == 1:
        one_func_commits.append(commits[v['1'][0]])

# do dumping...

