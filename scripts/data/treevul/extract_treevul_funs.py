import os

import git
import json
import subprocess
import pickle
from unidiff import PatchSet

local_repos_meta_info_json_path = '/data1/zhijietang/treevul_projects_retreive_results_updated.json'
treevul_commits_path = '/data1/zhijietang/treevul_filtered_commits.pkl'
diff_output_path = '/data1/zhijietang/treevul_filtered_diffs_v2/diffs'
failed_commit_list_dump_path = "/data1/zhijietang/treevul_filtered_diffs_v2/treevul_failed_commits.json"
cwe_path_list_dump_path = "/data1/zhijietang/treevul_filtered_diffs_v2/treevul_pathlist.json"
dump_base_path = "/data1/zhijietang/treevul_filtered_diffs_v2"
git_diff_dump_cmd_temp = 'git diff --no-index --unified=50000 --output={} {} {}'

temp_files_a_path = '/data1/zhijietang/temp/diff_test/a'
temp_files_b_path = '/data1/zhijietang/temp/diff_test/b'

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def dump_json(cont, file_path):
    with open(file_path, 'w') as f:
        json.dump(cont, f, indent=4)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def write_text(content, file_path):
    file_dir = os.path.join(*os.path.split(file_path)[:-1])
    if not os.path.exists(file_dir):
        os.system(f'mkdir -p {file_dir}')
    with open(file_path, 'w') as f:
        f.write(content)

accepted_file_ext_names = ['c', 'cpp', 'cc', 'c++']
filtered_file_ext_names = set()

def filter_files_in_diff_and_dump(diff, dump_path):
    os.system(f'rm -rf {temp_files_a_path}/*')
    os.system(f'rm -rf {temp_files_b_path}/*')
    patch_set = PatchSet(diff)
    file_index = 0
    for file in patch_set:
        file_ext_name = file.source_file.split('.')[-1]
        if file_ext_name in accepted_file_ext_names:
            # NOTE: Here we assumes the context window is large enough to hold the whole file, thus a file only has one hunk
            before_file_path = f'{temp_files_a_path}/{file.path}'
            after_file_path = f'{temp_files_b_path}/{file.path}'
            write_text(''.join(file[0].source), before_file_path)
            write_text(''.join(file[0].target), after_file_path)
            # write_text(file[0].source, f'{temp_files_a_path}/{file_index}')
            # write_text(file[0].target, f'{temp_files_b_path}/{file_index}')
            file_index += 1
        else:
            filtered_file_ext_names.add(file_ext_name)

    subprocess.run(git_diff_dump_cmd_temp.format(dump_path, temp_files_a_path, temp_files_b_path), shell=True, check=False)


local_repos_meta_infos = load_json(local_repos_meta_info_json_path)
treevul_commits = load_pickle(treevul_commits_path)
succeed, failed = 0, 0
failed_commits = []
cwe_path_lists = {}
for i, (commit_id, commits) in enumerate(treevul_commits.items()):
    print(i, commit_id)
    repo_name = commits[0]['repo']
    repo_name_as_path = '---'.join(repo_name.split('/'))
    try:
        project_path = '/'.join(local_repos_meta_infos[repo_name]['local_clone_dir'].split('/')[:-1])
        # subprocess.run(f'git config --global --add safe.directory {project_path}', shell=True, check=True)
        # repo = git.Repo(project_path)
        # diff_output = repo.git.diff(f'{commit_id}^!', unified=50000)
        # write_text(diff_output, f"{diff_output_path}/{repo_name_as_path}---{commit_id}.diff")
        # filter_files_in_diff_and_dump(diff_output, f"{diff_output_path}/{repo_name_as_path}---{commit_id}.diff")
        succeed += 1
    except Exception as e:
        print(f"Error: {e}")
        failed_commits.append(f"{repo_name}---{commit_id}")
        failed += 1
    finally:
        cwe_path_lists[f'{repo_name_as_path}---{commit_id}.diff'] = {
            'cwe_path': commits[0]['path_list'],
            'cve_list': commits[0]['cve_list']
        }

print('\n\n')
print('*'*50)
print(f'Success: {succeed} Fail: {failed}')
print('*'*50)
print('Failed commits:')
print(failed_commits)
print('*'*50)
print('Filtered Ext Names:')
print(filtered_file_ext_names)

# dump_json(failed_commits, failed_commit_list_dump_path)
# dump_json(list(filtered_file_ext_names), dump_base_path+"/filtered_ext_names.json")
dump_json(cwe_path_lists, cwe_path_list_dump_path)





