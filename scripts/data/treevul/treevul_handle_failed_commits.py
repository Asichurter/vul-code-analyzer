import subprocess
import json
import requests
import os
import time
import base64

failed_commits_stat_dumped_path = '/data1/zhijietang/treevul_filtered_diffs_v2/treevul_failed_commits.json'
new_failed_commits_stat_dumped_path = '/data1/zhijietang/treevul_filtered_diffs_v2/treevul_failed_commits_ver2.json'
commit_dump_base_path = '/data1/zhijietang/treevul_filtered_diffs_v2/failed_diffs'
temp_files_a_path = '/data1/zhijietang/temp/diff_test/a'
temp_files_b_path = '/data1/zhijietang/temp/diff_test/b'

commit_api_cmd_temp = 'curl -H "Accept: application/vnd.github+json" -H "Authorization: Bearer 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" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/{}/commits/{}'
api_full_url_cmd_temp = 'curl -H "Accept: application/vnd.github+json" -H "Authorization: Bearer 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" -H "X-GitHub-Api-Version: 2022-11-28" {}'
file_api_cmd_temp = 'curl -H "Accept: application/vnd.github+json" -H "Authorization: Bearer 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" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/{repo}/contents/{filename}?ref={commit_id}'
file_raw_url_temp = 'https://github.com/{repo}/raw/{sha}/{filename}'
git_diff_dump_cmd_temp = 'git diff --no-index --unified=50000 --output={} {} {}'

accepted_file_ext_names = ['c', 'cpp', 'cc', 'c++']

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def dump_json(cont, file_path, indent=None):
    with open(file_path, 'w') as f:
        json.dump(cont, f, indent=indent)

def write_text(content, file_path):
    file_dir = os.path.join(*os.path.split(file_path)[:-1])
    if not os.path.exists(file_dir):
        os.system(f'mkdir -p {file_dir}')
    with open(file_path, 'w') as f:
        f.write(content)


commits_to_process = load_json(failed_commits_stat_dumped_path)
succeed, failed = 0, 0
failed_commits = []

for i, commit in enumerate(commits_to_process):
    # commit = commit['commit']
    print(f'\n{i+1}/{len(commits_to_process)} {commit}')
    repo, commit_id = commit.split('---')
    commit_api_cmd = commit_api_cmd_temp.format(repo, commit_id)

    try:
        api_output = json.loads(subprocess.check_output(commit_api_cmd, shell=True))

        # Handle redis-cached problem
        if 'message' in api_output and api_output['message'] == 'Moved Permanently':
            api_output = json.loads(subprocess.check_output(api_full_url_cmd_temp.format(api_output['url']), shell=True))

        files = api_output['files']
        filenames = [file['filename'] for file in files]
        if len(api_output['parents']) != 1:
            print(f'Warning! Commit {commit} has not one parent: {api_output["parents"]}')
            failed += 1
            failed_commits.append({
                'commit': commit,
                'cause': f'Not one parent({len(api_output["parents"])})'
            })
            continue
        else:
            parent_commit_id = api_output['parents'][0]['sha']

        os.system(f'rm -rf {temp_files_a_path}/*')
        os.system(f'rm -rf {temp_files_b_path}/*')
        for j, filename in enumerate(filenames):
            file_ext_name = filename.split('.')[-1]
            if file_ext_name not in accepted_file_ext_names:
                continue

            before_api_output = json.loads(subprocess.check_output(file_api_cmd_temp.format(repo=repo, filename=filename, commit_id=parent_commit_id), shell=True))
            after_api_output = json.loads(subprocess.check_output(file_api_cmd_temp.format(repo=repo, filename=filename, commit_id=commit_id), shell=True))

            before_fetch_raw_output, after_fetch_raw_output = False, False
            if 'message' in before_api_output:
                msg = before_api_output['message']
                # Handle redis-cached problem
                if msg == 'Moved Permanently':
                    before_api_output = json.loads(subprocess.check_output(api_full_url_cmd_temp.format(before_api_output['url']), shell=True))
                elif msg == 'Not Found':
                    # Make a dummy output
                    before_api_output = {'content': ''}
                    before_fetch_raw_output = True
                else:
                    print(f'Warning: Unhandled message for before-change file: {msg}')
            if 'message' in after_api_output:
                msg = after_api_output['message']
                # Handle redis-cached problem
                if msg == 'Moved Permanently':
                    after_api_output = json.loads(subprocess.check_output(api_full_url_cmd_temp.format(after_api_output['url']), shell=True))
                elif msg == 'Not Found':
                    # Make a dummy output
                    after_api_output = {'content': ''}
                    after_fetch_raw_output = True
                else:
                    print(f'Warning: Unhandled message for after-change file: {msg}')

            before_file_content = base64.b64decode(before_api_output['content']).decode() if not before_fetch_raw_output else before_api_output['content']
            after_file_content = base64.b64decode(after_api_output['content']).decode() if not after_fetch_raw_output else after_api_output['content']

            #--------------------------------------------------------------------------------------------------
            # DEPRECATED: Pulling content from github.raw using requests
            # --------------------------------------------------------------------------------------------------
            # before_file_raw_url = file_raw_url_temp.format(repo=repo, sha=parent_commit_id, filename=filename)
            # after_file_raw_url = file_raw_url_temp.format(repo=repo, sha=commit_id, filename=filename)
            # print(f'before: {before_file_raw_url}, after: {after_file_raw_url}')
            # print(f'Get before-code: {before_file_raw_url}')
            # before_file_raw_results = requests.get(before_file_raw_url)
            # print(f'Get after-code: {after_file_raw_url}')
            # after_file_raw_results = requests.get(after_file_raw_url)
            # before_file_content = before_file_raw_results.content.decode()
            # after_file_content = after_file_raw_results.content.decode()
            # --------------------------------------------------------------------------------------------------

            write_text(before_file_content, f'{temp_files_a_path}/{filename}')
            write_text(after_file_content, f'{temp_files_b_path}/{filename}')

        repo_as_file_path = repo.replace('/', '---')
        commit_dump_path = f"{commit_dump_base_path}/{repo_as_file_path}---{commit_id}.diff"
        git_diff_dump_cmd = git_diff_dump_cmd_temp.format(commit_dump_path, temp_files_a_path, temp_files_b_path)
        subprocess.run(git_diff_dump_cmd, shell=True, check=False)

        succeed += 1
        time.sleep(3)

    except Exception as e:
        print(f'Error for Commit {commit}: {e}')
        failed += 1
        failed_commits.append({
            'commit': commit,
            'cause': f'Error: {e}'
        })

print('\n')
print("*"*50)
print(f'Succed: {succeed} Failed: {failed}')
print("*"*50)
dump_json(failed_commits, new_failed_commits_stat_dumped_path, 4)