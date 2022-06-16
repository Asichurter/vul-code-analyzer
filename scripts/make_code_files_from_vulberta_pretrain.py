import re
import os
from os.path import join
import pickle
import pandas as pd
from tqdm import tqdm
import subprocess

pretrained_file_path = '/data1/zhijietang/vul_data/VulBERTa_data/pretrain/drapgh.pkl'
dumped_path = '/data1/zhijietang/vul_data/graph_temp/cfiles/'
reformat_cmd = '/data1/zhijietang/miniconda3/lib/python3.8/site-packages/clang_format/data/bin/clang-format -i -style=file '
max_volume_per_folder = 10000


def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def dump_c_file(file_path, src_code):
    with open(file_path, 'w') as f:
        f.write(src_code)


def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def clean_c_code(c_src):
    # remove c/cpp style comment
    removed_com_src = comment_remover(c_src)

    # remove empty lines
    c_lines = removed_com_src.split('\n')
    new_lines = []
    for line in c_lines:
        if line.strip() != '':
            new_lines.append(line)
    return '\n'.join(new_lines)


def reformat_code(src_path):
    # reformat code
    subprocess.run(
        reformat_cmd + src_path,
        shell=True, check=True
    )


def check_if_dump(src):
    src_lines = src.split('\n')
    # filter too large file
    if len(src_lines) > 100:
        return False
    return True


def main():
    # [NOTE]
    # clear contents before actions
    os.system('rm -rf /data1/zhijietang/vul_data/graph_temp/cfiles/*')

    pretrained_data: pd.DataFrame = read_pickle(pretrained_file_path)
    print("len:", len(pretrained_data))
    dumped_count = 0
    for i in tqdm(range(len(pretrained_data))):
        vol = i // max_volume_per_folder
        # make volume folder
        vol_folder = join(dumped_path, 'vol' + str(vol))
        if not os.path.exists(vol_folder):
            os.mkdir(vol_folder)

        dumped_code_path = join(vol_folder, str(i) + '.c')
        row_function = pretrained_data.loc[i]['functionSource']
        cleaned_code = clean_c_code(row_function)

        if check_if_dump(cleaned_code):
            dump_c_file(dumped_code_path, cleaned_code)
            dumped_count += 1

        # reformat may heavily affect the speed of processing code
        # reformat_code(dumped_code_path)
        # if i == 20000:
        #     print('breaking')
        #     break

    print(f'Total dumped: {dumped_count}')


if __name__ == "__main__":
    main()