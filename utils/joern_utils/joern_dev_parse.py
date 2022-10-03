import os
import re
import csv
import subprocess
import shutil
import pickle
from tqdm import tqdm
import argparse

def clean_signature_line(code: str) -> str:
    code = re.sub(r'( |\t|\n)+', ' ', code)
    return code

def convert_func_signature_to_one_line(code_path=None, code=None, redump=True):
    """
    This function aims to convert the signature of a c/cpp function
    into the uniform one line form, with left brace in a new line.
    This function should be called before calling "joern-parse" to
    help fix the location shift bug of function signature identifiers.

    Example:
        seat_set_active_session (Seat *seat, Session *session)
        {
            ...
        }
    """
    if code_path is not None:
        with open(code_path, 'r') as f:
            text = f.read()
    else:
        text = code

    left_bracket_first_idx = text.find('{')
    signature_text = text[:left_bracket_first_idx]
    signature_text = clean_signature_line(signature_text).strip()
    text = signature_text + '\n' + text[left_bracket_first_idx:]

    if redump:
        with open(code_path, 'w') as f:
            f.write(text)
    else:
        return text

def preprocess_rows(rows):
    processed_rows = []
    for row in rows:
        processed_row = ''.join(row).split('\n')
        processed_rows.extend(processed_row)
    return processed_rows

def read_csv_as_list(path):
    rows = list(csv.reader(open(path, 'r')))
    # BugFix: Handle some rows span across multiple line items
    rows = preprocess_rows(rows)

    fields = ''.join(rows[0]).split('\t')
    list_rows = []
    for row in rows[1:]:
        values = ''.join(row).split('\t')
        assert len(values) == len(fields), f'len(values) != len(fields), value: {row}({len(values)}), field_len: {len(fields)}'
        list_rows.append(values)

    # return dict_rows
    return list_rows

def read_csv_as_dict(path):
    rows = list(csv.reader(open(path, 'r')))
    # BugFix: Handle some rows span across multiple line items
    rows = preprocess_rows(rows)

    fields = ''.join(rows[0]).split('\t')
    dict_rows = []
    for row in rows[1:]:
        values = ''.join(row).split('\t')
        assert len(values) == len(fields)
        dict_row = {k:v for k,v in zip(fields, values)}
        dict_rows.append(dict_row)

    return dict_rows

def process_one_file(file_path, file_name, tgt_base_path):
    no_ext_file_name, tgt_folder_path = copy_src_file_to_folder(file_path, file_name, tgt_base_path)
    status, tgt_parsed_path = joern_parse(no_ext_file_name, tgt_base_path, file_name)
    # Remove temp code folder
    shutil.rmtree(tgt_folder_path)
    # Move parsed results to root of the parsed folder
    for tgt_file in ['nodes.csv', 'edges.csv']:
        shutil.move(os.path.join(tgt_parsed_path, no_ext_file_name, file_name, tgt_file),
                    os.path.join(tgt_parsed_path, tgt_file))
    # Remove other folders in parsed folder
    shutil.rmtree(os.path.join(tgt_parsed_path, no_ext_file_name))

    nodes = read_csv_as_list(os.path.join(tgt_parsed_path, 'nodes.csv'))
    edges = read_csv_as_list(os.path.join(tgt_parsed_path, 'edges.csv'))
    file_parsed_res = {
        'file_path': file_path,
        'nodes': nodes,
        'edges': edges,
    }
    shutil.rmtree(tgt_parsed_path)
    return file_parsed_res


def process_one_vol(vol_base_path, vol_name, tgt_vol_base_path):
    vol_path = os.path.join(vol_base_path, vol_name)
    tgt_base_path = os.path.join(tgt_vol_base_path, vol_name)
    tgt_vol_file_name = os.path.join(tgt_vol_base_path, f'joern_parsed_raw_{vol_name}.pkl')
    if not os.path.exists(tgt_base_path):
        os.mkdir(tgt_base_path)

    vol_datas = []
    print(f'Processing {vol_name}')
    fail_processed_items = []
    for item in tqdm(os.listdir(vol_path)):
        item_file_path = os.path.join(vol_path, item)
        try:
            parsed_res = process_one_file(item_file_path, item, tgt_base_path)
            vol_datas.append(parsed_res)
        except Exception as e:
            print(f'Error when processing {item_file_path}: {e}')
            fail_processed_items.append(item_file_path)

    print(f'Failed to process items: ({len(fail_processed_items)} in total):')
    print(fail_processed_items)
    print(f'Dump to {tgt_vol_file_name}')
    with open(tgt_vol_file_name, 'wb') as f:
        pickle.dump(vol_datas, f)
    shutil.rmtree(tgt_base_path)


def copy_src_file_to_folder(src_file_path, file_name, tgt_base_path):
    no_ext_file_name = '.'.join(file_name.split('.')[:-1])
    tgt_folder_path = os.path.join(tgt_base_path, no_ext_file_name)
    if os.path.exists(tgt_folder_path):
        shutil.rmtree(tgt_folder_path)
    os.mkdir(tgt_folder_path)
    tgt_file_path = os.path.join(tgt_folder_path, file_name)
    shutil.copy(src_file_path, tgt_file_path)
    # ADD: Pre-process before "joern-parse", to handle "signature-shift" problem
    convert_func_signature_to_one_line(tgt_file_path)
    return no_ext_file_name, tgt_folder_path


def joern_parse(folder_name, folder_base_path, file_name):
    """
    Core joern func.
    Input is a folder containing code files, output is also a folder containing
    global and local parsing resutls.
    Return:
    - StatusCode:
        - 0: Sucess
        - 1: Error
        - 2: Warning
    """
    status_code = 0
    no_ext_file_name = '.'.join(file_name.split('.')[:-1])
    cmd = f'cd {folder_base_path} && /home/joern/joern-parse {folder_name}'  # cd to folder base dir to prevent long absolute path of parsed dir
    # os.system(cmd)
    parsed_res = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    parsed_err_msg = parsed_res.stderr.read().decode('UTF-8')
    # parsed_out_msg = parsed_res.stdout.read().decode('UTF-8')
    # print(f'Err msg: {parsed_err_msg}')
    # print(f'Out msg: {parsed_out_msg}')

    # Handle error situation
    if 'skipping' in parsed_err_msg:
        print(f'[Error] Error when parsing "{folder_name}", skipped')
        status_code = 1
    # Handle warnings, we continue to process but tag it with status=2
    if 'warning' in parsed_err_msg:
        print(f'[Warning] "{folder_name}": {parsed_err_msg}. Process goes on')
        status_code = 2

    src_parsed_path = os.path.join(folder_base_path, 'parsed')  # Since output_dir of joern-parse is cwd
    tgt_parsed_path = os.path.join(folder_base_path, f'parsed_{no_ext_file_name}_{status_code}')
    os.system(f'mv {src_parsed_path} {tgt_parsed_path}')
    return status_code, tgt_parsed_path


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

def test_copy_src_file_to_folder():
    src_file_name = '/data/cppfiles/vol0/812.cpp'
    file_name = '812-test.cpp'
    tgt_tar_base_path = '/data/joern_dev_analysis_results/vol0/'
    copy_src_file_to_folder(src_file_name, file_name, tgt_tar_base_path)
    print('Done')


def test_joern_parse():
    folder_name = '812-test'
    folder_base_path = '/data/joern_dev_analysis_results/vol0/'
    file_name = '812.cpp'
    status, _ = joern_parse(folder_name, folder_base_path, file_name)
    print(f'Done: {status}')


def test_joern_parse_err():
    folder_name = 'testErr1'
    folder_base_path = '/home/tests'
    file_name = 'testErr1.cpp'
    status, _ = joern_parse(folder_name, folder_base_path, file_name)
    print(f'Done: {status}')


def test_joern_parse_warn_continue():
    folder_name = 'testWarn1'
    folder_base_path = '/home/tests'
    file_name = 'testWarn1.cpp'
    status, _ = joern_parse(folder_name, folder_base_path, file_name)
    print(f'Done: {status}')


def test_joern_parse_warn_break():
    folder_name = 'testWarn2'
    folder_base_path = '/home/tests'
    file_name = 'testWarn2.cpp'
    status, _ = joern_parse(folder_name, folder_base_path, file_name)
    print(f'Done: {status}')


def test_process_one_file():
    file_path = '/data/cppfiles/vol0/811.cpp'
    file_name = '811.cpp'
    tgt_base_path = '/data/joern_dev_analysis_results/vol0'
    process_one_file(file_path, file_name, tgt_base_path)
    print('Done')


def test_process_one_vol():
    vol_base_path = '/data/cppfiles/'
    vol_name = 'voltest'
    tgt_vol_base_path = '/data/joern_dev_analysis_results'
    process_one_vol(vol_base_path, vol_name, tgt_vol_base_path)


if __name__ == '__main__':
    # test_copy_src_file_to_folder()
    # test_joern_parse()
    # test_process_one_file()
    # test_process_one_vol()
    # test_joern_parse_err()
    # test_joern_parse_warn_continue()
    # test_joern_parse_warn_break()

    parser = argparse.ArgumentParser()
    parser.add_argument('--vol', type=str, help='parse which vol')
    parser.add_argument('--vol_base_path', type=str, help='where the volumes locate')
    parser.add_argument('--tgt_vol_base_path', type=str, help='where the parsed volumes locate')
    args = parser.parse_args()

    process_one_vol(args.vol_base_path, args.vol, args.tgt_vol_base_path)


