
from copy import deepcopy
from tqdm import tqdm

from utils.file import load_text, dump_json
from utils.data_utils.data_clean import reformat_c_cpp_code

def split_gadgets(text_lines):
    split_line = "------------------------------"
    gadgets = []
    cur_lines = []
    for line in text_lines.split('\n'):
        if line == split_line:
            gadgets.append(deepcopy(cur_lines))
            cur_lines.clear()
        else:
            cur_lines.append(line)

    return gadgets

def process_gadget_code(gadget_lines):
    processed_lines_with_idx = []
    header_line = gadget_lines[0]
    label_line = gadget_lines[-1]

    for line in gadget_lines[1:-1]:
        splits = line.split()

        try:
            maybe_line_idx = int(splits[-1])
        except Exception as e:
            print(f'\n[Error] err: {e}\n')
            return None

        line_content = ' '.join(splits[:-1])
        processed_lines_with_idx.append([maybe_line_idx, line_content])

    processed_lines = [item[1] for item in sorted(processed_lines_with_idx, key=lambda x: x[0])]
    processed_code = '\n'.join(processed_lines)
    processed_code = reformat_c_cpp_code(processed_code, '/data1/zhijietang/temp/temp.tmp')

    return processed_code, int(label_line), header_line

def process_gadget_code_fixed(gadget_lines):
    processed_lines_with_idx = []
    header_line = gadget_lines[0]
    label_line = gadget_lines[-1]
    content_from_last_line = ''
    try:
        for line in gadget_lines[1:-1]:
            if content_from_last_line != '':
                line = f'{content_from_last_line} {line}'
            splits = line.split()
            maybe_line_idx = splits[-1]
            if maybe_line_idx.isdigit():
                line_content = ' '.join(splits[:-1])
                processed_lines_with_idx.append([int(maybe_line_idx), line_content])
                content_from_last_line = ''
            else:
                content_from_last_line += maybe_line_idx + ' '
    except Exception as e:
        print(f'\n[Fatal] err in fixed function: {e}')
        print('*'*50)
        print(f'Lines:\n{gadget_lines}')
        print('*' * 50)
        print()

    processed_lines = [item[1] for item in sorted(processed_lines_with_idx, key=lambda x: x[0])]
    processed_code = '\n'.join(processed_lines)
    processed_code = reformat_c_cpp_code(processed_code, '/data1/zhijietang/temp/temp.tmp')

    return processed_code, int(label_line), header_line

if __name__ == '__main__':
    # lines = ['header', 'a', 'b', 'c 1', 'd 2', 'e', 'f 3', '0']
    # items = process_gadget_code(lines)

    muvulpeeker_mvd_data_path = '/data1/zhijietang/vul_data/datasets/mu_vuldeepecker/mvd.txt'
    tgt_dump_data_path = '/data1/zhijietang/vul_data/datasets/mu_vuldeepecker/processed_gadgets.json'
    print('Loading data...')
    raw_txt = load_text(muvulpeeker_mvd_data_path)
    print('Splitting gadgets...')
    gadget_lines_list = split_gadgets(raw_txt)

    print('Processing gadgets...')
    gadgets = []
    for gadget_lines in tqdm(gadget_lines_list):
        gadget_items = process_gadget_code(gadget_lines)
        if gadget_items is None:
            gadget_items = process_gadget_code_fixed(gadget_lines)
        gadgets.append(gadget_items)

    print('Dumping...')
    dump_json(gadgets, tgt_dump_data_path)