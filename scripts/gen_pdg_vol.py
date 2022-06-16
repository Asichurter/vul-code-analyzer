import os
import subprocess
import re
import json
import sys
import logging
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

code_base_path = '/root/data/cppfiles/'
dump_path = '/root/data/packed_data_jsons/'
temp_dot_dir = '/root/data/temp/dot_temp_{vol}/'
temp_bin_path = '/root/data/temp/temp_pdg_{vol}.bin'
target_dot_filename = '0-pdg.dot'  # guessed filename in the output directory, needs to be validated


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vol', type=str, help='processed vol. number')
    return parser.parse_args()


def rm_dir(dir_path):
    os.system(f'rm -rf {dir_path}')


def load_json(json_path):
    with open(json_path, 'r', encoding='UTF-8') as f:
        j = json.load(f)
    return j


def dump_json(obj, path, indent=4, sort=False):
    with open(path, 'w', encoding='UTF-8') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, sort_keys=sort)


def convert_c_cpp_type(origin_file_path):
    with open(origin_file_path, 'r') as f:
        lines = f.readlines()

    name_list = origin_file_path.split('.')
    assert len(name_list) == 2, f'File path does not meet one-dot requirement: {origin_file_path}'
    file_name, ext_name = name_list
    if ext_name == 'c':
        new_file_path = file_name + '.cpp'
    else:
        new_file_path = file_name + '.c'

    with open(new_file_path, 'w') as f:
        f.writelines(lines)
    # remove original file
    os.system(f'rm {origin_file_path}')
    return new_file_path


joern_parse_cmd_temp = 'joern-parse {code_path} -o {parsed_bin_path} --namespaces std --language c'
joern_export_cmd_temp = 'joern-export {parsed_bin_path} --repr {repr} --out {out_dir} --format dot'
dot2json_cmd_temp = 'dot -Txdot_json -o {out_json_path} {in_dot_path}'


def extract_line_num(v_label):
    matched = re.match('.*<SUB>(.*)<\/SUB>', v_label)
    if matched is None:
        logger.warning(f"v_label={v_label} does not match the line number pattern") # handle the cases like vol0/217.cpp
        return None
    groups = matched.groups()
    if len(groups) > 1:
        # match one more line number group, error
        logger.warning(f"v_label={v_label} does not exactly matches one line number: {groups}!")
    elif len(groups) == 0:
        logger.error(f"v_label={v_label} does not exactly matches any line numbers: {groups}!")
        return None
    return int(groups[0])


def process_edges(dot_obj):
    """
        Normalize edges and specify edges' type, return a processed edge list.

        Input: PDG in dot-json format.
        Output: List of strings, with each as an edge of control, data or both dependency.
    """
    # compute vertice mapping from v_id to line_no.
    v_map = {}
    for v_obj in dot_obj['objects']:
        v_idx = int(v_obj['_gvid'])
        line_num = extract_line_num(v_obj['label'])
        v_map[v_idx] = line_num

    # recompute edges using normalized vertices
    control_edges = set()
    data_edges = set()
    out_edges = []
    for e_obj in dot_obj['edges']:
        tail, head = int(e_obj['tail']), int(e_obj['head'])
        map_tail, map_head = v_map[tail], v_map[head]
        # drop the edge connecting vertices without matching line number
        if map_tail is None or map_head is None:
            continue
        edge = f'{map_tail},{map_head}'  # edge format: "[tail],[head]"
        if 'CDG' in e_obj['label']:
            control_edges.add(edge)
        elif 'DDG' in e_obj['label']:
            data_edges.add(edge)
        else:
            logger.warning(f'Neither CDG or DDG edge: {e_obj}')

    # Merge control edges and data edges into unified edges.
    # Use a special id to identify different edges:
    #     1: Only data edges
    #     2: Only control edges:
    #     3: Both data and control edges
    #  Edge format: "[tail],[head] [edge_id]"
    for edge in control_edges:
        if edge in data_edges:
            data_edges.remove(edge)  # leave data-only edges in data's edge set
            out_edges.append(edge + ' 3')
        else:
            out_edges.append(edge + ' 2')
    for edge in data_edges:
        out_edges.append(edge + ' 1')
    return out_edges


def pack_file_data(code_file, edges):
    """
        Pack all the needed info of a function in a file.
        Contains:
            # code: Original Code tokens.
            # total_line: Number of total lines, i.e. number of total vertices in the PDG.
            # edges: Edge list with tail, head and speicified edge type, in string format.
    """
    file_data = {
        'edges': edges
    }
    with open(code_file, 'r') as f:
        lines = f.readlines()
        file_data['total_line'] = len(lines)
        file_data['code'] = ''.join(lines)
    return file_data


def process_dot_json(dot_json_path, code_file_path, dump_json_path):
    """
        Entrance for processing a PDG in json format and dump as a processed data file.
    """
    dot_PDG_json = load_json(dot_json_path)
    func_name = dot_PDG_json['name']
    logger.info(f'Func Name: {func_name}')
    edges = process_edges(dot_PDG_json)
    packed_data = pack_file_data(code_file_path, edges)
    dump_json(packed_data, dump_json_path)


def parse_pdg_from_code(vol, code_file_path, file_name):
    # delete temp files
    os.system(f"rm {os.path.join(temp_dot_dir, 'graph.json')}")
    rm_dir(temp_dot_dir)

    # joern parse, bin as output
    joern_parse_cmd = joern_parse_cmd_temp.format(
        code_path=code_file_path,
        parsed_bin_path=temp_bin_path
    )
    subprocess.run(joern_parse_cmd, shell=True, check=True)

    # joern export, dot files as output
    joern_export_cmd = joern_export_cmd_temp.format(
        parsed_bin_path=temp_bin_path,
        repr='pdg',
        out_dir=temp_dot_dir
    )
    subprocess.run(joern_export_cmd, shell=True, check=True)

    # check if target output dot file exists
    if not os.path.exists(os.path.join(temp_dot_dir, target_dot_filename)):
        logger.warning(f'File ({code_file_path} parse failed, no target dot file ({target_dot_filename}))')
        return False

    # convert dot file to json format
    dot2json_cmd = dot2json_cmd_temp.format(
        in_dot_path=os.path.join(temp_dot_dir, target_dot_filename),
        out_json_path=os.path.join(temp_dot_dir, 'graph.json')
    )
    subprocess.run(dot2json_cmd, shell=True, check=True)

    # process PDG dot file, pack needed data for code analysis as a unified file
    process_dot_json(
        dot_json_path=os.path.join(temp_dot_dir, 'graph.json'),
        code_file_path=os.path.join(temp_dot_dir, code_file_path),
        dump_json_path=os.path.join(dump_path, vol, file_name + '.json')
    )
    return True


def main():
    args = read_args()
    vol = 'vol' + args.vol
    global temp_bin_path, temp_dot_dir, logger
    temp_bin_path = temp_bin_path.format(vol=args.vol)
    temp_dot_dir = temp_dot_dir.format(vol=args.vol)

    fh = logging.FileHandler(f'/root/data/temp/gen_pdg_{vol}.log', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    count = 0
    # for vol in os.listdir(code_base_path):
    vol_path = os.path.join(code_base_path, vol)
    if not os.path.exists(os.path.join(dump_path, vol)):
        os.mkdir(os.path.join(dump_path, vol))

    for file in os.listdir(vol_path):
        print("*" * 30, '\n', f'Count: #{count}\n', "*" * 30, '\n')
        code_file_path = os.path.join(vol_path, file)
        file_name = file.split('.')[0]
        logger.info(f'Vol #{vol}, Count #{count}, path: {code_file_path}')

        try:
            succeed = parse_pdg_from_code(vol, code_file_path, file_name)
            if not succeed:
                code_file_path = convert_c_cpp_type(code_file_path)
                succeed_2nd = parse_pdg_from_code(vol, code_file_path, file_name)
                if not succeed_2nd:
                    logger.critical(f'File: {code_file_path} parsed failed with both c/cpp format')
        except Exception as e:
            logger.error(f'Error: {str(e)}')
        count += 1
        # sys.exit(0)


if __name__ == '__main__':
    main()
