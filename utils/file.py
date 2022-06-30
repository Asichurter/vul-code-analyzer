import json
import pickle
from typing import Dict
import os
from datetime import datetime


def load_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        j = json.load(f)
    return j

def dump_json(obj, path, indent=4, sort=False):
    with open(path, 'w', encoding='UTF-8') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, sort_keys=sort)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def read_dumped(file_path, dump_format=None):
    # guess file format when format is not given
    if dump_format is None:
        dump_format = file_path.split('.')[-1]

    if dump_format in ['pickle', 'pkl']:
        return load_pickle(file_path)
    elif dump_format == 'json':
        return load_json(file_path)
    else:
        raise ValueError(f'[read_dumped] Unsupported dump format: {dump_format}')


def save_evaluate_results(results: Dict,
                          other_configs: Dict,
                          save_json_path: str):
    results.update(other_configs)

    if os.path.exists(save_json_path):
        result_list = load_json(save_json_path)
    else:
        result_list = []

    # add time field
    results.update({
        'time': datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
    })

    result_list.append(results)
    dump_json(result_list, save_json_path)