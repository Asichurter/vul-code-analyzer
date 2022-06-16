import json
import pickle

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