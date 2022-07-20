from typing import Dict


def overwrite_dict(old_dict: Dict, new_dict: Dict):
    # Old dict is not a dict but an item, return new dict
    if type(old_dict) != dict:
        return new_dict
    for k,v in new_dict.items():
        if k not in old_dict or type(v) != dict:
            old_dict[k] = v
        else:
            # If target is dict, go recursive
           old_dict[k] = overwrite_dict(old_dict[k], v)
    return old_dict

def delete_dict_items(src_dict: Dict, del_dict: Dict):
    # Src dict is not a dict but an item, return None
    if type(src_dict) != dict:
        return None
    for k,v in del_dict.items():
        if k in src_dict:
            if type(v) != dict:
                del src_dict[k]
            else:
                src_dict[k] = delete_dict_items(src_dict[k], v)
    return src_dict

if __name__ == '__main__':
    src_dict = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4
    }
    del_dict = {'a': 1, 'd': 1}
    src_dict = delete_dict_items(src_dict, del_dict)
