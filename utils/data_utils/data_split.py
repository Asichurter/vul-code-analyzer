import random
import time
from typing import Iterable, Dict, List
import os

from utils.file import dump_json, dump_pickle


def split_cross_validation_data(data_items, cv):
    cv_num = len(data_items) // cv + 1
    random.seed(time.time() % 6355608 + 1)
    random.shuffle(data_items)

    data_items_cv_split = [data_items[cv_num*cv_i : cv_num*(cv_i+1)] for cv_i in range(cv)]
    for cv_i in range(cv):
        test_cv_set = data_items_cv_split[cv_i]
        train_cv_set = []
        for cv_j in range(cv):
            if cv_j != cv_i:
                train_cv_set.extend(data_items_cv_split[cv_j])

        random.shuffle(train_cv_set)
        random.shuffle(test_cv_set)
        yield train_cv_set, test_cv_set

def random_split(data_list, train_ratio, validate_ratio):
    """
    Return train/validate/test subset.
    """
    random.seed(time.time() % 6355608 + 1)
    random.shuffle(data_list)

    train_num = int(len(data_list) * train_ratio)
    validate_num = int(len(data_list) * validate_ratio)
    test_num = len(data_list) - train_num - validate_num

    return data_list[:train_num], \
           data_list[train_num:train_num+validate_num], \
           data_list[-test_num:]

def class_sensitive_random_split(data_list: Iterable[Dict], train_ratio, validate_ratio, label_key: str, shuffle=True):
    trains, validates, tests = [], [], []
    labels = set([d[label_key] for d in data_list])
    for label in labels:
        labeled_data = [d for d in data_list if d[label_key]==label]
        labeled_train, labeled_val, labeled_test = random_split(labeled_data, train_ratio, validate_ratio)
        trains += labeled_train
        validates += labeled_val
        tests += labeled_test

    if shuffle:
        random.shuffle(trains)
        random.shuffle(validates)
        random.shuffle(tests)

    return trains, validates, tests

def class_sensitive_random_split_by_list(data_list: Iterable[List], train_ratio, validate_ratio, label_index: int, shuffle=True):
    trains, validates, tests = [], [], []
    labels = set([d[label_index] for d in data_list])
    for label in labels:
        labeled_data = [d for d in data_list if d[label_index]==label]
        labeled_train, labeled_val, labeled_test = random_split(labeled_data, train_ratio, validate_ratio)
        trains += labeled_train
        validates += labeled_val
        tests += labeled_test

    if shuffle:
        random.shuffle(trains)
        random.shuffle(validates)
        random.shuffle(tests)

    return trains, validates, tests

def sample_groups(data_items: List[Dict], group_key: str, total: int, group_sample_ratio: Dict[str, float],
                  strict_sample_check: bool = False):
    groups = {}
    for item in data_items:
        key = str(item[group_key])
        if key not in group_sample_ratio:
            continue
        if key not in groups:
            groups[key] = []
        groups[key].append(item)

    sampled = []
    for g_name, group_items in groups.items():
        g_num = int(total * group_sample_ratio[g_name])
        if g_num > len(group_items):
            msg = f'Group "{g_name}" have no more than {g_num}({total} * {group_sample_ratio[g_name]}) items (only {len(group_items)}).'
            if strict_sample_check:
                assert False, msg
            else:
                print('\nWarning:' + msg + '\n')
        group_sampled = random.sample(group_items, g_num)
        sampled.extend(group_sampled)

    random.shuffle(sampled)
    return sampled


def dump_split_helper(dump_base_path, dump_format='json', *split_outputs):
    train, val, test = split_outputs
    if dump_format == 'json':
        dump_json(train, os.path.join(dump_base_path, 'train.json'))
        dump_json(val, os.path.join(dump_base_path, 'validate.json'))
        dump_json(test, os.path.join(dump_base_path, 'test.json'))
    elif dump_format == 'pkl':
        dump_pickle(train, os.path.join(dump_base_path, 'train.pkl'))
        dump_pickle(val, os.path.join(dump_base_path, 'validate.pkl'))
        dump_pickle(test, os.path.join(dump_base_path, 'test.pkl'))
    else:
        raise ValueError(f'dump_format={dump_format}')
