import random
import time
from typing import Iterable, Dict


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
