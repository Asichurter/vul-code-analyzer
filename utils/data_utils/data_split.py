import random
import time

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
