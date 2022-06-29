import os
import random
import time

from utils.file import load_json, dump_json

k_fold = 5
vul_data_path = '/data1/zhijietang/vul_data/datasets/reveal/vulnerables.json'
non_vul_data_path = '/data1/zhijietang/vul_data/datasets/reveal/non-vulnerables.json'
train_ratio = 0.8
validate_ratio = 0.1
test_ratio = 0.1

dump_base_path = '/data1/zhijietang/vul_data/datasets/reveal/cv'

def random_split(data_list):
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

def add_label_field(data_objs, label_key, label):
    for obj in data_objs:
        obj[label_key] = label
    return data_objs


def main():
    vul_data = load_json(vul_data_path)
    non_vul_data = load_json(non_vul_data_path)
    add_label_field(vul_data, 'vulnerable', 1)
    add_label_field(non_vul_data, 'vulnerable', 0)

    for ki in range(k_fold):
        vul_train, vul_val, vul_test = random_split(vul_data)
        non_vul_train, non_vul_val, non_vul_test = random_split(non_vul_data)
        train = vul_train + non_vul_train
        validate = vul_val + non_vul_val
        test = vul_test + non_vul_test

        cv_base_path = os.path.join(dump_base_path, f'cv_{ki}')
        os.mkdir(cv_base_path)
        dump_json(train, os.path.join(cv_base_path, 'train.json'))
        dump_json(validate, os.path.join(cv_base_path, 'validate.json'))
        dump_json(test, os.path.join(cv_base_path, 'test.json'))


if __name__ == '__main__':
    main()
