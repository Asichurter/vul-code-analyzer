import os
from tqdm import tqdm

from utils.data_utils.data_split import random_split
from utils.file import load_json, dump_json

k_fold = 5
# vul_data_path = '/data1/zhijietang/vul_data/datasets/reveal/vulnerables.json'
# non_vul_data_path = '/data1/zhijietang/vul_data/datasets/reveal/non-vulnerables.json'
vul_data_path = '/data1/zhijietang/vul_data/datasets/reveal/small/vulnerables'
non_vul_data_path = '/data1/zhijietang/vul_data/datasets/reveal/small/non-vulnerables'
train_ratio = 0.7
validate_ratio = 0.1
test_ratio = 0.2

# dump_base_path = '/data1/zhijietang/vul_data/datasets/reveal/random_split'
dump_base_path = '/data1/zhijietang/vul_data/datasets/reveal/small/random_split'

def add_label_field(data_objs, label_key, label):
    for obj in data_objs:
        obj[label_key] = label
    return data_objs


def main():

    vul_data = []
    for file in tqdm(os.listdir(vul_data_path)):
        data_item = load_json(os.path.join(vul_data_path, file))
        data_item['file'] = file
        vul_data.append(data_item)

    non_vul_data = []
    for file in tqdm(os.listdir(non_vul_data_path)):
        data_item = load_json(os.path.join(non_vul_data_path, file))
        data_item['file'] = file
        non_vul_data.append(data_item)

    # vul_data = load_json(vul_data_path)
    # non_vul_data = load_json(non_vul_data_path)

    add_label_field(vul_data, 'vulnerable', 1)
    add_label_field(non_vul_data, 'vulnerable', 0)

    for ki in range(k_fold):
        # todo: This is not split of cross validation, but just random split.
        vul_train, vul_val, vul_test = random_split(vul_data, train_ratio, validate_ratio)
        non_vul_train, non_vul_val, non_vul_test = random_split(non_vul_data, train_ratio, validate_ratio)
        train = vul_train + non_vul_train
        validate = vul_val + non_vul_val
        test = vul_test + non_vul_test

        cv_base_path = os.path.join(dump_base_path, f'split_{ki}')
        os.mkdir(cv_base_path)
        dump_json(train, os.path.join(cv_base_path, 'train.json'))
        dump_json(validate, os.path.join(cv_base_path, 'validate.json'))
        dump_json(test, os.path.join(cv_base_path, 'test.json'))


if __name__ == '__main__':
    main()
