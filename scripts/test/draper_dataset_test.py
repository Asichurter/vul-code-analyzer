import h5py
from tqdm import tqdm

from utils.file import dump_pickle, load_pickle

# labels = {}
for split in ['test', 'validate', 'train']:
    file = h5py.File(f'/data1/zhijietang/vul_data/datasets/draper-dataset/VDISC_{split}.hdf5', 'r')
    print(f"{split} total: {len(file['functionSource'])}")
    # split_labels = {}
    # for k in file.keys():
    #     if k == 'functionSource':
    #         continue
    #
    #     idxes = set()
    #     for i in tqdm(range(len(file[k]))):
    #         if file[k][i]:
    #             idxes.add(i)
    #     split_labels[k] = idxes
    #
    # labels[split] = split_labels

# dump_pickle(labels, '/data1/zhijietang/vul_data/draper-dataset/draper-label-stat.pkl')
#
labels = load_pickle('/data1/zhijietang/vul_data/datasets/draper-dataset/draper-label-stat.pkl')
for split in ['test', 'validate', 'train']:
    vul_sum = 0
    for vul_key in ['CWE-119', 'CWE-120', 'CWE-469', 'CWE-476', 'CWE-other']:
        print(f'Split={split}, vul_type={vul_key}, len={len(labels[split][vul_key])}')
        vul_sum += len(labels[split][vul_key])
    print(f'{split} Vul Sum: {vul_sum}')