from os.path import join as pjoin

from utils.file import read_dumped, dump_json
from utils.data_utils.data_split import dump_split_helper, sample_groups, random_split

src_data_base_path = '/data2/zhijietang/vul_data/datasets/Fan_et_al/vul_det/cleaned_splits/split_0'
tgt_data_base_path = '/data2/zhijietang/vul_data/datasets/Fan_et_al/vul_det/neuralpda_cleaned_splits'
cv = 10

full_datas = []
for split in ['train', 'validate', 'test']:
    split_data = read_dumped(pjoin(src_data_base_path, f'{split}.json'))
    full_datas.extend(split_data)

for ver in range(cv):
    print(f"# CV-{ver}")
    # Sample
    sampled_full_datas = sample_groups(full_datas, 'vul',
                                       total=41444,     # max num of 26.3% vul
                                       group_sample_ratio={'1': 0.263, '0': 0.737},
                                       strict_sample_check=True)
    # Split
    dump_split_helper(pjoin(tgt_data_base_path, f'split_{ver}'), 'json', *random_split(sampled_full_datas, 0.8, 0.1))