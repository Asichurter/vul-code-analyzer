import os
from tqdm import tqdm

from utils.file import load_json, dump_text

src_reveal_data_base = '/data1/zhijietang/vul_data/datasets/reveal'
tgt_dump_base = '/data1/zhijietang/vul_data/datasets/docker/reveal'

for dtype in ['vulnerables', 'non-vulnerables']:
    src_file = load_json(os.path.join(src_reveal_data_base, dtype + '.json'))
    print(f'processing {dtype}')
    for i, item in enumerate(tqdm(src_file)):
        code = item['code']
        hash = item['hash']
        dump_text(code ,os.path.join(tgt_dump_base, dtype, f"{i}_{hash}.c"))
