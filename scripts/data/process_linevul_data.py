import pandas
from tqdm import tqdm

from utils.file import dump_json
from utils.joern_utils.pretty_print_utils import print_code_with_line_num

src_file_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/linevul_splits/split_0/test.csv'
tgt_file_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/linevul_splits/split_0/test.json'

df = pandas.read_csv(src_file_path)

my_rows = []
for idx, row in tqdm(df.iterrows()):
    new_row = {'vul': int(row['target']),
               'index': idx,
               'code': row['processed_func'],
               'flaw_line': row['flaw_line'] if type(row['flaw_line']) != float else None,
               'flaw_line_index': row['flaw_line_index'] if type(row['flaw_line_index']) != float else None}
    my_rows.append(new_row)

dump_json(my_rows, tgt_file_path)
