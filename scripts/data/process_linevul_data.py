import pandas
from tqdm import tqdm

from utils.file import dump_json
from utils.joern_utils.pretty_print_utils import print_code_with_line_num

src_file_path = '/data1/zhijietang/projects/LineVul/data/big-vul_dataset/processed_data.csv'
tgt_file_path = ''

df = pandas.read_csv()

my_rows = []
for idx, row in tqdm(df.iterrows()):
    new_row = {'target': row['target'],
               'index': idx,
               'processed_func': row['processed_func'],
               'flaw_line': row['flaw_line'] if type(row['flaw_line']) != float else None,
               'flaw_line_index': row['flaw_line_index'] if type(row['flaw_line_index']) != float else None}
    my_rows.append(new_row)

dump_json(my_rows, tgt_file_path)
