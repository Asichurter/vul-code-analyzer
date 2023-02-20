
from utils.file import read_dumped, dump_json

data_file_path = '/data1/zhijietang/projects/PDBERT/data/datasets/extrinsic/vul_assess/validate.json'

def thin_item(item):
    return {
        'code': item['code'],
        'index': item['index'],
        'Complexity': item['Complexity'],
        'Availability': item['Availability'],
        'Confidentiality': item['Confidentiality'],
        'Integrity': item['Integrity'],
        'vul': item['vul']
    }

raw_datas = read_dumped(data_file_path)
new_datas = [thin_item(item) for item in raw_datas]

dump_json(new_datas, data_file_path)