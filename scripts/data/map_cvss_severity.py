import os

from utils.file import read_dumped, dump_json


def map_severity_inplace(data_path):
    print(f'\nReading {data_path}...')
    src_data = read_dumped(data_path)

    def map_severity(score):
        score = float(score)
        if score < 4:
            return 'Low'
        elif score < 7:
            return 'Medium'
        else:
            return 'High'

    print(f'Processing {data_path}...')
    for item in src_data:
        try:
            item['Severity'] = map_severity(item['Score'])
        except Exception as e:
            print(f'Error when map severity, score: {item["Score"]}, err: {e}')
            item['Severity'] = ''

    print(f'Dumping {data_path}...\n')
    dump_json(src_data, data_path)

if __name__ == '__main__':
    data_base_path = "/data1/zhijietang/vul_data/datasets/Fan_et_al/cvss_metric_pred/split_0/"
    file_names = ['train.json', 'validate.json', 'test.json']
    for file in file_names:
        data_path = os.path.join(data_base_path, file)
        map_severity_inplace(data_path)
