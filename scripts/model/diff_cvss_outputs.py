from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from common import *
from downstream import *
from utils.allennlp_utils.build_utils import build_dataset_reader_from_config
from utils.file import read_dumped, dump_json

class CvssPredictor(Predictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict)

def extract_data(_data):
    return {
        'id': _data[''],
        'cwe_id': _data['CWE ID'],
        'cve': _data['CVE Page'],
        'Availability': _data['Availability'],
        'Complexity': _data['Complexity'],
        'Confidentiality': _data['Confidentiality'],
        'Integrity': _data['Integrity'],
        "codeLink": _data['codeLink'],
        'file_name': _data['file_name'],
        'code': _data['code'],
    }

model_a_path = '/data1/zhijietang/vul_data/run_logs/cvss_metric/14/model.tar.gz'
model_b_path = '/data1/zhijietang/vul_data/run_logs/cvss_metric/25/model.tar.gz'
reader_config_path = '/data1/zhijietang/vul_data/run_logs/cvss_metric/14/config.json'
test_data_path = '/data1/zhijietang/vul_data/datasets/Fan_et_al/cvss_metric_pred/split_1/test.json'
tgt_dump_path = '/data1/zhijietang/temp/cvss_ablation_diff_results.json'

cuda = 7
comp_direction = False
comp_metric_key = 'macro_f1_mean'
diff_max_instances = 100

print('Building reader...')
reader = build_dataset_reader_from_config(reader_config_path)

print('Loading models...')
model_a = Model.from_archive(model_a_path)
model_b = Model.from_archive(model_b_path)
predictor_a = CvssPredictor(model_a, reader)
predictor_b = CvssPredictor(model_b, reader)

print('Reading data...')
test_datas = read_dumped(test_data_path)

diff_outputs = []

for data in test_datas:
    a_outputs = predictor_a.predict_json(data)
    a_metrics = model_a.get_metrics(True)
    b_outputs = predictor_b.predict_json(data)
    b_metrics = model_b.get_metrics(True)

    a_comp_metric = a_metrics[comp_metric_key]
    b_comp_metric = b_metrics[comp_metric_key]

    if (a_comp_metric < b_comp_metric) ^ comp_direction:
        diff_outputs.append({
            'data': extract_data(data),
            'a_metrics': a_metrics,
            'b_metrics': b_metrics
        })
        if len(diff_outputs) == diff_max_instances:
            break

print(f'Total {len(diff_outputs)} instances, dump to {tgt_dump_path}')
dump_json(diff_outputs, tgt_dump_path)

