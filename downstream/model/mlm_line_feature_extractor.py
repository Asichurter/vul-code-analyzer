import torch

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from allennlp.models import Model

from pretrain.comp.nn.line_extractor import LineExtractor
from utils.allennlp_utils.build_utils import build_dataset_reader_from_dict
from utils.file import load_json
from utils.dict import overwrite_dict, delete_dict_items

from pretrain import *

class MLMLineExtractor(Predictor):
    def __init__(self, model_archive_path,
                 reader_config_path,
                 line_extractor: LineExtractor,
                 overwrite_reader_config={},
                 delete_reader_config={},
                 cuda_device='cpu'):
        print(f'Loading model...')
        model = Model.from_archive(model_archive_path)
        model.line_extractor = line_extractor
        model = model.to(cuda_device)
        print(f'Loading reader...')
        reader_config = load_json(reader_config_path)['dataset_reader']
        reader_config = overwrite_dict(reader_config, overwrite_reader_config)
        reader_config = delete_dict_items(reader_config, delete_reader_config)
        reader = build_dataset_reader_from_dict(reader_config)
        print('Init predictor...')
        super().__init__(model, reader, frozen=True)

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text_fields = {
            'code': json_dict['code'],
        }
        ok, instance = self._dataset_reader.text_to_instance(text_fields, forward_type='line_features')
        if not ok:
            raise ValueError(f'Not ok instance: {instance.human_readable_dict()}')
        else:
            return instance

    def predict(self, code: str):
        return self.predict_json({
            'code': code
        })

if __name__ == '__main__':
    model_path = '/data1/zhijietang/vul_data/run_logs/pretrain/15/model.tar.gz'
    reader_config_path = '/data1/zhijietang/vul_data/run_logs/pretrain/15/config.json'
    overwrite_reader_config = {
        'type': 'raw_pdg_predict',
        'max_lines': 50,
        'code_max_tokens': 256,
        'code_tokenizer': {'max_length': 256},
        'identifier_key': None,
        # 'meta_data_keys': {'edges': 'edges', 'vulnerable': 'label', 'file': 'file'}
    }
    delete_reader_config = {
        'from_raw_data': 1,
        'pdg_max_vertice': 1
    }
    line_extractor = AvgLineExtractor(max_lines=50)
    extractor = MLMLineExtractor(model_path, reader_config_path, line_extractor,
                                 overwrite_reader_config, delete_reader_config,
                                 cuda_device=0)
    code = "static void inject_user ( void ) {\n size_t len ;\n len = strescape ( ( char * ) injectbuf , ( char * ) injectbuf ) ;\n if ( wdg_c1 -> flags & WDG_OBJ_FOCUSED ) {\n user_inject ( injectbuf , len , curr_conn , 1 ) ;\n }\n else if ( wdg_c2 -> flags & WDG_OBJ_FOCUSED ) {\n user_inject ( injectbuf , len , curr_conn , 2 ) ;\n }\n }"
    outputs = extractor.predict(code)


