############################################################
#
#   As "torch.nn.Module", not "Allennlp.models.Predictor".
#
############################################################
from typing import List, Union

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.models import Model

from downstream.model.feature_extraction.grad_predictor import GradPredictorV2
from pretrain.comp.nn.line_extractor import LineExtractor
from utils.allennlp_utils.build_utils import build_dataset_reader_from_dict
from utils.file import load_json
from utils.dict import overwrite_dict, delete_dict_items

from pretrain import *


class MLMLineExtractorV2(GradPredictorV2):
    def __init__(self, model_archive_path,
                 reader_config_path,
                 line_extractor: LineExtractor,
                 overwrite_reader_config={},
                 delete_reader_config={},
                 cuda_device: Union[str, int] = 'cpu',):
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
        super().__init__(model, reader)

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text_fields = {
            'code': json_dict['code'],
        }
        ok, instance = self.dataset_reader.text_to_instance(text_fields, forward_type='line_features')
        if not ok:
            raise ValueError(f'Not ok instance: {instance.human_readable_dict()}')
        else:
            return instance

    def predict_batch_with_grad(self, code_list: List[str], **kwargs):
        return self.predict_batch_json_with_grad(
            [
                {'code': code} for code in code_list
            ],
            **kwargs
        )



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
    extractor = MLMLineExtractorV2(model_path, reader_config_path, line_extractor,
                                 overwrite_reader_config, delete_reader_config,
                                 cuda_device=0)
    code1 = "static void inject_user ( void ) {\n size_t len ;\n len = strescape ( ( char * ) injectbuf , ( char * ) injectbuf ) ;\n if ( wdg_c1 -> flags & WDG_OBJ_FOCUSED ) {\n user_inject ( injectbuf , len , curr_conn , 1 ) ;\n }\n else if ( wdg_c2 -> flags & WDG_OBJ_FOCUSED ) {\n user_inject ( injectbuf , len , curr_conn , 2 ) ;\n }\n }"
    code2 = "static void option_import_marks ( const char * marks , int from_stream , int ignore_missing ) {\n if ( import_marks_file ) {\n if ( from_stream ) die ( \"Only one import-marks command allowed per stream\" ) ;\n if ( ! import_marks_file_from_stream ) read_marks ( ) ;\n }\n import_marks_file = make_fast_import_path ( marks ) ;\n safe_create_leading_directories_const ( import_marks_file ) ;\n import_marks_file_from_stream = from_stream ;\n import_marks_file_ignore_missing = ignore_missing ;\n }"
    # outputs = extractor.predict(code1)
    outputs = extractor.predict_batch_with_grad([code1, code2])


