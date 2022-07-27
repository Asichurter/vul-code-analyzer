from typing import List, Dict, Union

import torch

from allennlp.common import JsonDict
from allennlp.data import Instance, Batch
from allennlp.predictors import Predictor
from allennlp.models import Model
from allennlp.nn import util
from allennlp.common.checks import ConfigurationError

from pretrain.comp.nn.line_extractor import LineExtractor
from utils.allennlp_utils.build_utils import build_dataset_reader_from_dict
from utils.file import load_json
from utils.dict import overwrite_dict, delete_dict_items

from pretrain import *


class GradPredictorV2(torch.nn.Module):
    def __init__(self, model, reader):
        super(GradPredictorV2, self).__init__()
        self.model = model
        self.dataset_reader = reader
        # super().__init__(model, reader, frozen)

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        raise NotImplementedError

    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        instances = []
        for json_dict in json_dicts:
            instances.append(self._json_to_instance(json_dict))
        return instances

    def predict_batch_json_with_grad(self, inputs: List[JsonDict], **kwargs) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        return self.predict_batch_instance_with_grad(instances, **kwargs)

    def predict_batch_instance_with_grad(self, instances: List[Instance], **kwargs) -> List[JsonDict]:
        for instance in instances:
            self.dataset_reader.apply_token_indexers(instance)
        outputs = self.forward_on_instances_with_grad(instances, **kwargs)
        return outputs

    def forward_on_instances_with_grad(self, instances: List[Instance], separate_instances: bool = True) -> List[Dict[str, torch.Tensor]]:
        """
        We re-implement this method in predictor which is original from "Model" class for
        intergrated implementation of gradient prediction.

        Compared with original method, we:
        (1) Remove "torch.no_grad()" context.
        (2) Remove ".detach().cpu().numpy()".
        (3) Remove "human_readable()".
        (4) Replace all the self's method calls with self.model'.
        """
        batch_size = len(instances)
        # with torch.no_grad():
        cuda_device = self.model._get_prediction_device()
        dataset = Batch(instances)
        dataset.index_instances(self.model.vocab)
        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        outputs = self.model(**model_input)

        if separate_instances:
            instance_separated_output: List[Dict[str, torch.Tensor]] = [
                {} for _ in dataset.instances
            ]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self.model._maybe_warn_for_unseparable_batches(name)
                        continue
                    # output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self.model._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output
        else:
            return outputs


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


