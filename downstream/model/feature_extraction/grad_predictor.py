from typing import List, Dict

import torch
from allennlp.common import JsonDict
from allennlp.data import Instance, Batch
from allennlp.nn import util


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