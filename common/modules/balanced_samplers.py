from typing import List, Iterable, Sequence, Optional, Tuple
import random
from copy import deepcopy

from allennlp.data.fields import LabelField, TensorField
from allennlp.data.samplers.batch_sampler import BatchSampler
from allennlp.data.instance import Instance

from utils import GlobalLogger as mylogger

#todo: refactor class structure, to abstract a super class of iter and rand sampler
@BatchSampler.register('binary_duplicate_balanced_iter')
class BinaryDuplicateBalancedIterSampler(BatchSampler):
    def __init__(self,
                 batch_size: int,
                 majority_label_index: int = 0,
                 label_key: str = 'label',
                 major_instance_ratio_in_batch: float = 0.5):
        self._batch_size = batch_size
        self._majority_label_index = majority_label_index
        self._label_key = label_key
        self._major_instance_ratio_in_batch = major_instance_ratio_in_batch


    def _get_majority_and_minority_idxes(self,
                                         instances: Sequence[Instance],
                                         shuffle: bool = False) -> Tuple[List[int],List[int]]:
        major_instance_idxes = []
        minor_instance_idxes = []
        for inst_idx, instance in enumerate(instances):
            label_field: LabelField = instance.fields.get(self._label_key)
            assert label_field is not None, \
                f'[{self.__class__.__name__}._get_majority_and_minority_idxes] ' \
                                      f'None returned by label_key {self._label_key}, legal keys are: {instance.fields.keys()}'
            if isinstance(label_field, TensorField):
                assert len(label_field.tensor) == 1, f'Expect TensorField label with len=1, but got len={len(label_field.tensor)}'
                label_index = label_field.tensor.item()
            elif isinstance(label_field, LabelField):
                label_index = label_field._label_id
            else:
                raise ValueError(f'Unsupported field type of label. Expect to got LabelField or TensorField, but got {type(label_field)}')

            if label_index == self._majority_label_index:
                major_instance_idxes.append(inst_idx)
            else:
                minor_instance_idxes.append(inst_idx)

        if shuffle:
            random.shuffle(major_instance_idxes)
            random.shuffle(minor_instance_idxes)

        return major_instance_idxes, minor_instance_idxes


    def _major_num_per_batch(self):
        return int(self._batch_size * self._major_instance_ratio_in_batch)


    def _sample_batch_idxes_step(self,
                                 current_batch_idx: int,
                                 majority_instance_idxes: List[int],
                                 minority_instance_idxes: List[int],
                                 seed: int = None) -> List[int]:
        """
        We iterate over major instances without duplication during batching and make up a batch
        by randomly sampling minor instances, thus making duplication of minor instances.
        Every batch will consume a certain number of major instances, determined by batch_size
        and major_instance_ratio.
        """
        major_num_per_batch = self._major_num_per_batch()
        major_beg_idx = major_num_per_batch * current_batch_idx
        # Use min to avoid out of bound in tail batch
        major_end_idx = min(major_num_per_batch * (current_batch_idx + 1), len(majority_instance_idxes))
        # Append  majority instances to batch first
        batch_major_idxes = majority_instance_idxes[major_beg_idx:major_end_idx]
        batch_idxes: List[int] = deepcopy(batch_major_idxes)

        if seed is not None:
            random.seed(seed)

        # Randomly sampling minority instances to make up an entire batch.
        # This operation implicitly allows for duplication among different batches.
        # In this way, majority and minority instances are clustered in the batch
        batch_minor_idxes = []
        while len(batch_minor_idxes) < self._batch_size - len(batch_idxes):
            sampled_idx = random.choice(minority_instance_idxes)
            if sampled_idx not in batch_minor_idxes:
                batch_minor_idxes.append(sampled_idx)

        batch_idxes.extend(batch_minor_idxes)
        return batch_idxes


    def _get_num_batches_from_major_idxes(self, major_instance_idxes: List[int]):
        """
        batch_num = instance_num / major_instance_per_batch + 1.
        """
        major_num_per_batch = self._major_num_per_batch()
        # Here assumes drop_last is 'False', so include tail batch
        batch_num = len(major_instance_idxes) // major_num_per_batch + 1
        return batch_num


    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        majority_instance_idxes, minority_instance_idxes = \
            self._get_majority_and_minority_idxes(instances, shuffle=True)
        batch_num = self._get_num_batches_from_major_idxes(majority_instance_idxes)

        for batch_i in range(batch_num):
            step_batch = self._sample_batch_idxes_step(
                batch_i,
                majority_instance_idxes,
                minority_instance_idxes
            )
            # mylogger.debug('get_batch_indices',
            #                f'batch_idxes: {[(idx, instances[idx].fields["label"].label) for idx in step_batch]}')
            yield step_batch


    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        major_instance_idxes, _ = self._get_majority_and_minority_idxes(instances)
        return self._get_num_batches_from_major_idxes(major_instance_idxes)


    def get_batch_size(self) -> Optional[int]:
        return self._batch_size


@BatchSampler.register('binary_duplicate_balanced_rand')
class BinaryDuplicateBalancedRandSampler(BinaryDuplicateBalancedIterSampler):
    """
    Sampler used in CC2Vec and DeepJIT models.
    """

    def _sample_batch_idxes_step(self,
                                 current_batch_idx: int,
                                 majority_instance_idxes: List[int],
                                 minority_instance_idxes: List[int],
                                 seed: int = None) -> List[int]:
        """
        We just randomly sample batch_size/2 instances from both majority and minority
        classes to generate a batch, without iterating majority class.
        This means different batches may have duplicated samples for both majority and
        minority samples, compared to iter sampler which only duplicate minority class.
        """
        if seed is not None:
            random.seed(seed)

        major_num_in_batch = self._major_num_per_batch()
        minor_num_in_batch = self._batch_size - major_num_in_batch
        minor_idxes = random.sample(minority_instance_idxes, minor_num_in_batch)
        major_idxes = random.sample(majority_instance_idxes, major_num_in_batch)
        return major_idxes + minor_idxes

    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        majority_instance_idxes, minority_instance_idxes = \
            self._get_majority_and_minority_idxes(instances)
        batch_num = self.get_num_batches(instances)

        for batch_i in range(batch_num):
            step_batch = self._sample_batch_idxes_step(
                batch_i,
                majority_instance_idxes,
                minority_instance_idxes
            )
            # mylogger.debug('get_batch_indices',
            #                f'batch_idxes: {[(idx, instances[idx].fields["label"].label) for idx in step_batch]}')
            yield step_batch

    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        """
        Because we no longer iterate majority class, batch_num only depends on
        number of instances and batch_size, allowing not-full iteration of all
        instances in an epoch.
        """
        return len(instances) // self._batch_size + 1
