import torch
from typing import Iterable, Dict, Optional, List, Tuple
from tqdm import tqdm

from allennlp.data import Instance, Tokenizer, TokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, TextField, TensorField, ListField, LabelField

from common.modules.code_cleaner import CodeCleaner, TrivialCodeCleaner
from utils.downstream_utils.tokenize_utils import downstream_tokenize
from utils.file import read_dumped
from utils import GlobalLogger as mylogger

@DatasetReader.register('treevul_class_base')
class TreeVulClassBaseDatasetReader(DatasetReader):
    def __init__(self,
                 cwe_label_space: List[str],
                 code_tokenizer: Tokenizer,
                 code_indexer: TokenIndexer,
                 code_max_tokens: int,
                 code_namespace: str = "code_tokens",
                 code_cleaner: CodeCleaner = TrivialCodeCleaner(),
                 tokenizer_type: str = 'codebert',
                 model_mode: Optional[str] = None,
                 hunk_separator: str = '',
                 file_separator: str = '',
                 keep_hunk_hierarchy: bool = False,
                 keep_file_hierarchy: bool = True,      # Disabled now, default to True
                 debug: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.code_tokenizer = code_tokenizer
        self.code_indexers = {code_namespace: code_indexer}
        self.code_max_tokens = code_max_tokens
        self.code_cleaner = code_cleaner
        self.tokenizer_type = tokenizer_type
        self.model_mode = model_mode

        self.cwe_label_space = cwe_label_space
        self.cwe_label_map = {cwe:i for i,cwe in enumerate(cwe_label_space)}
        self.hunk_separator = hunk_separator
        self.file_separator = file_separator
        self.keep_hunk_hierarchy = keep_hunk_hierarchy
        self.keep_file_hierarchy = keep_file_hierarchy

        self.debug = debug
        self.total_read_instances = 0

    def _process_commit_files_as_field(self, files: List[Dict]) -> Optional[Field]:
        file_field_list = []
        for file in files:
            hunks_field = self._process_hunks_in_file(file)
            if hunks_field is not None:
                file_field_list.append(hunks_field)

        if len(file_field_list) == 0:
            return None
        else:
            return ListField(file_field_list)

    def _process_hunks_in_file(self, file) -> Optional[Field]:
        if self.keep_hunk_hierarchy:
            hunk_field_list = []
            # We only focus removed code to classify vul code into CWE classes
            for hunk in file['REM_DIFF']:
                if hunk != '':
                    hunk = self.code_cleaner.clean_code(hunk)
                    tokenized_hunk = downstream_tokenize(self.code_tokenizer, hunk, self.tokenizer_type, self.model_mode)
                    hunk_field = TextField(tokenized_hunk, self.code_indexers)
                    hunk_field_list.append(hunk_field)
            if len(hunk_field_list) == 0:
                return None
            else:
                return ListField(hunk_field_list)
        else:
            # Filter empty hunks
            valid_hunks = [hunk for hunk in file['REM_DIFF'] if hunk != '']
            if len(valid_hunks) == 0:
                return None
            else:
                # Hunks are concatenated with hunk separator
                hunks_str = self.hunk_separator.join(valid_hunks)
                hunks_str = self.code_cleaner.clean_code(hunks_str)
                tokenized_hunks = downstream_tokenize(self.code_tokenizer, hunks_str, self.tokenizer_type, self.model_mode)
                return TextField(tokenized_hunks, self.code_indexers)

    def text_to_instance(self, data_item: Dict) -> Tuple[bool, Optional[Instance]]:
        cwe_label = data_item['cwe_label']
        if cwe_label not in self.cwe_label_space:
            return False, None
        else:
            cwe_label_field = TensorField(torch.LongTensor([self.cwe_label_map[cwe_label]]))

        removed_code_field = self._process_commit_files_as_field(data_item['files'])
        # Skip insances where removed code is empty
        if removed_code_field is None:
            return False, None

        fields = {
            'code': removed_code_field,
            'label': cwe_label_field,
        }
        return True, Instance(fields)


    def _read(self, file_path) -> Iterable[Instance]:
        start_count = self.total_read_instances
        data = read_dumped(file_path)
        data = data[:150] if self.debug else data
        mylogger.info('reader', f'Total {len(data)} items in data before read')
        for item in tqdm(data):
            ok, instance =  self.text_to_instance(item)
            if ok:
                self.total_read_instances += 1
                yield instance
            else:
                pass
                # mylogger.debug('reader', f'Filter commit_id={item["cid"]}, CWE_label={item["cwe_label"]}')
        mylogger.info('reader', f'Total {self.total_read_instances - start_count} loaded by reader')