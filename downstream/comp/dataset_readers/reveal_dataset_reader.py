import torch
from typing import Iterable, Dict

from allennlp.data import Instance, Tokenizer, TokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField

from common.code_cleaner import CodeCleaner, TrivialCodeCleaner
from utils.file import load_json

class RevealDatasetReader(DatasetReader):
    def __init__(self,
                 code_tokenizer: Tokenizer,
                 code_indexer: TokenIndexer,
                 # max_lines: int,
                 code_max_tokens: int,
                 code_namespace: str = "code_tokens",
                 code_cleaner: CodeCleaner = TrivialCodeCleaner(),
                 **kwargs):
        super().__init__(**kwargs)
        # self.max_lines = max_lines
        self.code_tokenizer = code_tokenizer
        self.code_indexers = {code_namespace: code_indexer}  # or {"tokens": SingleIdTokenIndexer()}
        self.code_max_tokens = code_max_tokens
        self.code_cleaner = code_cleaner

    def text_to_instance(self, data_item: Dict) -> Instance:
        code = data_item['code']
        label = data_item['label']

        code = self.code_cleaner.clean_code(code)
        tokenized_code = self.code_tokenizer.tokenize(code)

        fields = {
            'code': TextField(tokenized_code, self.code_indexers),
            'label': TensorField(torch.LongTensor([label]))
        }
        return Instance(fields)


    def _read(self, file_path) -> Iterable[Instance]:
        data = load_json(file_path)
        for item in data:
            yield self.text_to_instance(item)