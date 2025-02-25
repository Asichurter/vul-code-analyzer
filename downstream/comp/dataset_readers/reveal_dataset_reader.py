import torch
from typing import Iterable, Dict, Optional
from tqdm import tqdm

from allennlp.data import Instance, Tokenizer, TokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, TensorField

from common.modules.code_cleaner import CodeCleaner, TrivialCodeCleaner
from utils.downstream_utils.tokenize_utils import downstream_tokenize
from utils.file import load_json


@DatasetReader.register('reveal_base')
class RevealBaseDatasetReader(DatasetReader):
    def __init__(self,
                 code_tokenizer: Tokenizer,
                 code_indexer: TokenIndexer,
                 # max_lines: int,
                 code_max_tokens: int,
                 code_namespace: str = "code_tokens",
                 code_cleaner: CodeCleaner = TrivialCodeCleaner(),
                 tokenizer_type: str = 'codebert',
                 model_mode: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        # self.max_lines = max_lines
        self.code_tokenizer = code_tokenizer
        self.code_indexers = {code_namespace: code_indexer}  # or {"tokens": SingleIdTokenIndexer()}
        self.code_max_tokens = code_max_tokens
        self.code_cleaner = code_cleaner
        self.tokenizer_type = tokenizer_type
        self.model_mode = model_mode

    def text_to_instance(self, data_item: Dict) -> Instance:
        code = data_item['code']
        label = data_item['vulnerable']

        code = self.code_cleaner.clean_code(code)
        tokenized_code = downstream_tokenize(self.code_tokenizer, code, self.tokenizer_type, self.model_mode)

        fields = {
            'code': TextField(tokenized_code, self.code_indexers),
            'label': TensorField(torch.LongTensor([label]))
        }
        return Instance(fields)


    def _read(self, file_path) -> Iterable[Instance]:
        data = load_json(file_path)
        for item in tqdm(data):
            yield self.text_to_instance(item)