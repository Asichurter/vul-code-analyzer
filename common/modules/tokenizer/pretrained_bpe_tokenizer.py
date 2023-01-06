from typing import List, Dict, Any, Optional

from allennlp.data import Token
from overrides import overrides
from tokenizers import Tokenizer as RealTokenizer
from allennlp.data.tokenizers import Tokenizer


@Tokenizer.register('pretrained_bpe')
class PretrainedBPETokenzizer(Tokenizer):
    def __init__(self,
                 tokenizer_config_path: str,
                 max_length: int = 512,
                 start_token: Optional[str] = None,
                 end_token: Optional[str] = None):
        self._tokenizer = RealTokenizer.from_file(tokenizer_config_path)
        self._max_length = max_length if max_length != -1 else 1e10
        self._start_token = start_token
        self._end_token = end_token

    def _maybe_add_start_end_tokens(self, tokens: List[Token]):
        if self._start_token is not None:
            tokens.insert(0, Token(self._start_token))
        if self._end_token is not None:
            tokens.append(Token(self._end_token))
        return tokens

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        raw_tokens = self._tokenizer.encode(text)
        tokens = []
        for token in raw_tokens.tokens[:self._max_length]:
            tokens.append(Token(token))
        tokens = self._maybe_add_start_end_tokens(tokens)
        return tokens

    def _to_params(self) -> Dict[str, Any]:
        return {'type': 'pretrained_bpe'}
