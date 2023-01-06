import re
from typing import List, Dict, Any, Optional

from allennlp.data import Token
from overrides import overrides
from allennlp.data.tokenizers import Tokenizer

from utils.allennlp_utils.tokenizer import tokenize_text


@Tokenizer.register('simple_split')
class SimpleSplitTokenizer(Tokenizer):
    """
    Tokenizing by splitting punctuations, camel case and other simple strategies.
    """
    def __init__(self,
                 max_length: int = 512,
                 start_token: Optional[str] = None,
                 end_token: Optional[str] = None):
        self._max_length = max_length if max_length != -1 else 1e10
        self._start_token = start_token
        self._end_token = end_token

    def _tokenize(self, text: str) -> List[Token]:
        new_text = tokenize_text(text)
        tokens = [Token(t) for t in new_text]
        return tokens

    def _maybe_add_start_end_tokens(self, tokens: List[Token]):
        if self._start_token is not None:
            tokens.insert(0, Token(self._start_token))
        if self._end_token is not None:
            tokens.append(Token(self._end_token))
        return tokens

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        tokens = self._tokenize(text)
        tokens = self._maybe_add_start_end_tokens(tokens)
        return tokens

    def _to_params(self) -> Dict[str, Any]:
        return {'type': 'simple_split'}

