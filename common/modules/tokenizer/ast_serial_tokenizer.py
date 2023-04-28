import re
from typing import List, Dict, Any, Optional

from allennlp.data import Token
from overrides import overrides
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer

from utils.pretrain_utils.ast_utils import dispatch_ast_serial_tokenize_method
from utils.data_utils.changed_func_extraction import parse_tree
from utils import GlobalLogger as mylogger


@Tokenizer.register('ast_serial')
class ASTSerialTokenizer(Tokenizer):
    """
        Given code text, this tokenizer first parse it into AST with tree_sitter,
        and serialize AST into sequence of AllenNLP tokens.

        This tokenizer will call outer implementations of serialization and tokenization,
        and acts only as a wrapper.
    """
    def __init__(self,
                 ast_serial_tokenize_method: str,
                 real_tokenizer: Tokenizer,
                 max_length: int = 512,
                 start_token: Optional[str] = "<s>",
                 end_token: Optional[str] = "</s>",
                 ast_serial_tokenize_params: Dict = {},
                 **kwargs):
        self._max_length = max_length if max_length != -1 else 1e10
        self._start_token = start_token
        self._end_token = end_token

        self._real_tokenizer = real_tokenizer
        self._ast_serial_tokenize_method = dispatch_ast_serial_tokenize_method(ast_serial_tokenize_method)
        self._ast_serial_tokenize_params = ast_serial_tokenize_params

        # Simple check
        if isinstance(real_tokenizer, PretrainedTransformerTokenizer):
            if real_tokenizer._add_special_tokens:
                mylogger.warning("PretrainedASTSerialTokenizer",
                                 f"'_add_special_tokens' is enabled for real tokenizer, this will bring in unexpected special tokens among the serialized sequence")

    def _tokenize(self, text: str) -> List[Token]:
        # Parse AST
        tree = parse_tree(text)
        # Serialize AST as sequence
        tokens = self._ast_serial_tokenize_method(tree.root_node, self._real_tokenizer, **self._ast_serial_tokenize_params)
        # Truncate by max_length
        truncate_tokens = tokens[:self._max_length]
        return truncate_tokens

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
        return {'type': 'ast_serial',
                'ast_serial_tokenize_method': self._ast_serial_tokenize_method,
                'max_length': self._max_length,
                'ast_serial_tokenize_params': self._ast_serial_tokenize_params}

