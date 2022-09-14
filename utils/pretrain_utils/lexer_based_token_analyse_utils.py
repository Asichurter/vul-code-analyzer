from functools import reduce
from typing import List
import re

from pygments.lexers.c_cpp import CppLexer

cpp_lexer = CppLexer()

def get_filtered_token_span_by_cpplexer(raw_code: str, filtered_types: List[str]):
    """
    Given raw c/cpp code, analyze with cpp lexer and return
    char spans that are not filtered by given token types.
    """
    lexer_tokens = list(cpp_lexer.get_tokens_unprocessed(raw_code))
    not_masked_spans = []
    lexer_token_char_span = [t[0] for t in lexer_tokens] + [len(raw_code)]
    for i,token in enumerate(lexer_tokens):
        token_type_str = str(token[1])
        should_filtered = reduce(lambda v,e: v or re.match(e, token_type_str) is not None, filtered_types, False)
        if not should_filtered:
            # Left-close right-open span
            not_masked_spans.append((lexer_token_char_span[i], lexer_token_char_span[i+1]))
    return not_masked_spans