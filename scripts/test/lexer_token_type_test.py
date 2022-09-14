from pygments.lexers.c_cpp import CppLexer
import os
from tqdm import tqdm
import re

from utils.file import load_text

cpp_lexer = CppLexer()
code_file_base_paths = [
    '/data1/zhijietang/vul_data/datasets/docker/cppfiles/vol100/',
    '/data1/zhijietang/vul_data/datasets/docker/cppfiles/vol200/',
]


def clean_code(code: str) -> str:
    # return re.sub(r'^\s+|\s+$|\s+(?=\s)', ' ', code)
    code = re.sub(r' +|\t+', ' ', code)
    return code

last_token_example = {}

type_set = set()
for code_file_base_path in code_file_base_paths:
    for item in tqdm(os.listdir(code_file_base_path)):
        fp = os.path.join(code_file_base_path, item)
        code_content = load_text(fp)
        code_content = clean_code(code_content)
        lexer_tokens = list(cpp_lexer.get_tokens_unprocessed(code_content))
        for token in lexer_tokens:
            type_set.add(str(token[1]))
            last_token_example[str(token[1])] = token[-1]
            if str(token[1]) == 'Token.Error':
                print(token)
