import re

from allennlp.common.registrable import Registrable


class CodeCleaner(Registrable):
    def basic_code_process(self, code: str):
        # Add missing new_line char to last line.
        if code[-1] != '\n':
            code += '\n'
        return code

    def clean_code(self, code: str) -> str:
        raise NotImplementedError


@CodeCleaner.register('trivial')
@CodeCleaner.register('default')
class TrivialCodeCleaner(CodeCleaner):
    def clean_code(self, code: str) -> str:
        code = self.basic_code_process(code)
        return code

@CodeCleaner.register('pre_line_truncate')
class PreLineTruncateCodeCleaner(CodeCleaner):
    def __init__(self, max_line: int):
        self.max_line = max_line

    def clean_code(self, code: str) -> str:
        truncate_code = '\n'.join(code.split('\n')[:self.max_line])
        truncate_code = self.basic_code_process(truncate_code)
        return truncate_code


@CodeCleaner.register('space_sub')
class SpaceSubCodeCleaner(CodeCleaner):
    """
    Replace multiple spaces and tabs(\s) with a single space.
    """
    def clean_code(self, code: str) -> str:
        code = self.basic_code_process(code)
        # return re.sub(r'^\s+|\s+$|\s+(?=\s)', ' ', code)
        code = re.sub(r' +|\t+', ' ', code)
        return code