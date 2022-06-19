import re

from allennlp.common.registrable import Registrable


class CodeCleaner(Registrable):
    def clean_code(self, code: str) -> str:
        raise NotImplementedError


@CodeCleaner.register('trivial')
@CodeCleaner.register('default')
class TrivialCodeCleaner(CodeCleaner):
    def clean_code(self, code: str) -> str:
        return code


@CodeCleaner.register('space_sub')
class SpaceSubCodeCleaner(CodeCleaner):
    """
    Replace multiple spaces and tabs(\s) with a single space.
    """
    def clean_code(self, code: str) -> str:
        # return re.sub(r'^\s+|\s+$|\s+(?=\s)', ' ', code)
        return re.sub(r' +|\t+', ' ', code)