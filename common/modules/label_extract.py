from typing import Dict

from allennlp.common.registrable import Registrable

class LabelExtractor(Registrable):
    def extract_label(self, obj: Dict):
        raise NotImplementedError

@LabelExtractor.register('key')
class KeyLabelExtractor(LabelExtractor):
    def __init__(self, key: str):
        self.key = key

    def extract_label(self, obj: Dict):
        return obj[self.key]

@LabelExtractor.register('treevul_cwe')
class TreeVulCweLabelExtractor(LabelExtractor):
    def __init__(self, cwe_layer: int = 0):
        self.cwe_layer = cwe_layer

    def extract_label(self, obj: Dict):
        return obj['cwe_path'][0][self.cwe_layer]