

from allennlp.models import Model

from downstream import *

model_archive_path = '/data1/zhijietang/vul_data/run_logs/reveal_vul_predict/1/model.tar.gz'

model = Model.from_archive(model_archive_path)