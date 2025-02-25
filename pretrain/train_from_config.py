import _jsonnet
import json
import torch
import sys
import json
import platform
from allennlp.commands.train import train_model_from_file

base_dir = json.load(open('global_vars.json'))[platform.node()]['base_dir']

sys.path.extend([f'/{base_dir}/zhijietang/projects/vul-code-analyzer'])

from utils.cmd_args import read_train_from_config_args
from utils.file import dump_json
from utils import GlobalLogger as mylogger

# For importing costumed modules
from pretrain import *
from common import *

args = read_train_from_config_args()

converted_json_file_path = f'/{base_dir}/zhijietang/temp/config.json'
serialization_dir = f'/{base_dir}' + '/zhijietang/vul_data/run_logs/pretrain/{}'

config_json = json.loads(_jsonnet.evaluate_file(args.config))

extra = config_json.pop('extra')
version = extra.get('version')
assert version is not None
serialization_dir = serialization_dir.format(version)

# add serial dir to callback parameters
for callback in config_json['trainer']['callbacks']:
    callback['serialization_dir'] = serialization_dir

dump_json(config_json, converted_json_file_path, indent=None)

# Manually set cuda device to avoid additional memory usage bug on GPU:0
# See https://github.com/pytorch/pytorch/issues/66203
cuda_device = config_json.get('trainer').get('cuda_device')
cuda_device = int(cuda_device)
torch.cuda.set_device(cuda_device)

mylogger.debug('train_from_config',
               f'Ver. = {version}, cuda_device = {cuda_device}')

# Instead of this python code, you would typically just call
# allennlp train [config_file] -s [serialization_dir]
print('start to train from file...')
ret = train_model_from_file(
    converted_json_file_path,
    serialization_dir,
    force=True,
    file_friendly_logging=not args.disable_friendly_logging,
    # include_package=['core'],   # all needed models can be imported within a single __init__ of a module
)
# Exit to release GPU memory
sys.exit(0)