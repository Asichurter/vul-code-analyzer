import _jsonnet
import json
import torch
import sys
import subprocess
from allennlp.commands.train import train_model_from_file

sys.path.extend(['/data1/zhijietang/projects/vul-code-analyzer'])

from utils.cmd_args import read_train_eval_from_config_args
from utils.file import dump_json
from utils import GlobalLogger as mylogger
from utils.cmd_args import make_cli_args

# For importing costumed modules
from downstream import *
from common import *

args = read_train_eval_from_config_args()

python_bin = '/data1/zhijietang/miniconda3/bin/python'
eval_script_path = f'/data1/zhijietang/projects/vul-code-analyzer/downstream/eval/{args.eval_script}.py'
converted_json_file_path = f'/data1/zhijietang/temp/config.json'
serialization_dir = '/data1/zhijietang/vul_data/run_logs/{}/{}/'

############### Do Preparations ###############
if args.add_rs:
    serialization_dir += 'rs_0'

config_json = json.loads(_jsonnet.evaluate_file(args.config))

extra = config_json.pop('extra')
version = extra.get('version')
assert version is not None
serialization_dir = serialization_dir.format(args.run_log_dir, version)

# add serial dir to callback parameters
for callback in config_json['trainer']['callbacks']:
    callback['serialization_dir'] = serialization_dir

dump_json(config_json, converted_json_file_path, indent=None)

mylogger.info('main', f"Args: {args}")

############### Do Train ###############
# Manually set cuda device to avoid additional memory usage bug on GPU:0
# See https://github.com/pytorch/pytorch/issues/66203
cuda_device = config_json.get('trainer').get('cuda_device')
cuda_device = int(cuda_device)
torch.cuda.set_device(cuda_device)

print('start to train from file...')
ret = train_model_from_file(
    converted_json_file_path,
    serialization_dir,
    force=True,
    # include_package=['core'],
)
# Try release GPU memory
del ret
torch.cuda.empty_cache()

############### Do Test ###############
for test_model_file_name in args.test_model_names.split(','):
    patience = 5
    mylogger.info('main', f'Start to test File {test_model_file_name}')
    test_cmd_args = {
        'data_base_path': args.data_base_path,
        'run_log_dir': args.run_log_dir,
        'version': int(version),
        'model_name': test_model_file_name,
        'data_file_name': args.data_file_name,
        'cuda': cuda_device,
        'average': args.average,
        'extra_averages': args.extra_averages,
        **json.loads(args.extra_eval_configs)
    }
    two_bar_args = {'dump_scores': ""}
    test_cmd_arg_str = make_cli_args(test_cmd_args, two_bar_args, skip_none=True)
    test_cmd = f"{python_bin} {eval_script_path}{test_cmd_arg_str}"

    # Try multiple times for test script running
    while patience > 0:
        try:
            subprocess.run(test_cmd, shell=True, check=True)
            torch.cuda.empty_cache()
            break
        except subprocess.CalledProcessError as e:
            mylogger.error('test', f'Error when run test script, err: {e}')
            patience -= 1
    if patience == 0:
        mylogger.error('test', f'Fail to run test, patience runs out. Cmd: {test_cmd}')

# Exit to release GPU memory
sys.exit(0)