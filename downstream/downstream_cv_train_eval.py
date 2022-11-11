from copy import deepcopy

import _jsonnet
import json
import torch
import sys
from allennlp.commands.train import train_model_from_file
import subprocess

sys.path.extend(['/data1/zhijietang/projects/vul-code-analyzer'])

from utils.cmd_args import read_reveal_cv_train_from_config_args, make_cli_args
from utils.file import dump_json, dump_text, load_text
from utils import GlobalLogger as mylogger
from downstream.scripts.aggre_multi_results import count_mean_metrics

# For importing costumed modules
from downstream import *
from common import *

args = read_reveal_cv_train_from_config_args()
data_file_name = args.test_file_name

converted_json_file_path = '/data1/zhijietang/temp/config.json'
temp_jsonnet_path = '/data1/zhijietang/temp/temp.jsonnet'
python_bin = '/data1/zhijietang/miniconda3/bin/python'
# eval_script_path = '/'.join(__file__.split('/')[:-1]) + f'/eval/{args.eval_script}.py'
eval_script_path = f'/data1/zhijietang/projects/vul-code-analyzer/downstream/eval/{args.eval_script}.py'

for split in range(args.cv):
    serialization_dir = f'/data1/zhijietang/vul_data/run_logs/{args.run_log_dir}/{args.version}/rs_{split}'
    split_jsonnet_config = load_text(args.config)
    split_jsonnet_config = split_jsonnet_config.replace('local split_index = 0;', f'local split_index = {split};')
    split_config = json.loads(_jsonnet.evaluate_snippet("", split_jsonnet_config))
    split_config.pop('extra')

    for callback in split_config['trainer']['callbacks']:
        callback['serialization_dir'] = serialization_dir
    dump_json(split_config, converted_json_file_path, indent=None)

    # Manually set cuda device to avoid additional memory usage bug on GPU:0
    # See https://github.com/pytorch/pytorch/issues/66203
    cuda_device = split_config.get('trainer').get('cuda_device')
    cuda_device = int(cuda_device)
    torch.cuda.set_device(cuda_device)

    mylogger.debug('train_from_config',
                   f'run_log_dir={args.run_log_dir}, ver={args.version}, split={split}, '
                   f'cuda_device={cuda_device}, serial_dir={serialization_dir}')

    if not args.no_train:
        print('start to train from file...')
        ret = train_model_from_file(
            converted_json_file_path,
            serialization_dir,
            force=True,
            file_friendly_logging=not args.disable_friendly_logging,
            # include_package=['core'],   # all needed models can be imported within a single __init__ of a module
        )
        del ret
        torch.cuda.empty_cache()

    for test_model_file_name in args.test_filenames.split(','):
        patience = 5
        mylogger.info('reveal_cv_helper', f'Start to test Version {args.version}, Split {split}, File {test_model_file_name}')
        test_cmd_args = {
            'version': args.version,
            'subfolder': args.subfolder,
            'subset': f'split_{split}',
            'model_name': test_model_file_name,
            'data_file_name': data_file_name,
            'run_log_dir': args.run_log_dir,
            'split': split,
            'cuda': cuda_device
        }
        test_cmd_arg_str = make_cli_args(test_cmd_args, {})
        test_cmd = f"{python_bin} {eval_script_path}{test_cmd_arg_str}"
        # test_cmg = f"{python_bin} {eval_script_path} -version {args.version} -subfolder {args.subfolder} -subset split_{split} " \
        #            f"-model_name {test_model_file_name} -data_file_name {data_file_name} -run_log_dir {args.run_log_dir} " \
        #            f"-split {split} -cuda {cuda_device}"

        # Try multiple times for test script running
        while patience > 0:
            try:
                subprocess.run(test_cmd, shell=True, check=True)
                torch.cuda.empty_cache()
                break
            except subprocess.CalledProcessError as e:
                mylogger.error('test', f'Error when run test script for split {split}, err: {e}')
                patience -= 1
        if patience == 0:
            mylogger.error('test', f'Fail to run test for split {split}, patience runs out.')

count_mean_metrics(args.run_log_dir, args.version, args.title, args.cv)

# Exit to release GPU memory
sys.exit(0)