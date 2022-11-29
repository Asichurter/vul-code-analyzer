import argparse
from typing import Dict


def make_cli_args(one_bar_args: Dict, two_bar_args: Dict):
    args = ''
    for argkey, argval in one_bar_args.items():
        if argval is None:
            arg = f' -{argkey}'
        else:
            arg = f' -{argkey} {argval}'
        args += arg
    for argkey, argval in two_bar_args.items():
        if argval is None:
            arg = f' --{argkey}'
        else:
            arg = f' --{argkey} {argval}'
        args += arg
    return args

def read_train_from_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='', help='config path of training')
    parser.add_argument('-run_log_dir', type=str, default='reveal_new', help='config path of training')
    parser.add_argument('--disable_friendly_logging', action='store_true', help='whether to disable file_friendly_logging of allennlp.train')
    return parser.parse_args()

def read_reveal_cv_train_from_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='config path of training')
    parser.add_argument('-run_log_dir', type=str, default='reveal_new', help='config path of training')
    parser.add_argument('-version', type=str, required=True, help='subfoler name under run log directory')
    parser.add_argument('-cv', type=int, default=5, help='number of cross-validation')
    parser.add_argument('-subfolder', required=True, type=str)
    parser.add_argument('-test_filenames', type=str, default='model.tar.gz', help='model file names after training, splitted by comma')
    parser.add_argument('-eval_script', type=str, default='evaluate_reveal.py', help='model file names after training, splitted by comma')
    parser.add_argument('-title', type=str, default='', help='title for reporting cv results')
    parser.add_argument('-test_file_name', type=str, default='test.json', help='filename of the tested data')
    parser.add_argument('--no_train', action='store_true', help='whether do train from scratch')
    parser.add_argument('--disable_friendly_logging', action='store_true', help='whether to disable file_friendly_logging of allennlp.train')
    return parser.parse_args()

def read_reveal_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', required=True, type=int)
    parser.add_argument('-subset', required=True, type=str, help='random split folder')
    parser.add_argument('-subfolder', required=True, type=str)
    parser.add_argument('-model_name', type=str, default='model.tar.gz')
    parser.add_argument('-data_file_name', type=str, default='test.json')
    parser.add_argument('-run_log_dir', default='reveal_new', type=str)
    parser.add_argument('-split', required=True, type=str)
    parser.add_argument('-cuda', type=int)
    return parser.parse_args()

def read_fan_vul_det_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', required=True, type=int)
    parser.add_argument('-subfolder', required=True, type=str, help="folder of which format, such as `splits`")
    parser.add_argument('-subset', required=True, type=str, help='which split to use')
    parser.add_argument('-model_name', type=str, default='model.tar.gz')
    parser.add_argument('-data_file_name', type=str, default='test.pkl')
    parser.add_argument('-run_log_dir', default='fan_vul_det', type=str)
    parser.add_argument('-cuda', type=int)
    parser.add_argument('-batch_size', default=64, type=int)
    return parser.parse_args()

def read_devign_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', required=True, type=int)
    parser.add_argument('-subfolder', required=True, type=str, help="folder of which format, such as `splits`")
    parser.add_argument('-subset', required=True, type=str, help='which split to use')
    parser.add_argument('-split', required=True, type=str)
    parser.add_argument('-model_name', type=str, default='model.tar.gz')
    parser.add_argument('-data_file_name', type=str, default='test.json')
    parser.add_argument('-run_log_dir', default='devign', type=str)
    parser.add_argument('-cuda', type=int)
    parser.add_argument('-batch_size', default=64, type=int)
    return parser.parse_args()

def read_treevul_classification_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', required=True, type=int)
    parser.add_argument('-subfolder', required=True, type=str, help="folder of which format, such as `splits`")
    parser.add_argument('-subset', required=True, type=str, help='which split to use')
    parser.add_argument('-split', required=True, type=str)
    parser.add_argument('-model_name', type=str, default='model.tar.gz')
    parser.add_argument('-data_file_name', type=str, default='test.json')
    parser.add_argument('-run_log_dir', default='treevul', type=str)
    parser.add_argument('-cuda', type=int)
    parser.add_argument('-batch_size', default=32, type=int)

    parser.add_argument('-average', type=str, default='macro')
    return parser.parse_args()

def read_aggre_eval_results_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', type=str)
    parser.add_argument('-run_log_dir', type=str)
    parser.add_argument('-title', type=str)
    return parser.parse_args()