import argparse
from typing import Dict


def make_cli_args(one_bar_args: Dict, two_bar_args: Dict, skip_none: bool = False):
    args = ''
    for argkey, argval in one_bar_args.items():
        if argval is None:
            if skip_none:
                continue
            arg = f' -{argkey}'
        else:
            arg = f' -{argkey} {argval}'
        args += arg
    for argkey, argval in two_bar_args.items():
        if argval is None:
            if skip_none:
                continue
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
    parser.add_argument('--add_rs', action='store_true', default=False, help='whether to add the rs_0 the tail of the serilization path')
    return parser.parse_args()

def read_train_eval_from_config_args():
    parser = argparse.ArgumentParser()
    # Neccessary configs
    parser.add_argument('-config', type=str, default='', help='config path of training')
    parser.add_argument('-run_log_dir', type=str, required=True, help='config path of training')
    parser.add_argument('-data_base_path', required=True, type=str)

    # Extra configs
    parser.add_argument('-eval_script', type=str, default='eval_classification', help='test script file to do test')
    parser.add_argument('-test_batch_size', default=32, type=int)
    parser.add_argument('-test_model_names', type=str, default='model.tar.gz', help="Model names to be tested, split by comma")
    parser.add_argument('-data_file_name', type=str, default='test.json')
    parser.add_argument('-average', required=True, type=str, help="average method for classification metric calculation")
    parser.add_argument('-extra_averages', default=None, type=str, help="Extra average methods, split by comma")
    parser.add_argument('-extra_eval_configs', default="{}", type=str, help="Json str to configure params to eval script")

    parser.add_argument('--dump_scores', action='store_true', default=False)
    parser.add_argument('--add_rs', action='store_true', default=False, help='whether to add the rs_0 the tail of the serilization path')
    return parser.parse_args()

def read_cv_train_from_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='config path of training')
    parser.add_argument('-run_log_dir', type=str, default='reveal_new', help='config path of training')
    parser.add_argument('-version', type=str, required=True, help='subfoler name under run log directory')
    parser.add_argument('-dataset', type=str, required=True, help='which dataset to test on')
    parser.add_argument('-average', type=str, required=True, help='average method for metric calculation, "binary", "macro" or "minor"...')
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

def read_classification_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', required=True, type=int)
    parser.add_argument('-run_log_dir', required=True, type=str)
    parser.add_argument('-data_base_path', default=None, type=str, help="You can specify this full path to ignore subfolder, subset options.")
    parser.add_argument('-dataset', default=None, type=str)
    parser.add_argument('-subfolder', default=None, type=str, help="folder of which format, such as `splits`")
    parser.add_argument('-subset', default=None, type=str, help='which split to use')
    parser.add_argument('-split', default=None, type=str)

    parser.add_argument('-model_name', type=str, default='model.tar.gz')
    parser.add_argument('-data_file_name', type=str, default='test.json')
    parser.add_argument('-cuda', type=int, default=0)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('--dump_scores', action='store_true', default=False)

    parser.add_argument('-average', required=True, type=str)
    parser.add_argument('-extra_averages', default=None, type=str, help="Extra average methods, split by comma")
    return parser.parse_args()

def read_multi_task_classification_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_log_dir', required=True, type=str)
    parser.add_argument('-version', required=True, type=int)
    parser.add_argument('-task_names', required=True, type=str, help="Task names joined with ','")
    parser.add_argument('-data_base_path', default=None, type=str, help="You can specify this full path to ignore subfolder, subset options.")

    # parser.add_argument('-dataset', required=True, type=str)
    # parser.add_argument('-subfolder', default=None, type=str, help="folder of which format, such as `splits`")
    # parser.add_argument('-subset', default=None, type=str, help='which split to use')

    parser.add_argument('-split', default=None, type=str)
    parser.add_argument('-cuda', type=int, default=0)
    parser.add_argument('-model_name', type=str, default='model.tar.gz')
    parser.add_argument('-data_file_name', type=str, default='test.json')
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('--dump_scores', action='store_true', default=False)
    parser.add_argument('--all_metrics', action='store_true', default=False)

    parser.add_argument('-average', type=str, default='macro')
    parser.add_argument('-extra_averages', default=None, type=str, help="Extra average methods, split by comma")
    return parser.parse_args()

def read_aggre_eval_results_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', type=str)
    parser.add_argument('-run_log_dir', type=str)
    parser.add_argument('-title', type=str)
    parser.add_argument('-cv', type=int, default=5)
    return parser.parse_args()

def read_pdg_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_log_dir', required=True, type=str)
    parser.add_argument('-version', required=True, type=int)
    parser.add_argument('-model_name', type=str, default='model.tar.gz')
    parser.add_argument('-data_base_path', required=True, type=str)
    parser.add_argument('-cuda', type=int, default=0)
    parser.add_argument('-vol_range', required=True, type=str, help="[left, right] format, left-close right-close interval")

    return parser.parse_args()









