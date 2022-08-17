import argparse

def read_train_from_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='', help='config path of training')
    return parser.parse_args()

def read_reveal_cv_train_from_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='config path of training')
    parser.add_argument('-run_log_dir', type=str, default='reveal_new', help='config path of training')
    parser.add_argument('-version', type=str, required=True, help='subfoler name under run log directory')
    parser.add_argument('-cv', type=int, default=5, help='number of cross-validation')
    parser.add_argument('-test_filenames', type=str, default='model.tar.gz', help='model file names after training, splitted by comma')
    parser.add_argument('-eval_filename', type=str, default='evaluate_reveal.py', help='model file names after training, splitted by comma')
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

def read_aggre_eval_results_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', type=str)
    parser.add_argument('-run_log_dir', type=str)
    parser.add_argument('-title', type=str)
    return parser.parse_args()