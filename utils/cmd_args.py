import argparse

def read_train_from_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='', help='config path of training')
    return parser.parse_args()

def read_reveal_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-version', type=int)
    parser.add_argument('-subset', type=str)
    parser.add_argument('-model_name', type=str, default='model.tar.gz')
    parser.add_argument('-data_file_name', type=str, default='test.json')
    parser.add_argument('-cuda', type=int)
    return parser.parse_args()