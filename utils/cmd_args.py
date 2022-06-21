import argparse

def read_train_from_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='', help='config path of training')
    return parser.parse_args()