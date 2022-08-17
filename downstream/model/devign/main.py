import argparse
import os
import pickle
import sys

sys.path.extend(['/data1/zhijietang/projects/vul-code-analyzer'])

from downstream.model.feature_extraction.mlm_line_feature_extractor import MLMLineExtractor
from downstream.model.feature_extraction.mlm_line_feature_extractor_v2 import MLMLineExtractorV2
from downstream.model.feature_extraction.cls_feature_extractor import ClsExtractorV2
from pretrain import AvgLineExtractor

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from downstream.model.devign.data_loader.dataset import DataSet
from downstream.model.devign.modules.model import DevignModel, GGNNSum, GGNNSumNew, GGNNMeanResidual, \
    GGNNMeanMixedResidual
from downstream.model.devign.modules.e2e_model import GGNNMeanEnd2End, GGNNMeanEnd2EndV2, NodeMean, ClsPooler
from downstream.model.devign.trainer import train
from downstream.model.devign.devign_utils import tally_param, debug
# from downstream.model.devign.devign_global_flag import global_cuda_device
from utils.stat import stat_model_param_number
from utils.seed import seed_everything

def create_node_feature_extractor(cuda_device):
    model_path = '/data1/zhijietang/vul_data/run_logs/pretrain/15/model.tar.gz'
    reader_config_path = '/data1/zhijietang/vul_data/run_logs/pretrain/15/config.json'
    overwrite_reader_config = {
        'type': 'raw_pdg_predict',
        'max_lines': 50,
        'code_max_tokens': 256,
        'code_tokenizer': {'max_length': 256},
        'identifier_key': None,
        # 'meta_data_keys': {'edges': 'edges', 'vulnerable': 'label', 'file': 'file'}
    }
    delete_reader_config = {
        'from_raw_data': 1,
        'pdg_max_vertice': 1
    }
    line_extractor = AvgLineExtractor(max_lines=50)
    # extractor = MLMLineExtractor(model_path, reader_config_path, line_extractor,
    #                              overwrite_reader_config, delete_reader_config,
    #                              cuda_device=global_cuda_device,
    #                              frozen=False)
    extractor = MLMLineExtractorV2(model_path, reader_config_path, line_extractor,
                                   overwrite_reader_config, delete_reader_config,
                                   cuda_device=cuda_device)
    return extractor

def create_cls_feature_extractor(cuda_device):
    model_path = '/data1/zhijietang/vul_data/run_logs/pretrain/15/model.tar.gz'
    reader_config_path = '/data1/zhijietang/vul_data/run_logs/pretrain/15/config.json'
    overwrite_reader_config = {
        'type': 'raw_pdg_predict',
        'max_lines': 50,
        'code_max_tokens': 256,
        'code_tokenizer': {'max_length': 256},
        'identifier_key': None,
        # 'meta_data_keys': {'edges': 'edges', 'vulnerable': 'label', 'file': 'file'}
    }
    delete_reader_config = {
        'from_raw_data': 1,
        'pdg_max_vertice': 1
    }
    extractor = ClsExtractorV2(model_path, reader_config_path,
                               overwrite_reader_config, delete_reader_config,
                               cuda_device=cuda_device)
    return extractor

if __name__ == '__main__':
    # torch.manual_seed(1000)
    # np.random.seed(1000)
    seed_everything(1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn', 'ggnn_new', 'ggnn_res', 'ggnn_mixed_res',
                                 'ggnn_end2end', 'node_mean', 'cls'], default='devign')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset for experiment.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the parser')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='target')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=128)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--patience', type=int, help='Early stop patience', default=100)
    parser.add_argument('--res_forward', type=bool, help='Add residual feature when forward', default=False)
    parser.add_argument('--dynamic_node', type=bool, help='Whether using dynamicn node features', default=True)
    parser.add_argument('--max_steps', type=int, default=1000000, help='Maximum training steps')
    parser.add_argument('--do_val', action='store_true', help='Do validation during training')
    parser.add_argument('--dump_key', type=str, required=True, help='Key to store when dumping results')
    parser.add_argument('--cuda', type=int, default=0, help='Cuda device')

    # parser.add_argument('--cuda', type=int, help='Cuda device', default=0)
    args = parser.parse_args()
    cuda_device = args.cuda

    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, 'processed.bin')
    if False and os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    else:
        dataset = DataSet(train_src=os.path.join(input_dir, 'train.pkl'), # 'train_GGNNinput.json'),
                          valid_src=os.path.join(input_dir, 'validate.pkl'), #  'valid_GGNNinput.json'),
                          test_src=os.path.join(input_dir, 'test.pkl'), #  'test_GGNNinput.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()
    # assert args.feature_size == dataset.feature_size, \
    #     f'Dataset contains different feature vector ({dataset.feature_size}) than argument feature size ({args.feature_size}). ' \
    #     'Either change the feature vector size in argument, or provide different dataset.'
    if args.model_type == 'ggnn':
        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    elif args.model_type == 'devign':
        model = DevignModel(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    elif args.model_type == 'ggnn_new':
        model = GGNNSumNew(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                           num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    elif args.model_type == 'ggnn_res':
        model = GGNNMeanResidual(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                                 num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    elif args.model_type == 'ggnn_mixed_res':
        model = GGNNMeanMixedResidual(input_dim=args.feature_size, output_dim=args.graph_embed_size,
                                      num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    elif args.model_type == 'ggnn_end2end':
        extractor = create_node_feature_extractor(cuda_device)
        model = GGNNMeanEnd2EndV2(extractor,
                                  input_dim=args.feature_size, output_dim=args.graph_embed_size,
                                  num_steps=args.num_steps, max_edge_types=dataset.max_edge_type,
                                  residual_forward=args.res_forward,
                                  dynamic_node_features=args.dynamic_node)
    elif args.model_type == 'node_mean':
        extractor = create_node_feature_extractor(cuda_device)
        model = NodeMean(extractor,
                         input_dim=args.feature_size, output_dim=args.graph_embed_size,
                         num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    elif args.model_type == 'cls':
        extractor = create_cls_feature_extractor(cuda_device)
        model = ClsPooler(extractor,
                          input_dim=args.feature_size, output_dim=args.graph_embed_size,
                          num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    else:
        raise ValueError(f'Unsupported model_type: {args.model_type}')


    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    stat_model_param_number(model)
    debug('#' * 100)
    model.cuda(cuda_device)
    loss_function = BCELoss(reduction='sum')
    optim = Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    # if args.model_type == 'ggnn_end2end':
    #     # Add parameters of line feature extractor to optimizer
    #     optim.add_param_group({
    #         'params': model.feature_extractor._model.parameters(),
    #         'lr': 1e-5
    #     })

    train(model=model, dataset=dataset, max_steps=args.max_steps, dev_every=128,
          loss_function=loss_function, optimizer=optim,
          save_path=model_dir + f'/{args.model_type}', max_patience=args.patience, log_every=None,
          do_val=args.do_val,
          dump_key=args.dump_key,
          cuda_device=cuda_device)
