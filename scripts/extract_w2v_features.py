import gensim
import os
import numpy
from tqdm import tqdm
import re

from utils.file import load_json, dump_pickle
from utils import GlobalLogger as mylogger

w2v_model_path = '/data1/zhijietang/vul_data/datasets/reveal/small/random_split/split_1/w2v.pkl'
src_data_base_path = '/data1/zhijietang/vul_data/datasets/reveal/small/random_split/split_1/'
tgt_data_dump_path = '/data1/zhijietang/vul_data/datasets/reveal/small/devign/w2v/rs_1'
embed_size = 300

data_files = ['train', 'validate', 'test']
line_feature_pooling = 'avg'

def adapt_devign_edge_format(edges, line_count):
    devign_edges = []
    for edge in edges:
        tail, head, etype = re.split(',| ', edge)
        tail, head, etype = int(tail), int(head), int(etype)
        if tail >= line_count or head >= line_count:
            continue
        if etype == 1 or etype == 3:
            devign_edges.append([tail, 0, head])
        if etype == 2 or etype == 3:
            devign_edges.append([tail, 1, head])

    return devign_edges

if __name__ == '__main__':
    w2v = gensim.models.Word2Vec.load(w2v_model_path)
    vocab = w2v.wv.key_to_index
    embeddings = w2v.wv.vectors

    for data_split in data_files:
        src_data_file = data_split + '.json'
        src_datas = load_json(os.path.join(src_data_base_path, src_data_file))
        tgt_datas = []
        print(data_split)
        for src_data in tqdm(src_datas):
            code = src_data['code']
            line_features = []
            lines = code.split('\n')    # Note using \n to split means \n will not appear in tokens
            for li, line in enumerate(lines):
                line_tokens = [t.strip() for t in line.split()]
                line_feature = numpy.zeros((embed_size,))
                for line_token in line_tokens:
                    if line_token in vocab:
                        line_feature += embeddings[vocab[line_token]]
                    else:
                        mylogger.warning('main', f'Token {line_token} not in vocab')

                if line_feature_pooling == 'avg':
                    if len(line_tokens) != 0:
                        line_feature /= len(line_tokens)
                    else:
                        mylogger.warning('main', f'Empty line detected: Line-idx: {li}, file: {src_data["file"]}')
                line_features.append(line_feature)

            line_features = numpy.stack(line_features, axis=0)
            graph = adapt_devign_edge_format(src_data['edges'], src_data['total_line'])
            if len(graph) < 3:
                mylogger.warning('main', f'file: {src_data["file"]} has less than 3 edges.')
            tgt_data = {
                'node_features': line_features,
                'graph': graph,
                'node_count': src_data['total_line'],
                'target': src_data['vulnerable'],
                'id': src_data['file'],
                'code': code
            }
            tgt_datas.append(tgt_data)

        tgt_file_name = data_split + '.pkl'
        dump_pickle(tgt_datas, os.path.join(tgt_data_dump_path, tgt_file_name))


