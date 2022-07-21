import argparse
from gensim.models import Word2Vec
import json
import os


def train(args):
    data_base_path = args.data_base_path
    files = args.data_files
    sentences = []
    for f in files:
        data = json.load(open(os.path.join(data_base_path, f)))
        for e in data:
            code = e['code']
            sentences.append([token.strip() for token in code.split()])
    wvmodel = Word2Vec(sentences, min_count=args.min_occ, workers=8, vector_size=args.embedding_size)
    print('Embedding Size : ', wvmodel.vector_size)
    # for i in range(args.epochs):
    wvmodel.train(sentences, total_examples=len(sentences), epochs=args.epochs)
    if not os.path.exists(args.save_model_dir):
        os.mkdir(args.save_model_dir)
    save_file_path = os.path.join(args.save_model_dir, args.model_name)
    wvmodel.save(save_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_path', type=str, required=True)
    parser.add_argument('--data_files', type=str, nargs='+', default=['train.json', 'validate.json', 'test.json'])
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('-bin', '--save_model_dir', type=str, required=True)
    parser.add_argument('-n', '--model_name', type=str, required=True)
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    args = parser.parse_args()
    train(args)